#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimensions
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of attention heads
    int patch_dim; // patch size for each token
    float dropout; // dropout in the ffn layers
    int img_width; // width of the image
    int img_height; // height of the image
    int patch_width; // width of the patch
    int patch_height; // height of the patch
    int n_classes; // the number of classes for the classification head
} Config;

typedef struct {
    float* patch2dim_weights; // (patch_dim, dim)
    float* pos_emb_weights; // (num_patches, dim)
    float* norm_att_weights; // (layer, dim) norm weights
    float* norm_ffn_weights; // (layer, dim) ffn weights
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; //(layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_heads * head_size)
    float* wv; // (layer, dim, n_heads * head_size)
    float* wo; // (layer, dim, n_heads * head_size)
    // floats for the ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, hidden_dim, dim)
    float* w_cls_token; // (dim,)
    // classifier weights for the logits, on the last layers
    float* wcls;
} TransformerWeights;

typedef struct {
    float* x; // activation at current time stamp (dim,)
    float* xb; // same, but inside a residual branch (dim,)
    float* xb2; // an additional buffer just for convenience (dim,)
    float* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float* q; // query (dim,)
    float* k; // key (dim,)
    float* v; // value (dim,)
    float* att; // buffer for scores/attention values (n_heads, seq_len)
    float* logits; // ouput logits
    // kv cache
    float* key_cache; // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;


typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffer for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));

    int num_patches = (int) (p->img_width / p->patch_width) * (p->img_height / p->patch_height);
    s->key_cache = calloc(p->n_layers * num_patches * p->dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * num_patches * p->dim, sizeof(float));
    s->att = calloc(p->n_layers * num_patches, sizeof(float));
    s->logits = calloc(p->n_classes, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->key_cache
        || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr) {
    // make sure the multiplication below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    int head_size = p->dim / p->n_heads;
    int num_patches = (int) (p->img_width / p->patch_width) * (p->img_height / p->patch_height);

    w->patch2dim_weights = ptr;
    ptr += p->patch_dim * p->dim;
    w->pos_emb_weights = ptr;
    ptr += (num_patches + 1) * p->dim;
    w->w_cls_token = ptr;
    ptr += p->dim;
    w->norm_att_weights = ptr;
    ptr += p->n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->norm_ffn_weights = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->wcls = ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
        int* fd, float** data, ssize_t* file_size) {
    FILE* file = fopen(checkpoint, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file%s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    float* weights_ptr = *data + sizeof(Config) / sizeof(float);
    memory_map_weights(weights, config, weights_ptr);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_tranformer(Transformer* t) {
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// Neural net blocks; the dynamics of the Transformer
void layer_norm(float* o, float* x, float* weight, int size) {
    // calculate the mean
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i];
    }
    float mean = ss / size;
    // now calculate the std deviation
    ss = 0.0f;
    for (int i = 0; i < size; i++) {
        float numerator = x - mean;
        numerator *= numerator;
        ss += numerator;
    }
    float std_dev = sqrtf(ss / size);
    for (int i = 0; i < size; i++) {
        o[i] = (x[i] - mean) / sqrtf(std_dev);
    }
}
