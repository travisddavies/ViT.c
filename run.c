#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <fcntl.h>
#define PI 3.141592654
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
    int img_channels; // the number of channels in the input image
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
    ptr += num_patches * p->dim;
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
void layer_norm(float* o, float* x, int size) {
    // calculate the mean
    float ss = 0.0f;
    const float epsilon = 1e-5f;
    for (int i = 0; i < size; i++) {
        ss += x[i];
    }
    float mean = ss / size;
    // now calculate the std deviation
    ss = 0.0f;
    for (int i = 0; i < size; i++) {
        float numerator = x[i] - mean;
        numerator *= numerator;
        ss += numerator;
    }
    float variance = ss / size;
    float std_dev = sqrtf(variance + epsilon);
    for (int i = 0; i < size; i++) {
        o[i] = (x[i] - mean) / std_dev;
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 0; i < size; i++) {
        if (max_val < x[i]) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void pos_encoding(float* xout, float* x, float* w, int n, int pos) {
    for (int i = 0; i < n; i++) {
        xout[i] = x[i] + w[i + n*pos];
    }
}

float* forward(Transformer* transformer, uint8_t* img, uint img_height, uint img_width) {
   // a few convenience variables
   Config* p = &transformer->config;
   TransformerWeights* w = &transformer->weights;
   RunState* s = &transformer->state;
   float* x = s->x;
   int dim = p->dim;
   int hidden_dim = p->hidden_dim;
   int head_size = dim / p->n_heads;

   if (img_height % p->patch_height != 0 || img_width % p->patch_width != 0) {
        fprintf(stderr, "Dimensions of image not divisable by the patch height");
        exit(EXIT_FAILURE);
   }

   int n_h = img_height / p->patch_height;
   int n_w = img_width / p->patch_width;
   int n_patches = n_h * n_w;
   int patch_dim = p->patch_width * p->patch_height * p->img_channels;
   for (int l = 0; l < p->n_layers; l++) {
       for (int patch_no = 0; patch_no < n_patches; patch_no++) {
           // extract a patch to become a 1D token
           for (int i = 0; i < p->patch_height; i++) {
               int src_idx = p->img_channels * (patch_no * p->patch_width + (i * img_width));
               int patch_idx = i * p->patch_width * p->img_channels;
               int segment_dim = p->patch_width * p->img_channels;
               memcpy(x + patch_idx, img + src_idx, segment_dim*sizeof(*x));
           }
           // then normalise the patch and feed it through a feed forward layer,
           // this acts like a linear transformation layer for the token to match
           // the dimensions of the transformer
           layer_norm(s->xb, s->x, patch_dim);
           matmul(s->x, s->xb, w->patch2dim_weights, patch_dim, p->dim);
           layer_norm(s->xb, s->x, p->dim);

           // now we apply positional encoding to the patch embedding
           for (int i = 0; i < p->dim; i++) {
               s->xb[i] += transformer->weights.pos_emb_weights[i + p->dim*patch_no];
           }
           matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
           matmul(s->q, s->xb, w->wk + l*dim*dim, dim, dim);
           matmul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

           // multihead attention. Iterate over all heads
           int h;
           for (h = 0; h < p->n_heads; h++) {
               // get the query vector for this head
               float* q = s->q + h * head_size;
               // attention scores for this head
               float* att = s->att + h * n_patches;
               // iterate over all timesteps, including the current one
               for (int t = 0; t <= n_patches; t++) {
                   // get the key vector for this head and at this timestep
                   float* k = s->k + h * head_size + t * head_size;
                   // calculate the attention score as the dot product of q and k
                   float score = 0.0f;
                   for (int i = 0; i < head_size; i++) {
                       score += q[i] * k[i];
                   }
                   score /= sqrtf(head_size);
                   // save the score to the attention buffer
                   att[t] = score;
               }
               // softmax the scores to get attention weights, from 0...pos inclusively
               softmax(att, n_patches*head_size);

               // weighted sum of the values, store back into xb
               float* xb = s->xb + h * head_size;
               memset(xb, 0, head_size * sizeof(float));
               for (int t = 0; t < n_patches; t++) {
                   // get the value vector for this head and at this timestep
                   float* v = s->v + h * head_size + t * head_size;
                   // get the attention weight for this timestep
                   float a = att[t];
                   // accumulate the weighted value into xb
                   for (int i = 0; i < head_size; i++) {
                       xb[i] += a * v[i];
                   }
               }
           }

           // final matmul to get the output of the attention
           matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

           // residual connection back into x
           for (int i = 0; i < n_patches*dim; i++) {
               x[i] += s->xb2[i];
           }


           // Now for FFN
           // 1. layer_norm
           // 2. linear(dim, hidden_dim)
           // 3. GELU(x)
           // 4. Dropout TODO:
           // 5. linear(hidden_dim, dim)
           // 6. Dropout TODO:
           // layer norm
           for (int i = 0; i < n_patches; i++) {
               layer_norm(x + i*p->dim, x + i*p->dim, dim);
           }
           matmul(s->hb, s->x, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
           // GELU activation function
           for (int i = 0; i < n_patches*p->hidden_dim; i++) {
               // GELU(x) = 0.5 * x * (1 + Tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
               float comp1 = 1 + tanhf(sqrtf(2/PI) * (s->hb[i] + 0.044715 * pow(s->hb[i], 3)));
               float comp2 = 0.5 * s-> hb[i] * comp1;
               s->hb[i] = comp2;
           }
           matmul(s->xb, s->hb, w->w2 + l+dim*hidden_dim, hidden_dim, dim);

           // residual connection back into x
           for (int i = 0; i < n_patches*dim; i++) {
               x[i] += s->xb[i];
           }

           for (int i = 0; i < n_patches; i++) {
               layer_norm(x + i*p->dim, x + i*p->dim, p->dim);
           }
       }
    }
    matmul(s->logits, x, w->wcls, p->dim, p->n_classes);
    return s->logits;
}
