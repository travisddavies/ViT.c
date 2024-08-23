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
    int patch_size; // patch size for each token
    float dropout; // dropout in the ffn layers
} Config;

typedef struct {
    float* norm_att_weights; // (layer, dim) norm weights
    float* norm_ffn_weights; // (layer, dim) ffn weights
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; //(layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_heads * head_size)
    float* wo; // (layer, dim, n_heads * head_size)
    // floats for the ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, hidden_dim, dim)
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
}
