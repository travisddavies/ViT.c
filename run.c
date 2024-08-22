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
