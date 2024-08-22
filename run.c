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

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int patch_size;
    float dropout;

} Config;

typedef struct {
    float* norm_att_weights;
    float* norm_ffn_weights;
    float* wq;
    float* wk;
    float* wo;
    // floats for the ffn
    float* w1;
    float* w2;
    // classifier weights for the logits, on the last layers
    float* wcls;
} TransformerWeights;
