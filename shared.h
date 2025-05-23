#ifndef SHARED
#define SHARED
#include <stdio.h>
#include <stdlib.h>

typedef double nn_float;

typedef struct {
    int in;
    int out;
    nn_float *weights;
    nn_float *bias;
} Linear;

typedef struct {
    int len;
    nn_float *data;
} Buffer;

#endif
