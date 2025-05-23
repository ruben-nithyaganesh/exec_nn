#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "shared.h"
#include "linear_exec.h"

Buffer load_buffer(unsigned char* filename) {
    FILE *file;
    file = fopen(filename, "r");
    if(!file) { printf("load_buffer: something went wrong\n"); }

    int length;
    fscanf(file, "%d\n", &length);

    nn_float* data = (nn_float *)malloc(sizeof(nn_float) * length);
    nn_float* bp = data;
    for(int i = 0; i < length; i++) {
        fscanf(file, "%f ", bp);
        bp++;
    }
     
    Buffer buffer;
    buffer.len = length;
    buffer.data = data;
    return buffer;
}

int main(int argc, unsigned char **argv) {

    unsigned char *f;
    if (argc <= 1) {
        return -1;
    }

    Buffer pytorch_out = load_buffer(argv[1]);

    // init output to zero
    int output_size = 1000;
    nn_float *output = (nn_float *)malloc(sizeof(nn_float) * output_size);
    for(int i = 0; i < output_size; i++) { output[i] = 0.0; }
    
    // fill input with ones
    int input_size = 100;
    nn_float *input = (nn_float *)malloc(sizeof(nn_float) * input_size);
    for(int i = 0; i < input_size; i++) { input[i] = 1.0; }
    
    double average = 0;
    for(int i = 0; i < 1000; i++) {
        struct timespec start;
        clock_gettime(CLOCK_REALTIME, &start);
        exec_linear_forward(input, output);
        struct timespec end;
        clock_gettime(CLOCK_REALTIME, &end);
        average += ((double)end.tv_nsec - (double)start.tv_nsec) / 100000000.0;
    }
    printf("average nano seconds: %f\n", average / 1000.0);

    average = 0;
    for(int i = 0; i < 1000; i++) {
        struct timespec start;
        clock_gettime(CLOCK_REALTIME, &start);
        linear_forward(input, output);
        struct timespec end;
        clock_gettime(CLOCK_REALTIME, &end);
        average += ((double)end.tv_nsec - (double)start.tv_nsec) / 100000000.0;
    }
    printf("average nano seconds: %f\n", average / 1000.0);
    //nn_float sum = 0;
    //for(int i = 0; i < 100; i++) {
    //    sum += output[i];
    //}
    //printf("\nsum: %.30f\n", sum);
    
    //for(int i = 0; i < pytorch_out.len; i++) {
    //    nn_float pytorch = pytorch_out.data[i];
    //    nn_float ours = output[i];
    //    printf("pytorch: %.5f\tours: %.5f\n", pytorch, ours);
    //}
}
