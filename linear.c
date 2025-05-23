#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "shared.h"
    
void linear_forward(Linear *linear, nn_float *in, nn_float *out) {
    for(int i = 0; i < linear->out; i++) {
        nn_float sum = 0;
        for(int j = 0; j < linear->in; j++) {
            sum += (linear->weights[linear->in * i + j] * in[j]);
        }
        sum += linear->bias[i];
        out[i] = sum;
    }
}

void generate_executable_linear(unsigned char *filename, Linear *linear) {
    unsigned char str[200];
    FILE *file;
    file = fopen(filename, "w");
    fprintf(file, "#include \"linear_exec.h\"\n");
    fprintf(file, "#include \"shared.h\"\n");
    fprintf(file, "void exec_linear_forward(nn_float *in, nn_float *out) {\n");
    fprintf(file, "\t// this code has been automatically generated\n");
    fprintf(file, "\tnn_float sum;\n");
    for(int i = 0; i < linear->out; i++) {
        fprintf(file, "\tsum = 0.0;\n");
        for(int j = 0; j < linear->in; j++) {
            sprintf(str, "\tsum += in[%d] *\t%.30f;\n", linear->weights[linear->in * i + j], j);
            fprintf(file, str);
        }
        sprintf(str, "\tsum += %.30f;\n", linear->bias[i]);
        fprintf(file, str);
        sprintf(str, "\tout[%d] = sum;\n", i);
        fprintf(file, str);
    }
    fprintf(file, "}\n");
    fclose(file);
}


int main() {
    FILE *linear_ckpt;
    linear_ckpt = fopen("linear_params.txt", "r");
    if(!linear_ckpt) { printf("something went wrong\n"); }
    
    // layer type
    unsigned char lt_string[20];
    fscanf(linear_ckpt, "%s\n", lt_string);
    printf("layer type: %s\n", lt_string);
    
    // input output count
    int in, out;
    fscanf(linear_ckpt, "%d %d\n", &out, &in);
    printf("in: %d out: %d\n", in, out);
    
    Linear linear;
    linear.in = in;
    linear.out = out;
    linear.weights = (nn_float *)malloc(in * out * sizeof(nn_float));   
    linear.bias = (nn_float *)malloc(out * sizeof(nn_float));   
    
    // load weight params
    nn_float param;
    nn_float *buff = linear.weights;
    for(int i = 0; i < in*out; i++) {
        fscanf(linear_ckpt, "%30f ", &param);
        //printf("param: %.30f\n", param);
        *buff = param;   
        buff++;
    }
    
    // load bias params
    buff = linear.bias;
    for(int i = 0; i < out; i++) {
        fscanf(linear_ckpt, "%30f ", &param);
        //printf("bias param: %.30f\n", param);
        *buff = param;   
        buff++;
    }
    fclose(linear_ckpt);
    
    printf("generating executable linear layer...\n");
    generate_executable_linear("linear_exec.c", &linear);

    // init output to zero
    int output_size = 1000;
    nn_float *output = (nn_float *)malloc(sizeof(nn_float) * output_size);
    for(int i = 0; i < output_size; i++) { output[i] = 0.0; }
    
    // fill input with ones
    int input_size = 100;
    nn_float *input = (nn_float *)malloc(sizeof(nn_float) * input_size);
    for(int i = 0; i < input_size; i++) { input[i] = 1.0; }

    struct timepsec start;
    struct timepsec end;
    for(int i = 0; i < 1000; i++) {
        clock_gettime(CLOCK_REALTIME, &start);  
        linear_forward(&linear, input, output);
        clock_gettime(CLOCK_REALTIME, &end);  
        double elapsed = end.
    }
    return 0;
}
