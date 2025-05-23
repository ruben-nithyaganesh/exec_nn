#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "shared.h"

#define VERBOSE

#ifndef VERBOSE
    #define DEBUG(s)
#endif
#ifdef VERBOSE
    #define DEBUG(s) s
#endif

typedef struct {
    int N, C, W, H;
    nn_float *data;
} Tensor;

typedef enum { ZEROS, REFLECT } Padding_Mode;
typedef struct { int width; int height; } Size;
typedef struct {
    nn_float *weights;
    nn_float *bias;
    int in_channels;
    int out_channels;
    Size kernel;
    Size stride;
    Size padding;
    Size dilation;
    int groups;
    Padding_Mode padding_mode;
} Conv2D;

Padding_Mode padding_mode(unsigned char* str) {
    if(strcmp("zeros", str) == 0) {
        return ZEROS;
    }
    if(strcmp("reflect", str) == 0) {
        return REFLECT;
    }
    printf("Unrecognised padding mode, defaulting to zeros");
    return ZEROS;
}


nn_float tensor_at(Tensor t, int n, int c, int h, int w) {
    return t.data[
        n * (t.C * t.W * t.H) +
        c * (t.W * t.H) +
        h * (t.W) +
        w
    ];
}

Conv2D load_conv2d(unsigned char *filename) {
    unsigned char str[200];
    FILE *file = fopen(filename, "r");
    if(!file) { printf("something went wrong\n"); }
    fscanf(file, "%s\n", str);
    printf("%s\n", str);
    
    int has_bias;
    Conv2D conv2d;
    fscanf(file, "%d %d (%d, %d) (%d, %d) (%d, %d) (%d, %d) %d %d %s\n",
        &conv2d.in_channels,
        &conv2d.out_channels,
        &conv2d.kernel.width,
        &conv2d.kernel.height,
        &conv2d.stride.width,
        &conv2d.stride.height,
        &conv2d.padding.width,
        &conv2d.padding.height,
        &conv2d.dilation.width,
        &conv2d.dilation.height,
        &conv2d.groups,
        &has_bias,
        str
    );
        DEBUG(
            printf("\tin_channels: %d\n", conv2d.in_channels);
            printf("\tout_channels: %d\n", conv2d.out_channels);
            printf("\tkernel.width: %d\n", conv2d.kernel.width);
            printf("\tkernel.height: %d\n", conv2d.kernel.height);
            printf("\tstride.width: %d\n", conv2d.stride.width);
            printf("\tstride.height: %d\n", conv2d.stride.height);
            printf("\tpadding.width: %d\n", conv2d.padding.width);
            printf("\tpadding.height: %d\n", conv2d.padding.height);
            printf("\tdilation.width: %d\n", conv2d.dilation.width);
            printf("\tdilation.height: %d\n", conv2d.dilation.height);
            printf("\tgroups: %d\n", conv2d.groups);
            printf("\tbias: %d\n", has_bias);
            conv2d.padding_mode = padding_mode(str);
            printf("\tpadding_mode_str: %s\n", str);
            printf("\tpadding_mode_enum: %d\n", conv2d.padding_mode);
        )
    
    // conv2d weights shape is (out_channels, in_channels / groups, kernel_size[0], kernel_size[1])
    int conv2d_weight_size = conv2d.out_channels * (conv2d.in_channels / conv2d.groups) * conv2d.kernel.width * conv2d.kernel.height;
    conv2d.weights = (nn_float *)malloc(conv2d_weight_size * sizeof(nn_float));
    nn_float *buf = conv2d.weights;
    for(int i = 0; i < conv2d_weight_size; i++) {
        fscanf(file, "%lf ", buf);
        DEBUG(printf("%.30lf ", conv2d.weights[i]);)
        buf++;
    }
    DEBUG(printf("\n"));
    if(has_bias) {
        // conv2d bias shape is (out_channels)
        int conv2d_bias_size = conv2d.out_channels;
        conv2d.bias = (nn_float *)malloc(conv2d_weight_size * sizeof(nn_float));
        nn_float *buf = conv2d.bias;
        for(int i = 0; i < conv2d_bias_size; i++) {
            fscanf(file, "%lf ", buf);
            DEBUG(printf("%.30lf ", conv2d.bias[i]));
            buf++;
        }
        DEBUG(printf("\n"));
    } else {
        conv2d.bias = 0;
    }

    return conv2d;
}

Tensor read_tensor(unsigned char* filename) {
    FILE *file;
    file = fopen(filename, "r");
    if(!file) { printf("something went wrong\n"); }
    Tensor t;
    fscanf(file, "%d %d %d %d \n", &t.N, &t.C, &t.W, &t.H);
    int input_size = t.N * t.C * t.W * t.H;
    t.data = (nn_float *)malloc(input_size * sizeof(nn_float));
    nn_float *b = t.data;
    for(int i = 0; i < input_size; i++) {
        fscanf(file, "%lf ", b);
        DEBUG(printf("%lf\n", t.data[i]));
        b++;
    }
    printf("\n");
    return t;
}

Tensor conv2d_forward(Conv2D *conv2d, Tensor input) {
  
    int batch_stride = input.C * input.W * input.H;
    int channel_stride = input.W * input.H;
    int row_stride = input.W;
    
    int conv_kernel_stride = conv2d->kernel.width * conv2d->kernel.height * conv2d->in_channels;
    int conv_y_offset = (conv2d->kernel.height / 2);
    int conv_x_offset = (conv2d->kernel.width / 2);
    
    Tensor output;
    output.N = input.N;
    output.C = conv2d->out_channels;
    output.H = input.H - (conv_y_offset * 2);
    output.W = input.H - (conv_x_offset * 2);
    int output_size = output.N * output.C * output.H * output.W;
    output.data = (nn_float *)malloc(output_size * sizeof(nn_float));
    int output_batch_stride = output.C * output.W * output.H;

    nn_float *output_buffer = output.data;
    nn_float sum = 0;
    for(int batch = 0; batch < input.N; batch++) {
        for(int out_channel = 0; out_channel < conv2d->out_channels; out_channel++) {
            nn_float *kernel = conv2d->weights + (out_channel * conv_kernel_stride);
            for(int image_y = conv_y_offset; image_y < input.H - conv_y_offset; image_y++) {
            for(int image_x = conv_x_offset; image_x < input.W - conv_x_offset; image_x++) {
                    sum = 0;
                    if(conv2d->bias != 0) {
                        sum = conv2d->bias[out_channel];
                    }
                    for(int in_channel = 0; in_channel < conv2d->in_channels; in_channel++) {
                        for(int y = 0; y < conv2d->kernel.height; y++) {
                            for(int x = 0; x < conv2d->kernel.width; x++) {
                                nn_float input_value = input.data[
                                    (batch * batch_stride) +
                                    (in_channel * channel_stride) +
                                    ((image_y - conv_y_offset + y) * row_stride) +
                                    image_x - conv_x_offset + x
                                ];
                                //int batch_dim = batch;
                                //int channel_dim = in_channel;
                                //int row_dim = image_y - conv_y_offset + y;
                                //int col_dim = image_x - conv_x_offset + x;
                                //nn_float input_value = tensor_at(input, batch_dim, channel_dim, row_dim, col_dim);
                                nn_float kernel_value = kernel[
                                    (conv2d->kernel.height * conv2d->kernel.width * in_channel) +
                                    (conv2d->kernel.width * y) + x
                                ];

                                sum += kernel_value * input_value;
                            }
                        }
                    }
                    *output_buffer = sum;
                    printf("%.30f\n", *output_buffer);
                    output_buffer++;
                }
            }
        }
    }
    return output;
}

int main() {
    Conv2D conv2d = load_conv2d("conv_params.txt");
    Tensor t = read_tensor("conv_input.txt");
    printf("%.30f\n", tensor_at(t, 0, 0, 3, 4));
    printf("%.30f\n", tensor_at(t, 0, 0, 4, 3));
    Tensor out = conv2d_forward(&conv2d, t);
    printf("N: %d, C: %d, W: %d, H: %d\n", out.N, out.C, out.W, out.H);
    Tensor output = read_tensor("conv_output.txt");
    int size = output.N * output.C * output.W * output.H;
    for(int i = 0; i < size; i++) {
       printf("ours: %s%.30lf\ttorch: %s%.30lf\n", (out.data[i] < 0) ? "" : " ", out.data[i], (output.data[i] < 0) ? "" : " ", output.data[i]);
    }
}
