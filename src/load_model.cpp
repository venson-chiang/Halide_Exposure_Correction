#include "load_model.h"
#include <Halide.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <float.h>
#include <fstream>
#include <iostream>

using namespace Halide;

Buffer<double> load_weight_buffer_from_file(const std::string &filename, std::vector<int> &shape) {
    
    Buffer<double> buffer(shape);
    int num_dims = shape[0] * shape[1] * shape[2] * shape[3];
    std::ifstream infile(filename, std::ios::binary);
    infile.read((char *)buffer.data(), num_dims * sizeof(double));
    infile.close();
    assert(!infile.fail());

    return buffer;
}

Buffer<double> load_bias_buffer_from_file(const std::string &filename, std::vector<int> &shape) {
    
    Buffer<double> buffer(shape);
    int num_dims = shape[0];
    std::ifstream infile(filename, std::ios::binary);
    infile.read((char *)buffer.data(), num_dims * sizeof(double));
    infile.close();
    assert(!infile.fail());
    
    return buffer;
}

std::vector<int> load_shape(const std::string &filename) {
    
    std::ifstream infile(filename, std::ios::binary);
    int num_dims = 0;
    infile.read(reinterpret_cast<char *>(&num_dims), sizeof(int));
    std::vector<int> dims(num_dims);
    infile.read((char *)dims.data(), num_dims * sizeof(int));
    infile.close();
    assert(!infile.fail());
    
    return dims;
}

Buffer<double> load_conv_params(std::string shapefile, std::string datafile) {
    std::vector<int> shape = load_shape(shapefile);
    assert(shape.size() == 4);
    
    return load_weight_buffer_from_file(datafile, shape);
}

Buffer<double> load_bias_params(std::string shapefile, std::string datafile) {
    std::vector<int> shape = load_shape(shapefile);
    assert(shape.size() == 1);
    
    return load_bias_buffer_from_file(datafile, shape);
}

std::vector<Buffer<double>> load_weight_buffer(std::string weight_dir) {

    std::vector<Buffer<double>> w_buffer;

    for (int i = 4; i > 0; i--) {

        // Encorder
        std::string l4_en1_conv1_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-1-Conv-1_wshape.data";
        std::string l4_en1_conv1_we_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-1-Conv-1_weight.data";
        Buffer<double> l4_en1_conv1 = load_conv_params(l4_en1_conv1_ws_file, l4_en1_conv1_we_file);
        w_buffer.push_back(l4_en1_conv1);
        std::string l4_en1_conv2_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-1-Conv-2_wshape.data";
        std::string l4_en1_conv2_we_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-1-Conv-2_weight.data";
        Buffer<double> l4_en1_conv2 = load_conv_params(l4_en1_conv2_ws_file, l4_en1_conv2_we_file);
        w_buffer.push_back(l4_en1_conv2);

        std::string l4_en2_conv1_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-2-Conv-1_wshape.data";
        std::string l4_en2_conv1_we_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-2-Conv-1_weight.data";
        Buffer<double> l4_en2_conv1 = load_conv_params(l4_en2_conv1_ws_file, l4_en2_conv1_we_file);
        w_buffer.push_back(l4_en2_conv1);
        std::string l4_en2_conv2_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-2-Conv-2_wshape.data";
        std::string l4_en2_conv2_we_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-2-Conv-2_weight.data";
        Buffer<double> l4_en2_conv2 = load_conv_params(l4_en2_conv2_ws_file, l4_en2_conv2_we_file);
        w_buffer.push_back(l4_en2_conv2);

        std::string l4_en3_conv1_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-3-Conv-1_wshape.data";
        std::string l4_en3_conv1_we_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-3-Conv-1_weight.data";
        Buffer<double> l4_en3_conv1 = load_conv_params(l4_en3_conv1_ws_file, l4_en3_conv1_we_file);
        w_buffer.push_back(l4_en3_conv1);
        std::string l4_en3_conv2_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-3-Conv-2_wshape.data";
        std::string l4_en3_conv2_we_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-3-Conv-2_weight.data";
        Buffer<double> l4_en3_conv2 = load_conv_params(l4_en3_conv2_ws_file, l4_en3_conv2_we_file);
        w_buffer.push_back(l4_en3_conv2);

        if (i == 4) {
            std::string l4_en4_conv1_ws_file = weight_dir+"/level_4-Encoder-Stage-4-Conv-1_wshape.data";
            std::string l4_en4_conv1_we_file = weight_dir+"/level_4-Encoder-Stage-4-Conv-1_weight.data";
            Buffer<double> l4_en4_conv1 = load_conv_params(l4_en4_conv1_ws_file, l4_en4_conv1_we_file);
            w_buffer.push_back(l4_en4_conv1);
            std::string l4_en4_conv2_ws_file = weight_dir+"/level_4-Encoder-Stage-4-Conv-2_wshape.data";
            std::string l4_en4_conv2_we_file = weight_dir+"/level_4-Encoder-Stage-4-Conv-2_weight.data";
            Buffer<double> l4_en4_conv2 = load_conv_params(l4_en4_conv2_ws_file, l4_en4_conv2_we_file);
            w_buffer.push_back(l4_en4_conv2);
        }

        // Bridge
        std::string l4_bd_conv1_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Bridge-Conv-1_wshape.data";
        std::string l4_bd_conv1_we_file = weight_dir+"/level_"+std::to_string(i)+"-Bridge-Conv-1_weight.data";
        Buffer<double> l4_bd_conv1 = load_conv_params(l4_bd_conv1_ws_file, l4_bd_conv1_we_file);
        w_buffer.push_back(l4_bd_conv1);
        std::string l4_bd_conv2_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Bridge-Conv-2_wshape.data";
        std::string l4_bd_conv2_we_file = weight_dir+"/level_"+std::to_string(i)+"-Bridge-Conv-2_weight.data";
        Buffer<double> l4_bd_conv2 = load_conv_params(l4_bd_conv2_ws_file, l4_bd_conv2_we_file);
        w_buffer.push_back(l4_bd_conv2);

        // Level4 Decorder
        std::string l4_de1_upconv_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-1-UpConv_wshape.data";
        std::string l4_de1_upconv_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-1-UpConv_weight.data";
        Buffer<double> l4_de1_upconv = load_conv_params(l4_de1_upconv_ws_file, l4_de1_upconv_we_file);
        w_buffer.push_back(l4_de1_upconv);

        std::string l4_de1_conv1_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-1-Conv-1_wshape.data";
        std::string l4_de1_conv1_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-1-Conv-1_weight.data";
        Buffer<double> l4_de1_conv1 = load_conv_params(l4_de1_conv1_ws_file, l4_de1_conv1_we_file);
        w_buffer.push_back(l4_de1_conv1);
        std::string l4_de1_conv2_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-1-Conv-2_wshape.data";
        std::string l4_de1_conv2_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-1-Conv-2_weight.data";
        Buffer<double> l4_de1_conv2 = load_conv_params(l4_de1_conv2_ws_file, l4_de1_conv2_we_file);
        w_buffer.push_back(l4_de1_conv2);

        std::string l4_de2_upconv_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-2-UpConv_wshape.data";
        std::string l4_de2_upconv_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-2-UpConv_weight.data";
        Buffer<double> l4_de2_upconv = load_conv_params(l4_de2_upconv_ws_file, l4_de2_upconv_we_file);
        w_buffer.push_back(l4_de2_upconv);

        std::string l4_de2_conv1_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-2-Conv-1_wshape.data";
        std::string l4_de2_conv1_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-2-Conv-1_weight.data";
        Buffer<double> l4_de2_conv1 = load_conv_params(l4_de2_conv1_ws_file, l4_de2_conv1_we_file);
        w_buffer.push_back(l4_de2_conv1);
        std::string l4_de2_conv2_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-2-Conv-2_wshape.data";
        std::string l4_de2_conv2_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-2-Conv-2_weight.data";
        Buffer<double> l4_de2_conv2 = load_conv_params(l4_de2_conv2_ws_file, l4_de2_conv2_we_file);
        w_buffer.push_back(l4_de2_conv2);

        std::string l4_de3_upconv_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-3-UpConv_wshape.data";
        std::string l4_de3_upconv_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-3-UpConv_weight.data";
        Buffer<double> l4_de3_upconv = load_conv_params(l4_de3_upconv_ws_file, l4_de3_upconv_we_file);
        w_buffer.push_back(l4_de3_upconv);

        std::string l4_de3_conv1_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-3-Conv-1_wshape.data";
        std::string l4_de3_conv1_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-3-Conv-1_weight.data";
        Buffer<double> l4_de3_conv1 = load_conv_params(l4_de3_conv1_ws_file, l4_de3_conv1_we_file);
        w_buffer.push_back(l4_de3_conv1);
        std::string l4_de3_conv2_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-3-Conv-2_wshape.data";
        std::string l4_de3_conv2_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-3-Conv-2_weight.data";
        Buffer<double> l4_de3_conv2 = load_conv_params(l4_de3_conv2_ws_file, l4_de3_conv2_we_file);
        w_buffer.push_back(l4_de3_conv2);

        if (i == 4) {
            std::string l4_de4_upconv_ws_file = weight_dir+"/level_4-Decoder-Stage-4-UpConv_wshape.data";
            std::string l4_de4_upconv_we_file = weight_dir+"/level_4-Decoder-Stage-4-UpConv_weight.data";
            Buffer<double> l4_de4_upconv = load_conv_params(l4_de4_upconv_ws_file, l4_de4_upconv_we_file);
            w_buffer.push_back(l4_de4_upconv);

            std::string l4_de4_conv1_ws_file = weight_dir+"/level_4-Decoder-Stage-4-Conv-1_wshape.data";
            std::string l4_de4_conv1_we_file = weight_dir+"/level_4-Decoder-Stage-4-Conv-1_weight.data";
            Buffer<double> l4_de4_conv1 = load_conv_params(l4_de4_conv1_ws_file, l4_de4_conv1_we_file);
            w_buffer.push_back(l4_de4_conv1);
            std::string l4_de4_conv2_ws_file = weight_dir+"/level_4-Decoder-Stage-4-Conv-2_wshape.data";
            std::string l4_de4_conv2_we_file = weight_dir+"/level_4-Decoder-Stage-4-Conv-2_weight.data";
            Buffer<double> l4_de4_conv2 = load_conv_params(l4_de4_conv2_ws_file, l4_de4_conv2_we_file);
            w_buffer.push_back(l4_de4_conv2);
        }

        // Final convolution
        std::string l4_final_conv_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Final-ConvolutionLayer_wshape.data";
        std::string l4_final_conv_we_file = weight_dir+"/level_"+std::to_string(i)+"-Final-ConvolutionLayer_weight.data";
        Buffer<double> l4_final_conv = load_conv_params(l4_final_conv_ws_file, l4_final_conv_we_file);
        w_buffer.push_back(l4_final_conv);
        
        if (i > 1) {
            // Upsampling
            std::string l4_upsample_ws_file = weight_dir+"/level_"+std::to_string(i)+"-upsampling_wshape.data";
            std::string l4_upsample_we_file = weight_dir+"/level_"+std::to_string(i)+"-upsampling_weight.data";
            Buffer<double> l4_upsample = load_conv_params(l4_upsample_ws_file, l4_upsample_we_file);
            w_buffer.push_back(l4_upsample);
        }
    }

    return w_buffer;
}

std::vector<Buffer<double>> load_bias_buffer(std::string weight_dir) {
    
    std::vector<Buffer<double>> w_buffer;

    for (int i = 4; i > 0; i--) {

        // Encorder
        std::string l4_en1_conv1_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-1-Conv-1_bshape.data";
        std::string l4_en1_conv1_we_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-1-Conv-1_bias.data";
        Buffer<double> l4_en1_conv1 = load_bias_params(l4_en1_conv1_ws_file, l4_en1_conv1_we_file);
        w_buffer.push_back(l4_en1_conv1);
        std::string l4_en1_conv2_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-1-Conv-2_bshape.data";
        std::string l4_en1_conv2_we_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-1-Conv-2_bias.data";
        Buffer<double> l4_en1_conv2 = load_bias_params(l4_en1_conv2_ws_file, l4_en1_conv2_we_file);
        w_buffer.push_back(l4_en1_conv2);

        std::string l4_en2_conv1_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-2-Conv-1_bshape.data";
        std::string l4_en2_conv1_we_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-2-Conv-1_bias.data";
        Buffer<double> l4_en2_conv1 = load_bias_params(l4_en2_conv1_ws_file, l4_en2_conv1_we_file);
        w_buffer.push_back(l4_en2_conv1);
        std::string l4_en2_conv2_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-2-Conv-2_bshape.data";
        std::string l4_en2_conv2_we_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-2-Conv-2_bias.data";
        Buffer<double> l4_en2_conv2 = load_bias_params(l4_en2_conv2_ws_file, l4_en2_conv2_we_file);
        w_buffer.push_back(l4_en2_conv2);

        std::string l4_en3_conv1_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-3-Conv-1_bshape.data";
        std::string l4_en3_conv1_we_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-3-Conv-1_bias.data";
        Buffer<double> l4_en3_conv1 = load_bias_params(l4_en3_conv1_ws_file, l4_en3_conv1_we_file);
        w_buffer.push_back(l4_en3_conv1);
        std::string l4_en3_conv2_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-3-Conv-2_bshape.data";
        std::string l4_en3_conv2_we_file = weight_dir+"/level_"+std::to_string(i)+"-Encoder-Stage-3-Conv-2_bias.data";
        Buffer<double> l4_en3_conv2 = load_bias_params(l4_en3_conv2_ws_file, l4_en3_conv2_we_file);
        w_buffer.push_back(l4_en3_conv2);

        if (i == 4) {
            std::string l4_en4_conv1_ws_file = weight_dir+"/level_4-Encoder-Stage-4-Conv-1_bshape.data";
            std::string l4_en4_conv1_we_file = weight_dir+"/level_4-Encoder-Stage-4-Conv-1_bias.data";
            Buffer<double> l4_en4_conv1 = load_bias_params(l4_en4_conv1_ws_file, l4_en4_conv1_we_file);
            w_buffer.push_back(l4_en4_conv1);
            std::string l4_en4_conv2_ws_file = weight_dir+"/level_4-Encoder-Stage-4-Conv-2_bshape.data";
            std::string l4_en4_conv2_we_file = weight_dir+"/level_4-Encoder-Stage-4-Conv-2_bias.data";
            Buffer<double> l4_en4_conv2 = load_bias_params(l4_en4_conv2_ws_file, l4_en4_conv2_we_file);
            w_buffer.push_back(l4_en4_conv2);
        }

        // Bridge
        std::string l4_bd_conv1_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Bridge-Conv-1_bshape.data";
        std::string l4_bd_conv1_we_file = weight_dir+"/level_"+std::to_string(i)+"-Bridge-Conv-1_bias.data";
        Buffer<double> l4_bd_conv1 = load_bias_params(l4_bd_conv1_ws_file, l4_bd_conv1_we_file);
        w_buffer.push_back(l4_bd_conv1);
        std::string l4_bd_conv2_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Bridge-Conv-2_bshape.data";
        std::string l4_bd_conv2_we_file = weight_dir+"/level_"+std::to_string(i)+"-Bridge-Conv-2_bias.data";
        Buffer<double> l4_bd_conv2 = load_bias_params(l4_bd_conv2_ws_file, l4_bd_conv2_we_file);
        w_buffer.push_back(l4_bd_conv2);

        // Decoder
        std::string l4_de1_upconv_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-1-UpConv_bshape.data";
        std::string l4_de1_upconv_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-1-UpConv_bias.data";
        Buffer<double> l4_de1_upconv = load_bias_params(l4_de1_upconv_ws_file, l4_de1_upconv_we_file);
        w_buffer.push_back(l4_de1_upconv);

        std::string l4_de1_conv1_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-1-Conv-1_bshape.data";
        std::string l4_de1_conv1_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-1-Conv-1_bias.data";
        Buffer<double> l4_de1_conv1 = load_bias_params(l4_de1_conv1_ws_file, l4_de1_conv1_we_file);
        w_buffer.push_back(l4_de1_conv1);
        std::string l4_de1_conv2_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-1-Conv-2_bshape.data";
        std::string l4_de1_conv2_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-1-Conv-2_bias.data";
        Buffer<double> l4_de1_conv2 = load_bias_params(l4_de1_conv2_ws_file, l4_de1_conv2_we_file);
        w_buffer.push_back(l4_de1_conv2);

        std::string l4_de2_upconv_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-2-UpConv_bshape.data";
        std::string l4_de2_upconv_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-2-UpConv_bias.data";
        Buffer<double> l4_de2_upconv = load_bias_params(l4_de2_upconv_ws_file, l4_de2_upconv_we_file);
        w_buffer.push_back(l4_de2_upconv);

        std::string l4_de2_conv1_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-2-Conv-1_bshape.data";
        std::string l4_de2_conv1_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-2-Conv-1_bias.data";
        Buffer<double> l4_de2_conv1 = load_bias_params(l4_de2_conv1_ws_file, l4_de2_conv1_we_file);
        w_buffer.push_back(l4_de2_conv1);
        std::string l4_de2_conv2_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-2-Conv-2_bshape.data";
        std::string l4_de2_conv2_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-2-Conv-2_bias.data";
        Buffer<double> l4_de2_conv2 = load_bias_params(l4_de2_conv2_ws_file, l4_de2_conv2_we_file);
        w_buffer.push_back(l4_de2_conv2);

        std::string l4_de3_upconv_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-3-UpConv_bshape.data";
        std::string l4_de3_upconv_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-3-UpConv_bias.data";
        Buffer<double> l4_de3_upconv = load_bias_params(l4_de3_upconv_ws_file, l4_de3_upconv_we_file);
        w_buffer.push_back(l4_de3_upconv);

        std::string l4_de3_conv1_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-3-Conv-1_bshape.data";
        std::string l4_de3_conv1_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-3-Conv-1_bias.data";
        Buffer<double> l4_de3_conv1 = load_bias_params(l4_de3_conv1_ws_file, l4_de3_conv1_we_file);
        w_buffer.push_back(l4_de3_conv1);
        std::string l4_de3_conv2_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-3-Conv-2_bshape.data";
        std::string l4_de3_conv2_we_file = weight_dir+"/level_"+std::to_string(i)+"-Decoder-Stage-3-Conv-2_bias.data";
        Buffer<double> l4_de3_conv2 = load_bias_params(l4_de3_conv2_ws_file, l4_de3_conv2_we_file);
        w_buffer.push_back(l4_de3_conv2);

        if (i == 4) {
            std::string l4_de4_upconv_ws_file = weight_dir+"/level_4-Decoder-Stage-4-UpConv_bshape.data";
            std::string l4_de4_upconv_we_file = weight_dir+"/level_4-Decoder-Stage-4-UpConv_bias.data";
            Buffer<double> l4_de4_upconv = load_bias_params(l4_de4_upconv_ws_file, l4_de4_upconv_we_file);
            w_buffer.push_back(l4_de4_upconv);

            std::string l4_de4_conv1_ws_file = weight_dir+"/level_4-Decoder-Stage-4-Conv-1_bshape.data";
            std::string l4_de4_conv1_we_file = weight_dir+"/level_4-Decoder-Stage-4-Conv-1_bias.data";
            Buffer<double> l4_de4_conv1 = load_bias_params(l4_de4_conv1_ws_file, l4_de4_conv1_we_file);
            w_buffer.push_back(l4_de4_conv1);
            std::string l4_de4_conv2_ws_file = weight_dir+"/level_4-Decoder-Stage-4-Conv-2_bshape.data";
            std::string l4_de4_conv2_we_file = weight_dir+"/level_4-Decoder-Stage-4-Conv-2_bias.data";
            Buffer<double> l4_de4_conv2 = load_bias_params(l4_de4_conv2_ws_file, l4_de4_conv2_we_file);
            w_buffer.push_back(l4_de4_conv2);
        }

        // Final Convolution 
        std::string l4_final_conv_ws_file = weight_dir+"/level_"+std::to_string(i)+"-Final-ConvolutionLayer_bshape.data";
        std::string l4_final_conv_we_file = weight_dir+"/level_"+std::to_string(i)+"-Final-ConvolutionLayer_bias.data";
        Buffer<double> l4_final_conv = load_bias_params(l4_final_conv_ws_file, l4_final_conv_we_file);
        w_buffer.push_back(l4_final_conv);

        if (i > 1) {
            // Upsampling
            std::string l4_upsample_ws_file = weight_dir+"/level_"+std::to_string(i)+"-upsampling_bshape.data";
            std::string l4_upsample_we_file = weight_dir+"/level_"+std::to_string(i)+"-upsampling_bias.data";
            Buffer<double> l4_upsample = load_bias_params(l4_upsample_ws_file, l4_upsample_we_file);
            w_buffer.push_back(l4_upsample);
        }
    }

    return w_buffer;
}