#include "model.h"
#include <Halide.h>

using namespace Halide;

Tensor Model(std::vector<Tensor> inputs, std::vector<Buffer<double>> w_buffer,
                                         std::vector<Buffer<double>> b_buffer) {
    
    // Level 4
    const WeightShape lv4_en1_conv1_ws = {24, 3, 3};
    const WeightShape lv4_en1_conv2_ws = {24, 3, 3};
    const WeightShape lv4_en2_conv1_ws = {48, 3, 3};
    const WeightShape lv4_en2_conv2_ws = {48, 3, 3};
    const WeightShape lv4_en3_conv1_ws = {96, 3, 3};
    const WeightShape lv4_en3_conv2_ws = {96, 3, 3};
    const WeightShape lv4_en4_conv1_ws = {192, 3, 3};
    const WeightShape lv4_en4_conv2_ws = {192, 3, 3};
    const WeightShape lv4_bd_conv1_ws = {384, 3, 3};
    const WeightShape lv4_bd_conv2_ws = {384, 3, 3};

    const WeightShape lv4_de1_upconv_ws = {192, 2, 2};
    const WeightShape lv4_de1_conv1_ws = {192, 3, 3};
    const WeightShape lv4_de1_conv2_ws = {192, 3, 3};
    const WeightShape lv4_de2_upconv_ws = {96, 2, 2};
    const WeightShape lv4_de2_conv1_ws = {96, 3, 3};
    const WeightShape lv4_de2_conv2_ws = {96, 3, 3};
    const WeightShape lv4_de3_upconv_ws = {48, 2, 2};
    const WeightShape lv4_de3_conv1_ws = {48, 3, 3};
    const WeightShape lv4_de3_conv2_ws = {48, 3, 3};
    const WeightShape lv4_de4_upconv_ws = {24, 2, 2};
    const WeightShape lv4_de4_conv1_ws = {24, 3, 3};
    const WeightShape lv4_de4_conv2_ws = {24, 3, 3};
    const WeightShape lv4_final_conv_ws = {3, 1, 1};
    const WeightShape lv4_upsample_ws = {3, 2, 2};

    Tensor lv4_en1_conv1;
    Tensor lv4_en1_lrelu1;
    Tensor lv4_en1_conv2;
    Tensor lv4_en1_lrelu2;
    Tensor lv4_en1_mpool;
    Tensor lv4_en2_conv1;
    Tensor lv4_en2_lrelu1;
    Tensor lv4_en2_conv2;
    Tensor lv4_en2_lrelu2;
    Tensor lv4_en2_mpool;
    Tensor lv4_en3_conv1;
    Tensor lv4_en3_lrelu1;
    Tensor lv4_en3_conv2;
    Tensor lv4_en3_lrelu2;
    Tensor lv4_en3_mpool;
    Tensor lv4_en4_conv1;
    Tensor lv4_en4_lrelu1;
    Tensor lv4_en4_conv2;
    Tensor lv4_en4_lrelu2;
    Tensor lv4_en4_mpool;

    Tensor lv4_bd_conv1;
    Tensor lv4_bd_lrelu1;
    Tensor lv4_bd_conv2;
    Tensor lv4_bd_lrelu2;

    Tensor lv4_de1_upconv;
    Tensor lv4_de1_uprelu;
    Tensor lv4_de1_concat;
    Tensor lv4_de1_conv1;
    Tensor lv4_de1_lrelu1;
    Tensor lv4_de1_conv2;
    Tensor lv4_de1_lrelu2;
    Tensor lv4_de2_upconv;
    Tensor lv4_de2_uprelu;
    Tensor lv4_de2_concat;
    Tensor lv4_de2_conv1;
    Tensor lv4_de2_lrelu1;
    Tensor lv4_de2_conv2;
    Tensor lv4_de2_lrelu2;
    Tensor lv4_de3_upconv;
    Tensor lv4_de3_uprelu;
    Tensor lv4_de3_concat;
    Tensor lv4_de3_conv1;
    Tensor lv4_de3_lrelu1;
    Tensor lv4_de3_conv2;
    Tensor lv4_de3_lrelu2;
    Tensor lv4_de4_upconv;
    Tensor lv4_de4_uprelu;
    Tensor lv4_de4_concat;
    Tensor lv4_de4_conv1;
    Tensor lv4_de4_lrelu1;
    Tensor lv4_de4_conv2;
    Tensor lv4_de4_lrelu2;

    Tensor lv4_final_conv;
    Tensor lv4_upsample;

    lv4_en1_conv1 = conv2D(inputs[3],     lv4_en1_conv1_ws, Func(w_buffer[0]), Func(b_buffer[0]), "level_4-Encoder-Stage-1-Conv-1");
    lv4_en1_lrelu1 = Leaky_relu(lv4_en1_conv1, "level_4-Encoder-Stage-1-L-ReLU-1");
    lv4_en1_conv2 = conv2D(lv4_en1_lrelu1, lv4_en1_conv2_ws, Func(w_buffer[1]), Func(b_buffer[1]), "level_4-Encoder-Stage-1-Conv-2");
    lv4_en1_lrelu2 = Leaky_relu(lv4_en1_conv2, "level_4-Encoder-Stage-1-L-ReLU-2");
    lv4_en1_mpool = max_pool(lv4_en1_lrelu2, "level_4-Encoder-Stage-1-MaxPool");
    
    lv4_en2_conv1 = conv2D(lv4_en1_mpool, lv4_en2_conv1_ws, Func(w_buffer[2]), Func(b_buffer[2]), "level_4-Encoder-Stage-2-Conv-1");
    lv4_en2_lrelu1 = Leaky_relu(lv4_en2_conv1, "level_4-Encoder-Stage-2-L-ReLU-1");
    lv4_en2_conv2 = conv2D(lv4_en2_lrelu1, lv4_en2_conv2_ws, Func(w_buffer[3]), Func(b_buffer[3]), "level_4-Encoder-Stage-2-Conv-2");
    lv4_en2_lrelu2 = Leaky_relu(lv4_en2_conv2, "level_4-Encoder-Stage-2-L-ReLU-2");
    lv4_en2_mpool = max_pool(lv4_en2_lrelu2, "level_4-Encoder-Stage-2-MaxPool");

    lv4_en3_conv1 = conv2D(lv4_en2_mpool, lv4_en3_conv1_ws, Func(w_buffer[4]), Func(b_buffer[4]), "level_4-Encoder-Stage-3-Conv-1");
    lv4_en3_lrelu1 = Leaky_relu(lv4_en3_conv1, "level_4-Encoder-Stage-3-L-ReLU-1");
    lv4_en3_conv2 = conv2D(lv4_en3_lrelu1, lv4_en3_conv2_ws, Func(w_buffer[5]), Func(b_buffer[5]), "level_4-Encoder-Stage-3-Conv-2");
    lv4_en3_lrelu2 = Leaky_relu(lv4_en3_conv2, "level_4-Encoder-Stage-3-L-ReLU-2");
    lv4_en3_mpool = max_pool(lv4_en3_lrelu2, "level_4-Encoder-Stage-3-MaxPool");

    lv4_en4_conv1 = conv2D(lv4_en3_mpool, lv4_en4_conv1_ws, Func(w_buffer[6]), Func(b_buffer[6]), "level_4-Encoder-Stage-4-Conv-1");
    lv4_en4_lrelu1 = Leaky_relu(lv4_en4_conv1, "level_4-Encoder-Stage-4-L-ReLU-1");
    lv4_en4_conv2 = conv2D(lv4_en4_lrelu1, lv4_en4_conv2_ws, Func(w_buffer[7]), Func(b_buffer[7]), "level_4-Encoder-Stage-4-Conv-2");
    lv4_en4_lrelu2 = Leaky_relu(lv4_en4_conv2, "level_4-Encoder-Stage-4-L-ReLU-2");
    lv4_en4_mpool = max_pool(lv4_en4_lrelu2, "level_4-Encoder-Stage-4-MaxPool");

    lv4_bd_conv1 = conv2D(lv4_en4_mpool, lv4_bd_conv1_ws, Func(w_buffer[8]), Func(b_buffer[8]), "level_4-Bridge-Conv-1");
    lv4_bd_lrelu1 = Leaky_relu(lv4_bd_conv1, "level_4-Bridge-L-ReLU-1");
    lv4_bd_conv2 = conv2D(lv4_bd_lrelu1,  lv4_bd_conv2_ws, Func(w_buffer[9]), Func(b_buffer[9]), "level_4-Bridge-Conv-2");
    lv4_bd_lrelu2 = Leaky_relu(lv4_bd_conv2, "level_4-Bridge-L-ReLU-2");

    lv4_de1_upconv = transposeConv2D(lv4_bd_lrelu2, lv4_de1_upconv_ws, Func(w_buffer[10]), Func(b_buffer[10]), "level_4-Decoder-Stage-1-UpConv");
    lv4_de1_uprelu = relu(lv4_de1_upconv, "level_4-Decoder-Stage-1-UpReLU");
    lv4_de1_concat = Concat(lv4_de1_uprelu, lv4_en4_lrelu2, "level_4-Decoder-Stage-1-DepthConcatenation");
    lv4_de1_conv1 = conv2D(lv4_de1_concat, lv4_de1_conv1_ws, Func(w_buffer[11]), Func(b_buffer[11]), "level_4-Decoder-Stage-1-Conv-1");
    lv4_de1_lrelu1 = Leaky_relu(lv4_de1_conv1, "level_4-Decoder-Stage-1-L-ReLU-1");
    lv4_de1_conv2 = conv2D(lv4_de1_lrelu1, lv4_de1_conv2_ws, Func(w_buffer[12]), Func(b_buffer[12]), "level_4-Decoder-Stage-1-Conv-2");
    lv4_de1_lrelu2 = Leaky_relu(lv4_de1_conv2, "level_4-Decoder-Stage-1-L-ReLU-2");

    lv4_de2_upconv = transposeConv2D(lv4_de1_lrelu2, lv4_de2_upconv_ws, Func(w_buffer[13]), Func(b_buffer[13]), "level_4-Decoder-Stage-2-UpConv");
    lv4_de2_uprelu = relu(lv4_de2_upconv, "level_4-Decoder-Stage-2-UpReLU");
    lv4_de2_concat = Concat(lv4_de2_uprelu, lv4_en3_lrelu2, "level_4-Decoder-Stage-2-DepthConcatenation");
    lv4_de2_conv1 = conv2D(lv4_de2_concat, lv4_de2_conv1_ws, Func(w_buffer[14]), Func(b_buffer[14]), "level_4-Decoder-Stage-2-Conv-1");
    lv4_de2_lrelu1 = Leaky_relu(lv4_de2_conv1, "level_4-Decoder-Stage-2-L-ReLU-1");
    lv4_de2_conv2 = conv2D(lv4_de2_lrelu1, lv4_de2_conv2_ws, Func(w_buffer[15]), Func(b_buffer[15]), "level_4-Decoder-Stage-2-Conv-2");
    lv4_de2_lrelu2 = Leaky_relu(lv4_de2_conv2, "level_4-Decoder-Stage-2-L-ReLU-2");

    lv4_de3_upconv = transposeConv2D(lv4_de2_lrelu2, lv4_de3_upconv_ws, Func(w_buffer[16]), Func(b_buffer[16]), "level_4-Decoder-Stage-3-UpConv");
    lv4_de3_uprelu = relu(lv4_de3_upconv, "level_4-Decoder-Stage-3-UpReLU");
    lv4_de3_concat = Concat(lv4_de3_uprelu, lv4_en2_lrelu2, "level_4-Decoder-Stage-3-DepthConcatenation");
    lv4_de3_conv1 = conv2D(lv4_de3_concat, lv4_de3_conv1_ws, Func(w_buffer[17]), Func(b_buffer[17]), "level_4-Decoder-Stage-3-Conv-1");
    lv4_de3_lrelu1 = Leaky_relu(lv4_de3_conv1, "level_4-Decoder-Stage-3-L-ReLU-1");
    lv4_de3_conv2 = conv2D(lv4_de3_lrelu1, lv4_de3_conv2_ws, Func(w_buffer[18]), Func(b_buffer[18]), "level_4-Decoder-Stage-3-Conv-2");
    lv4_de3_lrelu2 = Leaky_relu(lv4_de3_conv2, "level_4-Decoder-Stage-3-L-ReLU-2");

    lv4_de4_upconv = transposeConv2D(lv4_de3_lrelu2, lv4_de4_upconv_ws, Func(w_buffer[19]), Func(b_buffer[19]), "level_4-Decoder-Stage-4-UpConv");
    lv4_de4_uprelu = relu(lv4_de4_upconv, "level_4-Decoder-Stage-4-UpReLU");
    lv4_de4_concat = Concat(lv4_de4_uprelu, lv4_en1_lrelu2, "level_4-Decoder-Stage-4-DepthConcatenation");
    lv4_de4_conv1 = conv2D(lv4_de4_concat, lv4_de4_conv1_ws, Func(w_buffer[20]), Func(b_buffer[20]), "level_4-Decoder-Stage-4-Conv-1");
    lv4_de4_lrelu1 = Leaky_relu(lv4_de4_conv1, "level_4-Decoder-Stage-4-L-ReLU-1");
    lv4_de4_conv2 = conv2D(lv4_de4_lrelu1, lv4_de4_conv2_ws, Func(w_buffer[21]), Func(b_buffer[21]), "level_4-Decoder-Stage-4-Conv-2");
    lv4_de4_lrelu2 = Leaky_relu(lv4_de4_conv2, "level_4-Decoder-Stage-4-L-ReLU-2");

    lv4_final_conv = conv2D(lv4_de4_lrelu2, lv4_final_conv_ws, Func(w_buffer[22]), Func(b_buffer[22]), "level_4-Final-ConvolutionLayer");
    lv4_upsample = transposeConv2D(lv4_final_conv, lv4_upsample_ws, Func(w_buffer[23]), Func(b_buffer[23]), "level_4-upsampling");

    // Schedule
    lv4_en1_lrelu2.f.store_root();
    lv4_en2_lrelu2.f.store_root();
    lv4_en3_lrelu2.f.store_root();
    lv4_en4_lrelu2.f.store_root();
    lv4_upsample.f.store_root();
    
    // Level 3
    const WeightShape lv3_en1_conv1_ws = {24, 3, 3};
    const WeightShape lv3_en1_conv2_ws = {24, 3, 3};
    const WeightShape lv3_en2_conv1_ws = {48, 3, 3};
    const WeightShape lv3_en2_conv2_ws = {48, 3, 3};
    const WeightShape lv3_en3_conv1_ws = {96, 3, 3};
    const WeightShape lv3_en3_conv2_ws = {96, 3, 3};

    const WeightShape lv3_bd_conv1_ws = {192, 3, 3};
    const WeightShape lv3_bd_conv2_ws = {192, 3, 3};

    const WeightShape lv3_de1_upconv_ws = {96, 2, 2};
    const WeightShape lv3_de1_conv1_ws = {96, 3, 3};
    const WeightShape lv3_de1_conv2_ws = {96, 3, 3};
    const WeightShape lv3_de2_upconv_ws = {48, 2, 2};
    const WeightShape lv3_de2_conv1_ws = {48, 3, 3};
    const WeightShape lv3_de2_conv2_ws = {48, 3, 3};
    const WeightShape lv3_de3_upconv_ws = {24, 2, 2};
    const WeightShape lv3_de3_conv1_ws = {24, 3, 3};
    const WeightShape lv3_de3_conv2_ws = {24, 3, 3};
    const WeightShape lv3_final_conv_ws = {3, 1, 1};
    const WeightShape lv3_upsample_ws = {3, 2, 2};

    Tensor lv4_out_lv3_in;
    Tensor lv3_en1_conv1;
    Tensor lv3_en1_lrelu1;
    Tensor lv3_en1_conv2;
    Tensor lv3_en1_lrelu2;
    Tensor lv3_en1_mpool;
    Tensor lv3_en2_conv1;
    Tensor lv3_en2_lrelu1;
    Tensor lv3_en2_conv2;
    Tensor lv3_en2_lrelu2;
    Tensor lv3_en2_mpool;
    Tensor lv3_en3_conv1;
    Tensor lv3_en3_lrelu1;
    Tensor lv3_en3_conv2;
    Tensor lv3_en3_lrelu2;
    Tensor lv3_en3_mpool;

    Tensor lv3_bd_conv1;
    Tensor lv3_bd_lrelu1;
    Tensor lv3_bd_conv2;
    Tensor lv3_bd_lrelu2;

    Tensor lv3_de1_upconv;
    Tensor lv3_de1_uprelu;
    Tensor lv3_de1_concat;
    Tensor lv3_de1_conv1;
    Tensor lv3_de1_lrelu1;
    Tensor lv3_de1_conv2;
    Tensor lv3_de1_lrelu2;
    Tensor lv3_de2_upconv;
    Tensor lv3_de2_uprelu;
    Tensor lv3_de2_concat;
    Tensor lv3_de2_conv1;
    Tensor lv3_de2_lrelu1;
    Tensor lv3_de2_conv2;
    Tensor lv3_de2_lrelu2;
    Tensor lv3_de3_upconv;
    Tensor lv3_de3_uprelu;
    Tensor lv3_de3_concat;
    Tensor lv3_de3_conv1;
    Tensor lv3_de3_lrelu1;
    Tensor lv3_de3_conv2;
    Tensor lv3_de3_lrelu2;
    Tensor lv3_final_conv;
    Tensor lv3_reconstruct;
    Tensor lv3_upsample;

    lv4_out_lv3_in = Add(lv4_upsample, inputs[2], "out_L_4_in_L_3/in1");
    lv3_en1_conv1 = conv2D(lv4_out_lv3_in, lv3_en1_conv1_ws, Func(w_buffer[24]), Func(b_buffer[24]), "level_3-Encoder-Stage-1-Conv-1");
    lv3_en1_lrelu1 = Leaky_relu(lv3_en1_conv1, "level_3-Encoder-Stage-1-L-ReLU-1");
    lv3_en1_conv2 = conv2D(lv3_en1_lrelu1, lv3_en1_conv2_ws, Func(w_buffer[25]), Func(b_buffer[25]), "level_3-Encoder-Stage-1-Conv-2");
    lv3_en1_lrelu2 = Leaky_relu(lv3_en1_conv2, "level_3-Encoder-Stage-1-L-ReLU-2");
    lv3_en1_mpool = max_pool(lv3_en1_lrelu2, "level_3-Encoder-Stage-1-MaxPool");
    
    lv3_en2_conv1 = conv2D(lv3_en1_mpool, lv3_en2_conv1_ws, Func(w_buffer[26]), Func(b_buffer[26]), "level_3-Encoder-Stage-2-Conv-1");
    lv3_en2_lrelu1 = Leaky_relu(lv3_en2_conv1, "level_3-Encoder-Stage-2-L-ReLU-1");
    lv3_en2_conv2 = conv2D(lv3_en2_lrelu1, lv3_en2_conv2_ws, Func(w_buffer[27]), Func(b_buffer[27]), "level_3-Encoder-Stage-2-Conv-2");
    lv3_en2_lrelu2 = Leaky_relu(lv3_en2_conv2, "level_3-Encoder-Stage-2-L-ReLU-2");
    lv3_en2_mpool = max_pool(lv3_en2_lrelu2, "level_3-Encoder-Stage-2-MaxPool");

    lv3_en3_conv1 = conv2D(lv3_en2_mpool, lv3_en3_conv1_ws, Func(w_buffer[28]), Func(b_buffer[28]), "level_3-Encoder-Stage-3-Conv-1");
    lv3_en3_lrelu1 = Leaky_relu(lv3_en3_conv1, "level_3-Encoder-Stage-3-L-ReLU-1");
    lv3_en3_conv2 = conv2D(lv3_en3_lrelu1, lv3_en3_conv2_ws, Func(w_buffer[29]), Func(b_buffer[29]), "level_3-Encoder-Stage-3-Conv-2");
    lv3_en3_lrelu2 = Leaky_relu(lv3_en3_conv2, "level_3-Encoder-Stage-3-L-ReLU-2");
    lv3_en3_mpool = max_pool(lv3_en3_lrelu2, "level_3-Encoder-Stage-3-MaxPool");

    lv3_bd_conv1 = conv2D(lv3_en3_mpool, lv3_bd_conv1_ws, Func(w_buffer[30]), Func(b_buffer[30]), "level_3-Bridge-Conv-1");
    lv3_bd_lrelu1 = Leaky_relu(lv3_bd_conv1, "level_3-Bridge-L-ReLU-1");
    lv3_bd_conv2 = conv2D(lv3_bd_lrelu1,  lv3_bd_conv2_ws, Func(w_buffer[31]), Func(b_buffer[31]), "level_3-Bridge-Conv-2");
    lv3_bd_lrelu2 = Leaky_relu(lv3_bd_conv2, "level_3-Bridge-L-ReLU-2");

    lv3_de1_upconv = transposeConv2D(lv3_bd_lrelu2, lv3_de1_upconv_ws, Func(w_buffer[32]), Func(b_buffer[32]), "level_3-Decoder-Stage-1-UpConv");
    lv3_de1_uprelu = relu(lv3_de1_upconv, "level_3-Decoder-Stage-1-UpReLU");
    lv3_de1_concat = Concat(lv3_de1_uprelu, lv3_en3_lrelu2, "level_3-Decoder-Stage-1-DepthConcatenation");
    lv3_de1_conv1 = conv2D(lv3_de1_concat, lv3_de1_conv1_ws, Func(w_buffer[33]), Func(b_buffer[33]), "level_3-Decoder-Stage-1-Conv-1");
    lv3_de1_lrelu1 = Leaky_relu(lv3_de1_conv1, "level_3-Decoder-Stage-1-L-ReLU-1");
    lv3_de1_conv2 = conv2D(lv3_de1_lrelu1, lv3_de1_conv2_ws, Func(w_buffer[34]), Func(b_buffer[34]), "level_3-Decoder-Stage-1-Conv-2");
    lv3_de1_lrelu2 = Leaky_relu(lv3_de1_conv2, "level_3-Decoder-Stage-1-L-ReLU-2");

    lv3_de2_upconv = transposeConv2D(lv3_de1_lrelu2, lv3_de2_upconv_ws, Func(w_buffer[35]), Func(b_buffer[35]), "level_3-Decoder-Stage-2-UpConv");
    lv3_de2_uprelu = relu(lv3_de2_upconv, "level_3-Decoder-Stage-2-UpReLU");
    lv3_de2_concat = Concat(lv3_de2_uprelu, lv3_en2_lrelu2, "level_3-Decoder-Stage-2-DepthConcatenation");
    lv3_de2_conv1 = conv2D(lv3_de2_concat, lv3_de2_conv1_ws, Func(w_buffer[36]), Func(b_buffer[36]), "level_3-Decoder-Stage-2-Conv-1");
    lv3_de2_lrelu1 = Leaky_relu(lv3_de2_conv1, "level_3-Decoder-Stage-2-L-ReLU-1");
    lv3_de2_conv2 = conv2D(lv3_de2_lrelu1, lv3_de2_conv2_ws, Func(w_buffer[37]), Func(b_buffer[37]), "level_3-Decoder-Stage-2-Conv-2");
    lv3_de2_lrelu2 = Leaky_relu(lv3_de2_conv2, "level_3-Decoder-Stage-2-L-ReLU-2");

    lv3_de3_upconv = transposeConv2D(lv3_de2_lrelu2, lv3_de3_upconv_ws, Func(w_buffer[38]), Func(b_buffer[38]), "level_3-Decoder-Stage-3-UpConv");
    lv3_de3_uprelu = relu(lv3_de3_upconv, "level_3-Decoder-Stage-3-UpReLU");
    lv3_de3_concat = Concat(lv3_de3_uprelu, lv3_en1_lrelu2, "level_3-Decoder-Stage-3-DepthConcatenation");
    lv3_de3_conv1 = conv2D(lv3_de3_concat, lv3_de3_conv1_ws, Func(w_buffer[39]), Func(b_buffer[39]), "level_3-Decoder-Stage-3-Conv-1");
    lv3_de3_lrelu1 = Leaky_relu(lv3_de3_conv1, "level_3-Decoder-Stage-3-L-ReLU-1");
    lv3_de3_conv2 = conv2D(lv3_de3_lrelu1, lv3_de3_conv2_ws, Func(w_buffer[40]), Func(b_buffer[40]), "level_3-Decoder-Stage-3-Conv-2");
    lv3_de3_lrelu2 = Leaky_relu(lv3_de3_conv2, "level_3-Decoder-Stage-3-L-ReLU-2");
    
    lv3_final_conv = conv2D(lv3_de3_lrelu2, lv3_final_conv_ws, Func(w_buffer[41]), Func(b_buffer[41]), "level_3-Final-ConvolutionLayer");
    lv3_reconstruct = Add(lv3_final_conv, lv4_upsample, "level_3-reconstructLayer");
    lv3_upsample = transposeConv2D(lv3_reconstruct, lv3_upsample_ws, Func(w_buffer[42]), Func(b_buffer[42]), "level_3-upsampling");

    // Schedule
    lv3_en1_lrelu2.f.store_root();
    lv3_en2_lrelu2.f.store_root();
    lv3_en3_lrelu2.f.store_root();
    lv3_upsample.f.store_root();
    
    // Level 2
    const WeightShape lv2_en1_conv1_ws = {24, 3, 3};
    const WeightShape lv2_en1_conv2_ws = {24, 3, 3};
    const WeightShape lv2_en2_conv1_ws = {48, 3, 3};
    const WeightShape lv2_en2_conv2_ws = {48, 3, 3};
    const WeightShape lv2_en3_conv1_ws = {96, 3, 3};
    const WeightShape lv2_en3_conv2_ws = {96, 3, 3};

    const WeightShape lv2_bd_conv1_ws = {192, 3, 3};
    const WeightShape lv2_bd_conv2_ws = {192, 3, 3};

    const WeightShape lv2_de1_upconv_ws = {96, 2, 2};
    const WeightShape lv2_de1_conv1_ws = {96, 3, 3};
    const WeightShape lv2_de1_conv2_ws = {96, 3, 3};
    const WeightShape lv2_de2_upconv_ws = {48, 2, 2};
    const WeightShape lv2_de2_conv1_ws = {48, 3, 3};
    const WeightShape lv2_de2_conv2_ws = {48, 3, 3};
    const WeightShape lv2_de3_upconv_ws = {24, 2, 2};
    const WeightShape lv2_de3_conv1_ws = {24, 3, 3};
    const WeightShape lv2_de3_conv2_ws = {24, 3, 3};
    const WeightShape lv2_final_conv_ws = {3, 1, 1};
    const WeightShape lv2_upsample_ws = {3, 2, 2};

    Tensor lv3_out_lv2_in;
    Tensor lv2_en1_conv1;
    Tensor lv2_en1_lrelu1;
    Tensor lv2_en1_conv2;
    Tensor lv2_en1_lrelu2;
    Tensor lv2_en1_mpool;
    Tensor lv2_en2_conv1;
    Tensor lv2_en2_lrelu1;
    Tensor lv2_en2_conv2;
    Tensor lv2_en2_lrelu2;
    Tensor lv2_en2_mpool;
    Tensor lv2_en3_conv1;
    Tensor lv2_en3_lrelu1;
    Tensor lv2_en3_conv2;
    Tensor lv2_en3_lrelu2;
    Tensor lv2_en3_mpool;

    Tensor lv2_bd_conv1;
    Tensor lv2_bd_lrelu1;
    Tensor lv2_bd_conv2;
    Tensor lv2_bd_lrelu2;

    Tensor lv2_de1_upconv;
    Tensor lv2_de1_uprelu;
    Tensor lv2_de1_concat;
    Tensor lv2_de1_conv1;
    Tensor lv2_de1_lrelu1;
    Tensor lv2_de1_conv2;
    Tensor lv2_de1_lrelu2;
    Tensor lv2_de2_upconv;
    Tensor lv2_de2_uprelu;
    Tensor lv2_de2_concat;
    Tensor lv2_de2_conv1;
    Tensor lv2_de2_lrelu1;
    Tensor lv2_de2_conv2;
    Tensor lv2_de2_lrelu2;
    Tensor lv2_de3_upconv;
    Tensor lv2_de3_uprelu;
    Tensor lv2_de3_concat;
    Tensor lv2_de3_conv1;
    Tensor lv2_de3_lrelu1;
    Tensor lv2_de3_conv2;
    Tensor lv2_de3_lrelu2;
    Tensor lv2_final_conv;
    Tensor lv2_reconstruct;
    Tensor lv2_upsample;

    lv3_out_lv2_in = Add(lv3_upsample, inputs[1], "out_L_3_in_L_2/in1");
    lv2_en1_conv1 = conv2D(lv3_out_lv2_in, lv2_en1_conv1_ws, Func(w_buffer[43]), Func(b_buffer[43]), "level_2-Encoder-Stage-1-Conv-1");
    lv2_en1_lrelu1 = Leaky_relu(lv2_en1_conv1, "level_2-Encoder-Stage-1-L-ReLU-1");
    lv2_en1_conv2 = conv2D(lv2_en1_lrelu1, lv2_en1_conv2_ws, Func(w_buffer[44]), Func(b_buffer[44]), "level_2-Encoder-Stage-1-Conv-2");
    lv2_en1_lrelu2 = Leaky_relu(lv2_en1_conv2, "level_2-Encoder-Stage-1-L-ReLU-2");
    lv2_en1_mpool = max_pool(lv2_en1_lrelu2, "level_2-Encoder-Stage-1-MaxPool");
    
    lv2_en2_conv1 = conv2D(lv2_en1_mpool, lv2_en2_conv1_ws, Func(w_buffer[45]), Func(b_buffer[45]), "level_2-Encoder-Stage-2-Conv-1");
    lv2_en2_lrelu1 = Leaky_relu(lv2_en2_conv1, "level_2-Encoder-Stage-2-L-ReLU-1");
    lv2_en2_conv2 = conv2D(lv2_en2_lrelu1, lv2_en2_conv2_ws, Func(w_buffer[46]), Func(b_buffer[46]), "level_2-Encoder-Stage-2-Conv-2");
    lv2_en2_lrelu2 = Leaky_relu(lv2_en2_conv2, "level_2-Encoder-Stage-2-L-ReLU-2");
    lv2_en2_mpool = max_pool(lv2_en2_lrelu2, "level_2-Encoder-Stage-2-MaxPool");

    lv2_en3_conv1 = conv2D(lv2_en2_mpool, lv2_en3_conv1_ws, Func(w_buffer[47]), Func(b_buffer[47]), "level_2-Encoder-Stage-3-Conv-1");
    lv2_en3_lrelu1 = Leaky_relu(lv2_en3_conv1, "level_2-Encoder-Stage-3-L-ReLU-1");
    lv2_en3_conv2 = conv2D(lv2_en3_lrelu1, lv2_en3_conv2_ws, Func(w_buffer[48]), Func(b_buffer[48]), "level_2-Encoder-Stage-3-Conv-2");
    lv2_en3_lrelu2 = Leaky_relu(lv2_en3_conv2, "level_2-Encoder-Stage-3-L-ReLU-2");
    lv2_en3_mpool = max_pool(lv2_en3_lrelu2, "level_2-Encoder-Stage-3-MaxPool");
    
    lv2_bd_conv1 = conv2D(lv2_en3_mpool, lv2_bd_conv1_ws, Func(w_buffer[49]), Func(b_buffer[49]), "level_2-Bridge-Conv-1");
    lv2_bd_lrelu1 = Leaky_relu(lv2_bd_conv1, "level_2-Bridge-L-ReLU-1");
    lv2_bd_conv2 = conv2D(lv2_bd_lrelu1,  lv2_bd_conv2_ws, Func(w_buffer[50]), Func(b_buffer[50]), "level_2-Bridge-Conv-2");
    lv2_bd_lrelu2 = Leaky_relu(lv2_bd_conv2, "level_2-Bridge-L-ReLU-2");

    lv2_de1_upconv = transposeConv2D(lv2_bd_lrelu2, lv2_de1_upconv_ws, Func(w_buffer[51]), Func(b_buffer[51]), "level_2-Decoder-Stage-1-UpConv");
    lv2_de1_uprelu = relu(lv2_de1_upconv, "level_2-Decoder-Stage-1-UpReLU");
    lv2_de1_concat = Concat(lv2_de1_uprelu, lv2_en3_lrelu2, "level_2-Decoder-Stage-1-DepthConcatenation");
    lv2_de1_conv1 = conv2D(lv2_de1_concat, lv2_de1_conv1_ws, Func(w_buffer[52]), Func(b_buffer[52]), "level_2-Decoder-Stage-1-Conv-1");
    lv2_de1_lrelu1 = Leaky_relu(lv2_de1_conv1, "level_2-Decoder-Stage-1-L-ReLU-1");
    lv2_de1_conv2 = conv2D(lv2_de1_lrelu1, lv2_de1_conv2_ws, Func(w_buffer[53]), Func(b_buffer[53]), "level_2-Decoder-Stage-1-Conv-2");
    lv2_de1_lrelu2 = Leaky_relu(lv2_de1_conv2, "level_2-Decoder-Stage-1-L-ReLU-2");

    lv2_de2_upconv = transposeConv2D(lv2_de1_lrelu2, lv2_de2_upconv_ws, Func(w_buffer[54]), Func(b_buffer[54]), "level_2-Decoder-Stage-2-UpConv");
    lv2_de2_uprelu = relu(lv2_de2_upconv, "level_2-Decoder-Stage-2-UpReLU");
    lv2_de2_concat = Concat(lv2_de2_uprelu, lv2_en2_lrelu2, "level_2-Decoder-Stage-2-DepthConcatenation");
    lv2_de2_conv1 = conv2D(lv2_de2_concat, lv2_de2_conv1_ws, Func(w_buffer[55]), Func(b_buffer[55]), "level_2-Decoder-Stage-2-Conv-1");
    lv2_de2_lrelu1 = Leaky_relu(lv2_de2_conv1, "level_2-Decoder-Stage-2-L-ReLU-1");
    lv2_de2_conv2 = conv2D(lv2_de2_lrelu1, lv2_de2_conv2_ws, Func(w_buffer[56]), Func(b_buffer[56]), "level_2-Decoder-Stage-2-Conv-2");
    lv2_de2_lrelu2 = Leaky_relu(lv2_de2_conv2, "level_2-Decoder-Stage-2-L-ReLU-2");

    lv2_de3_upconv = transposeConv2D(lv2_de2_lrelu2, lv2_de3_upconv_ws, Func(w_buffer[57]), Func(b_buffer[57]), "level_2-Decoder-Stage-3-UpConv");
    lv2_de3_uprelu = relu(lv2_de3_upconv, "level_2-Decoder-Stage-3-UpReLU");
    lv2_de3_concat = Concat(lv2_de3_uprelu, lv2_en1_lrelu2, "level_2-Decoder-Stage-3-DepthConcatenation");
    lv2_de3_conv1 = conv2D(lv2_de3_concat, lv2_de3_conv1_ws, Func(w_buffer[58]), Func(b_buffer[58]), "level_2-Decoder-Stage-3-Conv-1");
    lv2_de3_lrelu1 = Leaky_relu(lv2_de3_conv1, "level_2-Decoder-Stage-3-L-ReLU-1");
    lv2_de3_conv2 = conv2D(lv2_de3_lrelu1, lv2_de3_conv2_ws, Func(w_buffer[59]), Func(b_buffer[59]), "level_2-Decoder-Stage-3-Conv-2");
    lv2_de3_lrelu2 = Leaky_relu(lv2_de3_conv2, "level_2-Decoder-Stage-3-L-ReLU-2");
    
    lv2_final_conv = conv2D(lv2_de3_lrelu2, lv2_final_conv_ws, Func(w_buffer[60]), Func(b_buffer[60]), "level_2-Final-ConvolutionLayer");
    lv2_reconstruct = Add(lv2_final_conv, lv3_upsample, "level_2-reconstructLayer");
    lv2_upsample = transposeConv2D(lv2_reconstruct, lv2_upsample_ws, Func(w_buffer[61]), Func(b_buffer[61]), "level_2-upsampling");

    // Schedule
    lv2_en1_lrelu2.f.store_root();
    lv2_en2_lrelu2.f.store_root();
    lv2_en3_lrelu2.f.store_root();
    lv2_upsample.f.store_root();
    
    // Level 1
    const WeightShape lv1_en1_conv1_ws = {16, 3, 3};
    const WeightShape lv1_en1_conv2_ws = {16, 3, 3};
    const WeightShape lv1_en2_conv1_ws = {32, 3, 3};
    const WeightShape lv1_en2_conv2_ws = {32, 3, 3};
    const WeightShape lv1_en3_conv1_ws = {64, 3, 3};
    const WeightShape lv1_en3_conv2_ws = {64, 3, 3};

    const WeightShape lv1_bd_conv1_ws = {128, 3, 3};
    const WeightShape lv1_bd_conv2_ws = {128, 3, 3};

    const WeightShape lv1_de1_upconv_ws = {64, 2, 2};
    const WeightShape lv1_de1_conv1_ws = {64, 3, 3};
    const WeightShape lv1_de1_conv2_ws = {64, 3, 3};
    const WeightShape lv1_de2_upconv_ws = {32, 2, 2};
    const WeightShape lv1_de2_conv1_ws = {32, 3, 3};
    const WeightShape lv1_de2_conv2_ws = {32, 3, 3};
    const WeightShape lv1_de3_upconv_ws = {16, 2, 2};
    const WeightShape lv1_de3_conv1_ws = {16, 3, 3};
    const WeightShape lv1_de3_conv2_ws = {16, 3, 3};
    const WeightShape lv1_final_conv_ws = {3, 1, 1};

    Tensor lv2_out_lv1_in;
    Tensor lv1_en1_conv1;
    Tensor lv1_en1_lrelu1;
    Tensor lv1_en1_conv2;
    Tensor lv1_en1_lrelu2;
    Tensor lv1_en1_mpool;
    Tensor lv1_en2_conv1;
    Tensor lv1_en2_lrelu1;
    Tensor lv1_en2_conv2;
    Tensor lv1_en2_lrelu2;
    Tensor lv1_en2_mpool;
    Tensor lv1_en3_conv1;
    Tensor lv1_en3_lrelu1;
    Tensor lv1_en3_conv2;
    Tensor lv1_en3_lrelu2;
    Tensor lv1_en3_mpool;

    Tensor lv1_bd_conv1;
    Tensor lv1_bd_lrelu1;
    Tensor lv1_bd_conv2;
    Tensor lv1_bd_lrelu2;

    Tensor lv1_de1_upconv;
    Tensor lv1_de1_uprelu;
    Tensor lv1_de1_concat;
    Tensor lv1_de1_conv1;
    Tensor lv1_de1_lrelu1;
    Tensor lv1_de1_conv2;
    Tensor lv1_de1_lrelu2;
    Tensor lv1_de2_upconv;
    Tensor lv1_de2_uprelu;
    Tensor lv1_de2_concat;
    Tensor lv1_de2_conv1;
    Tensor lv1_de2_lrelu1;
    Tensor lv1_de2_conv2;
    Tensor lv1_de2_lrelu2;
    Tensor lv1_de3_upconv;
    Tensor lv1_de3_uprelu;
    Tensor lv1_de3_concat;
    Tensor lv1_de3_conv1;
    Tensor lv1_de3_lrelu1;
    Tensor lv1_de3_conv2;
    Tensor lv1_de3_lrelu2;
    Tensor lv1_final_conv;
    Tensor lv1_reconstruct;

    lv2_out_lv1_in = Add(lv2_upsample, inputs[0], "out_L_2_in_L_1/in1");
    lv1_en1_conv1 = conv2D(lv2_out_lv1_in, lv1_en1_conv1_ws, Func(w_buffer[62]), Func(b_buffer[62]), "level_1-Encoder-Stage-1-Conv-1");
    lv1_en1_lrelu1 = Leaky_relu(lv1_en1_conv1, "level_1-Encoder-Stage-1-L-ReLU-1");
    lv1_en1_conv2 = conv2D(lv1_en1_lrelu1, lv1_en1_conv2_ws, Func(w_buffer[63]), Func(b_buffer[63]), "level_1-Encoder-Stage-1-Conv-2");
    lv1_en1_lrelu2 = Leaky_relu(lv1_en1_conv2, "level_1-Encoder-Stage-1-L-ReLU-2");
    lv1_en1_mpool = max_pool(lv1_en1_lrelu2, "level_1-Encoder-Stage-1-MaxPool");
    
    lv1_en2_conv1 = conv2D(lv1_en1_mpool, lv1_en2_conv1_ws, Func(w_buffer[64]), Func(b_buffer[64]), "level_1-Encoder-Stage-2-Conv-1");
    lv1_en2_lrelu1 = Leaky_relu(lv1_en2_conv1, "level_1-Encoder-Stage-2-L-ReLU-1");
    lv1_en2_conv2 = conv2D(lv1_en2_lrelu1, lv1_en2_conv2_ws, Func(w_buffer[65]), Func(b_buffer[65]), "level_1-Encoder-Stage-2-Conv-2");
    lv1_en2_lrelu2 = Leaky_relu(lv1_en2_conv2, "level_1-Encoder-Stage-2-L-ReLU-2");
    lv1_en2_mpool = max_pool(lv1_en2_lrelu2, "level_1-Encoder-Stage-2-MaxPool");

    lv1_en3_conv1 = conv2D(lv1_en2_mpool, lv1_en3_conv1_ws, Func(w_buffer[66]), Func(b_buffer[66]), "level_1-Encoder-Stage-3-Conv-1");
    lv1_en3_lrelu1 = Leaky_relu(lv1_en3_conv1, "level_1-Encoder-Stage-3-L-ReLU-1");
    lv1_en3_conv2 = conv2D(lv1_en3_lrelu1, lv1_en3_conv2_ws, Func(w_buffer[67]), Func(b_buffer[67]), "level_1-Encoder-Stage-3-Conv-2");
    lv1_en3_lrelu2 = Leaky_relu(lv1_en3_conv2, "level_1-Encoder-Stage-3-L-ReLU-2");
    lv1_en3_mpool = max_pool(lv1_en3_lrelu2, "level_1-Encoder-Stage-3-MaxPool");
    
    lv1_bd_conv1 = conv2D(lv1_en3_mpool, lv1_bd_conv1_ws, Func(w_buffer[68]), Func(b_buffer[68]), "level_1-Bridge-Conv-1");
    lv1_bd_lrelu1 = Leaky_relu(lv1_bd_conv1, "level_1-Bridge-L-ReLU-1");
    lv1_bd_conv2 = conv2D(lv1_bd_lrelu1,  lv1_bd_conv2_ws, Func(w_buffer[69]), Func(b_buffer[69]), "level_1-Bridge-Conv-2");
    lv1_bd_lrelu2 = Leaky_relu(lv1_bd_conv2, "level_1-Bridge-L-ReLU-2");

    lv1_de1_upconv = transposeConv2D(lv1_bd_lrelu2, lv1_de1_upconv_ws, Func(w_buffer[70]), Func(b_buffer[70]), "level_1-Decoder-Stage-1-UpConv");
    lv1_de1_uprelu = relu(lv1_de1_upconv, "level_1-Decoder-Stage-1-UpReLU");
    lv1_de1_concat = Concat(lv1_de1_uprelu, lv1_en3_lrelu2, "level_1-Decoder-Stage-1-DepthConcatenation");
    lv1_de1_conv1 = conv2D(lv1_de1_concat, lv1_de1_conv1_ws, Func(w_buffer[71]), Func(b_buffer[71]), "level_1-Decoder-Stage-1-Conv-1");
    lv1_de1_lrelu1 = Leaky_relu(lv1_de1_conv1, "level_1-Decoder-Stage-1-L-ReLU-1");
    lv1_de1_conv2 = conv2D(lv1_de1_lrelu1, lv1_de1_conv2_ws, Func(w_buffer[72]), Func(b_buffer[72]), "level_1-Decoder-Stage-1-Conv-2");
    lv1_de1_lrelu2 = Leaky_relu(lv1_de1_conv2, "level_1-Decoder-Stage-1-L-ReLU-2");

    lv1_de2_upconv = transposeConv2D(lv1_de1_lrelu2, lv1_de2_upconv_ws, Func(w_buffer[73]), Func(b_buffer[73]), "level_1-Decoder-Stage-2-UpConv");
    lv1_de2_uprelu = relu(lv1_de2_upconv, "level_1-Decoder-Stage-2-UpReLU");
    lv1_de2_concat = Concat(lv1_de2_uprelu, lv1_en2_lrelu2, "level_1-Decoder-Stage-2-DepthConcatenation");
    lv1_de2_conv1 = conv2D(lv1_de2_concat, lv1_de2_conv1_ws, Func(w_buffer[74]), Func(b_buffer[74]), "level_1-Decoder-Stage-2-Conv-1");
    lv1_de2_lrelu1 = Leaky_relu(lv1_de2_conv1, "level_1-Decoder-Stage-2-L-ReLU-1");
    lv1_de2_conv2 = conv2D(lv1_de2_lrelu1, lv1_de2_conv2_ws, Func(w_buffer[75]), Func(b_buffer[75]), "level_1-Decoder-Stage-2-Conv-2");
    lv1_de2_lrelu2 = Leaky_relu(lv1_de2_conv2, "level_1-Decoder-Stage-2-L-ReLU-2");

    lv1_de3_upconv = transposeConv2D(lv1_de2_lrelu2, lv1_de3_upconv_ws, Func(w_buffer[76]), Func(b_buffer[76]), "level_1-Decoder-Stage-3-UpConv");
    lv1_de3_uprelu = relu(lv1_de3_upconv, "level_1-Decoder-Stage-3-UpReLU");
    lv1_de3_concat = Concat(lv1_de3_uprelu, lv1_en1_lrelu2, "level_1-Decoder-Stage-3-DepthConcatenation");
    lv1_de3_conv1 = conv2D(lv1_de3_concat, lv1_de3_conv1_ws, Func(w_buffer[77]), Func(b_buffer[77]), "level_1-Decoder-Stage-3-Conv-1");
    lv1_de3_lrelu1 = Leaky_relu(lv1_de3_conv1, "level_1-Decoder-Stage-3-L-ReLU-1");
    lv1_de3_conv2 = conv2D(lv1_de3_lrelu1, lv1_de3_conv2_ws, Func(w_buffer[78]), Func(b_buffer[78]), "level_1-Decoder-Stage-3-Conv-2");
    lv1_de3_lrelu2 = Leaky_relu(lv1_de3_conv2, "level_1-Decoder-Stage-3-L-ReLU-2");
    
    lv1_final_conv = conv2D(lv1_de3_lrelu2, lv1_final_conv_ws, Func(w_buffer[79]), Func(b_buffer[79]), "level_1-Final-ConvolutionLayer");
    lv1_reconstruct = Add(lv1_final_conv, lv2_upsample, "level_1-reconstructLayer");

    // Schedule
    lv1_en1_lrelu2.f.store_root();
    lv1_en2_lrelu2.f.store_root();
    lv1_en3_lrelu2.f.store_root();
  
    return lv1_reconstruct;
}