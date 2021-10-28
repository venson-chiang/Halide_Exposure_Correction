#include <Halide.h>
#include <halide_image_io.h>

using namespace Halide::Tools;
using namespace Halide;

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <float.h>
#include <fstream>
#include <iostream>

#include "src/util.h"
#include "src/ml_tools.h"
#include "src/model.h"
#include "src/load_model.h"
#include "src/fit_and_slice.h"

int main(int argc, char **argv) {

    ///////////////////////////////////////////////////////////////////////////////////////
    // Step1. Arrange input images to get high_resolution_input and low_resolution_input tensors
    ///////////////////////////////////////////////////////////////////////////////////////
    std::string weight_dir = std::string(argv[1]);
    std::string im_dir = std::string(argv[2]) + "/" + std::string(argv[3]);
    std::string image_name = std::string(argv[3]);
    std::string output_name = image_name.substr(0, image_name.length() - 4);

    Buffer<uint8_t> input = load_image(im_dir);

    // Get high_resolution_input tensor.
    Tensor high_res_in;
    high_res_in.f = Func(input);
    high_res_in.shape = {input.width(), input.height(), 3};
    high_res_in.name = "high_res_in";
    std::cout << "high_res_in  shape = " << high_res_in.shape[0] << ", " << high_res_in.shape[1] << ", " << high_res_in.shape[2] << std::endl;
    
    // Get low_resolution_input tensor.
    int low_sz = 512;
    Tensor low_in = cubic_resize(high_res_in, low_sz);
    Buffer<uint8_t> low_in_buffer = low_in.f.realize(low_in.shape);
    save_image(low_in_buffer, "output_images/" + output_name + "_low_res_in.jpg");

    Tensor low_res_in;
    low_res_in.f = BoundaryConditions::repeat_edge(low_in_buffer, {{0, low_in.shape[0]}, {0, low_in.shape[1]}});
    low_res_in.shape = low_in.shape;
    std::cout << "low_res_in   shape = " << low_res_in.shape[0] << ", " << low_res_in.shape[1] << ", " << low_res_in.shape[2] << std::endl;
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Step2. Apply ML to get low_resolution_out tensors
    ///////////////////////////////////////////////////////////////////////////////////////

    // Repeat eadge padarray to make square tensor.
    Tensor padarray = pre_padarray(low_res_in, low_sz);

    // Calculate laplacian pyramid and arrage variable order to ML order {channel, width ,height}.
    int nlev = 4;
    std::vector<Tensor> pyramid;
    pyramid = laplacian_pyramid(padarray, nlev, 1.5f, 1.05f, "ML");
    
    // Import weights, bias to ML model 
    std::vector<Buffer<double>> w_buffer = load_weight_buffer(weight_dir);
    std::vector<Buffer<double>> b_buffer = load_bias_buffer(weight_dir);
    Tensor ml_out = Model(pyramid, w_buffer, b_buffer);
    
    // Remove pad array and arrange variable order back to Image order {width, height channel}.
    Tensor low_out = remove_padarray(ml_out, low_res_in.shape[0], low_res_in.shape[1]);
    Buffer<uint8_t> low_out_buffer = low_out.f.realize(low_out.shape);
    save_image(low_out_buffer, "output_images/" + output_name + "_low_res_out.jpg");

    Tensor low_res_out;
    low_res_out.f = Func(low_out_buffer);
    low_res_out.shape = low_out.shape;
    low_res_out.name = "low_res_out";
    std::cout << "low_res_out  shape = " << low_res_out.shape[0] << ", " << low_res_out.shape[1] << ", " << low_res_out.shape[2] << std::endl;
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Step3. Apply bgu processing to get high_resolution_output from low_res_in, low_res_out, high_res_in
    ///////////////////////////////////////////////////////////////////////////////////////
    
    Tensor high_out = fit_and_slice(low_res_out, low_res_in, high_res_in, 1.f/8.f, 16);
    Buffer<uint8_t> high_out_buffer = high_out.f.realize(high_out.shape);
    save_image(high_out_buffer, "output_images/" + output_name + "_high_res_out.jpg");

    Tensor high_res_out;
    high_res_out.f = Func(high_out_buffer);
    high_res_out.shape = high_res_in.shape;
    high_res_out.name = "high_res_out";
    std::cout << "high_res_out shape = " << high_res_out.shape[0] << ", " << high_res_out.shape[1] << ", " << high_res_out.shape[2] << std::endl;
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Step4. Fuse high_res_out and high_res_in to get exposure_correction image
    ///////////////////////////////////////////////////////////////////////////////////////
    
    Tensor fusion = exposure_fusion(high_res_out, high_res_in);
    Buffer<uint8_t> output = fusion.f.realize({high_res_out.shape});
    save_image(output, "output_images/" + output_name + "_exposure_correct.jpg");
    
    printf("Success!!\n");
}