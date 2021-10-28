#pragma once

#include <Halide.h>

using namespace  Halide;

Buffer<double> load_conv_params(std::string shapefile, std::string datafile);

std::vector<Buffer<double>> load_weight_buffer(std::string weight_dir);

std::vector<Buffer<double>> load_bias_buffer(std::string weight_dir);