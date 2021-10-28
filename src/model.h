#pragma once

#include <Halide.h>
#include <tuple>
#include "ml_tools.h"

Tensor Model(std::vector<Tensor> inputs, std::vector<Buffer<double>> w_buffer
                                       , std::vector<Buffer<double>> b_buffer);