#pragma once

#include <Halide.h>
#include <tuple>

using namespace Halide;

struct Tensor {
    Halide::Func f;
    std::vector<int> shape;
    std::string name;
};

struct WeightShape {
    int c;  // output channels
    int w;
    int h;
};

Tensor conv2D(const Tensor &input, const WeightShape &weight_shape, const Func &weight,
                const Func &bias, const std::string &name);

Tensor relu(const Tensor &input, const std::string &name);

Tensor Leaky_relu(const Tensor &input, const std::string &name);

Tensor max_pool(const Tensor &input, const std::string &name);

Tensor transposeConv2D(const Tensor &input, const WeightShape &weight_shape, const Func &weights,
                       const Func &bias, const std::string &name);

Tensor Concat(const Tensor &in1, const Tensor &in2, const std::string &name);

Tensor Add(const Tensor &t1, const Tensor &t2, const std::string &name);

