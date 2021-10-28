#include "ml_tools.h"

using namespace Halide;

Func pad(Func f, Expr width, Expr height) {

    Halide::Region bounds(f.dimensions());
    bounds[1].min = 0;
    bounds[1].extent = width;
    bounds[2].min = 0;
    bounds[2].extent = height;
    
    return Halide::BoundaryConditions::constant_exterior(f, 0.0f, bounds);
}

Tensor conv2D(const Tensor &input, const WeightShape &weight_shape, const Func &weights,
                 const Func &bias, const std::string &name) {
    Var c, i, j, ii, jj;
    int p = weight_shape.w / 2;

    Func padded;
    padded = pad(input.f, 512, 512);

    RDom r(0, input.shape[0], 0, weight_shape.w, 0, weight_shape.h);
    Func conv;
    conv(c, i, j) = sum(cast<float>(weights(r.y, r.z, r.x, c)) * padded(r.x, i + r.y - p, j + r.z - p))
                        + cast<float>(bias(c));

    Tensor output;
    output.f = conv;
    output.name = name;
    output.shape = {weight_shape.c, input.shape[1], input.shape[2]};

    // Schedule
    conv.compute_root().parallel(j).vectorize(i, 16);

    return output;
}

Tensor relu(const Tensor &input, const std::string &name) {

    Var c, i, j, ii, jj;

    Func relu;
    relu(c, i, j) = max(0.f, input.f(c, i, j));
    
    Tensor output;
    output.f = relu;
    output.shape = input.shape;
    output.name = name;

    // Schedule
    relu.compute_root().reorder(c, i, j).parallel(j).vectorize(i, 16);

    return output;

}

Tensor Leaky_relu(const Tensor &input, const std::string &name) {

    Var c, i, j, ii, jj;

    Func relu;
    relu(c, i, j) = select(input.f(c, i, j) > 0.f, input.f(c, i, j), 0.2f * input.f(c, i, j));
    
    Tensor output;
    output.f = relu;
    output.shape = input.shape;
    output.name = name;

    // Schedule
    relu.compute_root().reorder(c, i, j).parallel(j).vectorize(i, 16);

    return output;

}

Tensor max_pool(const Tensor &input, const std::string &name) {
    
    Var c, i, j, ii, jj;

    RDom r(0, 2, 0, 2);
    Func pool;
    pool(c, i, j) = maximum(input.f(c, 2*i+r.x, 2*j+r.y));

    Tensor output;
    output.f = pool;
    output.name = name;
    output.shape = {input.shape[0], input.shape[1]/2, input.shape[2]/2};

    // Schedule
    pool.compute_root().reorder(c, i, j).parallel(j).vectorize(i, 16);
    
    return output;
}

Tensor transposeConv2D(const Tensor &input, const WeightShape &weight_shape, const Func &weights,
                 const Func &bias, const std::string &name) {
    
    Var c, i, j, ii, jj;

    RDom r(0, input.shape[0]);
    Func conv;
    conv(c, i, j) = sum(cast<float>(weights(i%2, j%2, c, r)) * input.f(r, i/2, j/2))
                     + cast<float>(bias(c));
    
    Tensor output;
    output.f = conv;
    output.name = name;
    output.shape = {weight_shape.c, input.shape[1]*2, input.shape[2]*2};

    // Schedule
    conv.compute_root().parallel(j).vectorize(i, 16);

    return output;
}

Tensor Concat(const Tensor &in1, const Tensor &in2, const std::string &name) {

    Var c, i, j, ii, jj;

    Func concat;
    concat(c, i, j) = select(c < in1.shape[0], in1.f(c, i, j), 
                                               in2.f(c-in1.shape[0], i, j));

    Tensor output;
    output.f = concat;
    output.name = name;
    output.shape = {in1.shape[0] + in2.shape[0], in1.shape[1], in1.shape[2]};

    // Schedule
    concat.compute_root().reorder(c, i, j).parallel(j).vectorize(i, 16);

    return output;            
}

Tensor Add(const Tensor &t1, const Tensor &t2, const std::string &name) {
    
    Var c, i, j, io, jo, ii, jj;
    
    assert(t1.shape == t2.shape);
    Func summed;
    summed(c, i, j) = t1.f(c, i, j) + t2.f(c, i, j);
    Tensor output;
    output.f = summed;
    output.shape = t1.shape;
    output.name = name;

    // Schedule
    summed.compute_root().reorder(c, i, j).parallel(j).vectorize(i, 16);

    return output;
}