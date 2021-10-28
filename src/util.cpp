#include "util.h"
#include <Halide.h>
#include <math.h>

using namespace Halide;

Expr kenerl_cubic(Expr x) {
    Expr xx = cast<float>(abs(x));
    Expr xx2 = xx * xx;
    Expr xx3 = xx2 * xx;
    float a = -0.5f;

    return select(xx < 1.0f, (a + 2.0f) * xx3 - (a + 3.0f) * xx2 + 1,
           select(xx < 2.0f, a * xx3 - 5.0f * a * xx2 + 8.0f * a * xx - 4.0f * a,
                  0.0f));
}

// Halide Resize: reference to https://github.com/halide/Halide/blob/master/apps/resize/resize.cpp
Tensor cubic_resize(Tensor input, int low_sz) {
    
    Var x, y, c, k;
    Func as_float("as_float");
    Func unnormalized_kernel_x("unnormalized_kernel_x"), unnormalized_kernel_y("unnormalized_kernel_y");
    Func kernel_x("kernel_x"), kernel_y("kernel_y");
    Func kernel_sum_x, kernel_sum_y;
    Func resized_x, resized_y;

    as_float(x, y, c) = cast<float>(input.f(x, y, c));
    int cubic_tap = 4;
    
    float scale_factor = float(low_sz) / float(std::max(input.shape[0], input.shape[1]));
    int resized_w = ceil(input.shape[0]*scale_factor);
    int resized_h = ceil(input.shape[1]*scale_factor);

    float inverse_scale_factor = 1.f / scale_factor;
    float kernel_scaling = scale_factor;
    float inverse_kernel_scaling = inverse_scale_factor;
    float kernel_radius = 0.5f * cubic_tap * inverse_kernel_scaling;
    int kernel_taps = int(ceil(cubic_tap * inverse_kernel_scaling));

    Expr sourcex = (x + 0.5f) * inverse_scale_factor - 0.5f;
    Expr sourcey = (y + 0.5f) * inverse_scale_factor - 0.5f;

    Expr beginx = cast<int>(ceil(sourcex - kernel_radius));
    Expr beginy = cast<int>(ceil(sourcey - kernel_radius));
    beginx = clamp(beginx, 0, input.shape[0] - kernel_taps);
    beginy = clamp(beginy, 0, input.shape[1] - kernel_taps);

    RDom r(0, kernel_taps);
    
    unnormalized_kernel_x(x, k) = kenerl_cubic((k + beginx - sourcex) * kernel_scaling);
    unnormalized_kernel_y(y, k) = kenerl_cubic((k + beginy - sourcey) * kernel_scaling);

    kernel_sum_x(x) = sum(unnormalized_kernel_x(x, r), "kernel_sum_x");
    kernel_sum_y(y) = sum(unnormalized_kernel_y(y, r), "kernel_sum_y");

    kernel_x(x, k) = unnormalized_kernel_x(x, k) / kernel_sum_x(x);
    kernel_y(y, k) = unnormalized_kernel_y(y, k) / kernel_sum_y(y);

    Func resized("resized");
    resized_y(x, y, c) = sum(kernel_y(y, r) * as_float(x, r+beginy, c), "resized_y");
    resized_x(x, y, c) = sum(kernel_x(x, r) * resized_y(r+beginx, y, c), "resized_x");
    resized(x, y, c) = cast<uint8_t>(clamp(resized_x(x, y, c), 0.f, 255.f));

    Tensor output;
    output.f = resized;
    output.shape = {resized_w, resized_h, input.shape[2]};
    output.name = "resized";

    // Schedule
    unnormalized_kernel_x.compute_root();
    unnormalized_kernel_y.compute_root();
    kernel_x.compute_root();
    kernel_y.compute_root();
    resized.compute_root().parallel(y).vectorize(x, 16);

    return output;
}


Tensor pre_padarray(Tensor input, int low_sz) {

    int w = low_sz - input.shape[0];
    int h = low_sz - input.shape[1];

    Var x, y, c;
    Func prepad("pre_padarray");

    prepad(x, y, c) = cast<float>(input.f(x-w, y-h, c));

    Tensor output;
    output.f = prepad;
    output.shape = {low_sz, low_sz, input.shape[2]};
    output.name = "pre_padarray";
        
    // Schedule
    prepad.compute_root().parallel(y).vectorize(x, 16);

    return output; 
}

Func downsample(Func img, int w, int h, int ch) {

    Buffer<float> filter(5, 5);
    filter(0,0)=0.0025f; filter(1,0)=0.0125f; filter(2,0)=0.02f; filter(3,0)=0.0125f; filter(4,0)=0.0025f;
    filter(0,1)=0.0125f; filter(1,1)=0.0625f; filter(2,1)=0.10f; filter(3,1)=0.0625f; filter(4,1)=0.0125f;
    filter(0,2)=0.0200f; filter(1,2)=0.1000f; filter(2,2)=0.16f; filter(3,2)=0.1000f; filter(4,2)=0.0200f;
    filter(0,3)=0.0125f; filter(1,3)=0.0625f; filter(2,3)=0.10f; filter(3,3)=0.0625f; filter(4,3)=0.0125f;
    filter(0,4)=0.0025f; filter(1,4)=0.0125f; filter(2,4)=0.02f; filter(3,4)=0.0125f; filter(4,4)=0.0025f;
    
    Var x, y, c, i, j;
    Func img_mirror = BoundaryConditions::mirror_interior(img, {{0, w}, {0, h}});
    Func out1("d_out1");
    Func out2("d_out2");
    Func output("downsample");

    int f_sz = filter.width();
    RDom r(0, f_sz, 0, f_sz);

    if (ch == 1) {
        out1(x, y) += filter(r.x, r.y) * img_mirror(x-f_sz/2+r.x, y-f_sz/2+r.y);
        out2(x, y) += filter(r.x, r.y) * out1(x-f_sz/2+r.x, y-f_sz/2+r.y);
        output(i, j) = out2(2*i, 2*j);
    } else {
        out1(x, y, c) += filter(r.x, r.y) * img_mirror(x-f_sz/2+r.x, y-f_sz/2+r.y, c);
        out2(x, y, c) += filter(r.x, r.y) * out1(x-f_sz/2+r.x, y-f_sz/2+r.y, c);
        output(i, j ,c) = out2(2*i, 2*j, c);
    }

    // Schedule
    out1.compute_root().parallel(y).vectorize(x, 16);
    output.compute_root().parallel(j).vectorize(i, 16);

    return output;
}

Func upsample(Func img, int w, int h, int ch) {

    Buffer<float> filter(5, 5);
    filter(0,0)=0.0025f; filter(1,0)=0.0125f; filter(2,0)=0.02f; filter(3,0)=0.0125f; filter(4,0)=0.0025f;
    filter(0,1)=0.0125f; filter(1,1)=0.0625f; filter(2,1)=0.10f; filter(3,1)=0.0625f; filter(4,1)=0.0125f;
    filter(0,2)=0.0200f; filter(1,2)=0.1000f; filter(2,2)=0.16f; filter(3,2)=0.1000f; filter(4,2)=0.0200f;
    filter(0,3)=0.0125f; filter(1,3)=0.0625f; filter(2,3)=0.10f; filter(3,3)=0.0625f; filter(4,3)=0.0125f;
    filter(0,4)=0.0025f; filter(1,4)=0.0125f; filter(2,4)=0.02f; filter(3,4)=0.0125f; filter(4,4)=0.0025f;

    Var x, y, c, i, j;
    Func img_repeat = BoundaryConditions::repeat_edge(img, {{0, w}, {0, h}});
    Func temp;
    Func out1("u_out1");
    Func out2("u_out2");
    Func output("upsample");

    temp(x, y, c) = select(x%2 == 0 && y%2 == 0, 4.f * img_repeat(x/2-1, y/2-1, c), 0.f);

    int f_sz = filter.width();
    RDom r(0, f_sz, 0, f_sz);

    if (ch == 1) {
        out1(x, y) += filter(r.x, r.y) * temp(x+r.x, y+r.y);
        out2(x, y) += filter(r.x, r.y) * out1(x+r.x, y+r.y);
        output(i, j) = out2(i-2, j-2);
    } else {
        out1(x, y, c) += filter(r.x, r.y) * temp(x+r.x, y+r.y, c);
        out2(x, y, c) += filter(r.x, r.y) * out1(x+r.x, y+r.y, c);
        output(i, j, c) = out2(i-2, j-2, c);
    }

    // Schedule
    out1.compute_root().parallel(y).vectorize(x, 16);
    output.compute_root().parallel(j).vectorize(i, 16);

    return output;
}

std::vector<Tensor> laplacian_pyramid(Tensor input, int nlev, float s1, float s2, std::string type) {
    
    Func J = input.f;
    std::vector<Tensor> pyrm(nlev);

    Var x, y, c;
    int w_pyramid = input.shape[0];
    int h_pyramid = input.shape[1];
    for (int l = 0; l < nlev; l++) {
        Func down = downsample(J, w_pyramid, h_pyramid, 3);

        if (l < nlev-1) {
            Func up = upsample(down, w_pyramid/2, h_pyramid/2, 3);
            if (type == "ML") {
                pyrm[l].f(c, x, y) = (J(x, y, c) - up(x, y, c)) * s1;
                pyrm[l].shape = {input.shape[2], w_pyramid, h_pyramid};
            } else {
                pyrm[l].f(x, y, c) = (J(x, y, c) - up(x, y, c)) * s1;
                pyrm[l].shape = {w_pyramid, h_pyramid, input.shape[2]};
            }
        } else {
            if (type == "ML") {
                pyrm[l].f(c, x, y)= J(x, y, c) * s2;
                pyrm[l].shape = {input.shape[2], w_pyramid, h_pyramid};
            } else {
                pyrm[l].f(x, y, c)= J(x, y, c) * s2;
                pyrm[l].shape = {w_pyramid, h_pyramid, input.shape[2]};
            }
        }
        J = down;
        w_pyramid = (w_pyramid+1) / 2;
        h_pyramid = (h_pyramid+1) / 2;

        // Schedule
        pyrm[l].f.compute_root().reorder(c, x, y).parallel(y).vectorize(x, 16);
    }

    return pyrm;
}

Tensor remove_padarray(Tensor input, int w, int h) {
    
    Var x, y, c;
    Func rempad;
    rempad(x, y, c) = cast<uint8_t>(clamp(input.f(c, x+input.shape[1]-w, y+input.shape[2]-h), 0.f, 255.f));

    Tensor output;
    output.f = rempad;
    output.shape = {w, h, input.shape[0]};
    output.name = "remove_pad_array";

    // Schedule
    rempad.compute_root().parallel(y).vectorize(x, 16);

    return output;
}

std::vector<Tensor> gaussian_pyramid(Tensor input, int nlev) {
    
    std::vector<Tensor> pyramid(nlev);
    int w_pyramid = input.shape[0];
    int h_pyramid = input.shape[1];
    
    pyramid[0].f = input.f;
    pyramid[0].shape = {w_pyramid, h_pyramid};
    
    for (int l = 1; l < nlev; l++) {
        Func down = downsample(pyramid[l-1].f, w_pyramid, h_pyramid, 1);
        w_pyramid = (w_pyramid+1) / 2;
        h_pyramid = (h_pyramid+1) / 2;
        pyramid[l].f = down;
        pyramid[l].shape = {w_pyramid, h_pyramid};
    }

    return pyramid;
}

Tensor reconstruct_laplacian_pyramid(std::vector<Tensor> input) {

    int nlev = input.size();
    int w_pyramid = input[nlev-1].shape[0];
    int h_pyramid = input[nlev-1].shape[1];

    Var x("rec_x"), y("rec_y"), c("rec_c");
    std::vector<Func> pyr(nlev);
    pyr[nlev-1](x, y, c) = input[nlev-1].f(x, y, c);

    for (int l = nlev-2; l > -1; l--) {
        Func up = upsample(pyr[l+1], w_pyramid, h_pyramid, 3);
        pyr[l](x, y, c) = input[l].f(x, y, c) + up(x, y, c);

        w_pyramid = input[l].shape[0];
        h_pyramid = input[l].shape[1];

        // Scheule
        pyr[l].compute_root().reorder(c, x, y).parallel(y).vectorize(x, 16);
    }

    Func reconstruct("reconstruct");
    reconstruct(x, y, c) = cast<uint8_t>(clamp(pyr[0](x, y, c)*255.f, 0.f, 255.f));

    Tensor output;
    output.f = reconstruct;
    output.shape = input[0].shape;

    return output;
}

Func contrast(Func input) {
    
    Buffer<float> h(3, 3);
    h(0,0) = 0.f; h(1, 0) =  1.f; h(2,0) = 0.f;
    h(0,1) = 1.f; h(1, 1) = -4.f; h(2,1) = 1.f;
    h(0,2) = 0.f; h(1, 2) =  1.f; h(2,2) = 0.f;

    Var x("x"), y("y");
    Func gray("gray");
    gray(x, y) = input(x, y, 0)*0.25f + input(x, y, 1)*0.5f + input(x, y, 2)*0.25f;

    RDom r(0, 3, 0, 3);
    Func output("contrast");
    output(x, y) = abs(sum(h(r.x, r.y) * gray(x+r.x-1, y+r.y-1)));

    // Schedule
    gray.compute_root().parallel(y).vectorize(x, 16);
    output.compute_root().parallel(y).vectorize(x, 16);

    return output;
}

Func saturation(Func input) {
    
    Var x("x"), y("y");
    Func mu("mean");
    mu(x, y) = (input(x, y, 0) + input(x, y, 1) + input(x, y, 2)) / 3.f;

    Func output("saturation");
    output(x, y) = sqrt((pow(input(x, y, 0) - mu(x, y), 2) + 
                         pow(input(x, y, 1) - mu(x, y), 2) +
                         pow(input(x, y, 2) - mu(x, y), 2)) / 3.f);

    // Schedule
    mu.compute_root().parallel(y).vectorize(x, 16);
    output.compute_root().parallel(y).vectorize(x, 16);

    return output;
}

Func well_exposedness(Func input) {

    Var x("x"), y("y");
    Expr R = exp(-0.5f * pow(input(x, y, 0) - 0.5f, 2) / 0.04f);
    Expr G = exp(-0.5f * pow(input(x, y, 1) - 0.5f, 2) / 0.04f);
    Expr B = exp(-0.5f * pow(input(x, y, 2) - 0.5f, 2) / 0.04f);

    Func output("well_exposedness");
    output(x, y) = R * G * B;

    // Schedule
    output.compute_root().parallel(y).vectorize(x, 16);

    return output;
}

Tensor exposure_fusion(Tensor input1, Tensor input2) {

    Var x("x"), y("y"), c("c");
    
    int width = input1.shape[0];
    int height = input1.shape[1];

    Func input1_redge = BoundaryConditions::repeat_edge(input1.f, {{0, width}, {0, height}});
    Func input2_redge = BoundaryConditions::repeat_edge(input2.f, {{0, width}, {0, height}});

    Func in1_norm("in1_norm"), in2_norm("in2_norm");
    in1_norm(x, y, c) = input1_redge(x, y, c) / 255.f;
    in2_norm(x, y, c) = input2_redge(x, y, c) / 255.f;

    Tensor input1_norm, input2_norm;
    input1_norm.f = in1_norm;
    input1_norm.shape = input1.shape;
    input2_norm.f = in2_norm;
    input2_norm.shape = input2.shape;

    Func co1 = contrast(input1_norm.f);
    Func sa1 = saturation(input1_norm.f);
    Func ex1 = well_exposedness(input1_norm.f);
    Func co2 = contrast(input2_norm.f);
    Func sa2 = saturation(input2_norm.f);
    Func ex2 = well_exposedness(input2_norm.f);

    Func prod1("prod1"), prod2("prod2"), w1("w1"), w2("w2");
    prod1(x, y) = co1(x, y) * sa1(x, y) * ex1(x, y) + 1e-12f;
    prod2(x, y) = co2(x, y) * sa2(x, y) * ex2(x, y) + 1e-12f;

    w1(x, y) = prod1(x, y) / (prod1(x, y) + prod2(x, y));
    w2(x, y) = prod2(x, y) / (prod1(x, y) + prod2(x, y));

    Tensor weight1, weight2;
    weight1.f = w1;
    weight1.shape = {input1.shape[0], input1.shape[1]};
    weight2.f = w2;
    weight2.shape = {input2.shape[0], input2.shape[1]};

    int nlev = floor(log(std::min(width, height)) / log(2.f));
    std::cout << "nlev=" << nlev << std::endl;

    std::vector<Tensor> pyrW1, pyrI1, pyrW2, pyrI2, pyr(nlev);
    pyrW1 = gaussian_pyramid(weight1, nlev);
    pyrI1 = laplacian_pyramid(input1_norm, nlev, 1.f, 1.f, "None");
    pyrW2 = gaussian_pyramid(weight2, nlev);
    pyrI2 = laplacian_pyramid(input2_norm, nlev, 1.f, 1.f, "None");

    for (int i = 0; i < nlev; i++) {
        pyr[i].f(x, y, c) = pyrW1[i].f(x, y) * pyrI1[i].f(x, y, c) + 
                            pyrW2[i].f(x, y) * pyrI2[i].f(x, y, c);
        pyr[i].shape = pyrI2[i].shape;

        // Schedule
        pyr[i].f.compute_root().parallel(y).vectorize(x, 16);
    }

    // Schedule
    w1.compute_root().parallel(y).vectorize(x, 16);
    w2.compute_root().parallel(y).vectorize(x, 16);

    return reconstruct_laplacian_pyramid(pyr);
}

