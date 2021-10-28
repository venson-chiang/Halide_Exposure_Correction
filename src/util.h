#pragma once

#include <Halide.h>
#include "ml_tools.h"

using namespace Halide;

Tensor cubic_resize(Tensor input, int low_sz);

Tensor pre_padarray(Tensor input, int low_sz);

Func downsample(Func img, int w, int h, int ch);

Func upsample(Func img, int w, int h, int ch);

std::vector<Tensor> laplacian_pyramid(Tensor input, int nlev, float s1, float s2, std::string type);

Tensor remove_padarray(Tensor input, int w, int h);

Tensor exposure_fusion(Tensor input1, Tensor input2);