#pragma once

#include <Halide.h>
#include <cstdio>
#include "ml_tools.h"

using namespace Halide;
using Halide::ConciseCasts::f32;
using Halide::ConciseCasts::i32;

Tensor fit_and_slice(Tensor low_res_out, Tensor low_res_in, Tensor high_res_in, float r_sigma, int s_sigma);