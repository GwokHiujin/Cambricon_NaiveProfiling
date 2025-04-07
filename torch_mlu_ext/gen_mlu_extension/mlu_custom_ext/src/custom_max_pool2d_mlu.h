#pragma once
#include <torch/extension.h>
torch::Tensor max_pool2d_mlu(torch::Tensor x, int kernel_size, int stride, int padding, int dilation);