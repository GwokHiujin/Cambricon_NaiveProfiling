#pragma once
#include <torch/extension.h>
torch::Tensor maxpool1d_mlu_forward(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation);