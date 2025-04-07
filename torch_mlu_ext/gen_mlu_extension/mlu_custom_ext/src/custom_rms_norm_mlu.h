#pragma once
#include <torch/extension.h>
torch::Tensor rms_norm_mlu(torch::Tensor x, float eps);