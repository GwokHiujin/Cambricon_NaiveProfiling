#pragma once
#include <torch/extension.h>
torch::Tensor cumprod_mlu(torch::Tensor x);