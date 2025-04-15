#pragma once
#include <torch/extension.h>
torch::Tensor matmul_5_mlu(torch::Tensor a, torch::Tensor b);