#pragma once
#include <torch/extension.h>
torch::Tensor matmul_8_mlu(torch::Tensor A, torch::Tensor B);