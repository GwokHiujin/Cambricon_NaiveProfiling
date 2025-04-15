#pragma once
#include <torch/extension.h>
torch::Tensor matmul_2_mlu(torch::Tensor A, torch::Tensor B);