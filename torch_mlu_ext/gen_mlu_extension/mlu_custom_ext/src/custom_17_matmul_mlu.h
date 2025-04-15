#pragma once
#include <torch/extension.h>
torch::Tensor matmul_7_mlu(torch::Tensor A, torch::Tensor B_T);