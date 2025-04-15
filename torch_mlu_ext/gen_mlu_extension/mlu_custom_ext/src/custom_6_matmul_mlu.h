#pragma once
#include <torch/extension.h>
torch::Tensor matmul_3_mlu(torch::Tensor A, torch::Tensor B);