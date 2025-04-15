#pragma once
#include <torch/extension.h>
torch::Tensor matmul_4_mlu(torch::Tensor A, torch::Tensor B);