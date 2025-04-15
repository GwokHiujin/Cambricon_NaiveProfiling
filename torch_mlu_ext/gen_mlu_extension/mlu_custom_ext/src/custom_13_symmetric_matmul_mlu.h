#pragma once
#include <torch/extension.h>
torch::Tensor symmetric_matmul_mlu(torch::Tensor A, torch::Tensor B);