#pragma once
#include <torch/extension.h>
torch::Tensor matmul_6_mlu(torch::Tensor A, torch::Tensor B);