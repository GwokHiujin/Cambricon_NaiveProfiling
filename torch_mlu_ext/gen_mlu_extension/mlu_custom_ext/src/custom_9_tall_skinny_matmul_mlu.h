#pragma once
#include <torch/extension.h>
torch::Tensor tall_skinny_matmul_mlu(torch::Tensor A, torch::Tensor B);