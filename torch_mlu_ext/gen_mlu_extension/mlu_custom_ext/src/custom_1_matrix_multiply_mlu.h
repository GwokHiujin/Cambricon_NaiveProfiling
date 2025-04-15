#pragma once
#include <torch/extension.h>
torch::Tensor matrix_multiply_mlu(torch::Tensor A, torch::Tensor B);