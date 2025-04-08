#pragma once
#include <torch/extension.h>
torch::Tensor matrix_scalar_mul_mlu(torch::Tensor A, double s);