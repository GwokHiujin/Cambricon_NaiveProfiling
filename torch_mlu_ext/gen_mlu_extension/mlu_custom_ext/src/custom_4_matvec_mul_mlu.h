#pragma once
#include <torch/extension.h>
torch::Tensor matvec_mul_mlu(torch::Tensor A, torch::Tensor B);