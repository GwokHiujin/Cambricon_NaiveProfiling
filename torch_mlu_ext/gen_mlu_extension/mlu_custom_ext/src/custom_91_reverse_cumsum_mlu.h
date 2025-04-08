#pragma once
#include <torch/extension.h>
torch::Tensor reverse_cumsum_mlu(torch::Tensor x);