#pragma once
#include <torch/extension.h>
torch::Tensor l2_normalize_mlu(torch::Tensor x);