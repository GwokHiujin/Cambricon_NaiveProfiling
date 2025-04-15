#pragma once
#include <torch/extension.h>
torch::Tensor softplus_mlu(torch::Tensor input);