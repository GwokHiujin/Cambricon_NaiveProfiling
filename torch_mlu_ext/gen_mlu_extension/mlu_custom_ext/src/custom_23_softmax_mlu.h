#pragma once
#include <torch/extension.h>
torch::Tensor softmax_mlu(torch::Tensor input);