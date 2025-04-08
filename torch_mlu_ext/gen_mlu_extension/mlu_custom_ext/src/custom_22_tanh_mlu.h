#pragma once
#include <torch/extension.h>
torch::Tensor tanh_mlu(torch::Tensor input);