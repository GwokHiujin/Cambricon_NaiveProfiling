#pragma once
#include <torch/extension.h>
torch::Tensor sigmoid_mlu_forward(torch::Tensor input);