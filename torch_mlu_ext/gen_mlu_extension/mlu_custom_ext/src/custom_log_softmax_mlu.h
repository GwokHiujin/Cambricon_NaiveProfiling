#pragma once
#include <torch/extension.h>
torch::Tensor log_softmax_mlu(torch::Tensor input, int dim);