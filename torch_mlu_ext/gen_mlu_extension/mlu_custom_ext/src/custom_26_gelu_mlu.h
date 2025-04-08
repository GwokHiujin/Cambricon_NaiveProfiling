#pragma once
#include <torch/extension.h>
torch::Tensor gelu_mlu(torch::Tensor input);