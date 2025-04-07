#pragma once
#include <torch/extension.h>
torch::Tensor hardsigmoid_mlu(torch::Tensor input);