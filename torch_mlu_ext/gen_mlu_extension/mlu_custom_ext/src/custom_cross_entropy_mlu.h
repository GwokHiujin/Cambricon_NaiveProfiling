#pragma once
#include <torch/extension.h>
torch::Tensor cross_entropy_mlu(torch::Tensor predictions, torch::Tensor targets);