#pragma once
#include <torch/extension.h>
torch::Tensor mse_mlu(torch::Tensor predictions, torch::Tensor targets);