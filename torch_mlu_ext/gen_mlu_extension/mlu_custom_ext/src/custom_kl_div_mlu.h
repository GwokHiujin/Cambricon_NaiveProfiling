#pragma once
#include <torch/extension.h>
torch::Tensor kl_div_mlu(torch::Tensor log_predictions, torch::Tensor targets);