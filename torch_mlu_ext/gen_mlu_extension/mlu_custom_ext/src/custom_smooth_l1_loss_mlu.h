#pragma once
#include <torch/extension.h>
torch::Tensor smooth_l1_loss_mlu(torch::Tensor predictions, torch::Tensor targets);