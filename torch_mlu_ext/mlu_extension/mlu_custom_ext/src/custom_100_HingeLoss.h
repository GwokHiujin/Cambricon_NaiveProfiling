#pragma once
#include <torch/extension.h>
torch::Tensor hinge_loss_mlu(torch::Tensor predictions, torch::Tensor targets);