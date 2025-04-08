#pragma once
#include <torch/extension.h>
torch::Tensor triplet_margin_loss_mlu(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, double margin);