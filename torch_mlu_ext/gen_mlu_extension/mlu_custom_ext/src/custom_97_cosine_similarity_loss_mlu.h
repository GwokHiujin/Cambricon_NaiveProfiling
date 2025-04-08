#pragma once
#include <torch/extension.h>
torch::Tensor cosine_similarity_loss_mlu(torch::Tensor predictions, torch::Tensor targets);