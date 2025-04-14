#pragma once
#include <torch/extension.h>
torch::Tensor maxpool1d_mlu_forward(torch::Tensor input);