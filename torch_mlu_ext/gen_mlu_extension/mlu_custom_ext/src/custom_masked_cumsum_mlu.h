#pragma once
#include <torch/extension.h>
torch::Tensor masked_cumsum_mlu(torch::Tensor x, torch::Tensor mask);