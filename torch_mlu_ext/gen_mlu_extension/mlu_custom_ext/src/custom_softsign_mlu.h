#pragma once
#include <torch/extension.h>
torch::Tensor softsign_mlu(torch::Tensor x);