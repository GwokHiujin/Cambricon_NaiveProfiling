#pragma once
#include <torch/extension.h>
torch::Tensor swish_mlu(torch::Tensor input);