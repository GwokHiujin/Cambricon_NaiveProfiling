#pragma once
#include <torch/extension.h>
torch::Tensor scan_mlu(torch::Tensor input);