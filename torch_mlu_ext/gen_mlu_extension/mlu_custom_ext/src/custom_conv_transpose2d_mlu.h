#pragma once
#include <torch/extension.h>
torch::Tensor conv_transpose2d_mlu(
    torch::Tensor input,
    torch::Tensor weight,
    std::tuple<int, int> stride,
    std::tuple<int, int> padding
);