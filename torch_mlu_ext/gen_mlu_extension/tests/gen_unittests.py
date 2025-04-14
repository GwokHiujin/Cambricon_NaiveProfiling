
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_mlu
import copy
import mlu_custom_ext
import unittest

class TestMLU(unittest.TestCase):

    def test_cosine_similarity_loss_mlu(self):

        batch_size = 128
        input_shape = (4096,)
        dim = 1
        predictions_cpu = torch.randn(batch_size, *input_shape)
        targets_cpu = torch.randn(batch_size, *input_shape)
        predictions_mlu = predictions_cpu.to("mlu")
        targets_mlu = targets_cpu.to("mlu")
        cosine_sim = F.cosine_similarity(predictions_cpu, targets_cpu, dim=1)
        result_cpu = cosine_sim
        result_mlu = mlu_custom_ext.ops.cosine_similarity_loss_mlu(predictions_mlu, targets_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_leaky_relu_mlu(self):

        batch_size = 16
        dim = 16384
        negative_slope = 0.01
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.leaky_relu(x_cpu, negative_slope)
        result_mlu = mlu_custom_ext.ops.leaky_relu_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_hardsigmoid_mlu(self):

        batch_size = 16
        dim = 16384
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.hardsigmoid(x_cpu)
        result_mlu = mlu_custom_ext.ops.hardsigmoid_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_maxpool1d_mlu_forward(self):

        M = 256
        K = 131072
        A_cpu = torch.randn(M, K)
        B_cpu = torch.randn(K, 1)
        A_mlu = A_cpu.to("mlu")
        B_mlu = B_cpu.to("mlu")
        result_cpu = torch.matmul(A_cpu, B_cpu)
        result_mlu = mlu_custom_ext.ops.maxpool1d_mlu_forward(A_mlu, B_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_gelu_mlu(self):

        batch_size = 16
        dim = 16384
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.gelu(x_cpu)
        result_mlu = mlu_custom_ext.ops.gelu_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_cumprod_mlu(self):

        batch_size = 128
        input_shape = (4000,)
        dim = 1
        x_cpu = torch.randn(batch_size, *input_shape)
        x_mlu = x_cpu.to("mlu")
        result_cpu = torch.cumprod(x_cpu, dim=dim)
        result_mlu = mlu_custom_ext.ops.cumprod_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_product_reduction_mlu(self):

        batch_size = 16
        dim1 = 256
        dim2 = 256
        reduction_dim = 1
        x_cpu = torch.randn(batch_size, dim1, dim2)
        x_mlu = x_cpu.to("mlu")
        result_cpu = torch.prod(x_cpu, dim=reduction_dim)
        result_mlu = mlu_custom_ext.ops.product_reduction_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_tanh_mlu(self):

        batch_size = 16
        dim = 16384
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = torch.tanh(x_cpu)
        result_mlu = mlu_custom_ext.ops.tanh_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_softmax_mlu(self):

        batch_size = 16
        dim = 16384
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.softmax(x_cpu, dim=1)
        result_mlu = mlu_custom_ext.ops.softmax_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_sum_reduction_mlu(self):

        batch_size = 16
        dim1 = 256
        dim2 = 256
        reduce_dim = 1
        x_cpu = torch.randn(batch_size, dim1, dim2)
        x_mlu = x_cpu.to("mlu")
        result_cpu = torch.sum(x_cpu, dim=reduce_dim, keepdim=True)
        result_mlu = mlu_custom_ext.ops.sum_reduction_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_l2_normalize_mlu(self):

        batch_size = 16
        dim = 16384
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.normalize(x_cpu, p=2, dim=1)
        result_mlu = mlu_custom_ext.ops.l2_normalize_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_scan_mlu(self):

        batch_size = 128
        input_shape = (4000,)  # Example shape (arbitrary)
        dim = 1
        x_cpu = torch.randn(batch_size, *input_shape)
        x_mlu = x_cpu.to("mlu")
        result_cpu = torch.cumsum(x_cpu, dim=dim)
        result_mlu = mlu_custom_ext.ops.scan_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_relu_mlu(self):

        batch_size = 16
        dim = 16384
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.relu(x_cpu)
        result_mlu = mlu_custom_ext.ops.relu_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_selu_forward_mlu(self):

        batch_size = 16
        dim = 16384
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.selu(x_cpu)
        result_mlu = mlu_custom_ext.ops.selu_forward_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_reverse_cumsum_mlu(self):

        batch_size = 128
        input_shape = (4000,)
        dim = 1
        x_cpu = torch.randn(batch_size, *input_shape)
        x_mlu = x_cpu.to("mlu")
        result_cpu = torch.cumsum(x_cpu.flip(dim), dim=dim).flip(dim)
        result_mlu = mlu_custom_ext.ops.reverse_cumsum_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_triplet_margin_loss_mlu(self):

        batch_size = 128
        input_shape = (4096,)
        dim = 1
        margin = 1.0
        anchor_cpu = torch.randn(batch_size, *input_shape)
        positive_cpu = torch.randn(batch_size, *input_shape)
        negative_cpu = torch.randn(batch_size, *input_shape)
        anchor_mlu = anchor_cpu.to("mlu")
        positive_mlu = positive_cpu.to("mlu")
        negative_mlu = negative_cpu.to("mlu")
        result_cpu = F.triplet_margin_loss(anchor_cpu, positive_cpu, negative_cpu, margin=margin)
        result_mlu = mlu_custom_ext.ops.triplet_margin_loss_mlu(anchor_mlu, positive_mlu, negative_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_matrix_scalar_mul_mlu(self):

        M = 128     # Origin: 16384
        N = 4096
        # FIXME: Found that when M*N is too large, torch_tensor.to("mlu") will fail
        A_cpu = torch.randn(M, N)
        s_cpu = 3.14
        A_mlu = A_cpu.to("mlu")
        result_cpu = A_cpu * s_cpu
        result_mlu = mlu_custom_ext.ops.matrix_scalar_mul_mlu(A_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_maxpool1d_mlu_forward(self):

        batch_size = 16
        features = 64
        sequence_length = 128
        kernel_size = 4
        stride = 2
        padding = 2
        dilation = 3
        return_indices = False
        x_cpu = torch.randn(batch_size, features, sequence_length)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.max_pool1d(
        x_cpu,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=return_indices,
        )
        result_mlu = mlu_custom_ext.ops.maxpool1d_mlu_forward(
        x_mlu,
        )
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_elu_forward_mlu(self):

        batch_size = 16
        dim = 16384
        alpha = 1.0
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.elu(x_cpu, alpha=alpha)
        result_mlu = mlu_custom_ext.ops.elu_forward_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_mse_mlu(self):

        batch_size = 128
        input_shape = (4096,)
        dim = 1
        predictions_cpu = torch.randn(batch_size, *input_shape)
        targets_cpu = torch.randn(batch_size, *input_shape)
        predictions_mlu = predictions_cpu.to("mlu")
        targets_mlu = targets_cpu.to("mlu")
        result_cpu = F.mse_loss(predictions_cpu, targets_cpu, reduction="mean")
        result_mlu = mlu_custom_ext.ops.mse_mlu(predictions_mlu, targets_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)
if __name__ == "__main__":
    unittest.main()
