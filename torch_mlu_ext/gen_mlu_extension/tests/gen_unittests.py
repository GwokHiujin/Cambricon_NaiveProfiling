
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch_mlu
import copy
import mlu_custom_ext
import unittest

class TestMLU(unittest.TestCase):
     
    # def test_matmul_6_mlu(self):

    #     M = 1024
    #     K = 4096
    #     N = 2048
    #     A_cpu = torch.randn(K, M)
    #     B_cpu = torch.randn(K, N)
    #     A_mlu = A_cpu.T.to("mlu")
    #     B_mlu = B_cpu.to("mlu")
    #     result_cpu = torch.matmul(A_cpu.T, B_cpu)
    #     result_mlu = mlu_custom_ext.ops.matmul_6_mlu(A_mlu, B_mlu)
    #     np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_mse_mlu(self):

        # M = 16384
        M = 4096
        N = 16
        A_cpu = torch.randn(M, N)
        B_cpu = torch.randn(N, M)
        A_mlu = A_cpu.to("mlu")
        B_mlu = B_cpu.to("mlu")
        result_cpu = torch.matmul(A_cpu, B_cpu)
        result_mlu = mlu_custom_ext.ops.mse_mlu(A_mlu, B_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_cosine_similarity_loss_mlu(self):

        batch_size = 128
        input_shape = (512,)
        dim = 1
        predictions_cpu = torch.randn(batch_size, *input_shape)
        targets_cpu = torch.randn(batch_size, *input_shape)
        predictions_mlu = predictions_cpu.to("mlu")
        targets_mlu = targets_cpu.to("mlu")
        cosine_sim = F.cosine_similarity(predictions_cpu, targets_cpu, dim=1)
        result_cpu = 1 - cosine_sim
        result_mlu = mlu_custom_ext.ops.cosine_similarity_loss_mlu(predictions_mlu, targets_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_leaky_relu_mlu(self):

        batch_size = 16
        # dim = 16384
        dim = 4096
        negative_slope = 0.01
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.leaky_relu(x_cpu, negative_slope)
        result_mlu = mlu_custom_ext.ops.leaky_relu_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_scan_mlu(self):

        # M = 8205
        # K = 2949
        # N = 5921
        M = 128
        K = 128
        N = 128
        A_cpu = torch.randn(M, K)
        B_cpu = torch.randn(K, N)
        A_mlu = A_cpu.to("mlu")
        B_mlu = B_cpu.to("mlu")
        result_cpu = torch.matmul(A_cpu, B_cpu)
        result_mlu = mlu_custom_ext.ops.scan_mlu(A_mlu, B_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_hardsigmoid_mlu(self):

        batch_size = 16
        # dim = 16384
        dim = 4096
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.hardsigmoid(x_cpu)
        result_mlu = mlu_custom_ext.ops.hardsigmoid_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)
    
    # def test_matmul_2_mlu(self):

    #     # M = 1024
    #     # K = 4096
    #     # N = 2048
    #     M = 256
    #     K = 1024
    #     N = 512
    #     A_cpu = torch.randn(M, K)
    #     B_cpu = torch.randn(K, N)
    #     A_mlu = A_cpu.to("mlu")
    #     B_mlu = B_cpu.to("mlu")
    #     result_cpu = torch.matmul(A_cpu, B_cpu)
    #     result_mlu = mlu_custom_ext.ops.matmul_2_mlu(A_mlu, B_mlu)
    #     np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)
    
    # def test_matmul_3_mlu(self):

    #     M = 256
    #     N = 256
    #     K = 131072
    #     A_cpu = torch.randn(M, K)
    #     B_cpu = torch.randn(K, N)
    #     A_mlu = A_cpu.to("mlu")
    #     B_mlu = B_cpu.to("mlu")
    #     result_cpu = torch.matmul(A_cpu, B_cpu)
    #     result_mlu = mlu_custom_ext.ops.matmul_3_mlu(A_mlu, B_mlu)
    #     np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)
    
    # def test_matmul_4_mlu(self):

    #     # M = 16384
    #     # N = 16384
    #     M = 4096
    #     N = 4096
    #     K = 32
    #     A_cpu = torch.randn(M, K)
    #     B_cpu = torch.randn(K, N)
    #     A_mlu = A_cpu.to("mlu")
    #     B_mlu = B_cpu.to("mlu")
    #     result_cpu = torch.matmul(A_cpu, B_cpu)
    #     result_mlu = mlu_custom_ext.ops.matmul_4_mlu(A_mlu, B_mlu)
    #     np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)
     
    # def test_matmul_5_mlu(self):

    #     M = 8205
    #     K = 2949
    #     N = 5921
    #     A_cpu = torch.randn(M, K)
    #     B_cpu = torch.randn(K, N)
    #     A_mlu = A_cpu.to("mlu")
    #     B_mlu = B_cpu.to("mlu")
    #     result_cpu = torch.matmul(A_cpu, B_cpu)
    #     result_mlu = mlu_custom_ext.ops.matmul_5_mlu(A_mlu, B_mlu)
    #     np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

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
        # dim = 16384
        dim = 4096
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

    # def test_matmul_7_mlu(self):

    #     M = 1024
    #     K = 4096
    #     N = 2048
    #     A_cpu = torch.randn(M, K)
    #     B_cpu = torch.randn(N, K)
    #     A_mlu = A_cpu.to("mlu")
    #     B_mlu = B_cpu.T.to("mlu")
    #     result_cpu = torch.matmul(A_cpu, B_cpu.T)
    #     result_mlu = mlu_custom_ext.ops.matmul_7_mlu(A_mlu, B_mlu)
    #     np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_tanh_mlu(self):

        batch_size = 16
        # dim = 16384
        dim = 4096
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = torch.tanh(x_cpu)
        result_mlu = mlu_custom_ext.ops.tanh_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_softplus_mlu(self):

        batch_size = 16
        # dim = 16384
        dim = 4096
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.softplus(x_cpu)
        result_mlu = mlu_custom_ext.ops.softplus_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)
    
    def test_tensor_matrix_multiply_mlu(self):

        b = 16
        i = 32
        j = 64
        l = 32
        k = 64
        A_cpu = torch.randn(b, i, j, l)
        B_cpu = torch.randn(l, k)
        A_mlu = A_cpu.to("mlu")
        B_mlu = B_cpu.to("mlu")
        result_cpu = torch.einsum("bijl,lk->bijk", A_cpu, B_cpu)
        result_mlu = mlu_custom_ext.ops.tensor_matrix_multiply_mlu(A_mlu, B_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)
    
    def test_symmetric_matmul_mlu(self):

        N = 256
        A_cpu = torch.randn(N, N)
        A_cpu = (A_cpu + A_cpu.T) / 2  # Ensure symmetry
        B_cpu = torch.randn(N, N)
        B_cpu = (B_cpu + B_cpu.T) / 2  # Ensure symmetry
        A_mlu = A_cpu.to("mlu")
        B_mlu = B_cpu.to("mlu")
        result_cpu = torch.matmul(A_cpu, B_cpu)
        result_mlu = mlu_custom_ext.ops.symmetric_matmul_mlu(A_mlu, B_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_symmetric_matmul_mlu(self):

        N = 256
        A_cpu = torch.randn(N, N)
        B_cpu = torch.randn(N, N)
        A_mlu = A_cpu.to("mlu")
        B_mlu = B_cpu.to("mlu")
        result_cpu = torch.matmul(A_cpu, B_cpu)
        result_mlu = mlu_custom_ext.ops.symmetric_matmul_mlu(A_mlu, B_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_softmax_mlu(self):

        batch_size = 16
        # dim = 16384
        dim = 4096
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

    def test_new_gelu_mlu(self):

        batch_size = 2000
        dim = 2000
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = (
        0.5
        * x_cpu
        * (
        1.0
        + torch.tanh(math.sqrt(2.0 / math.pi) * (x_cpu + 0.044715 * torch.pow(x_cpu, 3.0)))
        )
        )
        result_mlu = mlu_custom_ext.ops.new_gelu_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_l2_normalize_mlu(self):

        batch_size = 16
        # dim = 16384
        dim = 4096
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.normalize(x_cpu, p=2, dim=1)
        result_mlu = mlu_custom_ext.ops.l2_normalize_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_scan_mlu(self):

        batch_size = 128
        # input_shape = (4000,)  # Example shape (arbitrary)
        input_shape = (512,)
        dim = 1
        x_cpu = torch.randn(batch_size, *input_shape)
        x_mlu = x_cpu.to("mlu")
        result_cpu = torch.cumsum(x_cpu, dim=dim)
        result_mlu = mlu_custom_ext.ops.scan_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    # def test_matmul_5_mlu(self):

    #     M = 256
    #     N = 256
    #     # K = 131072
    #     K = 4096
    #     A_cpu = torch.randn(M, K)
    #     B_cpu = torch.randn(K, N)
    #     A_mlu = A_cpu.to("mlu")
    #     B_mlu = B_cpu.to("mlu")
    #     result_cpu = torch.matmul(A_cpu, B_cpu)
    #     result_mlu = mlu_custom_ext.ops.matmul_5_mlu(A_mlu, B_mlu)
    #     np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_relu_mlu(self):

        batch_size = 16
        # dim = 16384
        dim = 4096
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.relu(x_cpu)
        result_mlu = mlu_custom_ext.ops.relu_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_selu_forward_mlu(self):

        batch_size = 16
        # dim = 16384
        dim = 4096
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
        input_shape = (512,)
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

        # M = 16384
        # N = 4096
        M = 128
        N = 4096
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
        # dim = 16384
        dim = 4096
        alpha = 1.0
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.elu(x_cpu, alpha=alpha)
        result_mlu = mlu_custom_ext.ops.elu_forward_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    # def test_matmul_8_mlu(self):

    #     # M = 1024
    #     # K = 4096
    #     # N = 2048
    #     M = 256
    #     K = 1024
    #     N = 512
    #     A_cpu = torch.randn(K, M)
    #     B_cpu = torch.randn(N, K)
    #     A_mlu = A_cpu.T.to("mlu")
    #     B_mlu = B_cpu.T.to("mlu")
    #     result_cpu = torch.matmul(A_cpu.T, B_cpu.T)
    #     result_mlu = mlu_custom_ext.ops.matmul_8_mlu(A_mlu, B_mlu)
    #     np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_mse_mlu(self):

        batch_size = 128
        input_shape = (512,)
        dim = 1
        predictions_cpu = torch.randn(batch_size, *input_shape)
        targets_cpu = torch.randn(batch_size, *input_shape)
        predictions_mlu = predictions_cpu.to("mlu")
        targets_mlu = targets_cpu.to("mlu")
        result_cpu = F.mse_loss(predictions_cpu, targets_cpu, reduction="mean")
        result_mlu = mlu_custom_ext.ops.mse_mlu(predictions_mlu, targets_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_leaky_relu_mlu(self):

        batch_size = 16
        dim = 16384
        negative_slope = 0.01
        A_cpu = torch.randn(batch_size, dim)
        A_mlu = A_cpu.to("mlu")
        result_cpu = F.leaky_relu(A_cpu, negative_slope)
        result_mlu = mlu_custom_ext.ops.leaky_relu_mlu(A_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)
     
if __name__ == "__main__":
    unittest.main()
