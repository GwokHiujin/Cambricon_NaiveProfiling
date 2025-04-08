
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
        result_cpu = torch.mean(1 - cosine_sim)
        result_mlu = mlu_custom_ext.ops.cosine_similarity_loss_mlu(predictions_mlu, targets_mlu)
        result_mlu = torch.mean(1 - cosine_sim)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_hardsigmoid_mlu(self):

        batch_size = 16
        dim = 16384
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.hardsigmoid(x_cpu)
        result_mlu = mlu_custom_ext.ops.hardsigmoid_mlu(x_mlu)
        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)

    def test_gelu_mlu(self):

        batch_size = 16
        dim = 16384
        x_cpu = torch.randn(batch_size, dim)
        x_mlu = x_cpu.to("mlu")
        result_cpu = F.gelu(x_cpu)
        result_mlu = mlu_custom_ext.ops.gelu_mlu(x_mlu)
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
