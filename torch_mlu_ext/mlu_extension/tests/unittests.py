import torch
import numpy as np
import torch_mlu
import copy
import mlu_custom_ext
import unittest


class TestMLU(unittest.TestCase):
    """
    test sigmoid
    """

    def test_sigmoid(self, shapes=[(1, 256)]):
        for shape in shapes:
            x_cpu = torch.randn(shape)
            x_mlu = x_cpu.to("mlu")
            y_mlu = mlu_custom_ext.ops.active_sigmoid_mlu(x_mlu)
            y_cpu = x_cpu.sigmoid()
            print(y_mlu.cpu())
            np.testing.assert_array_almost_equal(y_mlu.cpu(), y_cpu, decimal=3)
    
    """
    test HingeLoss
    """

    def test_hinge_loss(self, shapes=[(256,)]):
        for shape in shapes:
            predictions_cpu = torch.randn(shape)
            predictions_mlu = predictions_cpu.to("mlu")
            targets_cpu = torch.randint(0, 2, shape).float() * 2 - 1
            targets_mlu = targets_cpu.to("mlu")

            output_mlu = mlu_custom_ext.ops.hinge_loss_mlu(predictions_mlu, targets_mlu)
            output_cpu = torch.mean(torch.clamp(1 - predictions_cpu * targets_cpu, min=0))

            print(output_mlu.cpu())
            np.testing.assert_array_almost_equal(output_mlu.cpu(), output_cpu, decimal=3)


if __name__ == "__main__":
    unittest.main()
