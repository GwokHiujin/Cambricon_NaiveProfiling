import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_mlu
import copy
import mlu_custom_ext
import unittest


class TestMLU(unittest.TestCase):
    """
    Sample: test sigmoid
    """

    def test_sigmoid(self, shapes=[(1, 256)]):
        for shape in shapes:
            x_cpu = torch.randn(shape)
            x_mlu = x_cpu.to("mlu")
            y_mlu = mlu_custom_ext.ops.active_sigmoid_mlu(x_mlu)
            y_cpu = x_cpu.sigmoid()
            print(y_mlu.cpu())
            np.testing.assert_array_almost_equal(y_mlu.cpu(), y_cpu, decimal=3)


if __name__ == "__main__":
    unittest.main()
