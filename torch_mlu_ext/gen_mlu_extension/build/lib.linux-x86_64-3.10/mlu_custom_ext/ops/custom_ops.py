
import torch
from torch import Tensor


def mse_mlu(predictions:Tensor,targets:Tensor,):
    return torch.ops.mlu_custom_ext.mse_mlu(predictions,targets,)

def elu_forward_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.elu_forward_mlu(input,)

def softmax_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.softmax_mlu(input,)

def maxpool1d_mlu_forward(input:Tensor,):
    return torch.ops.mlu_custom_ext.maxpool1d_mlu_forward(input,)

def sum_reduction_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.sum_reduction_mlu(x,)

def cumprod_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.cumprod_mlu(x,)

def scan_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.scan_mlu(input,)

def l2_normalize_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.l2_normalize_mlu(x,)

def matrix_scalar_mul_mlu(A:Tensor,):
    return torch.ops.mlu_custom_ext.matrix_scalar_mul_mlu(A,)

def hardsigmoid_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.hardsigmoid_mlu(input,)

def matvec_mul_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matvec_mul_mlu(A,B,)

def relu_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.relu_mlu(input,)

def cosine_similarity_loss_mlu(predictions:Tensor,targets:Tensor,):
    return torch.ops.mlu_custom_ext.cosine_similarity_loss_mlu(predictions,targets,)

def leaky_relu_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.leaky_relu_mlu(input,)

def reverse_cumsum_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.reverse_cumsum_mlu(x,)

def gelu_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.gelu_mlu(x,)

def tanh_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.tanh_mlu(input,)

def product_reduction_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.product_reduction_mlu(input,)

def triplet_margin_loss_mlu(anchor:Tensor,positive:Tensor,negative:Tensor,):
    return torch.ops.mlu_custom_ext.triplet_margin_loss_mlu(anchor,positive,negative,)

def selu_forward_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.selu_forward_mlu(x,)