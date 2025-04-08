
import torch
from torch import Tensor


def mse_mlu(predictions:Tensor,targets:Tensor,):
    return torch.ops.mlu_custom_ext.mse_mlu(predictions,targets,)

def softmax_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.softmax_mlu(input,)

def scan_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.scan_mlu(input,)

def l2_normalize_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.l2_normalize_mlu(x,)

def hardsigmoid_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.hardsigmoid_mlu(input,)

def matvec_mul_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matvec_mul_mlu(A,B,)

def gelu_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.gelu_mlu(input,)

def relu_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.relu_mlu(input,)

def cosine_similarity_loss_mlu(predictions:Tensor,targets:Tensor,):
    return torch.ops.mlu_custom_ext.cosine_similarity_loss_mlu(predictions,targets,)

def reverse_cumsum_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.reverse_cumsum_mlu(x,)

def tanh_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.tanh_mlu(input,)

def selu_forward_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.selu_forward_mlu(x,)