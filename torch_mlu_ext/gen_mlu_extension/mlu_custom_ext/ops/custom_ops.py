
import torch
from torch import Tensor


def mse_mlu(predictions:Tensor,targets:Tensor,):
    return torch.ops.mlu_custom_ext.mse_mlu(predictions,targets,)

def softplus_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.softplus_mlu(input,)

def elu_forward_mlu(input:Tensor,alpha:double,):
    return torch.ops.mlu_custom_ext.elu_forward_mlu(input,alpha,)

def softmax_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.softmax_mlu(input,)

def maxpool1d_mlu_forward(input:Tensor,kernel_size:int64_t,stride:int64_t,padding:int64_t,dilation:int64_t,):
    return torch.ops.mlu_custom_ext.maxpool1d_mlu_forward(input,kernel_size,stride,padding,dilation,)

def sum_reduction_mlu(x:Tensor,dim:int64_t,):
    return torch.ops.mlu_custom_ext.sum_reduction_mlu(x,dim,)

def cumprod_mlu(x:Tensor,dim:int64_t,):
    return torch.ops.mlu_custom_ext.cumprod_mlu(x,dim,)

def scan_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.scan_mlu(input,)

def l2_normalize_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.l2_normalize_mlu(x,)

def matrix_scalar_mul_mlu(A:Tensor,s:double,):
    return torch.ops.mlu_custom_ext.matrix_scalar_mul_mlu(A,s,)

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

def leaky_relu_mlu(input:Tensor,negative_slope:double,):
    return torch.ops.mlu_custom_ext.leaky_relu_mlu(input,negative_slope,)

def reverse_cumsum_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.reverse_cumsum_mlu(x,)

def gelu_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.gelu_mlu(x,)

def tanh_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.tanh_mlu(input,)

def product_reduction_mlu(input:Tensor,reduction_dim:int64_t,):
    return torch.ops.mlu_custom_ext.product_reduction_mlu(input,reduction_dim,)

def triplet_margin_loss_mlu(anchor:Tensor,positive:Tensor,negative:Tensor,margin:double,):
    return torch.ops.mlu_custom_ext.triplet_margin_loss_mlu(anchor,positive,negative,margin,)

def selu_forward_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.selu_forward_mlu(x,)