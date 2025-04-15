
import torch
from torch import Tensor


def frobenius_norm_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.frobenius_norm_mlu(input,)

def mse_mlu(predictions:Tensor,targets:Tensor,):
    return torch.ops.mlu_custom_ext.mse_mlu(predictions,targets,)

def softplus_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.softplus_mlu(input,)

def elu_forward_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.elu_forward_mlu(input,)

def softmax_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.softmax_mlu(input,)

def matmul_2_mlu(a:Tensor,b:Tensor,):
    return torch.ops.mlu_custom_ext.matmul_2_mlu(a,b,)

def maxpool1d_mlu_forward(input:Tensor,):
    return torch.ops.mlu_custom_ext.maxpool1d_mlu_forward(input,)

def sum_reduction_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.sum_reduction_mlu(x,)

def matmul_3_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matmul_3_mlu(A,B,)

def cumprod_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.cumprod_mlu(x,)

def tensor_matrix_multiply_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.tensor_matrix_multiply_mlu(A,B,)

def matmul_4_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matmul_4_mlu(A,B,)

def matmul_5_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matmul_5_mlu(A,B,)

def batch_norm_mlu(input:Tensor,gamma:Tensor,beta:Tensor,running_mean:Tensor,running_var:Tensor,):
    return torch.ops.mlu_custom_ext.batch_norm_mlu(input,gamma,beta,running_mean,running_var,)

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

def tall_skinny_matmul_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.tall_skinny_matmul_mlu(A,B,)

def gelu_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.gelu_mlu(input,)

def matmul_6_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matmul_6_mlu(A,B,)

def relu_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.relu_mlu(input,)

def cosine_similarity_loss_mlu(predictions:Tensor,targets:Tensor,):
    return torch.ops.mlu_custom_ext.cosine_similarity_loss_mlu(predictions,targets,)

def leaky_relu_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.leaky_relu_mlu(input,)

def reverse_cumsum_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.reverse_cumsum_mlu(x,)

def new_gelu_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.new_gelu_mlu(x,)

def symmetric_matmul_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.symmetric_matmul_mlu(A,B,)

def tanh_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.tanh_mlu(input,)

def matmul_7_mlu(A:Tensor,B_T:Tensor,):
    return torch.ops.mlu_custom_ext.matmul_7_mlu(A,B_T,)

def matrix_multiply_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matrix_multiply_mlu(A,B,)

def matmul_8_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matmul_8_mlu(A,B,)

def product_reduction_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.product_reduction_mlu(input,)

def triplet_margin_loss_mlu(anchor:Tensor,positive:Tensor,negative:Tensor,):
    return torch.ops.mlu_custom_ext.triplet_margin_loss_mlu(anchor,positive,negative,)

def selu_forward_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.selu_forward_mlu(x,)