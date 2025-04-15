
import torch
from torch import Tensor


def frobenius_norm_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.frobenius_norm_mlu(input,)

def mse_mlu(predictions:Tensor,targets:Tensor,):
    return torch.ops.mlu_custom_ext.mse_mlu(predictions,targets,)

def softplus_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.softplus_mlu(input,)

def elu_forward_mlu(input:Tensor,alpha:float,):
    return torch.ops.mlu_custom_ext.elu_forward_mlu(input,alpha,)

def softmax_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.softmax_mlu(input,)

def matmul_mlu(a:Tensor,b:Tensor,):
    return torch.ops.mlu_custom_ext.matmul_mlu(a,b,)

def maxpool1d_mlu_forward(input:Tensor,kernel_size:int,stride:int,padding:int,dilation:int,):
    return torch.ops.mlu_custom_ext.maxpool1d_mlu_forward(input,kernel_size,stride,padding,dilation,)

def sum_reduction_mlu(x:Tensor,dim:int,):
    return torch.ops.mlu_custom_ext.sum_reduction_mlu(x,dim,)

def layer_norm_mlu(input:Tensor,weight:Tensor,bias:Tensor,):
    return torch.ops.mlu_custom_ext.layer_norm_mlu(input,weight,bias,)

def matmul_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matmul_mlu(A,B,)

def cumprod_mlu(x:Tensor,dim:int64_t,):
    return torch.ops.mlu_custom_ext.cumprod_mlu(x,dim,)

def tensor_matrix_multiply_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.tensor_matrix_multiply_mlu(A,B,)

def matmul_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matmul_mlu(A,B,)

def matmul_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matmul_mlu(A,B,)

def batch_norm_mlu(input:Tensor,gamma:Tensor,beta:Tensor,running_mean:Tensor,running_var:Tensor,epsilon:float,):
    return torch.ops.mlu_custom_ext.batch_norm_mlu(input,gamma,beta,running_mean,running_var,epsilon,)

def scan_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.scan_mlu(input,)

def argmin_mlu(input:Tensor,dim:int,):
    return torch.ops.mlu_custom_ext.argmin_mlu(input,dim,)

def l2_normalize_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.l2_normalize_mlu(x,)

def matrix_scalar_mul_mlu(A:Tensor,s:float,):
    return torch.ops.mlu_custom_ext.matrix_scalar_mul_mlu(A,s,)

def hardsigmoid_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.hardsigmoid_mlu(input,)

def matvec_mul_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matvec_mul_mlu(A,B,)

def tall_skinny_matmul_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.tall_skinny_matmul_mlu(A,B,)

def gelu_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.gelu_mlu(input,)

def matmul_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matmul_mlu(A,B,)

def relu_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.relu_mlu(input,)

def cosine_similarity_loss_mlu(predictions:Tensor,targets:Tensor,):
    return torch.ops.mlu_custom_ext.cosine_similarity_loss_mlu(predictions,targets,)

def leaky_relu_mlu(input:Tensor,negative_slope:float,):
    return torch.ops.mlu_custom_ext.leaky_relu_mlu(input,negative_slope,)

def reverse_cumsum_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.reverse_cumsum_mlu(x,)

def new_gelu_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.new_gelu_mlu(x,)

def symmetric_matmul_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.symmetric_matmul_mlu(A,B,)

def tanh_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.tanh_mlu(input,)

def matmul_mlu(A:Tensor,B_T:Tensor,M:int,N:int,K:int,):
    return torch.ops.mlu_custom_ext.matmul_mlu(A,B_T,M,N,K,)

def matrix_multiply_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matrix_multiply_mlu(A,B,)

def matmul_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matmul_mlu(A,B,)

def product_reduction_mlu(input:Tensor,reduction_dim:int,):
    return torch.ops.mlu_custom_ext.product_reduction_mlu(input,reduction_dim,)

def triplet_margin_loss_mlu(anchor:Tensor,positive:Tensor,negative:Tensor,margin:float,):
    return torch.ops.mlu_custom_ext.triplet_margin_loss_mlu(anchor,positive,negative,margin,)

def selu_forward_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.selu_forward_mlu(x,)