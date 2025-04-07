
import torch
from torch import Tensor


def mse_mlu(predictions:Tensor,targets:Tensor,):
    return torch.ops.mlu_custom_ext.mse_mlu(predictions,targets,)

def softplus_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.softplus_mlu(input,)

def elu_forward_mlu(input:Tensor,alpha:float,):
    return torch.ops.mlu_custom_ext.elu_forward_mlu(input,alpha,)

def mean_reduction_mlu(input:Tensor,dim:int,):
    return torch.ops.mlu_custom_ext.mean_reduction_mlu(input,dim,)

def softmax_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.softmax_mlu(input,)

def maxpool1d_mlu_forward(input:Tensor,kernel_size:int,stride:int,padding:int,dilation:int,):
    return torch.ops.mlu_custom_ext.maxpool1d_mlu_forward(input,kernel_size,stride,padding,dilation,)

def sum_reduction_mlu(x:Tensor,dim:int,):
    return torch.ops.mlu_custom_ext.sum_reduction_mlu(x,dim,)

def layer_norm_mlu(input:Tensor,weight:Tensor,bias:Tensor,):
    return torch.ops.mlu_custom_ext.layer_norm_mlu(input,weight,bias,)

def l1_norm_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.l1_norm_mlu(x,)

def cumprod_mlu(x:Tensor,dim:int64_t,):
    return torch.ops.mlu_custom_ext.cumprod_mlu(x,dim,)

def cross_entropy_mlu(predictions:Tensor,targets:Tensor,):
    return torch.ops.mlu_custom_ext.cross_entropy_mlu(predictions,targets,)

def swish_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.swish_mlu(input,)

def scan_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.scan_mlu(input,)

def max_pool2d_mlu(x:Tensor,kernel_size:int,stride:int,padding:int,dilation:int,):
    return torch.ops.mlu_custom_ext.max_pool2d_mlu(x,kernel_size,stride,padding,dilation,)

def masked_cumsum_mlu(x:Tensor,mask:Tensor,):
    return torch.ops.mlu_custom_ext.masked_cumsum_mlu(x,mask,)

def l2_normalize_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.l2_normalize_mlu(x,)

def softsign_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.softsign_mlu(x,)

def matrix_scalar_mul_mlu(A:Tensor,s:float,):
    return torch.ops.mlu_custom_ext.matrix_scalar_mul_mlu(A,s,)

def conv_transpose2d_mlu(input:Tensor,weight:Tensor,stride:std::tuple<int, int>,padding:std::tuple<int, int>,):
    return torch.ops.mlu_custom_ext.conv_transpose2d_mlu(input,weight,stride,padding,)

def hardsigmoid_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.hardsigmoid_mlu(input,)

def smooth_l1_loss_mlu(predictions:Tensor,targets:Tensor,):
    return torch.ops.mlu_custom_ext.smooth_l1_loss_mlu(predictions,targets,)

def matvec_mul_mlu(A:Tensor,B:Tensor,):
    return torch.ops.mlu_custom_ext.matvec_mul_mlu(A,B,)

def gelu_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.gelu_mlu(input,)

def relu_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.relu_mlu(input,)

def cosine_similarity_loss_mlu(predictions:Tensor,targets:Tensor,):
    return torch.ops.mlu_custom_ext.cosine_similarity_loss_mlu(predictions,targets,)

def kl_div_mlu(log_predictions:Tensor,targets:Tensor,):
    return torch.ops.mlu_custom_ext.kl_div_mlu(log_predictions,targets,)

def leaky_relu_mlu(input:Tensor,negative_slope:float,):
    return torch.ops.mlu_custom_ext.leaky_relu_mlu(input,negative_slope,)

def sigmoid_mlu_forward(input:Tensor,):
    return torch.ops.mlu_custom_ext.sigmoid_mlu_forward(input,)

def reverse_cumsum_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.reverse_cumsum_mlu(x,)

def gelu_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.gelu_mlu(x,)

def rms_norm_mlu(x:Tensor,eps:float,):
    return torch.ops.mlu_custom_ext.rms_norm_mlu(x,eps,)

def tanh_mlu(input:Tensor,):
    return torch.ops.mlu_custom_ext.tanh_mlu(input,)

def log_softmax_mlu(input:Tensor,dim:int,):
    return torch.ops.mlu_custom_ext.log_softmax_mlu(input,dim,)

def product_reduction_mlu(input:Tensor,reduction_dim:int,):
    return torch.ops.mlu_custom_ext.product_reduction_mlu(input,reduction_dim,)

def triplet_margin_loss_mlu(anchor:Tensor,positive:Tensor,negative:Tensor,margin:float,):
    return torch.ops.mlu_custom_ext.triplet_margin_loss_mlu(anchor,positive,negative,margin,)

def selu_forward_mlu(x:Tensor,):
    return torch.ops.mlu_custom_ext.selu_forward_mlu(x,)