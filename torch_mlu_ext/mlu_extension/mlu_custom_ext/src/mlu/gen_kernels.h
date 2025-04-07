#pragma once
#include <cnrt.h>

// Auto-generated declarations
void frobenius_norm_kernel_entry(float* v1, float* v2, float v3, int32_t v4);
void square_sum_kernel_entry(float* v1, float* v2, int32_t v3);
void mse_kernel_entry(float* v1, float* v2, float* v3, int32_t v4);
void softplus_kernel_entry(float* v1, float* v2, int32_t v3);
void elu_forward_kernel_entry(float* v1, float* v2, float v3, int32_t v4);
void mean_reduction_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5);
void softmax_kernel_batch_entry(float* v1, float* v2, int32_t v3, int32_t v4);
void maxpool1d_cuda_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, int32_t v8, int32_t v9, int32_t v10);
void sum_reduction_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5);
void layer_norm_kernel_entry(float* v1, float* v2, float* v3, float* v4, float* v5, float* v6, int32_t v7, int32_t v8, int32_t v9, int32_t v10);
void compute_mean_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6);
void l1_norm_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4);
void cumprod_kernel_entry(float* v1, float* v2, int64_t v3);
void swish_kernel_entry(float* v1, float* v2, int32_t v3);
void batch_norm_kernel_entry(float* v1, float* v2, float* v3, float* v4, float* v5, float* v6, int32_t v7, int32_t v8, int32_t v9, int32_t v10, float v11);
void scan_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4);
void max_pool2d_cuda_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, int32_t v8, int32_t v9, int32_t v10, int32_t v11, int32_t v12);
void l2_normalize_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4);
void softsign_kernel_entry(float* v1, float* v2, int32_t v3);
void matrix_scalar_mul_kernel_entry(float* v1, float v2, float* v3, int32_t v4);
void conv_transpose2d_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, int32_t v8, int32_t v9, int32_t v10, int32_t v11, int32_t v12, int32_t v13, int32_t v14, int32_t v15, int32_t v16);
void hardsigmoid_kernel_entry(float* v1, float* v2, int32_t v3);
void smooth_l1_loss_kernel_launcher_entry(float* v1, float* v2, float* v3, int32_t v4);
void matvec_mul_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5);
void gelu_kernel_entry(float* v1, float* v2, int32_t v3);
void relu_kernel_entry(float* v1, float* v2, int32_t v3);
void cosine_similarity_loss_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5);
void kl_div_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5);
void leaky_relu_kernel_entry(float* v1, float* v2, float v3, int32_t v4);
void sigmoid_kernel_entry(float* v1, float* v2, int32_t v3);
void reverse_cumsum_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4);
void gelu_kernel_entry(float* v1, float* v2, int32_t v3);
void rms_norm_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, float v7);
void tanh_kernel_entry(float* v1, float* v2, int32_t v3);
void log_softmax_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4);
void product_reduction_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6);
void triplet_margin_loss_kernel_entry(float* v1, float* v2, float* v3, float v4, float* v5, int32_t v6);
void selu_forward_kernel_entry(float* v1, float* v2, int32_t v3);
