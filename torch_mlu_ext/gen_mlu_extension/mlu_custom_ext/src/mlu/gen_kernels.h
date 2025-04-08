#pragma once
#include <cnrt.h>

// Auto-generated declarations
void mse_kernel_entry(float* v1, float* v2, float* v3, int32_t v4);
void elu_forward_kernel_entry(float* v1, float* v2, float v3, int32_t v4);
void softmax_kernel_batch_entry(float* v1, float* v2, int32_t v3, int32_t v4);
void maxpool1d_cuda_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, int32_t v8, int32_t v9, int32_t v10);
void sum_reduction_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5);
void cumprod_kernel_entry(float* v1, float* v2, int64_t v3);
void batch_norm_kernel_entry(float* v1, float* v2, float* v3, float* v4, float* v5, float* v6, int32_t v7, int32_t v8, int32_t v9, int32_t v10, float v11);
void scan_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4);
void l2_normalize_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4);
void matrix_scalar_mul_kernel_entry(float* v1, float v2, float* v3, int32_t v4);
void hardsigmoid_kernel_entry(float* v1, float* v2, int32_t v3);
void matvec_mul_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5);
void gelu_kernel_entry(float* v1, float* v2, int32_t v3);
void relu_kernel_entry(float* v1, float* v2, int32_t v3);
void cosine_similarity_loss_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5);
void leaky_relu_kernel_entry(float* v1, float* v2, float v3, int32_t v4);
void reverse_cumsum_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4);
void tanh_kernel_entry(float* v1, float* v2, int32_t v3);
void product_reduction_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6);
void triplet_margin_loss_kernel_entry(float* v1, float* v2, float* v3, float v4, float* v5, int32_t v6);
void selu_forward_kernel_entry(float* v1, float* v2, int32_t v3);
