#pragma once
#include <cnrt.h>

// Auto-generated declarations
void mse_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int size);
void softplus_kernel_entry(float* v1, float* v2, int32_t v3, int size);
void elu_forward_kernel_entry(float* v1, float* v2, float v3, int32_t v4, int size);
void softmax_kernel_batch_entry(float* v1, float* v2, int32_t v3, int32_t v4, int size);
void maxpool1d_cuda_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, int32_t v8, int32_t v9, int32_t v10, int size);
void sum_reduction_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5, int size);
void cumprod_kernel_entry(float* v1, float* v2, int64_t v3, int size);
void batch_norm_kernel_entry(float* v1, float* v2, float* v3, float* v4, float* v5, float* v6, int32_t v7, int32_t v8, int32_t v9, int32_t v10, float v11, int size);
void scan_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int size);
void l2_normalize_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int size);
void matrix_scalar_mul_kernel_entry(float* v1, float v2, float* v3, int32_t v4, int size);
void hardsigmoid_kernel_entry(float* v1, float* v2, int32_t v3, int size);
void matvec_mul_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5, int size);
void gelu_kernel_entry(float* v1, float* v2, int32_t v3, int size);
void relu_kernel_entry(float* v1, float* v2, int32_t v3, int size);
void cosine_similarity_loss_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5, int size);
void leaky_relu_kernel_entry(float* v1, float* v2, float v3, int32_t v4, int size);
void reverse_cumsum_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int size);
void gelu_kernel_entry(float* v1, float* v2, int32_t v3, int size);
void tanh_kernel_entry(float* v1, float* v2, int32_t v3, int size);
void product_reduction_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int size);
void triplet_margin_loss_kernel_entry(float* v1, float* v2, float* v3, float v4, float* v5, int32_t v6, int size);
void selu_forward_kernel_entry(float* v1, float* v2, int32_t v3, int size);
