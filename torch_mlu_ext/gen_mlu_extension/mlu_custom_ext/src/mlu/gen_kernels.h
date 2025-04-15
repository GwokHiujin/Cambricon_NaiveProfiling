#pragma once
#include <cnrt.h>

// Auto-generated declarations
void frobenius_norm_kernel_entry(float* v1, float* v2, float v3, int32_t v4, int elem_num);
void mse_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int elem_num);
void softplus_kernel_entry(float* v1, float* v2, int32_t v3, int elem_num);
void elu_forward_kernel_entry(float* v1, float* v2, float v3, int32_t v4, int elem_num);
void softmax_kernel_batch_entry(float* v1, float* v2, int32_t v3, int32_t v4, int elem_num);
void matmul_2_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5, int32_t v6, int elem_num);
void maxpool1d_cuda_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, int32_t v8, int32_t v9, int32_t v10, int elem_num);
void sum_reduction_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5, int elem_num);
void MatMulKernel_3_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5, int32_t v6, int elem_num);
void cumprod_kernel_entry(float* v1, float* v2, int64_t v3, int elem_num);
void tensor_matrix_multiply_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, int32_t v8, int elem_num);
void MatMulKernel_6_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5, int32_t v6, int elem_num);
void matmul_4_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5, int32_t v6, int elem_num);
void batch_norm_kernel_entry(float* v1, float* v2, float* v3, float* v4, float* v5, float* v6, int32_t v7, int32_t v8, int32_t v9, int32_t v10, float v11, int elem_num);
void scan_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int elem_num);
void l2_normalize_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int elem_num);
void matrix_scalar_mul_kernel_entry(float* v1, float v2, float* v3, int32_t v4, int elem_num);
void hardsigmoid_kernel_entry(float* v1, float* v2, int32_t v3, int elem_num);
void matvec_mul_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5, int elem_num);
void tall_skinny_matmul_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5, int elem_num);
void gelu_kernel_entry(float* v1, float* v2, int32_t v3, int elem_num);
void MatMulKernel_8_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5, int32_t v6, int elem_num);
void relu_kernel_entry(float* v1, float* v2, int32_t v3, int elem_num);
void cosine_similarity_loss_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5, int elem_num);
void leaky_relu_kernel_entry(float* v1, float* v2, float v3, int32_t v4, int elem_num);
void reverse_cumsum_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int elem_num);
void new_gelu_kernel_entry(float* v1, float* v2, int32_t v3, int elem_num);
void symmetric_matmul_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int elem_num);
void tanh_kernel_entry(float* v1, float* v2, int32_t v3, int elem_num);
void matmul_5_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5, int32_t v6, int elem_num);
void matrix_multiply_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int elem_num);
void matmul_7_kernel_entry(float* v1, float* v2, float* v3, int32_t v4, int32_t v5, int32_t v6, int elem_num);
void product_reduction_kernel_entry(float* v1, float* v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int elem_num);
void triplet_margin_loss_kernel_entry(float* v1, float* v2, float* v3, float v4, float* v5, int32_t v6, int elem_num);
void selu_forward_kernel_entry(float* v1, float* v2, int32_t v3, int elem_num);
