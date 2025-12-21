#include "plast/kernels/cpu/binary_backward_kernels.h"
#include <immintrin.h>

#define SIMD_WIDTH 8

void plast_cpu_mul_backward_kernel_float(float* grad_in1, float* grad_in2, const float* grad_out,
                                         const float* in1, const float* in2, size_t num_elements)
{
    size_t i = 0;

    for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH)
    {
        __m256 dout = _mm256_loadu_ps(grad_out + i);
        __m256 a = _mm256_loadu_ps(in1 + i);
        __m256 b = _mm256_loadu_ps(in2 + i);
        __m256 da = _mm256_loadu_ps(grad_in1 + i);
        __m256 db = _mm256_loadu_ps(grad_in2 + i);

        da = _mm256_add_ps(_mm256_mul_ps(dout, b), da);
        db = _mm256_add_ps(_mm256_mul_ps(dout, a), db);

        _mm256_storeu_ps(grad_in1 + i, da);
        _mm256_storeu_ps(grad_in2 + i, db);
    }

    for (; i < num_elements; ++i)
    {
        grad_in1[i] += grad_out[i] * in2[i];
        grad_in2[i] += grad_out[i] * in1[i];
    }
}

void plast_cpu_mul_backward_kernel_int32(int32_t* grad_in1, int32_t* grad_in2,
                                         const int32_t* grad_out, const int32_t* in1,
                                         const int32_t* in2, size_t num_elements)
{
    for (size_t i = 0; i < num_elements; ++i)
    {
        grad_in1[i] += grad_out[i] * in2[i];
        grad_in2[i] += grad_out[i] * in1[i];
    }
}
