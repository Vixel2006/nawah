#include "immintrin.h"
#include "plast/kernels/cpu/binary_backward_kernels.h"
#include <immintrin.h>

#define SIMD_WIDTH 8

void plast_cpu_sub_backward_kernel_float(float* grad_in1, float* grad_in2, const float* grad_out,
                                         size_t num_elements)
{
    size_t i = 0;

    for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH)
    {
        __m256 dout = _mm256_loadu_ps(grad_out + i);
        __m256 din1 = _mm256_loadu_ps(grad_in1 + i);
        __m256 din2 = _mm256_loadu_ps(grad_in2 + i);

        din1 = _mm256_add_ps(dout, din1);
        din2 = _mm256_sub_ps(din2, dout);

        _mm256_storeu_ps(grad_in1 + i, din1);
        _mm256_storeu_ps(grad_in2 + i, din2);
    }

    for (; i < num_elements; ++i)
    {
        grad_in1[i] += grad_out[i];
        grad_in2[i] -= grad_out[i];
    }
}

void plast_cpu_sub_backward_kernel_int32(int32_t* grad_in1, int32_t* grad_in2,
                                         const int32_t* grad_out, size_t num_elements)
{
}
