const std = @import("std");

pub const MAX_NDIM: u64 = 8;

pub const C_Tensor = extern struct {
    grad: ?*C_Tensor,
    creator: ?*anyopaque,
    data: ?*anyopaque,
    shape: [MAX_NDIM]u64,
    strides: [MAX_NDIM]u64,
    ndim: u64,
    dtype: u32,
    device: u32,
    requires_grad: bool,
};

pub const KernelParams = extern struct {
    dim: u64,
    keepdim: u64,
    fval: f64,
};

pub fn toCTensor(
    data_ptr: ?*anyopaque,
    shape: *const [MAX_NDIM]u64,
    strides: *const [MAX_NDIM]u64,
    ndim: u64,
    requires_grad: bool,
) C_Tensor {
    return .{
        .grad = null,
        .creator = null,
        .data = data_ptr,
        .shape = shape.*,
        .strides = strides.*,
        .ndim = ndim,
        .dtype = DTYPE_FLOAT32,
        .device = 0,
        .requires_grad = requires_grad,
    };
}

pub const DTYPE_FLOAT32: u32 = 1;

//  CPU Kernels

pub extern fn zeros_cpu(t: *C_Tensor, num_elements: u64) void;
pub extern fn ones_cpu(t: *C_Tensor, num_elements: u64) void;

pub extern fn add_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn add_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn sub_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn sub_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn mul_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn mul_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn div_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn div_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;

pub extern fn neg_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn neg_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn exp_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn exp_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn log_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn log_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn leaky_relu_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn leaky_relu_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn sin_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn sin_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn cos_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn cos_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn abs_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn abs_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;

pub extern fn matmul_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn matmul_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;

pub extern fn sum_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn sum_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn mean_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn mean_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn max_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn max_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn min_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn min_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;

pub extern fn matmul_relu_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn matmul_relu_cpu_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;

//  CUDA GPU Kernels

pub extern fn zeros_cuda(t: *C_Tensor, num_elements: u64) void;
pub extern fn ones_cuda(t: *C_Tensor, num_elements: u64) void;

pub extern fn add_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn add_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn sub_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn sub_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn mul_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn mul_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn div_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn div_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;

pub extern fn neg_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn neg_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn exp_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn exp_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn log_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn log_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn leaky_relu_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn leaky_relu_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn sin_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn sin_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn cos_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn cos_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn abs_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn abs_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;

pub extern fn matmul_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn matmul_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;

pub extern fn sum_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn sum_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn mean_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn mean_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn max_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn max_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;
pub extern fn min_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn min_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;

pub extern fn matmul_relu_cuda_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn matmul_relu_cuda_backward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void;

// Direct launch Runtime API & raw kernels
pub const Dim3 = extern struct {
    x: c_uint = 1,
    y: c_uint = 1,
    z: c_uint = 1,
};
pub extern fn cudaLaunchKernel(
    func: ?*const anyopaque,
    gridDim: Dim3,
    blockDim: Dim3,
    args: [*]const ?*anyopaque,
    sharedMem: usize,
    stream: ?*anyopaque,
) callconv(.c) c_int;

// Binary CUDA kernels
pub extern fn add_cuda_forward_float_contig_kernel() void;
pub extern fn add_cuda_forward_float_non_contig_kernel() void;
pub extern fn add_cuda_backward_float_contig_kernel() void;
pub extern fn add_cuda_backward_float_non_contig_kernel() void;

pub extern fn sub_cuda_forward_float_contig_kernel() void;
pub extern fn sub_cuda_forward_float_non_contig_kernel() void;
pub extern fn sub_cuda_backward_float_contig_kernel() void;
pub extern fn sub_cuda_backward_float_non_contig_kernel() void;

pub extern fn mul_cuda_forward_float_contig_kernel() void;
pub extern fn mul_cuda_forward_float_non_contig_kernel() void;
pub extern fn mul_cuda_backward_float_contig_kernel() void;
pub extern fn mul_cuda_backward_float_non_contig_kernel() void;

pub extern fn div_cuda_forward_float_contig_kernel() void;
pub extern fn div_cuda_forward_float_non_contig_kernel() void;
pub extern fn div_cuda_backward_float_contig_kernel() void;
pub extern fn div_cuda_backward_float_non_contig_kernel() void;

// Unary CUDA kernels
pub extern fn leaky_relu_cuda_forward_float_contig_kernel() void;
pub extern fn leaky_relu_cuda_forward_float_non_contig_kernel() void;
pub extern fn leaky_relu_cuda_backward_float_contig_kernel() void;
pub extern fn leaky_relu_cuda_backward_float_non_contig_kernel() void;

pub extern fn exp_cuda_forward_float_contig_kernel() void;
pub extern fn exp_cuda_forward_float_non_contig_kernel() void;
pub extern fn exp_cuda_backward_float_contig_kernel() void;
pub extern fn exp_cuda_backward_float_non_contig_kernel() void;

pub extern fn log_cuda_forward_float_contig_kernel() void;
pub extern fn log_cuda_forward_float_non_contig_kernel() void;
pub extern fn log_cuda_backward_float_contig_kernel() void;
pub extern fn log_cuda_backward_float_non_contig_kernel() void;

pub extern fn sin_cuda_forward_float_contig_kernel() void;
pub extern fn sin_cuda_forward_float_non_contig_kernel() void;
pub extern fn sin_cuda_backward_float_contig_kernel() void;
pub extern fn sin_cuda_backward_float_non_contig_kernel() void;

pub extern fn cos_cuda_forward_float_contig_kernel() void;
pub extern fn cos_cuda_forward_float_non_contig_kernel() void;
pub extern fn cos_cuda_backward_float_contig_kernel() void;
pub extern fn cos_cuda_backward_float_non_contig_kernel() void;

pub extern fn abs_cuda_forward_float_contig_kernel() void;
pub extern fn abs_cuda_forward_float_non_contig_kernel() void;
pub extern fn abs_cuda_backward_float_contig_kernel() void;
pub extern fn abs_cuda_backward_float_non_contig_kernel() void;

pub extern fn neg_cuda_forward_float_contig_kernel() void;
pub extern fn neg_cuda_forward_float_non_contig_kernel() void;
pub extern fn neg_cuda_backward_float_contig_kernel() void;
pub extern fn neg_cuda_backward_float_non_contig_kernel() void;

// Direct Matmul & Fused Ops
pub extern fn matmul_cpu_forward_direct(
    a_data: [*]const f32, a_shape: [*]const u64, a_strides: [*]const u64, a_ndim: u64,
    b_data: [*]const f32, b_shape: [*]const u64, b_strides: [*]const u64, b_ndim: u64,
    c_data: [*]f32, c_shape: [*]const u64, c_strides: [*]const u64, c_ndim: u64,
) void;
pub extern fn matmul_cpu_backward_direct(
    a_data: [*]const f32, a_shape: [*]const u64, a_strides: [*]const u64, a_ndim: u64,
    da_data: ?[*]f32, da_strides: [*]const u64, a_requires_grad: bool,
    b_data: [*]const f32, b_shape: [*]const u64, b_strides: [*]const u64, b_ndim: u64,
    db_data: ?[*]f32, db_strides: [*]const u64, b_requires_grad: bool,
    dc_data: [*]const f32, dc_shape: [*]const u64, dc_strides: [*]const u64, dc_ndim: u64,
) void;
pub extern fn matmul_cuda_forward_direct(
    a_data: ?*const anyopaque, a_shape: [*]const u64, a_strides: [*]const u64, a_ndim: u64,
    b_data: ?*const anyopaque, b_shape: [*]const u64, b_strides: [*]const u64, b_ndim: u64,
    c_data: ?*anyopaque, c_shape: [*]const u64, c_strides: [*]const u64, c_ndim: u64,
) void;
pub extern fn matmul_cuda_backward_direct(
    a_data: ?*const anyopaque, a_shape: [*]const u64, a_strides: [*]const u64, a_ndim: u64,
    da_data: ?*anyopaque, da_strides: [*]const u64, a_requires_grad: bool,
    b_data: ?*const anyopaque, b_shape: [*]const u64, b_strides: [*]const u64, b_ndim: u64,
    db_data: ?*anyopaque, db_strides: [*]const u64, b_requires_grad: bool,
    dc_data: ?*const anyopaque, dc_shape: [*]const u64, dc_strides: [*]const u64, dc_ndim: u64,
) void;

pub extern fn matmul_relu_cpu_forward_direct(
    a_data: [*]const f32, a_shape: [*]const u64, a_strides: [*]const u64, a_ndim: u64,
    b_data: [*]const f32, b_shape: [*]const u64, b_strides: [*]const u64, b_ndim: u64,
    c_data: [*]f32, c_shape: [*]const u64, c_strides: [*]const u64, c_ndim: u64, alpha: f32,
) void;
pub extern fn matmul_relu_cpu_backward_direct(
    a_data: [*]const f32, a_shape: [*]const u64, a_strides: [*]const u64, a_ndim: u64,
    da_data: ?[*]f32, da_strides: [*]const u64, a_requires_grad: bool,
    b_data: [*]const f32, b_shape: [*]const u64, b_strides: [*]const u64, b_ndim: u64,
    db_data: ?[*]f32, db_strides: [*]const u64, b_requires_grad: bool,
    c_data: [*]const f32, c_shape: [*]const u64, c_strides: [*]const u64, c_ndim: u64,
    dc_data: [*]const f32, dc_shape: [*]const u64, dc_strides: [*]const u64, dc_ndim: u64, alpha: f32,
) void;
pub extern fn matmul_relu_cuda_forward_direct(
    a_data: ?*const anyopaque, a_shape: [*]const u64, a_strides: [*]const u64, a_ndim: u64,
    b_data: ?*const anyopaque, b_shape: [*]const u64, b_strides: [*]const u64, b_ndim: u64,
    c_data: ?*anyopaque, c_shape: [*]const u64, c_strides: [*]const u64, c_ndim: u64, alpha: f32,
) void;
pub extern fn matmul_relu_cuda_backward_direct(
    a_data: ?*const anyopaque, a_shape: [*]const u64, a_strides: [*]const u64, a_ndim: u64,
    da_data: ?*anyopaque, da_strides: [*]const u64, a_requires_grad: bool,
    b_data: ?*const anyopaque, b_shape: [*]const u64, b_strides: [*]const u64, b_ndim: u64,
    db_data: ?*anyopaque, db_strides: [*]const u64, b_requires_grad: bool,
    c_data: ?*const anyopaque, c_shape: [*]const u64, c_strides: [*]const u64, c_ndim: u64,
    dc_data: ?*const anyopaque, dc_shape: [*]const u64, dc_strides: [*]const u64, dc_ndim: u64, alpha: f32,
) void;

// Direct Reduction Ops (CUDA)
pub extern fn sum_cuda_forward_direct(
    a_data: ?*const anyopaque, a_shape: [*]const u64, a_strides: [*]const u64, a_ndim: u64, a_is_contig: bool,
    c_data: ?*anyopaque, c_shape: [*]const u64, c_strides: [*]const u64, c_ndim: u64, dim: u64, keepdim: bool,
) void;
pub extern fn mean_cuda_forward_direct(
    a_data: ?*const anyopaque, a_shape: [*]const u64, a_strides: [*]const u64, a_ndim: u64, a_is_contig: bool,
    c_data: ?*anyopaque, c_shape: [*]const u64, c_strides: [*]const u64, c_ndim: u64, dim: u64, keepdim: bool,
) void;
pub extern fn max_cuda_forward_direct(
    a_data: ?*const anyopaque, a_shape: [*]const u64, a_strides: [*]const u64, a_ndim: u64, a_is_contig: bool,
    c_data: ?*anyopaque, c_shape: [*]const u64, c_strides: [*]const u64, c_ndim: u64, dim: u64, keepdim: bool,
) void;
pub extern fn min_cuda_forward_direct(
    a_data: ?*const anyopaque, a_shape: [*]const u64, a_strides: [*]const u64, a_ndim: u64, a_is_contig: bool,
    c_data: ?*anyopaque, c_shape: [*]const u64, c_strides: [*]const u64, c_ndim: u64, dim: u64, keepdim: bool,
) void;

pub extern fn sum_mean_cuda_backward_direct(
    dc_data: ?*const anyopaque, dc_shape: [*]const u64, dc_strides: [*]const u64, dc_ndim: u64,
    da_data: ?*anyopaque, da_shape: [*]const u64, da_strides: [*]const u64, da_ndim: u64, da_is_contig: bool,
    dim: u64, keepdim: bool, is_mean: bool,
) void;
pub extern fn max_min_cuda_backward_direct(
    dc_data: ?*const anyopaque, dc_shape: [*]const u64, dc_strides: [*]const u64, dc_ndim: u64,
    da_data: ?*anyopaque, da_shape: [*]const u64, da_strides: [*]const u64, da_ndim: u64, da_is_contig: bool,
    a_data_fwd: ?*const anyopaque, c_data_fwd: ?*const anyopaque,
    dim: u64, keepdim: bool,
) void;

// Direct CPU kernels (Binary & Unary)
pub extern fn add_cpu_forward_float_contig_kernel(a: [*]const f32, b: [*]const f32, c: [*]f32, num_elements: u64) void;
pub extern fn add_cpu_forward_float_kernel(
    a: [*]const f32, a_strides: [*]const u64, a_shape: [*]const u64, a_ndim: u64,
    b: [*]const f32, b_strides: [*]const u64, b_shape: [*]const u64, b_ndim: u64,
    c: [*]f32, c_strides: [*]const u64, c_shape: [*]const u64, c_ndim: u64,
) void;
pub extern fn add_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, b: [*]const f32, da: ?[*]f32, db: ?[*]f32, num_elements: u64) void;
pub extern fn add_cpu_backward_float_kernel(
    dout_data: [*]const f32, dout_strides: [*]const u64, dout_shape: [*]const u64, dout_ndim: u64,
    da_data: ?[*]f32, da_strides: [*]const u64, da_shape: [*]const u64, da_ndim: u64,
    db_data: ?[*]f32, db_strides: [*]const u64, db_shape: [*]const u64, db_ndim: u64,
    a_data: [*]const f32, a_strides: [*]const u64,
    b_data: [*]const f32, b_strides: [*]const u64,
) void;

pub extern fn sub_cpu_forward_float_contig_kernel(a: [*]const f32, b: [*]const f32, c: [*]f32, num_elements: u64) void;
pub extern fn sub_cpu_forward_float_kernel(
    a: [*]const f32, a_strides: [*]const u64, a_shape: [*]const u64, a_ndim: u64,
    b: [*]const f32, b_strides: [*]const u64, b_shape: [*]const u64, b_ndim: u64,
    c: [*]f32, c_strides: [*]const u64, c_shape: [*]const u64, c_ndim: u64,
) void;
pub extern fn sub_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, b: [*]const f32, da: ?[*]f32, db: ?[*]f32, num_elements: u64) void;
pub extern fn sub_cpu_backward_float_kernel(
    dout_data: [*]const f32, dout_strides: [*]const u64, dout_shape: [*]const u64, dout_ndim: u64,
    da_data: ?[*]f32, da_strides: [*]const u64, da_shape: [*]const u64, da_ndim: u64,
    db_data: ?[*]f32, db_strides: [*]const u64, db_shape: [*]const u64, db_ndim: u64,
    a_data: [*]const f32, a_strides: [*]const u64,
    b_data: [*]const f32, b_strides: [*]const u64,
) void;

pub extern fn mul_cpu_forward_float_contig_kernel(a: [*]const f32, b: [*]const f32, c: [*]f32, num_elements: u64) void;
pub extern fn mul_cpu_forward_float_kernel(
    a: [*]const f32, a_strides: [*]const u64, a_shape: [*]const u64, a_ndim: u64,
    b: [*]const f32, b_strides: [*]const u64, b_shape: [*]const u64, b_ndim: u64,
    c: [*]f32, c_strides: [*]const u64, c_shape: [*]const u64, c_ndim: u64,
) void;
pub extern fn mul_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, b: [*]const f32, da: ?[*]f32, db: ?[*]f32, num_elements: u64) void;
pub extern fn mul_cpu_backward_float_kernel(
    dout_data: [*]const f32, dout_strides: [*]const u64, dout_shape: [*]const u64, dout_ndim: u64,
    da_data: ?[*]f32, da_strides: [*]const u64, da_shape: [*]const u64, da_ndim: u64,
    db_data: ?[*]f32, db_strides: [*]const u64, db_shape: [*]const u64, db_ndim: u64,
    a_data: [*]const f32, a_strides: [*]const u64,
    b_data: [*]const f32, b_strides: [*]const u64,
) void;

pub extern fn div_cpu_forward_float_contig_kernel(a: [*]const f32, b: [*]const f32, c: [*]f32, num_elements: u64) void;
pub extern fn div_cpu_forward_float_kernel(
    a: [*]const f32, a_strides: [*]const u64, a_shape: [*]const u64, a_ndim: u64,
    b: [*]const f32, b_strides: [*]const u64, b_shape: [*]const u64, b_ndim: u64,
    c: [*]f32, c_strides: [*]const u64, c_shape: [*]const u64, c_ndim: u64,
) void;
pub extern fn div_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, b: [*]const f32, da: ?[*]f32, db: ?[*]f32, num_elements: u64) void;
pub extern fn div_cpu_backward_float_kernel(
    dout_data: [*]const f32, dout_strides: [*]const u64, dout_shape: [*]const u64, dout_ndim: u64,
    da_data: ?[*]f32, da_strides: [*]const u64, da_shape: [*]const u64, da_ndim: u64,
    db_data: ?[*]f32, db_strides: [*]const u64, db_shape: [*]const u64, db_ndim: u64,
    a_data: [*]const f32, a_strides: [*]const u64,
    b_data: [*]const f32, b_strides: [*]const u64,
) void;

pub extern fn neg_cpu_forward_float_contig_kernel(a: [*]const f32, c: [*]f32, num_elements: u64) void;
pub extern fn neg_cpu_forward_float_non_contig_kernel(a: [*]const f32, a_strides: [*]const u64, c: [*]f32, c_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;
pub extern fn neg_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, da: ?[*]f32, num_elements: u64) void;
pub extern fn neg_cpu_backward_float_non_contig_kernel(dout: [*]const f32, dout_strides: [*]const u64, a: [*]const f32, a_strides: [*]const u64, da: ?[*]f32, da_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;

pub extern fn exp_cpu_forward_float_contig_kernel(a: [*]const f32, c: [*]f32, num_elements: u64) void;
pub extern fn exp_cpu_forward_float_non_contig_kernel(a: [*]const f32, a_strides: [*]const u64, c: [*]f32, c_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;
pub extern fn exp_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, da: ?[*]f32, num_elements: u64) void;
pub extern fn exp_cpu_backward_float_non_contig_kernel(dout: [*]const f32, dout_strides: [*]const u64, a: [*]const f32, a_strides: [*]const u64, da: ?[*]f32, da_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;

pub extern fn log_cpu_forward_float_contig_kernel(a: [*]const f32, c: [*]f32, num_elements: u64) void;
pub extern fn log_cpu_forward_float_non_contig_kernel(a: [*]const f32, a_strides: [*]const u64, c: [*]f32, c_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;
pub extern fn log_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, da: ?[*]f32, num_elements: u64) void;
pub extern fn log_cpu_backward_float_non_contig_kernel(dout: [*]const f32, dout_strides: [*]const u64, a: [*]const f32, a_strides: [*]const u64, da: ?[*]f32, da_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;

pub extern fn leaky_relu_cpu_forward_float_contig_kernel(a: [*]const f32, c: [*]f32, num_elements: u64, alpha: f32) void;
pub extern fn leaky_relu_cpu_forward_float_non_contig_kernel(a: [*]const f32, a_strides: [*]const u64, c: [*]f32, c_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64, alpha: f32) void;
pub extern fn leaky_relu_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, da: ?[*]f32, num_elements: u64, alpha: f32) void;
pub extern fn leaky_relu_cpu_backward_float_non_contig_kernel(dout: [*]const f32, dout_strides: [*]const u64, a: [*]const f32, a_strides: [*]const u64, da: ?[*]f32, da_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64, alpha: f32) void;

pub extern fn sin_cpu_forward_float_contig_kernel(a: [*]const f32, c: [*]f32, num_elements: u64) void;
pub extern fn sin_cpu_forward_float_non_contig_kernel(a: [*]const f32, a_strides: [*]const u64, c: [*]f32, c_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;
pub extern fn sin_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, da: ?[*]f32, num_elements: u64) void;
pub extern fn sin_cpu_backward_float_non_contig_kernel(dout: [*]const f32, dout_strides: [*]const u64, a: [*]const f32, a_strides: [*]const u64, da: ?[*]f32, da_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;

pub extern fn cos_cpu_forward_float_contig_kernel(a: [*]const f32, c: [*]f32, num_elements: u64) void;
pub extern fn cos_cpu_forward_float_non_contig_kernel(a: [*]const f32, a_strides: [*]const u64, c: [*]f32, c_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;
pub extern fn cos_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, da: ?[*]f32, num_elements: u64) void;
pub extern fn cos_cpu_backward_float_non_contig_kernel(dout: [*]const f32, dout_strides: [*]const u64, a: [*]const f32, a_strides: [*]const u64, da: ?[*]f32, da_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;

pub extern fn abs_cpu_forward_float_contig_kernel(a: [*]const f32, c: [*]f32, num_elements: u64) void;
pub extern fn abs_cpu_forward_float_non_contig_kernel(a: [*]const f32, a_strides: [*]const u64, c: [*]f32, c_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;
pub extern fn abs_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, da: ?[*]f32, num_elements: u64) void;
pub extern fn abs_cpu_backward_float_non_contig_kernel(dout: [*]const f32, dout_strides: [*]const u64, a: [*]const f32, a_strides: [*]const u64, da: ?[*]f32, da_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;

// Direct CPU kernels (Reduction)
pub extern fn sum_cpu_forward_float_contig_kernel(a: [*]const f32, c: [*]f32, num_elements: u64) void;
pub extern fn sum_cpu_forward_float_non_contig_kernel(a_data: [*]const f32, a_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64, c_data: [*]f32) void;
pub extern fn sum_cpu_forward_float_dim_kernel(a_data: [*]const f32, a_strides: [*]const u64, a_shape: [*]const u64, a_ndim: u64, c_data: [*]f32, c_strides: [*]const u64, c_shape: [*]const u64, c_ndim: u64, dim: u64, keepdim: bool) void;
pub extern fn sum_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, da: [*]f32, num_elements: u64) void;
pub extern fn sum_cpu_backward_float_non_contig_kernel(dout_data: [*]const f32, da_data: [*]f32, da_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;
pub extern fn sum_cpu_backward_float_dim_kernel(a_data: [*]const f32, a_strides: [*]const u64, a_shape: [*]const u64, a_ndim: u64, c_data: [*]const f32, c_strides: [*]const u64, c_shape: [*]const u64, c_ndim: u64, dc_data: [*]const f32, da_data: [*]f32, da_strides: [*]const u64, dim: u64, keepdim: bool) void;

pub extern fn mean_cpu_forward_float_contig_kernel(a: [*]const f32, c: [*]f32, num_elements: u64) void;
pub extern fn mean_cpu_forward_float_non_contig_kernel(a_data: [*]const f32, a_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64, c_data: [*]f32) void;
pub extern fn mean_cpu_forward_float_dim_kernel(a_data: [*]const f32, a_strides: [*]const u64, a_shape: [*]const u64, a_ndim: u64, c_data: [*]f32, c_strides: [*]const u64, c_shape: [*]const u64, c_ndim: u64, dim: u64, keepdim: bool) void;
pub extern fn mean_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, da: [*]f32, num_elements: u64) void;
pub extern fn mean_cpu_backward_float_non_contig_kernel(dout_data: [*]const f32, da_data: [*]f32, da_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;
pub extern fn mean_cpu_backward_float_dim_kernel(a_data: [*]const f32, a_strides: [*]const u64, a_shape: [*]const u64, a_ndim: u64, c_data: [*]const f32, c_strides: [*]const u64, c_shape: [*]const u64, c_ndim: u64, dc_data: [*]const f32, da_data: [*]f32, da_strides: [*]const u64, dim: u64, keepdim: bool) void;

pub extern fn max_cpu_forward_float_contig_kernel(a: [*]const f32, c: [*]f32, num_elements: u64) void;
pub extern fn max_cpu_forward_float_non_contig_kernel(a_data: [*]const f32, a_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64, c_data: [*]f32) void;
pub extern fn max_cpu_forward_float_dim_kernel(a_data: [*]const f32, a_strides: [*]const u64, a_shape: [*]const u64, a_ndim: u64, c_data: [*]f32, c_strides: [*]const u64, c_shape: [*]const u64, c_ndim: u64, dim: u64, keepdim: bool) void;
pub extern fn max_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, da: [*]f32, num_elements: u64) void;
pub extern fn max_cpu_backward_float_non_contig_kernel(a_data: [*]const f32, a_strides: [*]const u64, c_data: [*]const f32, dc_data: [*]const f32, da_data: [*]f32, da_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;
pub extern fn max_cpu_backward_float_dim_kernel(a_data: [*]const f32, a_strides: [*]const u64, a_shape: [*]const u64, a_ndim: u64, c_data: [*]const f32, c_strides: [*]const u64, c_shape: [*]const u64, c_ndim: u64, dc_data: [*]const f32, da_data: [*]f32, da_strides: [*]const u64, dim: u64, keepdim: bool) void;

pub extern fn min_cpu_forward_float_contig_kernel(a: [*]const f32, c: [*]f32, num_elements: u64) void;
pub extern fn min_cpu_forward_float_non_contig_kernel(a_data: [*]const f32, a_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64, c_data: [*]f32) void;
pub extern fn min_cpu_forward_float_dim_kernel(a_data: [*]const f32, a_strides: [*]const u64, a_shape: [*]const u64, a_ndim: u64, c_data: [*]f32, c_strides: [*]const u64, c_shape: [*]const u64, c_ndim: u64, dim: u64, keepdim: bool) void;
pub extern fn min_cpu_backward_float_contig_kernel(dout: [*]const f32, a: [*]const f32, da: [*]f32, num_elements: u64) void;
pub extern fn min_cpu_backward_float_non_contig_kernel(a_data: [*]const f32, a_strides: [*]const u64, c_data: [*]const f32, dc_data: [*]const f32, da_data: [*]f32, da_strides: [*]const u64, shape: [*]const u64, ndim: u64, num_elements: u64) void;
pub extern fn min_cpu_backward_float_dim_kernel(a_data: [*]const f32, a_strides: [*]const u64, a_shape: [*]const u64, a_ndim: u64, c_data: [*]const f32, c_strides: [*]const u64, c_shape: [*]const u64, c_ndim: u64, dc_data: [*]const f32, da_data: [*]f32, da_strides: [*]const u64, dim: u64, keepdim: bool) void;

