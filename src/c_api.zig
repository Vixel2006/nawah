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

pub extern fn zeros_cpu(t: *C_Tensor, num_elements: u64) void;
pub extern fn ones_cpu(t: *C_Tensor, num_elements: u64) void;

pub extern fn add_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn sub_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn mul_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn div_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;

pub extern fn neg_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn exp_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn log_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn leaky_relu_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn sin_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn cos_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn abs_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;

pub extern fn matmul_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;

pub extern fn sum_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
pub extern fn mean_cpu_forward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void;
