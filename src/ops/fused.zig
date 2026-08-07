const c_api = @import("../c_api.zig");
const C_Tensor = c_api.C_Tensor;
const KernelParams = c_api.KernelParams;
const Function = @import("../function.zig").Function;

pub const Fused = enum {
    matmul_relu,
};

pub fn getFusedFn(op: Fused, dtype: u32, device: u32) Function {
    _ = dtype;
    return switch (device) {
        0 => switch (op) {
            .matmul_relu => .{ .forward = c_api.matmul_relu_cpu_forward, .backward = c_api.matmul_relu_cpu_backward },
        },
        1 => switch (op) {
            .matmul_relu => .{ .forward = c_api.matmul_relu_cuda_forward, .backward = c_api.matmul_relu_cuda_backward },
        },
        else => @panic("unknown device"),
    };
}
