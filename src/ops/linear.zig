const c_api = @import("../c_api.zig");
const C_Tensor = c_api.C_Tensor;
const KernelParams = c_api.KernelParams;
const Function = @import("../function.zig").Function;

pub const Linear = enum {
    matmul,
};

pub fn getLinearFn(op: Linear, dtype: u32, device: u32) Function {
    _ = dtype;
    return switch (device) {
        0 => switch (op) {
            .matmul => .{ .forward = c_api.matmul_cpu_forward, .backward = c_api.matmul_cpu_backward },
        },
        1 => switch (op) {
            .matmul => .{ .forward = c_api.matmul_cuda_forward, .backward = c_api.matmul_cuda_backward },
        },
        else => @panic("unknown device"),
    };
}
