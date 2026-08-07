const c_api = @import("../c_api.zig");
const C_Tensor = c_api.C_Tensor;
const KernelParams = c_api.KernelParams;
const Function = @import("../function.zig").Function;

pub const ElementWise = enum {
    add,
    sub,
    mul,
    div,
    relu,
    exp,
    log,
    sin,
    cos,
    abs,
    neg,
};

pub fn getElementWiseFn(op: ElementWise, dtype: u32, device: u32) Function {
    _ = dtype;
    return switch (device) {
        0 => switch (op) {
            .add => .{ .forward = c_api.add_cpu_forward, .backward = c_api.add_cpu_backward },
            .sub => .{ .forward = c_api.sub_cpu_forward, .backward = c_api.sub_cpu_backward },
            .mul => .{ .forward = c_api.mul_cpu_forward, .backward = c_api.mul_cpu_backward },
            .div => .{ .forward = c_api.div_cpu_forward, .backward = c_api.div_cpu_backward },
            .relu => .{ .forward = c_api.leaky_relu_cpu_forward, .backward = c_api.leaky_relu_cpu_backward },
            .exp => .{ .forward = c_api.exp_cpu_forward, .backward = c_api.exp_cpu_backward },
            .log => .{ .forward = c_api.log_cpu_forward, .backward = c_api.log_cpu_backward },
            .sin => .{ .forward = c_api.sin_cpu_forward, .backward = c_api.sin_cpu_backward },
            .cos => .{ .forward = c_api.cos_cpu_forward, .backward = c_api.cos_cpu_backward },
            .abs => .{ .forward = c_api.abs_cpu_forward, .backward = c_api.abs_cpu_backward },
            .neg => .{ .forward = c_api.neg_cpu_forward, .backward = c_api.neg_cpu_backward },
        },
        1 => switch (op) {
            .add => .{ .forward = c_api.add_cuda_forward, .backward = c_api.add_cuda_backward },
            .sub => .{ .forward = c_api.sub_cuda_forward, .backward = c_api.sub_cuda_backward },
            .mul => .{ .forward = c_api.mul_cuda_forward, .backward = c_api.mul_cuda_backward },
            .div => .{ .forward = c_api.div_cuda_forward, .backward = c_api.div_cuda_backward },
            .relu => .{ .forward = c_api.leaky_relu_cuda_forward, .backward = c_api.leaky_relu_cuda_backward },
            .exp => .{ .forward = c_api.exp_cuda_forward, .backward = c_api.exp_cuda_backward },
            .log => .{ .forward = c_api.log_cuda_forward, .backward = c_api.log_cuda_backward },
            .sin => .{ .forward = c_api.sin_cuda_forward, .backward = c_api.sin_cuda_backward },
            .cos => .{ .forward = c_api.cos_cuda_forward, .backward = c_api.cos_cuda_backward },
            .abs => .{ .forward = c_api.abs_cuda_forward, .backward = c_api.abs_cuda_backward },
            .neg => .{ .forward = c_api.neg_cuda_forward, .backward = c_api.neg_cuda_backward },
        },
        else => @panic("unknown device"),
    };
}
