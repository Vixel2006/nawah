const c_api = @import("../c_api.zig");
const Function = @import("../function.zig").Function;

pub const Reduce = enum {
    sum,
    mean,
    max,
    min,
};

pub fn getReduceFn(op: Reduce, dtype: u32, device: u32) Function {
    _ = dtype;
    _ = device;
    return switch (op) {
        .sum => .{ .forward = sumForward, .backward = sumBackward },
        .mean => .{ .forward = meanForward, .backward = meanBackward },
        .max => .{ .forward = maxForward, .backward = maxBackward },
        .min => .{ .forward = minForward, .backward = minBackward },
    };
}

fn sumForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.sum_cpu_forward(inputs, output, params);
}
fn sumBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.sum_cpu_backward(inputs, output, params);
}

fn meanForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.mean_cpu_forward(inputs, output, params);
}
fn meanBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.mean_cpu_backward(inputs, output, params);
}

fn maxForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.max_cpu_forward(inputs, output, params);
}
fn maxBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.max_cpu_backward(inputs, output, params);
}

fn minForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.min_cpu_forward(inputs, output, params);
}
fn minBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.min_cpu_backward(inputs, output, params);
}
