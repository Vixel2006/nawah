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
    _ = device;
    return switch (op) {
        .add => .{ .forward = addForward, .backward = addBackward },
        .sub => .{ .forward = subForward, .backward = subBackward },
        .mul => .{ .forward = mulForward, .backward = mulBackward },
        .div => .{ .forward = divForward, .backward = divBackward },
        .relu => .{ .forward = reluForward, .backward = reluBackward },
        .exp => .{ .forward = expForward, .backward = expBackward },
        .log => .{ .forward = logForward, .backward = logBackward },
        .sin => .{ .forward = sinForward, .backward = sinBackward },
        .cos => .{ .forward = cosForward, .backward = cosBackward },
        .abs => .{ .forward = absForward, .backward = absBackward },
        .neg => .{ .forward = negForward, .backward = negBackward },
    };
}

fn addForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.add_cpu_forward(inputs, output, params);
}
fn addBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.add_cpu_backward(inputs, output, params);
}

fn subForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.sub_cpu_forward(inputs, output, params);
}
fn subBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.sub_cpu_backward(inputs, output, params);
}

fn mulForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.mul_cpu_forward(inputs, output, params);
}
fn mulBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.mul_cpu_backward(inputs, output, params);
}

fn divForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.div_cpu_forward(inputs, output, params);
}
fn divBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.div_cpu_backward(inputs, output, params);
}

fn reluForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.leaky_relu_cpu_forward(inputs, output, params);
}
fn reluBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.leaky_relu_cpu_backward(inputs, output, params);
}

fn expForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.exp_cpu_forward(inputs, output, params);
}
fn expBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.exp_cpu_backward(inputs, output, params);
}

fn logForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.log_cpu_forward(inputs, output, params);
}
fn logBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.log_cpu_backward(inputs, output, params);
}

fn sinForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.sin_cpu_forward(inputs, output, params);
}
fn sinBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.sin_cpu_backward(inputs, output, params);
}

fn cosForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.cos_cpu_forward(inputs, output, params);
}
fn cosBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.cos_cpu_backward(inputs, output, params);
}

fn absForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.abs_cpu_forward(inputs, output, params);
}
fn absBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.abs_cpu_backward(inputs, output, params);
}

fn negForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.neg_cpu_forward(inputs, output, params);
}
fn negBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.neg_cpu_backward(inputs, output, params);
}
