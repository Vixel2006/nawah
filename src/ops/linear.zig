const c_api = @import("../c_api.zig");
const C_Tensor = c_api.C_Tensor;
const KernelParams = c_api.KernelParams;
const Function = @import("../function.zig").Function;

pub const Linear = enum {
    matmul,
};

pub fn getLinearFn(op: Linear, dtype: u32, device: u32) Function {
    _ = dtype;
    _ = device;
    return switch (op) {
        .matmul => .{ .forward = matmulForward, .backward = matmulBackward },
    };
}

fn matmulForward(inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.matmul_cpu_forward(inputs, output, params);
}
fn matmulBackward(inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) void {
    c_api.matmul_cpu_backward(inputs, output, params);
}
