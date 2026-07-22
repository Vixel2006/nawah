const c_api = @import("../c_api.zig");
const C_Tensor = c_api.C_Tensor;
const KernelParams = c_api.KernelParams;
const Function = @import("../function.zig").Function;

pub const Fused = enum {
    matmul_relu,
};

pub fn getFusedFn(op: Fused, dtype: u32, device: u32) Function {
    _ = dtype;
    switch (device) {
        0 => {},
        1 => @panic("cuda kernels not wired"),
        else => @panic("unknown device"),
    }
    return switch (op) {
        .matmul_relu => .{ .forward = matmulReluForward, .backward = matmulReluBackward },
    };
}

fn matmulReluForward(inputs: [*c]?*const C_Tensor, output: *C_Tensor, params: KernelParams) void {
    c_api.matmul_cpu_forward(inputs, output, params);
    var relu_inputs = [_]?*const C_Tensor{output};
    c_api.leaky_relu_cpu_forward(&relu_inputs, output, params);
}

fn matmulReluBackward(inputs: [*c]?*const C_Tensor, output: *const C_Tensor, params: KernelParams) void {
    _ = inputs;
    _ = output;
    _ = params;
}
