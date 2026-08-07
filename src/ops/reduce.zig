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
    return switch (device) {
        0 => switch (op) {
            .sum => .{ .forward = c_api.sum_cpu_forward, .backward = c_api.sum_cpu_backward },
            .mean => .{ .forward = c_api.mean_cpu_forward, .backward = c_api.mean_cpu_backward },
            .max => .{ .forward = c_api.max_cpu_forward, .backward = c_api.max_cpu_backward },
            .min => .{ .forward = c_api.min_cpu_forward, .backward = c_api.min_cpu_backward },
        },
        1 => switch (op) {
            .sum => .{ .forward = c_api.sum_cuda_forward, .backward = c_api.sum_cuda_backward },
            .mean => .{ .forward = c_api.mean_cuda_forward, .backward = c_api.mean_cuda_backward },
            .max => .{ .forward = c_api.max_cuda_forward, .backward = c_api.max_cuda_backward },
            .min => .{ .forward = c_api.min_cuda_forward, .backward = c_api.min_cuda_backward },
        },
        else => @panic("unknown device"),
    };
}
