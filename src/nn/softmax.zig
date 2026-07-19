const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const functions = @import("../ops/functions.zig");

pub fn Softmax(comptime T: type) type {
    return struct {
        dim: ?u64,
        keepdim: bool,

        pub fn init(dim: ?u64, keepdim: bool) @This() {
            return .{ .dim = dim, .keepdim = keepdim };
        }

        pub fn call(_: *@This(), x: *Tensor(T)) !*Tensor(T) {
            const max_val = try functions.max(T, x, null, false);
            const shifted = try functions.sub(T, x, max_val);
            const e = try functions.exp(T, shifted);
            const sum = try functions.sum(T, e, null, false);
            return functions.div(T, e, sum);
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}
