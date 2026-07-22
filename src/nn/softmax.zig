const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const functions = @import("../ops/functions.zig");
const Device = @import("../device.zig").Device;

pub fn Softmax(comptime T: type) type {
    return struct {
        dev: *Device,
        dim: ?u64,
        keepdim: bool,

        pub fn init(dev: *Device, dim: ?u64, keepdim: bool) @This() {
            return .{ .dev = dev, .dim = dim, .keepdim = keepdim };
        }

        pub fn call(self: *@This(), gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
            const max_val = try functions.max(T, gpa, x, self.dim, self.keepdim);
            const shifted = try functions.sub(T, gpa, x, max_val);
            const e = try functions.exp(T, gpa, shifted);
            const sum = try functions.sum(T, gpa, e, self.dim, self.keepdim);
            return functions.div(T, gpa, e, sum);
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}
