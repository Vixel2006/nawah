const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Device = @import("../device.zig").Device;

pub fn Flatten(comptime T: type) type {
    return struct {
        dev: *Device,
        start_dim: u64,

        pub fn init(dev: *Device, start_dim: u64) @This() {
            return .{ .dev = dev, .start_dim = start_dim };
        }

        pub fn call(self: *@This(), gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
            var outer: u64 = 1;
            var i: u64 = 0;
            while (i < self.start_dim) : (i += 1) {
                outer *= x.shape[@as(usize, @intCast(i))];
            }
            var inner: u64 = 1;
            while (i < x.ndim) : (i += 1) {
                inner *= x.shape[@as(usize, @intCast(i))];
            }
            const shape = [_]u64{ outer, inner };
            var out = try gpa.create(Tensor(T));
            out.* = try Tensor(T).init(self.dev, shape[0..], x.requires_grad);
            out.data = x.data;
            return out;
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}
