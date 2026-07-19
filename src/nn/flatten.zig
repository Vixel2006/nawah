const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;

pub fn Flatten(comptime T: type) type {
    return struct {
        start_dim: u64,

        pub fn init(start_dim: u64) @This() {
            return .{ .start_dim = start_dim };
        }

        pub fn forward(self: *@This(), x: *Tensor(T), alloc: std.mem.Allocator) !*Tensor(T) {
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
            var out = try alloc.create(Tensor(T));
            out.* = try Tensor(T).init(alloc, shape[0..], x.requires_grad);
            out.data = x.data;
            return out;
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}
