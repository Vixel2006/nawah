const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const functions = @import("../ops/functions.zig");

pub fn Dropout(comptime T: type) type {
    return struct {
        p: f64,
        rng: std.Random,

        pub fn init(p: f64, rng: std.Random) @This() {
            return .{ .p = p, .rng = rng };
        }

        pub fn forward(self: *@This(), x: *Tensor(T), alloc: std.mem.Allocator) !*Tensor(T) {
            const scale = 1.0 / (1.0 - self.p);
            var numel: u64 = 1;
            for (x.shape[0..x.ndim]) |dim| numel *= dim;
            var mask = try Tensor(T).init(alloc, x.shape[0..x.ndim], false);
            mask.data = try alloc.alloc(T, numel);
            for (0..numel) |i| {
                mask.data.?[i] = if (self.rng.float(f64) < self.p) 0 else @as(T, @floatCast(scale));
            }
            const mask_t = try alloc.create(Tensor(T));
            mask_t.* = mask;
            return functions.mul(T, alloc, x, mask_t);
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}
