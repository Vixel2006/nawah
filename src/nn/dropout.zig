const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const functions = @import("../ops/functions.zig");
const Device = @import("../device.zig").Device;

pub fn Dropout(comptime T: type) type {
    return struct {
        dev: *Device,
        p: f64,
        rng: std.Random,

        pub fn init(dev: *Device, p: f64, rng: std.Random) @This() {
            return .{ .dev = dev, .p = p, .rng = rng };
        }

        pub fn call(self: *@This(), gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
            const dev_alloc = self.dev.allocator();
            const scale = 1.0 / (1.0 - self.p);
            var numel: u64 = 1;
            for (x.shape[0..x.ndim]) |dim| numel *= dim;

            var mask = try Tensor(T).init(self.dev, x.shape[0..x.ndim], false);
            const mem = dev_alloc.alloc(numel * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory;
            mask.data = @as([*]T, @ptrCast(@alignCast(mem)))[0..numel];
            for (0..numel) |i| {
                mask.data.?[i] = if (self.rng.float(f64) < self.p) 0 else @as(T, @floatCast(scale));
            }

            const mask_t = try gpa.create(Tensor(T));
            mask_t.* = mask;
            return functions.mul(T, gpa, x, mask_t);
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}
