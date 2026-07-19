const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;

pub fn Optimizer(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        lr: T,
        params: std.ArrayList(*Tensor(T)),

        pub fn init(alloc: std.mem.Allocator, opts: struct {
            lr: T = 0.01,
        }) Self {
            return .{
                .alloc = alloc,
                .lr = opts.lr,
                .params = .empty,
            };
        }

        pub fn deinit(self: *Self) void {
            self.params.deinit(self.alloc);
        }

        pub fn addParams(self: *Self, p: []*Tensor(T)) !void {
            try self.params.appendSlice(self.alloc, p);
        }

        pub fn setLr(self: *Self, lr: T) void {
            self.lr = lr;
        }

        pub fn step(self: *Self) void {
            for (self.params.items) |p| {
                if (p.grad) |g| {
                    const pd = p.data.?;
                    const gd = g.data.?;
                    for (pd, gd) |*val, grad_val| {
                        val.* -= self.lr * grad_val;
                    }
                }
            }
        }

        pub fn zeroGrad(self: *Self) void {
            for (self.params.items) |p| {
                if (p.grad) |g| {
                    @memset(g.data.?, 0);
                }
            }
        }
    };
}
