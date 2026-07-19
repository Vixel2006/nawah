const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const functions = @import("../ops/functions.zig");

pub fn ReLU(comptime T: type) type {
    return struct {
        pub fn call(_: *@This(), x: *Tensor(T)) !*Tensor(T) {
            return functions.relu(T, x);
        }
        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }
        pub fn deinit(_: *@This()) void {}
    };
}

pub fn Tanh(comptime T: type) type {
    return struct {
        pub fn call(_: *@This(), x: *Tensor(T)) !*Tensor(T) {
            const alloc = x.alloc;
            const two = try functions.add(T, x, x);
            const e2x = try functions.exp(T, two);
            const one = try Tensor(T).ones(alloc, x.shape[0..x.ndim], false);
            const one_t = try alloc.create(Tensor(T));
            one_t.* = one;
            const num = try functions.sub(T, e2x, one_t);
            const den = try functions.add(T, e2x, one_t);
            return functions.div(T, num, den);
        }
        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }
        pub fn deinit(_: *@This()) void {}
    };
}

pub fn Sigmoid(comptime T: type) type {
    return struct {
        pub fn call(_: *@This(), x: *Tensor(T)) !*Tensor(T) {
            const alloc = x.alloc;
            const neg_x = try functions.neg(T, x);
            const e = try functions.exp(T, neg_x);
            const one = try Tensor(T).ones(alloc, x.shape[0..x.ndim], false);
            const one_t = try alloc.create(Tensor(T));
            one_t.* = one;
            const den = try functions.add(T, e, one_t);
            return functions.div(T, one_t, den);
        }
        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }
        pub fn deinit(_: *@This()) void {}
    };
}
