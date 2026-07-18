const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const functions = @import("../ops/functions.zig");

pub fn ReLU(comptime T: type) type {
    return struct {
        pub fn forward(_: *@This(), x: *Tensor(T), alloc: std.mem.Allocator) !*Tensor(T) {
            return functions.relu(T, alloc, x);
        }
        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }
        pub fn deinit(_: *@This()) void {}
    };
}

pub fn Tanh(comptime T: type) type {
    return struct {
        pub fn forward(_: *@This(), x: *Tensor(T), alloc: std.mem.Allocator) !*Tensor(T) {
            const two = try functions.add(T, alloc, x, x);
            const e2x = try functions.exp(T, alloc, two);
            const one = try Tensor(T).ones(alloc, x.shape[0..x.ndim], false);
            const one_t = try alloc.create(Tensor(T));
            one_t.* = one;
            const num = try functions.sub(T, alloc, e2x, one_t);
            const den = try functions.add(T, alloc, e2x, one_t);
            return functions.div(T, alloc, num, den);
        }
        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }
        pub fn deinit(_: *@This()) void {}
    };
}

pub fn Sigmoid(comptime T: type) type {
    return struct {
        pub fn forward(_: *@This(), x: *Tensor(T), alloc: std.mem.Allocator) !*Tensor(T) {
            const neg_x = try functions.neg(T, alloc, x);
            const e = try functions.exp(T, alloc, neg_x);
            const one = try Tensor(T).ones(alloc, x.shape[0..x.ndim], false);
            const one_t = try alloc.create(Tensor(T));
            one_t.* = one;
            const den = try functions.add(T, alloc, e, one_t);
            return functions.div(T, alloc, one_t, den);
        }
        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }
        pub fn deinit(_: *@This()) void {}
    };
}
