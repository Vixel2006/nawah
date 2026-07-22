const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const functions = @import("../ops/functions.zig");
const Device = @import("../device.zig").Device;

pub fn ReLU(comptime T: type) type {
    return struct {
        dev: *Device,

        pub fn init(dev: *Device) @This() {
            return .{ .dev = dev };
        }

        pub fn call(_: *@This(), gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
            return functions.relu(T, gpa, x);
        }
        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }
        pub fn deinit(_: *@This()) void {}
    };
}

pub fn Tanh(comptime T: type) type {
    return struct {
        dev: *Device,

        pub fn init(dev: *Device) @This() {
            return .{ .dev = dev };
        }

        pub fn call(self: *@This(), gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
            const two = try functions.add(T, gpa, x, x);
            const e2x = try functions.exp(T, gpa, two);
            const one_t = try gpa.create(Tensor(T));
            one_t.* = try Tensor(T).ones(self.dev, x.shape[0..x.ndim], false);
            const num = try functions.sub(T, gpa, e2x, one_t);
            const den = try functions.add(T, gpa, e2x, one_t);
            return functions.div(T, gpa, num, den);
        }
        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }
        pub fn deinit(_: *@This()) void {}
    };
}

pub fn Sigmoid(comptime T: type) type {
    return struct {
        dev: *Device,

        pub fn init(dev: *Device) @This() {
            return .{ .dev = dev };
        }

        pub fn call(self: *@This(), gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
            const neg_x = try functions.neg(T, gpa, x);
            const e = try functions.exp(T, gpa, neg_x);
            const one_t = try gpa.create(Tensor(T));
            one_t.* = try Tensor(T).ones(self.dev, x.shape[0..x.ndim], false);
            const den = try functions.add(T, gpa, e, one_t);
            return functions.div(T, gpa, one_t, den);
        }
        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }
        pub fn deinit(_: *@This()) void {}
    };
}
