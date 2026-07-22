const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const functions = @import("../ops/functions.zig");
const Device = @import("../device.zig").Device;

fn sqrtHelper(comptime T: type, gpa: std.mem.Allocator, dev: *Device, x: *Tensor(T)) !*Tensor(T) {
    const half_t = try gpa.create(Tensor(T));
    half_t.* = try Tensor(T).fromData(dev, &.{1}, &.{0.5}, false);
    const log_x = try functions.log(T, gpa, x);
    const half_log = try functions.mul(T, gpa, log_x, half_t);
    return functions.exp(T, gpa, half_log);
}

const Reduction = enum { mean, sum, none };

fn reduceLoss(comptime T: type, gpa: std.mem.Allocator, loss: *Tensor(T), reduction: Reduction) !*Tensor(T) {
    return switch (reduction) {
        .mean => functions.mean(T, gpa, loss, null, false),
        .sum => functions.sum(T, gpa, loss, null, false),
        .none => loss,
    };
}

pub fn MSELoss(comptime T: type) type {
    return struct {
        dev: *Device,
        reduction: Reduction,

        pub fn init(dev: *Device, reduction: Reduction) @This() {
            return .{ .dev = dev, .reduction = reduction };
        }

        pub fn call(self: *@This(), gpa: std.mem.Allocator, input: *Tensor(T), target: *Tensor(T)) !*Tensor(T) {
            const diff = try functions.sub(T, gpa, input, target);
            const sq = try functions.mul(T, gpa, diff, diff);
            return reduceLoss(T, gpa, sq, self.reduction);
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}

pub fn L1Loss(comptime T: type) type {
    return struct {
        dev: *Device,
        reduction: Reduction,

        pub fn init(dev: *Device, reduction: Reduction) @This() {
            return .{ .dev = dev, .reduction = reduction };
        }

        pub fn call(self: *@This(), gpa: std.mem.Allocator, input: *Tensor(T), target: *Tensor(T)) !*Tensor(T) {
            const diff = try functions.abs(T, gpa, try functions.sub(T, gpa, input, target));
            return reduceLoss(T, gpa, diff, self.reduction);
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}

pub fn BCELoss(comptime T: type) type {
    return struct {
        dev: *Device,
        reduction: Reduction,

        pub fn init(dev: *Device, reduction: Reduction) @This() {
            return .{ .dev = dev, .reduction = reduction };
        }

        pub fn call(self: *@This(), gpa: std.mem.Allocator, input: *Tensor(T), target: *Tensor(T)) !*Tensor(T) {
            const ones = try gpa.create(Tensor(T));
            ones.* = try Tensor(T).ones(self.dev, input.shape[0..input.ndim], false);

            const log_input = try functions.log(T, gpa, input);
            const neg_input = try functions.sub(T, gpa, ones, input);
            const log_one_minus = try functions.log(T, gpa, neg_input);
            const neg_target = try functions.sub(T, gpa, ones, target);

            const term1 = try functions.mul(T, gpa, target, log_input);
            const term2 = try functions.mul(T, gpa, neg_target, log_one_minus);
            const loss = try functions.neg(T, gpa, try functions.add(T, gpa, term1, term2));
            return reduceLoss(T, gpa, loss, self.reduction);
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}

pub fn BCEWithLogitsLoss(comptime T: type) type {
    return struct {
        dev: *Device,
        reduction: Reduction,

        pub fn init(dev: *Device, reduction: Reduction) @This() {
            return .{ .dev = dev, .reduction = reduction };
        }

        pub fn call(self: *@This(), gpa: std.mem.Allocator, input: *Tensor(T), target: *Tensor(T)) !*Tensor(T) {
            const ones = try gpa.create(Tensor(T));
            ones.* = try Tensor(T).ones(self.dev, input.shape[0..input.ndim], false);

            const neg_input = try functions.neg(T, gpa, input);
            const exp_neg = try functions.exp(T, gpa, neg_input);
            const sig = try functions.div(T, gpa, ones, try functions.add(T, gpa, exp_neg, ones));

            return try BCELoss(T).init(self.dev, self.reduction).call(gpa, sig, target);
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}

pub fn CosineEmbeddingLoss(comptime T: type) type {
    return struct {
        dev: *Device,
        margin: T,
        dim: u64,
        reduction: Reduction,

        pub fn init(dev: *Device, opts: struct {
            margin: T = 0.0,
            dim: u64 = std.math.maxInt(u64),
            reduction: Reduction = .mean,
        }) @This() {
            return .{
                .dev = dev,
                .margin = opts.margin,
                .dim = if (opts.dim == std.math.maxInt(u64)) std.math.maxInt(u64) else opts.dim,
                .reduction = opts.reduction,
            };
        }

        pub fn call(self: *@This(), gpa: std.mem.Allocator, x1: *Tensor(T), x2: *Tensor(T), target: *Tensor(T)) !*Tensor(T) {
            const dim: u64 = if (self.dim == std.math.maxInt(u64)) x1.ndim - 1 else self.dim;

            const dot = try functions.sum(T, gpa, try functions.mul(T, gpa, x1, x2), dim, true);
            const x1_sq = try functions.sum(T, gpa, try functions.mul(T, gpa, x1, x1), dim, true);
            const x2_sq = try functions.sum(T, gpa, try functions.mul(T, gpa, x2, x2), dim, true);
            const norm1 = try sqrtHelper(T, gpa, self.dev, x1_sq);
            const norm2 = try sqrtHelper(T, gpa, self.dev, x2_sq);
            const cos_sim = try functions.div(T, gpa, dot, try functions.mul(T, gpa, norm1, norm2));

            const cos_shape = cos_sim.shape[0..cos_sim.ndim];
            const ones = try gpa.create(Tensor(T));
            ones.* = try Tensor(T).ones(self.dev, cos_shape, false);
            const two_t = try gpa.create(Tensor(T));
            two_t.* = try Tensor(T).fromData(self.dev, &.{1}, &.{2.0}, false);

            const pos_weight = try functions.div(T, gpa, try functions.add(T, gpa, target, ones), two_t);
            const neg_weight = try functions.div(T, gpa, try functions.sub(T, gpa, ones, target), two_t);

            const pos_loss = try functions.mul(T, gpa, pos_weight, try functions.sub(T, gpa, ones, cos_sim));
            const margin_t = try gpa.create(Tensor(T));
            margin_t.* = try Tensor(T).fromData(self.dev, &.{1}, &.{self.margin}, false);
            const neg_loss = try functions.mul(T, gpa, neg_weight, try functions.relu(T, gpa, try functions.sub(T, gpa, cos_sim, margin_t)));

            const loss = try functions.add(T, gpa, pos_loss, neg_loss);
            return reduceLoss(T, gpa, loss, self.reduction);
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}
