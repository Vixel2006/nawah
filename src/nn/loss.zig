const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const functions = @import("../ops/functions.zig");

fn sqrtHelper(comptime T: type, x: *Tensor(T)) !*Tensor(T) {
    const alloc = x.alloc;
    const half_t = try alloc.create(Tensor(T));
    half_t.* = try Tensor(T).fromData(alloc, &.{1}, &.{0.5}, false);
    const log_x = try functions.log(T, x);
    const half_log = try functions.mul(T, log_x, half_t);
    return functions.exp(T, half_log);
}

const Reduction = enum { mean, sum, none };

fn reduceLoss(comptime T: type, loss: *Tensor(T), reduction: Reduction) !*Tensor(T) {
    return switch (reduction) {
        .mean => functions.mean(T, loss, null, false),
        .sum => functions.sum(T, loss, null, false),
        .none => loss,
    };
}

pub fn MSELoss(comptime T: type) type {
    return struct {
        reduction: Reduction,

        pub fn init(reduction: Reduction) @This() {
            return .{ .reduction = reduction };
        }

        pub fn call(self: *@This(), input: *Tensor(T), target: *Tensor(T)) !*Tensor(T) {
            const diff = try functions.sub(T, input, target);
            const sq = try functions.mul(T, diff, diff);
            return reduceLoss(T, sq, self.reduction);
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}

pub fn L1Loss(comptime T: type) type {
    return struct {
        reduction: Reduction,

        pub fn init(reduction: Reduction) @This() {
            return .{ .reduction = reduction };
        }

        pub fn call(self: *@This(), input: *Tensor(T), target: *Tensor(T)) !*Tensor(T) {
            const diff = try functions.abs(T, try functions.sub(T, input, target));
            return reduceLoss(T, diff, self.reduction);
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}

pub fn BCELoss(comptime T: type) type {
    return struct {
        reduction: Reduction,

        pub fn init(reduction: Reduction) @This() {
            return .{ .reduction = reduction };
        }

        pub fn call(self: *@This(), input: *Tensor(T), target: *Tensor(T)) !*Tensor(T) {
            const alloc = input.alloc;
            const ones = try alloc.create(Tensor(T));
            ones.* = try Tensor(T).ones(alloc, input.shape[0..input.ndim], false);

            const log_input = try functions.log(T, input);
            const neg_input = try functions.sub(T, ones, input);
            const log_one_minus = try functions.log(T, neg_input);
            const neg_target = try functions.sub(T, ones, target);

            const term1 = try functions.mul(T, target, log_input);
            const term2 = try functions.mul(T, neg_target, log_one_minus);
            const loss = try functions.neg(T, try functions.add(T, term1, term2));
            return reduceLoss(T, loss, self.reduction);
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}

pub fn BCEWithLogitsLoss(comptime T: type) type {
    return struct {
        reduction: Reduction,

        pub fn init(reduction: Reduction) @This() {
            return .{ .reduction = reduction };
        }

        pub fn call(self: *@This(), input: *Tensor(T), target: *Tensor(T)) !*Tensor(T) {
            const alloc = input.alloc;
            const ones = try alloc.create(Tensor(T));
            ones.* = try Tensor(T).ones(alloc, input.shape[0..input.ndim], false);

            const neg_input = try functions.neg(T, input);
            const exp_neg = try functions.exp(T, neg_input);
            const sig = try functions.div(T, ones, try functions.add(T, exp_neg, ones));

            return try BCELoss(T).init(self.reduction).call(sig, target);
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}

pub fn CosineEmbeddingLoss(comptime T: type) type {
    return struct {
        margin: T,
        dim: u64,
        reduction: Reduction,

        pub fn init(opts: struct {
            margin: T = 0.0,
            dim: u64 = std.math.maxInt(u64),
            reduction: Reduction = .mean,
        }) @This() {
            return .{
                .margin = opts.margin,
                .dim = if (opts.dim == std.math.maxInt(u64)) std.math.maxInt(u64) else opts.dim,
                .reduction = opts.reduction,
            };
        }

        pub fn call(self: *@This(), x1: *Tensor(T), x2: *Tensor(T), target: *Tensor(T)) !*Tensor(T) {
            const alloc = x1.alloc;
            const dim: u64 = if (self.dim == std.math.maxInt(u64)) x1.ndim - 1 else self.dim;

            const dot = try functions.sum(T, try functions.mul(T, x1, x2), dim, true);
            const x1_sq = try functions.sum(T, try functions.mul(T, x1, x1), dim, true);
            const x2_sq = try functions.sum(T, try functions.mul(T, x2, x2), dim, true);
            const norm1 = try sqrtHelper(T, x1_sq);
            const norm2 = try sqrtHelper(T, x2_sq);
            const cos_sim = try functions.div(T, dot, try functions.mul(T, norm1, norm2));

            const cos_shape = cos_sim.shape[0..cos_sim.ndim];
            const ones = try alloc.create(Tensor(T));
            ones.* = try Tensor(T).ones(alloc, cos_shape, false);
            const two_t = try alloc.create(Tensor(T));
            two_t.* = try Tensor(T).fromData(alloc, &.{1}, &.{2.0}, false);

            const pos_weight = try functions.div(T, try functions.add(T, target, ones), two_t);
            const neg_weight = try functions.div(T, try functions.sub(T, ones, target), two_t);

            const pos_loss = try functions.mul(T, pos_weight, try functions.sub(T, ones, cos_sim));
            const margin_t = try alloc.create(Tensor(T));
            margin_t.* = try Tensor(T).fromData(alloc, &.{1}, &.{self.margin}, false);
            const neg_loss = try functions.mul(T, neg_weight, try functions.relu(T, try functions.sub(T, cos_sim, margin_t)));

            const loss = try functions.add(T, pos_loss, neg_loss);
            return reduceLoss(T, loss, self.reduction);
        }

        pub fn parameters(_: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{};
        }

        pub fn deinit(_: *@This()) void {}
    };
}
