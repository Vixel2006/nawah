const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const functions = @import("../ops/functions.zig");

fn sqrtHelper(comptime T: type, alloc: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
    const half_t = try alloc.create(Tensor(T));
    half_t.* = try Tensor(T).fromData(alloc, &.{1}, &.{0.5}, false);
    const log_x = try functions.log(T, alloc, x);
    const half_log = try functions.mul(T, alloc, log_x, half_t);
    return functions.exp(T, alloc, half_log);
}

pub fn BatchNorm1d(comptime T: type) type {
    return struct {
        alloc: std.mem.Allocator,
        num_features: u64,
        eps: f64,
        momentum: f64,
        gamma: *Tensor(T),
        beta: *Tensor(T),
        running_mean: *Tensor(T),
        running_var: *Tensor(T),
        training: bool,

        pub fn init(alloc: std.mem.Allocator, num_features: u64, eps: f64, momentum: f64) !@This() {
            const gamma = try alloc.create(Tensor(T));
            gamma.* = try Tensor(T).ones(alloc, &.{num_features}, true);
            const beta = try alloc.create(Tensor(T));
            beta.* = try Tensor(T).zeros(alloc, &.{num_features}, true);
            const rm = try alloc.create(Tensor(T));
            rm.* = try Tensor(T).zeros(alloc, &.{num_features}, false);
            const rv = try alloc.create(Tensor(T));
            rv.* = try Tensor(T).ones(alloc, &.{num_features}, false);
            return .{
                .alloc = alloc,
                .num_features = num_features,
                .eps = eps,
                .momentum = momentum,
                .gamma = gamma,
                .beta = beta,
                .running_mean = rm,
                .running_var = rv,
                .training = true,
            };
        }

        pub fn forward(self: *@This(), x: *Tensor(T), alloc: std.mem.Allocator) !*Tensor(T) {
            const mean = try functions.mean(T, alloc, x, 0, true);
            const centered = try functions.sub(T, alloc, x, mean);
            const sq = try functions.mul(T, alloc, centered, centered);
            const var_t = try functions.mean(T, alloc, sq, 0, true);
            const eps_ptr = try alloc.create(Tensor(T));
            eps_ptr.* = try Tensor(T).fromData(alloc, &.{1}, &.{@as(T, @floatCast(self.eps))}, false);
            const var_eps = try functions.add(T, alloc, var_t, eps_ptr);
            const std_t = try sqrtHelper(T, alloc, var_eps);
            const normalized = try functions.div(T, alloc, centered, std_t);
            const scaled = try functions.mul(T, alloc, normalized, self.gamma);
            return functions.add(T, alloc, scaled, self.beta);
        }

        pub fn parameters(self: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{ self.gamma, self.beta };
        }

        pub fn deinit(self: *@This()) void {
            self.gamma.deinit();
            self.alloc.destroy(self.gamma);
            self.beta.deinit();
            self.alloc.destroy(self.beta);
            self.running_mean.deinit();
            self.alloc.destroy(self.running_mean);
            self.running_var.deinit();
            self.alloc.destroy(self.running_var);
        }
    };
}
