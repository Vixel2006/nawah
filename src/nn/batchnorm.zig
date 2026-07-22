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

pub fn BatchNorm1d(comptime T: type) type {
    return struct {
        dev: *Device,
        num_features: u64,
        eps: f64,
        momentum: f64,
        gamma: *Tensor(T),
        beta: *Tensor(T),
        running_mean: *Tensor(T),
        running_var: *Tensor(T),
        training: bool,

        pub fn init(dev: *Device, num_features: u64, eps: f64, momentum: f64) !@This() {
            const alloc = dev.allocator();
            const gamma = try alloc.create(Tensor(T));
            gamma.* = try Tensor(T).ones(dev, &.{num_features}, true);
            gamma.device = @enumFromInt(dev.kind());
            const beta = try alloc.create(Tensor(T));
            beta.* = try Tensor(T).zeros(dev, &.{num_features}, true);
            beta.device = @enumFromInt(dev.kind());
            const rm = try alloc.create(Tensor(T));
            rm.* = try Tensor(T).zeros(dev, &.{num_features}, false);
            rm.device = @enumFromInt(dev.kind());
            const rv = try alloc.create(Tensor(T));
            rv.* = try Tensor(T).ones(dev, &.{num_features}, false);
            rv.device = @enumFromInt(dev.kind());
            return .{
                .dev = dev,
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

        pub fn call(self: *@This(), gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
            const mean = try functions.mean(T, gpa, x, 0, true);
            const centered = try functions.sub(T, gpa, x, mean);
            const sq = try functions.mul(T, gpa, centered, centered);
            const var_t = try functions.mean(T, gpa, sq, 0, true);
            const eps_ptr = try gpa.create(Tensor(T));
            eps_ptr.* = try Tensor(T).fromData(self.dev, &.{1}, &.{@as(T, @floatCast(self.eps))}, false);
            const var_eps = try functions.add(T, gpa, var_t, eps_ptr);
            const std_t = try sqrtHelper(T, gpa, self.dev, var_eps);
            const normalized = try functions.div(T, gpa, centered, std_t);
            const scaled = try functions.mul(T, gpa, normalized, self.gamma);
            return functions.add(T, gpa, scaled, self.beta);
        }

        pub fn parameters(self: *@This(), _: std.mem.Allocator) ![]*Tensor(T) {
            return &.{ self.gamma, self.beta };
        }

        pub fn deinit(self: *@This()) void {
            const alloc = self.dev.allocator();
            self.gamma.deinit(null);
            alloc.destroy(self.gamma);
            self.beta.deinit(null);
            alloc.destroy(self.beta);
            self.running_mean.deinit(null);
            alloc.destroy(self.running_mean);
            self.running_var.deinit(null);
            alloc.destroy(self.running_var);
        }
    };
}
