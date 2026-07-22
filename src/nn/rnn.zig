const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const functions = @import("../ops/functions.zig");
const Device = @import("../device.zig").Device;

pub fn RNN(comptime T: type) type {
    return struct {
        dev: *Device,
        input_size: u64,
        hidden_size: u64,
        num_layers: u64,
        w_ih: *Tensor(T),
        w_hh: *Tensor(T),
        b_ih: ?*Tensor(T),
        b_hh: ?*Tensor(T),

        pub fn init(dev: *Device, rng: std.Random, input_size: u64, hidden_size: u64, num_layers: u64, bias: bool) !@This() {
            const alloc = dev.allocator();
            const w_ih_t = try alloc.create(Tensor(T));
            w_ih_t.* = try Tensor(T).kaimingUniform(dev, &.{ input_size, hidden_size }, true, rng);
            w_ih_t.device = @enumFromInt(dev.kind());
            const w_hh_t = try alloc.create(Tensor(T));
            w_hh_t.* = try Tensor(T).kaimingUniform(dev, &.{ hidden_size, hidden_size }, true, rng);
            w_hh_t.device = @enumFromInt(dev.kind());
            var b_ih_t: ?*Tensor(T) = null;
            var b_hh_t: ?*Tensor(T) = null;
            if (bias) {
                const bih = try alloc.create(Tensor(T));
                bih.* = try Tensor(T).zeros(dev, &.{hidden_size}, true);
                bih.device = @enumFromInt(dev.kind());
                b_ih_t = bih;
                const bhh = try alloc.create(Tensor(T));
                bhh.* = try Tensor(T).zeros(dev, &.{hidden_size}, true);
                bhh.device = @enumFromInt(dev.kind());
                b_hh_t = bhh;
            }
            return .{
                .dev = dev,
                .input_size = input_size,
                .hidden_size = hidden_size,
                .num_layers = num_layers,
                .w_ih = w_ih_t,
                .w_hh = w_hh_t,
                .b_ih = b_ih_t,
                .b_hh = b_hh_t,
            };
        }

        pub fn call(self: *@This(), gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
            const batch_size = x.shape[0];
            const h_t = try gpa.create(Tensor(T));
            h_t.* = try Tensor(T).zeros(self.dev, &.{ batch_size, self.hidden_size }, x.requires_grad);
            h_t.device = @enumFromInt(self.dev.kind());

            const ih = try functions.matmul(T, gpa, x, self.w_ih);
            const hh = try functions.matmul(T, gpa, h_t, self.w_hh);
            var pre = try functions.add(T, gpa, ih, hh);
            if (self.b_ih) |b_ih| {
                const with_b_ih = try functions.add(T, gpa, pre, b_ih);
                if (self.b_hh) |b_hh| {
                    pre = try functions.add(T, gpa, with_b_ih, b_hh);
                } else {
                    pre = with_b_ih;
                }
            }

            const two = try functions.add(T, gpa, pre, pre);
            const e2x = try functions.exp(T, gpa, two);
            const one_t = try gpa.create(Tensor(T));
            one_t.* = try Tensor(T).ones(self.dev, pre.shape[0..pre.ndim], false);
            one_t.device = @enumFromInt(self.dev.kind());
            const num = try functions.sub(T, gpa, e2x, one_t);
            const den = try functions.add(T, gpa, e2x, one_t);
            return functions.div(T, gpa, num, den);
        }

        pub fn parameters(self: *@This(), gpa: std.mem.Allocator) ![]*Tensor(T) {
            var list = std.ArrayList(*Tensor(T)).init(gpa);
            try list.append(self.w_ih);
            try list.append(self.w_hh);
            if (self.b_ih) |b| try list.append(b);
            if (self.b_hh) |b| try list.append(b);
            return list.toOwnedSlice();
        }

        pub fn deinit(self: *@This()) void {
            const alloc = self.dev.allocator();
            self.w_ih.deinit(null);
            alloc.destroy(self.w_ih);
            self.w_hh.deinit(null);
            alloc.destroy(self.w_hh);
            if (self.b_ih) |b| {
                b.deinit(null);
                alloc.destroy(b);
            }
            if (self.b_hh) |b| {
                b.deinit(null);
                alloc.destroy(b);
            }
        }
    };
}
