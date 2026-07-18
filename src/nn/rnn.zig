const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const functions = @import("../ops/functions.zig");

pub fn RNN(comptime T: type) type {
    return struct {
        alloc: std.mem.Allocator,
        input_size: u64,
        hidden_size: u64,
        num_layers: u64,
        w_ih: *Tensor(T),
        w_hh: *Tensor(T),
        b_ih: ?*Tensor(T),
        b_hh: ?*Tensor(T),

        pub fn init(alloc: std.mem.Allocator, rng: std.Random, input_size: u64, hidden_size: u64, num_layers: u64, bias: bool) !@This() {
            const w_ih_t = try alloc.create(Tensor(T));
            w_ih_t.* = try Tensor(T).kaimingUniform(alloc, &.{ input_size, hidden_size }, true, rng);
            const w_hh_t = try alloc.create(Tensor(T));
            w_hh_t.* = try Tensor(T).kaimingUniform(alloc, &.{ hidden_size, hidden_size }, true, rng);
            var b_ih_t: ?*Tensor(T) = null;
            var b_hh_t: ?*Tensor(T) = null;
            if (bias) {
                const bih = try alloc.create(Tensor(T));
                bih.* = try Tensor(T).zeros(alloc, &.{hidden_size}, true);
                b_ih_t = bih;
                const bhh = try alloc.create(Tensor(T));
                bhh.* = try Tensor(T).zeros(alloc, &.{hidden_size}, true);
                b_hh_t = bhh;
            }
            return .{
                .alloc = alloc,
                .input_size = input_size,
                .hidden_size = hidden_size,
                .num_layers = num_layers,
                .w_ih = w_ih_t,
                .w_hh = w_hh_t,
                .b_ih = b_ih_t,
                .b_hh = b_hh_t,
            };
        }

        pub fn forward(self: *@This(), x: *Tensor(T), alloc: std.mem.Allocator) !*Tensor(T) {
            const batch_size = x.shape[0];
            const h_t = try alloc.create(Tensor(T));
            h_t.* = try Tensor(T).zeros(alloc, &.{ batch_size, self.hidden_size }, x.requires_grad);

            const seq_len = x.ndim;
            _ = seq_len;
            const ih = try functions.matmul(T, alloc, x, self.w_ih);
            const hh = try functions.matmul(T, alloc, h_t, self.w_hh);
            var pre = try functions.add(T, alloc, ih, hh);
            if (self.b_ih) |b_ih| {
                const with_b_ih = try functions.add(T, alloc, pre, b_ih);
                if (self.b_hh) |b_hh| {
                    pre = try functions.add(T, alloc, with_b_ih, b_hh);
                } else {
                    pre = with_b_ih;
                }
            }

            const two = try functions.add(T, alloc, pre, pre);
            const e2x = try functions.exp(T, alloc, two);
            const one_t = try alloc.create(Tensor(T));
            one_t.* = try Tensor(T).ones(alloc, pre.shape[0..pre.ndim], false);
            const num = try functions.sub(T, alloc, e2x, one_t);
            const den = try functions.add(T, alloc, e2x, one_t);
            return functions.div(T, alloc, num, den);
        }

        pub fn parameters(self: *@This(), alloc: std.mem.Allocator) ![]*Tensor(T) {
            var list = std.ArrayList(*Tensor(T)).init(alloc);
            try list.append(alloc, self.w_ih);
            try list.append(alloc, self.w_hh);
            if (self.b_ih) |b| try list.append(alloc, b);
            if (self.b_hh) |b| try list.append(alloc, b);
            return list.items;
        }

        pub fn deinit(self: *@This()) void {
            self.w_ih.deinit();
            self.alloc.destroy(self.w_ih);
            self.w_hh.deinit();
            self.alloc.destroy(self.w_hh);
            if (self.b_ih) |b| {
                b.deinit();
                self.alloc.destroy(b);
            }
            if (self.b_hh) |b| {
                b.deinit();
                self.alloc.destroy(b);
            }
        }
    };
}
