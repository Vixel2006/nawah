const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Graph = @import("../graph.zig").Graph;
const functions = @import("../ops/functions.zig");

fn scalarLike(comptime T: type, alloc: std.mem.Allocator, val: T) !*Tensor(T) {
    const t = try alloc.create(Tensor(T));
    t.* = try Tensor(T).fromData(alloc, &.{1}, &.{val}, false);
    return t;
}

fn sqrtTensor(comptime T: type, x: *Tensor(T)) !*Tensor(T) {
    const half = try scalarLike(T, x.alloc, 0.5);
    return functions.exp(T, try functions.mul(T, try functions.log(T, x), half));
}

pub fn Adam(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        params: std.ArrayList(*Tensor(T)),
        m_bufs: std.ArrayList([]T),
        v_bufs: std.ArrayList([]T),
        t: u64,
        temp_arena: std.heap.ArenaAllocator,

        pub fn init(alloc: std.mem.Allocator, opts: struct {
            lr: T = 0.001,
            beta1: T = 0.9,
            beta2: T = 0.999,
            eps: T = 1e-8,
        }) Self {
            return .{
                .alloc = alloc,
                .lr = opts.lr,
                .beta1 = opts.beta1,
                .beta2 = opts.beta2,
                .eps = opts.eps,
                .params = .empty,
                .m_bufs = .empty,
                .v_bufs = .empty,
                .t = 0,
                .temp_arena = std.heap.ArenaAllocator.init(alloc),
            };
        }

        pub fn deinit(self: *Self) void {
            self.temp_arena.deinit();
            for (self.m_bufs.items) |m| self.alloc.free(m);
            for (self.v_bufs.items) |v| self.alloc.free(v);
            self.m_bufs.deinit(self.alloc);
            self.v_bufs.deinit(self.alloc);
            self.params.deinit(self.alloc);
        }

        pub fn addParams(self: *Self, p: []*Tensor(T)) !void {
            try self.params.appendSlice(self.alloc, p);
            try self.m_bufs.ensureUnusedCapacity(self.alloc, p.len);
            try self.v_bufs.ensureUnusedCapacity(self.alloc, p.len);
            for (p) |param| {
                const n = param.data.?.len;
                const m = try self.alloc.alloc(T, n);
                const v = try self.alloc.alloc(T, n);
                @memset(m, 0);
                @memset(v, 0);
                try self.m_bufs.append(self.alloc, m);
                try self.v_bufs.append(self.alloc, v);
            }
        }

        pub fn step(self: *Self) !void {
            self.t += 1;
            const arena = self.temp_arena.allocator();
            const b1 = self.beta1;
            const b2 = self.beta2;
            const eps = self.eps;
            const lr = self.lr;
            const bc1 = 1 - std.math.pow(T, b1, @as(T, @floatFromInt(self.t)));
            const bc2 = 1 - std.math.pow(T, b2, @as(T, @floatFromInt(self.t)));

            for (self.params.items, self.m_bufs.items, self.v_bufs.items) |p, m, v| {
                const g = p.grad orelse continue;
                const gd = g.data.?;

                for (m, v, gd) |*m_val, *v_val, grad| {
                    m_val.* = b1 * m_val.* + (1 - b1) * grad;
                    v_val.* = b2 * v_val.* + (1 - b2) * grad * grad;
                }

                const m_t = try arena.create(Tensor(T));
                m_t.* = try Tensor(T).fromData(arena, p.shape[0..p.ndim], m, false);
                const v_t = try arena.create(Tensor(T));
                v_t.* = try Tensor(T).fromData(arena, p.shape[0..p.ndim], v, false);

                const bc1_t = scalarLike(T, arena, bc1) catch continue;
                const bc2_t = scalarLike(T, arena, bc2) catch continue;
                const eps_t = scalarLike(T, arena, eps) catch continue;
                const lr_t = scalarLike(T, arena, lr) catch continue;

                const m_hat = functions.div(T, m_t, bc1_t) catch continue;
                const v_hat = functions.div(T, v_t, bc2_t) catch continue;
                const sqrt_v = sqrtTensor(T, v_hat) catch continue;
                const denom = functions.add(T, sqrt_v, eps_t) catch continue;
                const correction = functions.div(T, m_hat, denom) catch continue;
                const step_t = functions.mul(T, correction, lr_t) catch continue;
                const new_p = functions.sub(T, p, step_t) catch continue;

                var graph = Graph(T).init(arena);
                graph.dag(new_p.creator.?);
                graph.forward();
                @memcpy(p.data.?, new_p.data.?);
            }

            _ = self.temp_arena.reset(.retain_capacity);
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
