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

pub fn RMSprop(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        lr: T,
        alpha: T,
        eps: T,
        momentum: T,
        params: std.ArrayList(*Tensor(T)),
        sq_avg: std.ArrayList([]T),
        mom_bufs: std.ArrayList([]T),
        temp_arena: std.heap.ArenaAllocator,

        pub fn init(alloc: std.mem.Allocator, opts: struct {
            lr: T = 0.01,
            alpha: T = 0.99,
            eps: T = 1e-8,
            momentum: T = 0.0,
        }) Self {
            return .{
                .alloc = alloc,
                .lr = opts.lr,
                .alpha = opts.alpha,
                .eps = opts.eps,
                .momentum = opts.momentum,
                .params = .empty,
                .sq_avg = .empty,
                .mom_bufs = .empty,
                .temp_arena = std.heap.ArenaAllocator.init(alloc),
            };
        }

        pub fn deinit(self: *Self) void {
            self.temp_arena.deinit();
            for (self.sq_avg.items) |s| self.alloc.free(s);
            for (self.mom_bufs.items) |b| self.alloc.free(b);
            self.sq_avg.deinit(self.alloc);
            self.mom_bufs.deinit(self.alloc);
            self.params.deinit(self.alloc);
        }

        pub fn addParams(self: *Self, p: []*Tensor(T)) !void {
            try self.params.appendSlice(self.alloc, p);
            try self.sq_avg.ensureUnusedCapacity(self.alloc, p.len);
            try self.mom_bufs.ensureUnusedCapacity(self.alloc, p.len);
            for (p) |param| {
                const n = param.data.?.len;
                const s = try self.alloc.alloc(T, n);
                const mb = try self.alloc.alloc(T, n);
                @memset(s, 0);
                @memset(mb, 0);
                try self.sq_avg.append(self.alloc, s);
                try self.mom_bufs.append(self.alloc, mb);
            }
        }

        pub fn step(self: *Self) !void {
            const arena = self.temp_arena.allocator();
            const alpha = self.alpha;
            const eps = self.eps;
            const lr = self.lr;
            const mom = self.momentum;

            for (self.params.items, self.sq_avg.items, self.mom_bufs.items) |p, sq, mb| {
                const g = p.grad orelse continue;
                const gd = g.data.?;

                for (sq, gd) |*sq_val, grad| {
                    sq_val.* = alpha * sq_val.* + (1 - alpha) * grad * grad;
                }

                const sq_t = try arena.create(Tensor(T));
                sq_t.* = try Tensor(T).fromData(arena, p.shape[0..p.ndim], sq, false);
                const lr_t = scalarLike(T, arena, lr) catch continue;
                const eps_t = scalarLike(T, arena, eps) catch continue;

                const rms = sqrtTensor(T, sq_t) catch continue;
                const denom = functions.add(T, rms, eps_t) catch continue;
                const raw_step = functions.div(T, g, denom) catch continue;

                var graph = Graph(T).init(arena);
                graph.dag(raw_step.creator.?);
                graph.forward();

                if (mom > 0) {
                    for (mb, raw_step.data.?) |*mb_val, step_val| {
                        mb_val.* = mom * mb_val.* + lr * step_val;
                    }
                    const mb_t = try arena.create(Tensor(T));
                    mb_t.* = try Tensor(T).fromData(arena, p.shape[0..p.ndim], mb, false);
                    const new_p = functions.sub(T, p, mb_t) catch continue;

                    graph.dag(new_p.creator.?);
                    graph.forward();
                    @memcpy(p.data.?, new_p.data.?);
                } else {
                    const step_t = functions.mul(T, raw_step, lr_t) catch continue;
                    const new_p = functions.sub(T, p, step_t) catch continue;

                    graph.dag(new_p.creator.?);
                    graph.forward();
                    @memcpy(p.data.?, new_p.data.?);
                }
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
