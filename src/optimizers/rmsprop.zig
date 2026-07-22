const std = @import("std");
const Device = @import("../device.zig").Device;
const Tensor = @import("../tensor.zig").Tensor;
const Graph = @import("../graph.zig").Graph;
const functions = @import("../ops/functions.zig");

fn scalarLike(comptime T: type, gpa: std.mem.Allocator, dev: *Device, val: T) !*Tensor(T) {
    const t = try gpa.create(Tensor(T));
    t.* = try Tensor(T).fromData(dev, &.{1}, &.{val}, false);
    return t;
}

fn sqrtTensor(comptime T: type, gpa: std.mem.Allocator, dev: *Device, x: *Tensor(T)) !*Tensor(T) {
    const half = try scalarLike(T, gpa, dev, 0.5);
    return functions.exp(T, gpa, try functions.mul(T, gpa, try functions.log(T, gpa, x), half));
}

pub fn RMSprop(comptime T: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        dev: *Device,
        lr: T,
        alpha: T,
        eps: T,
        momentum: T,
        params: std.ArrayList(*Tensor(T)),
        sq_avg: std.ArrayList([]T),
        mom_bufs: std.ArrayList([]T),
        temp_arena: std.heap.ArenaAllocator,

        pub fn init(gpa: std.mem.Allocator, dev: *Device, opts: struct {
            lr: T = 0.01,
            alpha: T = 0.99,
            eps: T = 1e-8,
            momentum: T = 0.0,
        }) Self {
            return .{
                .gpa = gpa,
                .dev = dev,
                .lr = opts.lr,
                .alpha = opts.alpha,
                .eps = opts.eps,
                .momentum = opts.momentum,
                .params = .empty,
                .sq_avg = .empty,
                .mom_bufs = .empty,
                .temp_arena = std.heap.ArenaAllocator.init(gpa),
            };
        }

        pub fn deinit(self: *Self) void {
            const dev_alloc = self.dev.allocator();
            for (self.sq_avg.items) |s| dev_alloc.free(@as([*]u8, @ptrCast(s.ptr)), s.len * @sizeOf(T), @alignOf(T));
            for (self.mom_bufs.items) |b| dev_alloc.free(@as([*]u8, @ptrCast(b.ptr)), b.len * @sizeOf(T), @alignOf(T));
            self.sq_avg.deinit(self.gpa);
            self.mom_bufs.deinit(self.gpa);
            self.params.deinit(self.gpa);
            self.temp_arena.deinit();
        }

        pub fn addParams(self: *Self, p: []*Tensor(T)) !void {
            try self.params.appendSlice(self.gpa, p);
            try self.sq_avg.ensureUnusedCapacity(self.gpa, p.len);
            try self.mom_bufs.ensureUnusedCapacity(self.gpa, p.len);
            const dev_alloc = self.dev.allocator();
            for (p) |param| {
                const n = param.data.?.len;
                const s_mem = dev_alloc.alloc(n * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory;
                const s = @as([*]T, @ptrCast(@alignCast(s_mem)))[0..n];
                const mb_mem = dev_alloc.alloc(n * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory;
                const mb = @as([*]T, @ptrCast(@alignCast(mb_mem)))[0..n];
                @memset(s, 0);
                @memset(mb, 0);
                try self.sq_avg.append(self.gpa, s);
                try self.mom_bufs.append(self.gpa, mb);
            }
        }

        pub fn step(self: *Self) !void {
            const arena_alloc = self.temp_arena.allocator();
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

                const sq_t = try arena_alloc.create(Tensor(T));
                sq_t.* = try Tensor(T).fromData(self.dev, p.shape[0..p.ndim], sq, false);
                const lr_t = try scalarLike(T, arena_alloc, self.dev, lr);
                const eps_t = try scalarLike(T, arena_alloc, self.dev, eps);

                const rms = try sqrtTensor(T, arena_alloc, self.dev, sq_t);
                const denom = try functions.add(T, arena_alloc, rms, eps_t);
                const raw_step = try functions.div(T, arena_alloc, g, denom);

                var graph = Graph(T).init(arena_alloc);
                graph.dag(raw_step.creator.?);
                graph.forward();

                if (mom > 0) {
                    for (mb, raw_step.data.?) |*mb_val, step_val| {
                        mb_val.* = mom * mb_val.* + lr * step_val;
                    }
                    const mb_t = try arena_alloc.create(Tensor(T));
                    mb_t.* = try Tensor(T).fromData(self.dev, p.shape[0..p.ndim], mb, false);
                    const new_p = try functions.sub(T, arena_alloc, p, mb_t);

                    graph.dag(new_p.creator.?);
                    graph.forward();
                    @memcpy(p.data.?, new_p.data.?);
                } else {
                    const step_t = try functions.mul(T, arena_alloc, raw_step, lr_t);
                    const new_p = try functions.sub(T, arena_alloc, p, step_t);

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
