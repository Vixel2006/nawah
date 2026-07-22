const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Graph = @import("../graph.zig").Graph;
const functions = @import("../ops/functions.zig");
const Device = @import("../device.zig").Device;

fn scalarLike(comptime T: type, gpa: std.mem.Allocator, dev: *Device, val: T) !*Tensor(T) {
    const t = try gpa.create(Tensor(T));
    t.* = try Tensor(T).fromData(dev, &.{1}, &.{val}, false);
    return t;
}

fn sqrtTensor(comptime T: type, gpa: std.mem.Allocator, dev: *Device, x: *Tensor(T)) !*Tensor(T) {
    const half = try scalarLike(T, gpa, dev, 0.5);
    return functions.exp(T, gpa, try functions.mul(T, gpa, try functions.log(T, gpa, x), half));
}

pub fn Adam(comptime T: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        dev: *Device,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        params: std.ArrayList(*Tensor(T)),
        m_bufs: std.ArrayList([]T),
        v_bufs: std.ArrayList([]T),
        t: u64,
        temp_arena: std.heap.ArenaAllocator,

        pub fn init(gpa: std.mem.Allocator, dev: *Device, opts: struct {
            lr: T = 0.001,
            beta1: T = 0.9,
            beta2: T = 0.999,
            eps: T = 1e-8,
        }) Self {
            return .{
                .gpa = gpa,
                .dev = dev,
                .lr = opts.lr,
                .beta1 = opts.beta1,
                .beta2 = opts.beta2,
                .eps = opts.eps,
                .params = .empty,
                .m_bufs = .empty,
                .v_bufs = .empty,
                .t = 0,
                .temp_arena = std.heap.ArenaAllocator.init(gpa),
            };
        }

        pub fn deinit(self: *Self) void {
            const dev_alloc = self.dev.allocator();
            for (self.m_bufs.items) |m| dev_alloc.free(@as([*]u8, @ptrCast(m.ptr)), m.len * @sizeOf(T), @alignOf(T));
            for (self.v_bufs.items) |v| dev_alloc.free(@as([*]u8, @ptrCast(v.ptr)), v.len * @sizeOf(T), @alignOf(T));
            self.m_bufs.deinit(self.gpa);
            self.v_bufs.deinit(self.gpa);
            self.params.deinit(self.gpa);
            self.temp_arena.deinit();
        }

        pub fn addParams(self: *Self, p: []*Tensor(T)) !void {
            try self.params.appendSlice(self.gpa, p);
            try self.m_bufs.ensureUnusedCapacity(self.gpa, p.len);
            try self.v_bufs.ensureUnusedCapacity(self.gpa, p.len);
            const dev_alloc = self.dev.allocator();
            for (p) |param| {
                const n = param.data.?.len;
                const m_mem = dev_alloc.alloc(n * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory;
                const m = @as([*]T, @ptrCast(@alignCast(m_mem)))[0..n];
                const v_mem = dev_alloc.alloc(n * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory;
                const v = @as([*]T, @ptrCast(@alignCast(v_mem)))[0..n];
                @memset(m, 0);
                @memset(v, 0);
                try self.m_bufs.append(self.gpa, m);
                try self.v_bufs.append(self.gpa, v);
            }
        }

        pub fn step(self: *Self) !void {
            self.t += 1;
            const arena_alloc = self.temp_arena.allocator();
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

                const m_t = try arena_alloc.create(Tensor(T));
                m_t.* = try Tensor(T).fromData(self.dev, p.shape[0..p.ndim], m, false);
                const v_t = try arena_alloc.create(Tensor(T));
                v_t.* = try Tensor(T).fromData(self.dev, p.shape[0..p.ndim], v, false);

                const bc1_t = scalarLike(T, arena_alloc, self.dev, bc1) catch continue;
                const bc2_t = scalarLike(T, arena_alloc, self.dev, bc2) catch continue;
                const eps_t = scalarLike(T, arena_alloc, self.dev, eps) catch continue;
                const lr_t = scalarLike(T, arena_alloc, self.dev, lr) catch continue;

                const m_hat = functions.div(T, arena_alloc, m_t, bc1_t) catch continue;
                const v_hat = functions.div(T, arena_alloc, v_t, bc2_t) catch continue;
                const sqrt_v = sqrtTensor(T, arena_alloc, self.dev, v_hat) catch continue;
                const denom = functions.add(T, arena_alloc, sqrt_v, eps_t) catch continue;
                const correction = functions.div(T, arena_alloc, m_hat, denom) catch continue;
                const step_t = functions.mul(T, arena_alloc, correction, lr_t) catch continue;
                const new_p = functions.sub(T, arena_alloc, p, step_t) catch continue;

                var graph = Graph(T).init(arena_alloc);
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
