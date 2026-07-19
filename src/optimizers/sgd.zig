const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Graph = @import("../graph.zig").Graph;
const functions = @import("../ops/functions.zig");

fn scalarLike(comptime T: type, arena: std.mem.Allocator, val: T) !*Tensor(T) {
    const t = try arena.create(Tensor(T));
    t.* = try Tensor(T).fromData(arena, &.{1}, &.{val}, false);
    return t;
}

pub fn SGD(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        lr: T,
        momentum: T,
        nesterov: bool,
        params: std.ArrayList(*Tensor(T)),
        velocity: std.ArrayList([]T),
        temp_arena: std.heap.ArenaAllocator,

        pub fn init(alloc: std.mem.Allocator, opts: struct {
            lr: T = 0.01,
            momentum: T = 0.0,
            nesterov: bool = false,
        }) Self {
            return .{
                .alloc = alloc,
                .lr = opts.lr,
                .momentum = opts.momentum,
                .nesterov = opts.nesterov,
                .params = .empty,
                .velocity = .empty,
                .temp_arena = std.heap.ArenaAllocator.init(alloc),
            };
        }

        pub fn deinit(self: *Self) void {
            self.temp_arena.deinit();
            for (self.velocity.items) |v| self.alloc.free(v);
            self.velocity.deinit(self.alloc);
            self.params.deinit(self.alloc);
        }

        pub fn addParams(self: *Self, p: []*Tensor(T)) !void {
            try self.params.appendSlice(self.alloc, p);
            try self.velocity.ensureUnusedCapacity(self.alloc, p.len);
            for (p) |param| {
                const n = param.data.?.len;
                const v = try self.alloc.alloc(T, n);
                @memset(v, 0);
                try self.velocity.append(self.alloc, v);
            }
        }

        pub fn setLr(self: *Self, lr: T) void {
            self.lr = lr;
        }

        pub fn step(self: *Self) !void {
            const arena = self.temp_arena.allocator();

            for (self.params.items, self.velocity.items) |p, v| {
                const g = p.grad orelse continue;
                const gd = g.data.?;
                const mom = self.momentum;

                if (mom > 0) {
                    for (v, gd) |*vel, grad| {
                        vel.* = mom * vel.* + grad;
                    }
                    const vel_t = try arena.create(Tensor(T));
                    vel_t.* = try Tensor(T).fromData(arena, p.shape[0..p.ndim], v, false);
                    const lr_t = scalarLike(T, arena, self.lr) catch continue;
                    const step_t = functions.mul(T, vel_t, lr_t) catch continue;
                    const new_p = functions.sub(T, p, step_t) catch continue;
                    var graph = Graph(T).init(arena);
                    graph.dag(new_p.creator.?);
                    graph.forward();
                    @memcpy(p.data.?, new_p.data.?);
                } else {
                    const lr_t = scalarLike(T, arena, self.lr) catch continue;
                    const step_t = functions.mul(T, g, lr_t) catch continue;
                    const new_p = functions.sub(T, p, step_t) catch continue;
                    var graph = Graph(T).init(arena);
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
