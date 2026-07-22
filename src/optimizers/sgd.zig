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

pub fn SGD(comptime T: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        dev: *Device,
        lr: T,
        momentum: T,
        nesterov: bool,
        params: std.ArrayList(*Tensor(T)),
        velocity: std.ArrayList([]T),
        temp_arena: std.heap.ArenaAllocator,

        pub fn init(gpa: std.mem.Allocator, dev: *Device, opts: struct {
            lr: T = 0.01,
            momentum: T = 0.0,
            nesterov: bool = false,
        }) Self {
            return .{
                .gpa = gpa,
                .dev = dev,
                .lr = opts.lr,
                .momentum = opts.momentum,
                .nesterov = opts.nesterov,
                .params = .empty,
                .velocity = .empty,
                .temp_arena = std.heap.ArenaAllocator.init(gpa),
            };
        }

        pub fn deinit(self: *Self) void {
            const dev_alloc = self.dev.allocator();
            for (self.velocity.items) |v| {
                dev_alloc.free(@as([*]u8, @ptrCast(v.ptr)), v.len * @sizeOf(T), @alignOf(T));
            }
            self.velocity.deinit(self.gpa);
            self.params.deinit(self.gpa);
            self.temp_arena.deinit();
        }

        pub fn addParams(self: *Self, p: []*Tensor(T)) !void {
            try self.params.appendSlice(self.gpa, p);
            try self.velocity.ensureUnusedCapacity(self.gpa, p.len);
            const dev_alloc = self.dev.allocator();
            for (p) |param| {
                const n = param.data.?.len;
                const v_mem = dev_alloc.alloc(n * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory;
                const v = @as([*]T, @ptrCast(@alignCast(v_mem)))[0..n];
                @memset(v, 0);
                try self.velocity.append(self.gpa, v);
            }
        }

        pub fn setLr(self: *Self, lr: T) void {
            self.lr = lr;
        }

        pub fn step(self: *Self) !void {
            const arena_alloc = self.temp_arena.allocator();
            const mom = self.momentum;
            const lr = self.lr;

            for (self.params.items, self.velocity.items) |p, v| {
                const g = p.grad orelse continue;
                const gd = g.data.?;

                if (mom > 0) {
                    for (v, gd) |*vel, grad| {
                        vel.* = mom * vel.* + grad;
                    }
                    const vel_t = try arena_alloc.create(Tensor(T));
                    vel_t.* = try Tensor(T).fromData(self.dev, p.shape[0..p.ndim], v, false);
                    const lr_t = try scalarLike(T, arena_alloc, self.dev, lr);
                    const step_t = try functions.mul(T, arena_alloc, vel_t, lr_t);
                    const new_p = try functions.sub(T, arena_alloc, p, step_t);
                    var graph = Graph(T).init(arena_alloc);
                    graph.dag(new_p.creator.?);
                    graph.forward();
                    @memcpy(p.data.?, new_p.data.?);
                } else {
                    const lr_t = try scalarLike(T, arena_alloc, self.dev, lr);
                    const step_t = try functions.mul(T, arena_alloc, g, lr_t);
                    const new_p = try functions.sub(T, arena_alloc, p, step_t);
                    var graph = Graph(T).init(arena_alloc);
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
