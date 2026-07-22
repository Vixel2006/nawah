const std = @import("std");
const assert = std.debug.assert;
const PlastAllocator = @import("allocator.zig").PlastAllocator;
const Node = @import("node.zig").Node;
const Graph = @import("graph.zig").Graph;
const Device = @import("device.zig").Device;

pub const MAX_NDIM = 8;

pub const DeviceKind = enum(u32) {
    CPU = 0,
    CUDA = 1,
};

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        dev: *Device,
        ndim: u64,
        shape: [MAX_NDIM]u64,
        strides: [MAX_NDIM]u64,
        data: ?[]T = null,
        grad: ?*Self = null,
        creator: ?*Node(T) = null,
        device: DeviceKind = .CPU,
        requires_grad: bool = false,

        pub fn init(dev: *Device, shape: []const u64, requires_grad: bool) !Self {
            assert(shape.len > 0);
            assert(shape.len <= MAX_NDIM);

            const ndim: u64 = @intCast(shape.len);
            var shape_arr: [MAX_NDIM]u64 = undefined;
            for (shape, 0..) |s, i| {
                assert(s > 0);
                shape_arr[i] = s;
            }
            const strides = computeStrides(&shape_arr, ndim);
            return .{
                .dev = dev,
                .ndim = ndim,
                .shape = shape_arr,
                .strides = strides,
                .device = @enumFromInt(dev.kind()),
                .requires_grad = requires_grad,
            };
        }

        pub fn zeros(dev: *Device, shape: []const u64, requires_grad: bool) !Self {
            var t = try init(dev, shape, requires_grad);
            const n = numel(&t.shape, t.ndim);
            const dev_alloc = dev.allocator();
            const mem = dev_alloc.alloc(n * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory;
            t.data = @as([*]T, @ptrCast(@alignCast(mem)))[0..n];
            @memset(t.data.?, 0);
            return t;
        }

        pub fn ones(dev: *Device, shape: []const u64, requires_grad: bool) !Self {
            var t = try init(dev, shape, requires_grad);
            const n = numel(&t.shape, t.ndim);
            const dev_alloc = dev.allocator();
            const mem = dev_alloc.alloc(n * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory;
            t.data = @as([*]T, @ptrCast(@alignCast(mem)))[0..n];
            @memset(t.data.?, 1);
            return t;
        }

        pub fn fromData(dev: *Device, shape: []const u64, data: []const T, requires_grad: bool) !Self {
            var t = try init(dev, shape, requires_grad);
            const n = numel(&t.shape, t.ndim);
            assert(data.len == n);
            const dev_alloc = dev.allocator();
            const mem = dev_alloc.alloc(data.len * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory;
            const dup = @as([*]T, @ptrCast(@alignCast(mem)))[0..data.len];
            @memcpy(dup, data);
            t.data = dup;
            return t;
        }

        pub fn uniform(dev: *Device, shape: []const u64, requires_grad: bool, rng: std.Random) !Self {
            var t = try init(dev, shape, requires_grad);
            const n = numel(&t.shape, t.ndim);
            const dev_alloc = dev.allocator();
            const mem = dev_alloc.alloc(n * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory;
            const buf = @as([*]T, @ptrCast(@alignCast(mem)))[0..n];
            for (buf) |*v| v.* = rng.float(T);
            t.data = buf;
            return t;
        }

        pub fn kaimingUniform(dev: *Device, shape: []const u64, requires_grad: bool, rng: std.Random) !Self {
            var t = try init(dev, shape, requires_grad);
            const n = numel(&t.shape, t.ndim);
            const bound = std.math.sqrt(6.0 / @as(T, @floatFromInt(shape[0])));
            const dev_alloc = dev.allocator();
            const mem = dev_alloc.alloc(n * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory;
            const buf = @as([*]T, @ptrCast(@alignCast(mem)))[0..n];
            for (buf) |*v| {
                v.* = rng.float(T) * 2 * bound - bound;
            }
            t.data = buf;
            return t;
        }

        pub fn deinit(self: *Self, gpa: ?std.mem.Allocator) void {
            if (self.data) |d| {
                const dev_alloc = self.dev.allocator();
                if (d.len > 0) dev_alloc.free(@as([*]u8, @ptrCast(d.ptr)), d.len * @sizeOf(T), @alignOf(T));
                self.data = null;
            }
            if (self.grad) |g| {
                g.deinit(gpa);
                if (gpa) |allocator| {
                    allocator.destroy(g);
                }
                self.grad = null;
            }
        }

        pub fn realize(self: *Self, gpa: std.mem.Allocator) void {
            const creator = self.creator orelse return;
            var graph = Graph(T).init(gpa);
            defer graph.deinit();
            graph.dag(creator);
            graph.forward();
        }

        pub fn backward(self: *Self, gpa: std.mem.Allocator) void {
            const creator = self.creator orelse return;
            self.realize(gpa);
            const grad = gpa.create(Tensor(T)) catch return;
            grad.* = Tensor(T).ones(self.dev, self.shape[0..self.ndim], false) catch return;
            self.grad = grad;
            var graph = Graph(T).init(gpa);
            defer graph.deinit();
            graph.dag(creator);
            graph.backward() catch |err| {
                std.log.err("Error during backward: {any}", .{err});
            };
        }

        pub fn asSlice(self: *const Self) []const T {
            assert(self.data != null);
            return self.data.?[0..numel(&self.shape, self.ndim)];
        }

        pub fn isContiguous(self: *const Self) bool {
            if (self.ndim == 0) return true;
            var expected: u64 = 1;
            var i = self.ndim - 1;
            while (true) {
                if (self.strides[i] != expected) return false;
                if (i == 0) break;
                expected *= self.shape[i];
                i -= 1;
            }
            return true;
        }
    };
}

fn computeStrides(shape: *const [MAX_NDIM]u64, ndim: u64) [MAX_NDIM]u64 {
    var strides: [MAX_NDIM]u64 = undefined;
    if (ndim == 0) return strides;
    strides[ndim - 1] = 1;
    var i = ndim - 1;
    while (i > 0) {
        i -= 1;
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

fn numel(shape: *const [MAX_NDIM]u64, ndim: u64) u64 {
    var n: u64 = 1;
    for (shape[0..ndim]) |dim| n *= dim;
    return n;
}
