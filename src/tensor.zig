const std = @import("std");
const assert = std.debug.assert;

pub const MAX_NDIM = 8;
pub const Device = enum(u32) {
    CPU = 0,
    CUDA = 1,
};

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        ndim: u64,
        shape: [MAX_NDIM]u64,
        strides: [MAX_NDIM]u64,
        data: ?[]T = null,
        grad: ?*Self = null,
        device: Device = .CPU,
        requires_grad: bool = false,

        pub fn init(allocator: std.mem.Allocator, shape: []const u64, requires_grad: bool) !Self {
            const ndim: u64 = @intCast(shape.len);
            var shape_arr: [MAX_NDIM]u64 = undefined;
            for (shape, 0..) |s, i| shape_arr[i] = s;
            const strides = compute_strides(&shape_arr, ndim);
            return .{
                .alloc = allocator,
                .ndim = ndim,
                .shape = shape_arr,
                .strides = strides,
                .requires_grad = requires_grad,
            };
        }

        pub fn zeros(allocator: std.mem.Allocator, shape: []const u64, requires_grad: bool) !Self {
            var t = try init(allocator, shape, requires_grad);
            const n = numel(&t.shape, t.ndim);
            t.data = try allocator.alloc(T, n);
            @memset(t.data.?, 0);
            return t;
        }

        pub fn ones(allocator: std.mem.Allocator, shape: []const u64, requires_grad: bool) !Self {
            var t = try init(allocator, shape, requires_grad);
            const n = numel(&t.shape, t.ndim);
            t.data = try allocator.alloc(T, n);
            @memset(t.data.?, 1);
            return t;
        }

        pub fn fromData(allocator: std.mem.Allocator, shape: []const u64, data: []const T, requires_grad: bool) !Self {
            var t = try init(allocator, shape, requires_grad);
            const n = numel(&t.shape, t.ndim);
            assert(data.len == n);
            t.data = try allocator.dupe(T, data);
            return t;
        }

        pub fn uniform(allocator: std.mem.Allocator, shape: []const u64, requires_grad: bool, rng: std.Random) !Self {
            var t = try init(allocator, shape, requires_grad);
            const n = numel(&t.shape, t.ndim);
            const buf = try allocator.alloc(T, n);
            for (buf) |*v| v.* = rng.float(T);
            t.data = buf;
            return t;
        }

        pub fn deinit(self: *Self) void {
            if (self.data) |d| self.alloc.free(d);
            if (self.grad) |g| {
                g.deinit();
                self.alloc.destroy(g);
            }
        }

        pub fn asSlice(self: *const Self) []const T {
            return self.data.?[0..numel(&self.shape, self.ndim)];
        }

        fn isContiguous(self: *const Self) bool {
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

fn compute_strides(shape: *const [MAX_NDIM]u64, ndim: u64) [MAX_NDIM]u64 {
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
