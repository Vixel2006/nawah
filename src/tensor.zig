const std = @import("std");
const assert = std.debug.assert;

const MAX_NDIM: u16 = 8;

const Device = enum(u32) {
    CPU = 0,
    CUDA = 1,
};

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        ndim: u32,
        shape: []u32,
        strides: []u32,
        data: ?[]T = null,
        grad: ?*Tensor(T) = null,
        device: Device = .CPU,

        /// Initializing the tensor with its own metadata only.
        /// tensors will be realized lazily when we realize it.
        /// this way we don't initialize memory that we don't use.
        pub fn init(allocator: std.mem.Allocator, shape: []const u32, requires_grad: bool) !Self {
            const ndim: u32 = @intCast(shape.len);
            const shape_copy = try allocator.dupe(u32, shape);
            const strides: []u32 = try compute_strides(allocator, shape_copy);

            // TODO: Here we maybe need to have global arrays for tensors that requires_grad.
            // and tensors that are not. this way we can have a more efficient implementation.
            // as we will not have the data wasted on the requires_grad flag. if not then
            // just add the requires_grad flag to the tensor
            _ = requires_grad;

            return .{
                .alloc = allocator,
                .ndim = ndim,
                .shape = shape_copy,
                .strides = strides,
            };
        }

        pub fn zeros(allocator: std.mem.Allocator, shape: []const u32, requires_grad: bool) !Self {
            var t = try init(allocator, shape, requires_grad);
            const n = numel(shape);
            t.data = try allocator.alloc(T, n);
            @memset(t.data.?, 0);
            return t;
        }

        pub fn ones(allocator: std.mem.Allocator, shape: []const u32, requires_grad: bool) !Self {
            var t = try init(allocator, shape, requires_grad);
            const n = numel(shape);
            t.data = try allocator.alloc(T, n);
            @memset(t.data.?, 1);
            return t;
        }

        pub fn uniform(allocator: std.mem.Allocator, shape: []const u32, requires_grad: bool, rng: std.Random) !Self {
            var t = try init(allocator, shape, requires_grad);
            const n = numel(shape);
            const data = try allocator.alloc(T, n);
            for (data) |*v| v.* = rng.float(T);
            t.data = data;
            return t;
        }

        pub fn fromData(allocator: std.mem.Allocator, shape: []const u32, data: []const T, requires_grad: bool) !Self {
            var t = try init(allocator, shape, requires_grad);
            t.data = try allocator.dupe(T, data);
            return t;
        }

        pub fn deinit(self: *Self) void {
            if (self.data) |data| self.alloc.free(data);
            if (self.grad) |grad| grad.deinit();
        }

        fn is_contiguous(self: *const Self) bool {
            if (self.ndim == 0) return true;
            var expected_stride: u32 = 1;
            var i: u32 = self.ndim - 1;
            while (true) {
                if (self.strides[i] != expected_stride) return false;
                if (i == 0) break;
                expected_stride *= self.shape[i];
                i -= 1;
            }
            return true;
        }
    };
}

fn compute_strides(allocator: std.mem.Allocator, shape: []const u32) ![]u32 {
    const ndim = shape.len;
    const strides = try allocator.alloc(u32, ndim);
    if (ndim == 0) return strides;
    strides[ndim - 1] = 1;
    var i: u32 = @intCast(ndim - 1);
    while (i > 0) {
        i -= 1;
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

fn numel(shape: []const u32) u32 {
    var num_elements: u32 = 1;
    for (shape) |dim| {
        num_elements *= dim;
    }
    return num_elements;
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var t = try Tensor(f32).ones(arena.allocator(), &.{ 4, 4 }, false);
    defer t.deinit();

    std.debug.print("{any}", .{t.data});
}

test "initializing a tensor" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var t = try Tensor(f32).init(arena.allocator(), &.{ 1, 2, 3 }, true);
    defer t.deinit();

    try std.testing.expectEqual(3, t.ndim);
    try std.testing.expect(std.mem.eql(u32, t.shape, &.{ 1, 2, 3 }));
    try std.testing.expect(std.mem.eql(u32, t.strides, &.{ 6, 3, 1 }));
}

test "zeros initializer" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var t = try Tensor(f32).zeros(alloc, &.{ 2, 3 }, false);
    defer t.deinit();

    try std.testing.expectEqual(2, t.ndim);
    try std.testing.expect(std.mem.eql(u32, t.shape, &.{ 2, 3 }));
    try std.testing.expect(t.data != null);
    for (t.data.?[0..6]) |v| try std.testing.expectEqual(@as(f32, 0), v);
}
test "ones initializer" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var t = try Tensor(f32).ones(arena.allocator(), &.{ 2, 3 }, false);
    defer t.deinit();

    try std.testing.expectEqual(2, t.ndim);
    for (t.data.?[0..6]) |v| try std.testing.expectEqual(@as(f32, 1), v);
}
test "uniform initializer" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var t = try Tensor(f32).uniform(arena.allocator(), &.{ 4, 4 }, false, rng);
    defer t.deinit();

    try std.testing.expectEqual(16, t.shape[0] * t.shape[1]);
    // verify values are in [0, 1)
    for (t.data.?[0..16]) |v| {
        try std.testing.expect(v >= 0 and v < 1);
    }
}
test "fromData initializer" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var t = try Tensor(f32).fromData(arena.allocator(), &.{ 2, 3 }, &data, false);
    defer t.deinit();

    try std.testing.expectEqual(2, t.ndim);
    try std.testing.expect(std.mem.eql(f32, t.data.?[0..6], &data));
}
