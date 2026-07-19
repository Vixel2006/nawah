const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const functions = @import("../ops/functions.zig");
const builtin = @import("builtin");

fn defaultSeed() u64 {
    if (builtin.os.tag == .linux) {
        const Timespec = extern struct { sec: i64, nsec: i64 };
        var ts: Timespec = undefined;
        if (std.os.linux.clock_gettime(std.os.linux.CLOCK.REALTIME, @ptrCast(&ts)) == 0) {
            return @as(u64, @intCast(ts.sec)) ^ @as(u64, @intCast(ts.nsec));
        }
    }
    return 0;
}

pub fn Linear(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        weight: *Tensor(T),
        bias: ?*Tensor(T),
        in_features: u64,
        out_features: u64,

        pub fn init(alloc: std.mem.Allocator, in_features: u64, out_features: u64, opts: struct {
            bias: bool = true,
            seed: ?u64 = null,
        }) !Self {
            var rng = if (opts.seed) |s| std.Random.DefaultPrng.init(s) else std.Random.DefaultPrng.init(defaultSeed());
            const bound = std.math.sqrt(6.0 / @as(T, @floatFromInt(in_features)));

            const weight = try alloc.create(Tensor(T));
            weight.* = try Tensor(T).zeros(alloc, &.{ in_features, out_features }, true);
            for (weight.data.?) |*v| {
                v.* = rng.random().float(T) * 2 * bound - bound;
            }

            const bias: ?*Tensor(T) = if (opts.bias) blk: {
                const b = try alloc.create(Tensor(T));
                b.* = try Tensor(T).zeros(alloc, &.{out_features}, true);
                break :blk b;
            } else null;

            return .{
                .alloc = alloc,
                .weight = weight,
                .bias = bias,
                .in_features = in_features,
                .out_features = out_features,
            };
        }

        pub fn deinit(self: *Self) void {
            self.weight.deinit();
            self.alloc.destroy(self.weight);
            if (self.bias) |b| {
                b.deinit();
                self.alloc.destroy(b);
            }
        }

        pub fn call(self: *Self, x: *Tensor(T)) !*Tensor(T) {
            var out = try functions.matmul(T, x, self.weight);
            if (self.bias) |b| {
                out = try functions.add(T, out, b);
            }
            return out;
        }

        pub fn parameters(self: *Self, allocator: std.mem.Allocator) ![]*Tensor(T) {
            if (self.bias) |b| {
                var params = try allocator.alloc(*Tensor(T), 2);
                params[0] = self.weight;
                params[1] = b;
                return params;
            }
            var params = try allocator.alloc(*Tensor(T), 1);
            params[0] = self.weight;
            return params;
        }
    };
}

const testing = std.testing;

test "Linear — init and forward" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const rng_seed: u64 = 42;
    var linear = try Linear(f32).init(alloc, 4, 3, .{ .seed = rng_seed });
    defer linear.deinit();

    try testing.expect(linear.in_features == 4);
    try testing.expect(linear.out_features == 3);
    try testing.expect(linear.weight.shape[0] == 4);
    try testing.expect(linear.weight.shape[1] == 3);
    try testing.expect(linear.bias != null);
    try testing.expect(linear.bias.?.shape[0] == 3);

    var x = try Tensor(f32).ones(alloc, &.{2, 4}, false);
    const out = try linear.call(&x);
    try testing.expect(out.shape[0] == 2);
    try testing.expect(out.shape[1] == 3);
}

test "Linear — parameters" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var linear_with_bias = try Linear(f32).init(alloc, 4, 3, .{});
    defer linear_with_bias.deinit();
    {
        const params = try linear_with_bias.parameters(alloc);
        defer alloc.free(params);
        try testing.expect(params.len == 2);
    }

    var linear_no_bias = try Linear(f32).init(alloc, 4, 3, .{ .bias = false });
    defer linear_no_bias.deinit();
    {
        const params = try linear_no_bias.parameters(alloc);
        defer alloc.free(params);
        try testing.expect(params.len == 1);
    }
}
