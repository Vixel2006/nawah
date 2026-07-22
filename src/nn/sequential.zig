const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Device = @import("../device.zig").Device;

pub fn Sequential(comptime T: type) type {
    return struct {
        const Self = @This();

        const VTable = struct {
            ctx: *anyopaque,
            callFn: *const fn (ctx: *anyopaque, gpa: std.mem.Allocator, x: *Tensor(T)) anyerror!*Tensor(T),
            paramsFn: *const fn (ctx: *anyopaque, gpa: std.mem.Allocator) anyerror![]*Tensor(T),
            deinitFn: *const fn (ctx: *anyopaque) void,
        };

        gpa: std.mem.Allocator,
        dev: *Device,
        layers: std.ArrayList(VTable),

        pub fn init(gpa: std.mem.Allocator, dev: *Device) Self {
            return .{ .gpa = gpa, .dev = dev, .layers = std.ArrayList(VTable).init(gpa) };
        }

        pub fn deinit(self: *Self) void {
            for (self.layers.items) |layer| {
                layer.deinitFn(layer.ctx);
            }
            self.layers.deinit();
        }

        pub fn add(self: *Self, comptime LayerType: type, layer: *LayerType) !void {
            const wrapper = struct {
                fn call(ctx: *anyopaque, gpa: std.mem.Allocator, x: *Tensor(T)) anyerror!*Tensor(T) {
                    return @as(*LayerType, @ptrCast(@alignCast(ctx))).call(gpa, x);
                }
                fn params(ctx: *anyopaque, gpa: std.mem.Allocator) anyerror![]*Tensor(T) {
                    return @as(*LayerType, @ptrCast(@alignCast(ctx))).parameters(gpa);
                }
                fn deinit(ctx: *anyopaque) void {
                    @as(*LayerType, @ptrCast(@alignCast(ctx))).deinit();
                }
            };
            try self.layers.append(.{
                .ctx = @ptrCast(layer),
                .callFn = wrapper.call,
                .paramsFn = wrapper.params,
                .deinitFn = wrapper.deinit,
            });
        }

        pub fn call(self: *Self, gpa: std.mem.Allocator, x: *Tensor(T)) anyerror!*Tensor(T) {
            var out = x;
            for (self.layers.items) |layer| {
                out = try layer.callFn(layer.ctx, gpa, out);
            }
            return out;
        }

        pub fn parameters(self: *Self, gpa: std.mem.Allocator) ![]*Tensor(T) {
            var list = std.ArrayList(*Tensor(T)).init(gpa);
            defer list.deinit();
            for (self.layers.items) |layer| {
                const p = try layer.paramsFn(layer.ctx, gpa);
                try list.appendSlice(p);
                if (p.len > 0) gpa.free(p);
            }
            return list.toOwnedSlice();
        }
    };
}

// ── Tests ────────────────────────────────────────────────────────────────────

const Linear = @import("linear.zig").Linear;
const testing = std.testing;

test "Sequential — single linear layer" {
    const gpa = testing.allocator;
    var dev = try Device.init(.cpu);
    defer dev.deinit();

    var linear = try Linear(f32).init(gpa, &dev, 4, 3, .{ .seed = 42 });

    var seq = Sequential(f32).init(gpa, &dev);
    defer seq.deinit();

    try seq.add(Linear(f32), &linear);

    var x = try Tensor(f32).ones(&dev, &.{ 2, 4 }, false);
    defer x.deinit(gpa);
    const out = try seq.call(gpa, &x);
    try testing.expect(out.shape[0] == 2);
    try testing.expect(out.shape[1] == 3);
}

test "Sequential — parameters" {
    const gpa = testing.allocator;
    var dev = try Device.init(.cpu);
    defer dev.deinit();

    var l1 = try Linear(f32).init(gpa, &dev, 4, 3, .{ .seed = 1 });
    var l2 = try Linear(f32).init(gpa, &dev, 3, 2, .{ .seed = 2 });

    var seq = Sequential(f32).init(gpa, &dev);
    defer seq.deinit();
    try seq.add(Linear(f32), &l1);
    try seq.add(Linear(f32), &l2);

    const params = try seq.parameters(gpa);
    defer gpa.free(params);
    try testing.expect(params.len == 4);
}
