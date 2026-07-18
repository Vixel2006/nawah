const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;

pub fn Sequential(comptime T: type) type {
    return struct {
        const Self = @This();

        const VTable = struct {
            ctx: *anyopaque,
            forwardFn: *const fn (ctx: *anyopaque, x: *Tensor(T), alloc: std.mem.Allocator) anyerror!*Tensor(T),
            paramsFn: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]*Tensor(T),
            deinitFn: *const fn (ctx: *anyopaque) void,
        };

        alloc: std.mem.Allocator,
        layers: std.ArrayList(VTable),

        pub fn init(alloc: std.mem.Allocator) Self {
            return .{ .alloc = alloc, .layers = .empty };
        }

        pub fn deinit(self: *Self) void {
            for (self.layers.items) |layer| {
                layer.deinitFn(layer.ctx);
            }
            self.layers.deinit(self.alloc);
        }

        pub fn add(self: *Self, comptime LayerType: type, layer: *LayerType) !void {
            const wrapper = struct {
                fn forward(ctx: *anyopaque, x: *Tensor(T), alloc: std.mem.Allocator) anyerror!*Tensor(T) {
                    return @as(*LayerType, @ptrCast(@alignCast(ctx))).forward(x, alloc);
                }
                fn params(ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]*Tensor(T) {
                    return @as(*LayerType, @ptrCast(@alignCast(ctx))).parameters(allocator);
                }
                fn deinit(ctx: *anyopaque) void {
                    @as(*LayerType, @ptrCast(@alignCast(ctx))).deinit();
                }
            };
            try self.layers.append(self.alloc, .{
                .ctx = @ptrCast(layer),
                .forwardFn = wrapper.forward,
                .paramsFn = wrapper.params,
                .deinitFn = wrapper.deinit,
            });
        }

        pub fn forward(self: *Self, x: *Tensor(T), allocator: std.mem.Allocator) !*Tensor(T) {
            var out = x;
            for (self.layers.items) |layer| {
                out = try layer.forwardFn(layer.ctx, out, allocator);
            }
            return out;
        }

        pub fn parameters(self: *Self, allocator: std.mem.Allocator) ![]*Tensor(T) {
            var list: std.ArrayList(*Tensor(T)) = .empty;
            defer list.deinit(allocator);
            for (self.layers.items) |layer| {
                const p = try layer.paramsFn(layer.ctx, allocator);
                for (p) |param| {
                    try list.append(allocator, param);
                }
                allocator.free(p);
            }
            return list.toOwnedSlice(allocator);
        }
    };
}

const Linear = @import("linear.zig").Linear;
const testing = std.testing;

test "Sequential — single linear layer" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var linear = try Linear(f32).init(alloc, 4, 3, .{ .seed = 42 });

    var seq = Sequential(f32).init(alloc);
    defer seq.deinit();

    try seq.add(Linear(f32), &linear);

    var x = try Tensor(f32).ones(alloc, &.{2, 4}, false);
    const out = try seq.forward(&x, alloc);
    try testing.expect(out.shape[0] == 2);
    try testing.expect(out.shape[1] == 3);
}

test "Sequential — parameters" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var l1 = try Linear(f32).init(alloc, 4, 3, .{ .seed = 1 });
    var l2 = try Linear(f32).init(alloc, 3, 2, .{ .seed = 2 });

    var seq = Sequential(f32).init(alloc);
    defer seq.deinit();
    try seq.add(Linear(f32), &l1);
    try seq.add(Linear(f32), &l2);

    const params = try seq.parameters(alloc);
    defer alloc.free(params);
    try testing.expect(params.len == 4);
}
