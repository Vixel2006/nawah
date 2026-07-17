const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const assert = std.debug.assert;

pub fn Op(comptime T: type) type {
    return struct {
        const Self = @This();

        ctx: *anyopaque,
        vtable: *const VTable,

        pub const VTable = struct {
            forward: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, inputs: []const *Tensor(T)) anyerror!*Tensor(T),
            backward: *const fn (ctx: *anyopaque, grad_output: *Tensor(T)) void,
        };

        pub fn init(ctx: anytype, comptime impl: type) Self {
            const Ptr = @TypeOf(ctx);
            comptime { assert(@typeInfo(Ptr) == .pointer); }
            const vtable: *const VTable = &VTable{
                .forward = struct {
                    fn f(ptr: *anyopaque, a: std.mem.Allocator, ins: []const *Tensor(T)) anyerror!*Tensor(T) {
                        return impl.forward(@as(Ptr, @ptrCast(@alignCast(ptr))), a, ins);
                    }
                }.f,
                .backward = struct {
                    fn f(ptr: *anyopaque, g: *Tensor(T)) void {
                        impl.backward(@as(Ptr, @ptrCast(@alignCast(ptr))), g);
                    }
                }.f,
            };
            return .{ .ctx = @ptrCast(ctx), .vtable = vtable };
        }

        pub fn forward(self: Self, allocator: std.mem.Allocator, inputs: []const *Tensor(T)) !*Tensor(T) {
            return self.vtable.forward(self.ctx, allocator, inputs);
        }

        pub fn backward(self: Self, grad_output: *Tensor(T)) void {
            self.vtable.backward(self.ctx, grad_output);
        }
    };
}

const testing = std.testing;

test "Op vtable dispatch — forward and backward" {
    const AddImpl = struct {
        pub fn forward(_: *@This(), allocator: std.mem.Allocator, inputs: []const *Tensor(f32)) !*Tensor(f32) {
            const a = inputs[0].data.?;
            const b = inputs[1].data.?;
            const result = try Tensor(f32).zeros(allocator, inputs[0].shape[0..inputs[0].ndim], false);
            for (result.data.?, a, b) |*r, x, y| r.* = x + y;
            const ptr = try allocator.create(Tensor(f32));
            ptr.* = result;
            return ptr;
        }
        pub fn backward(_: *@This(), grad_output: *Tensor(f32)) void {
            _ = grad_output;
        }
    };

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var a = try Tensor(f32).fromData(alloc, &.{ 3 }, &[_]f32{ 1, 2, 3 }, false);
    var b = try Tensor(f32).fromData(alloc, &.{ 3 }, &[_]f32{ 4, 5, 6 }, false);

    var impl = AddImpl{};
    const op = Op(f32).init(&impl, AddImpl);

    const out = try op.forward(alloc, &.{ &a, &b });

    try testing.expect(std.mem.eql(f32, out.data.?, &[_]f32{ 5, 7, 9 }));

    op.backward(out);
}

test "Op vtable dispatch — stateless op with no context" {
    const MulByTwoImpl = struct {
        pub fn forward(_: *@This(), allocator: std.mem.Allocator, inputs: []const *Tensor(f32)) !*Tensor(f32) {
            const src = inputs[0].data.?;
            const result = try Tensor(f32).zeros(allocator, inputs[0].shape[0..inputs[0].ndim], false);
            for (result.data.?, src) |*r, x| r.* = x * 2;
            const ptr = try allocator.create(Tensor(f32));
            ptr.* = result;
            return ptr;
        }
        pub fn backward(_: *@This(), grad_output: *Tensor(f32)) void {
            _ = grad_output;
        }
    };

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var t = try Tensor(f32).fromData(alloc, &.{ 2, 2 }, &[_]f32{ 1, 2, 3, 4 }, false);

    var impl = MulByTwoImpl{};
    const op = Op(f32).init(&impl, MulByTwoImpl);

    const out = try op.forward(alloc, &.{ &t });

    try testing.expect(std.mem.eql(f32, out.data.?, &[_]f32{ 2, 4, 6, 8 }));
}
