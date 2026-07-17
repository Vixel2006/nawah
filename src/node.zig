const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Op = @import("op.zig").Op;
const assert = std.debug.assert;

pub fn Node(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        inputs: []*Tensor(T),
        output: *Tensor(T),
        grad: ?*Tensor(T) = null,
        op: Op(T),

        pub fn init(alloc: std.mem.Allocator, inputs: []*Tensor(T), output: *Tensor(T), op: Op(T)) Self {
            return .{ .alloc = alloc, .inputs = inputs, .output = output, .op = op };
        }

        pub fn deinit(self: *Self) void {
            self.alloc.free(self.inputs);
            self.output.deinit();
            self.alloc.destroy(self.output);
            if (self.grad) |g| {
                g.deinit();
                self.alloc.destroy(g);
            }
        }

        pub fn forward(self: *Self, allocator: std.mem.Allocator) !*Tensor(T) {
            self.output = try self.op.forward(allocator, self.inputs);
            return self.output;
        }

        pub fn backward(self: *Self) void {
            if (self.grad) |g| self.op.backward(g);
        }
    };
}

const testing = std.testing;

test "Node init and forward" {
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

    var a = try Tensor(f32).fromData(alloc, &.{ 4 }, &[_]f32{ 1, 2, 3, 4 }, false);
    var b = try Tensor(f32).fromData(alloc, &.{ 4 }, &[_]f32{ 5, 6, 7, 8 }, false);

    const dummy = try alloc.create(Tensor(f32));
    dummy.* = try Tensor(f32).zeros(alloc, &.{ 4 }, false);

    var impl = AddImpl{};
    const op = Op(f32).init(&impl, AddImpl);

    const inputs = try alloc.alloc(*Tensor(f32), 2);
    inputs[0] = &a;
    inputs[1] = &b;

    var node = Node(f32).init(alloc, inputs, dummy, op);
    const out = try node.forward(alloc);

    try testing.expect(out == node.output);
    try testing.expect(std.mem.eql(f32, out.data.?, &[_]f32{ 6, 8, 10, 12 }));

    node.deinit();
}

test "Node backward triggers op backward" {
    const BackwardCheck = struct {
        called: bool,

        pub fn forward(_: *@This(), allocator: std.mem.Allocator, inputs: []const *Tensor(f32)) !*Tensor(f32) {
            const result = try Tensor(f32).zeros(allocator, inputs[0].shape[0..inputs[0].ndim], false);
            const ptr = try allocator.create(Tensor(f32));
            ptr.* = result;
            return ptr;
        }
        pub fn backward(self: *@This(), grad_output: *Tensor(f32)) void {
            _ = grad_output;
            self.called = true;
        }
    };

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const t = try alloc.create(Tensor(f32));
    t.* = try Tensor(f32).zeros(alloc, &.{ 2 }, false);
    const dummy = try alloc.create(Tensor(f32));
    dummy.* = try Tensor(f32).zeros(alloc, &.{ 2 }, false);
    const grad = try alloc.create(Tensor(f32));
    grad.* = try Tensor(f32).zeros(alloc, &.{ 2 }, false);

    var check = BackwardCheck{ .called = false };
    const op = Op(f32).init(&check, BackwardCheck);

    const inputs = try alloc.alloc(*Tensor(f32), 1);
    inputs[0] = t;

    var node = Node(f32).init(alloc, inputs, dummy, op);
    _ = try node.forward(alloc);

    node.grad = grad;
    node.backward();

    try testing.expect(check.called);

    node.deinit();
}
