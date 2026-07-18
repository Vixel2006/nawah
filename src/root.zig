const std = @import("std");

pub const tensor = @import("tensor.zig");
pub const node = @import("node.zig");
pub const op = @import("op.zig");
pub const graph = @import("graph.zig");
pub const c_api = @import("c_api.zig");
pub const functions = @import("ops/functions.zig");

test {
    _ = @import("tensor.zig");
    _ = @import("op.zig");
    _ = @import("node.zig");
    _ = @import("graph.zig");
    _ = @import("ops/functions.zig");
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // Create input tensors
    var a = try tensor.Tensor(f32).fromData(alloc, &.{3}, &[_]f32{ 1, 2, 3 }, true);
    var b = try tensor.Tensor(f32).fromData(alloc, &.{3}, &[_]f32{ 4, 5, 6 }, true);

    // Lazy ops — no computation yet
    const c = try functions.add(f32, alloc, &a, &b);
    const d = try functions.mul(f32, alloc, c, &a);
    const e = try functions.relu(f32, alloc, d);

    // Realize — builds DAG and runs forward pass
    e.realize();

    std.debug.print("a:  {any}\n", .{a.data.?});
    std.debug.print("b:  {any}\n", .{b.data.?});
    std.debug.print("c = a + b:  {any}\n", .{c.data.?});
    std.debug.print("d = c * a:  {any}\n", .{d.data.?});
    std.debug.print("e = relu(d): {any}\n", .{e.data.?});

    // Backward pass
    e.backward();
    std.debug.print("backward done\n", .{});
    if (a.grad) |g| {
        std.debug.print("a.grad: {any}\n", .{g.data.?});
    }
    if (b.grad) |g| {
        std.debug.print("b.grad: {any}\n", .{g.data.?});
    }
}
