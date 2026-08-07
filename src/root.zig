const std = @import("std");

pub const tensor = @import("tensor.zig");
pub const uop = @import("uop.zig");
pub const op = @import("op.zig");
pub const graph = @import("graph.zig");
pub const c_api = @import("c_api.zig");
pub const functions = @import("ops/functions.zig");
pub const alc = @import("allocator.zig");
pub const device = @import("device.zig");
pub const Device = device.Device;

test {
    _ = @import("tensor.zig");
    _ = @import("op.zig");
    _ = @import("uop.zig");
    _ = @import("graph.zig");
    _ = @import("ops/functions.zig");
    _ = @import("device.zig");
    _ = @import("optimizer/mod.zig");
}

pub fn main() !void {
    var dev = try Device.init(.cuda);
    defer dev.deinit();

    const alloc = std.heap.page_allocator;

    const T = f32;

    var x = try tensor.Tensor(T).fromData(&dev, &.{ 4, 2 }, &[_]T{ 0, 0, 0, 1, 1, 0, 1, 1 }, true);
    var y = try tensor.Tensor(T).fromData(&dev, &.{ 4, 2 }, &[_]T{ 1, 1, 1, 0, 0, 1, 0, 0 }, true);

    const z = try functions.add(T, alloc, &x, &y);
    var a = try functions.mul(T, alloc, &x, z);
    try a.backward(alloc);

    std.debug.print("x: {any}\n", .{tensor.slice(f32, x.dev, x.data.?, x.shape, x.ndim)});
    std.debug.print("y: {any}\n", .{tensor.slice(f32, y.dev, y.data.?, y.shape, y.ndim)});
    std.debug.print("z: {any}\n", .{tensor.slice(f32, z.dev, z.data.?, z.shape, z.ndim)});
    std.debug.print("a: {any}\n", .{tensor.slice(f32, a.dev, a.data.?, a.shape, a.ndim)});

    std.debug.print("a grad: {any}\n", .{tensor.slice(f32, a.dev, a.grad.?, a.shape, a.ndim)});
    std.debug.print("z grad: {any}\n", .{tensor.slice(f32, z.dev, z.grad.?, z.shape, z.ndim)});
    std.debug.print("x grad: {any}\n", .{tensor.slice(f32, x.dev, x.grad.?, x.shape, x.ndim)});
    std.debug.print("y grad: {any}\n", .{tensor.slice(f32, y.dev, y.grad.?, y.shape, y.ndim)});
}
