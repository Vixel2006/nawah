const std = @import("std");

pub const tensor = @import("tensor.zig");
pub const node = @import("node.zig");
pub const op = @import("op.zig");

test {
    _ = @import("tensor.zig");
    _ = @import("op.zig");
    _ = @import("node.zig");
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var t = try tensor.Tensor(f32).ones(arena.allocator(), &.{ 4, 4 }, false);
    defer t.deinit();

    std.debug.print("{any}", .{t.data});
}
