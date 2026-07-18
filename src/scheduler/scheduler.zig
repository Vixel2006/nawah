const std = @import("std");
const Graph = @import("../graph.zig").Graph;
const fusion = @import("fusion.zig");
const Jit = @import("jit.zig").JIT;

pub const JIT = Jit;

pub fn Scheduler(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        jit: ?*JIT(T),
        jit_mode: bool,

        pub fn init(alloc: std.mem.Allocator, jit: ?*JIT(T)) Self {
            return .{ .alloc = alloc, .jit = jit, .jit_mode = false };
        }

        pub fn setJitMode(self: *Self, mode: bool) void {
            self.jit_mode = mode;
        }

        pub fn forward(self: *Self, graph: *Graph(T)) void {
            if (self.jit_mode) {
                _ = fusion.optimize(T, graph, self.alloc) catch {};
            }
            graph.forward();
        }

        pub fn backward(self: *Self, graph: *Graph(T)) void {
            if (self.jit_mode) {
                _ = fusion.optimize(T, graph, self.alloc) catch {};
            }
            graph.backward() catch |err| {
                std.debug.print("Error during backward: {any}\n", .{err});
            };
        }
    };
}
