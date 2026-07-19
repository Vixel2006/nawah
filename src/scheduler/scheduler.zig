const std = @import("std");
const Graph = @import("../graph.zig").Graph;
const Tensor = @import("../tensor.zig").Tensor;
const fusion = @import("fusion.zig");
const Jit = @import("jit.zig").JIT;

pub const JIT = Jit;

pub fn Scheduler(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        jit: ?*JIT(T),
        jit_mode: bool,
        graph: Graph(T),

        pub fn init(alloc: std.mem.Allocator, opts: struct {
            jit: ?*JIT(T) = null,
        }) Self {
            return .{
                .alloc = alloc,
                .jit = opts.jit,
                .jit_mode = false,
                .graph = Graph(T).init(alloc),
            };
        }

        pub fn deinit(self: *Self) void {
            self.graph.deinit();
        }

        pub fn setJit(self: *Self, j: ?*JIT(T)) void {
            self.jit = j;
        }

        pub fn setJitMode(self: *Self, mode: bool) void {
            self.jit_mode = mode;
        }

        fn optimize(self: *Self) void {
            if (self.jit_mode) {
                _ = fusion.optimize(T, &self.graph, self.alloc) catch {};
            }
        }

        pub fn forward(self: *Self, tensor: *Tensor(T)) void {
            self.graph.dag(tensor.creator.?);
            self.optimize();
            self.graph.forward();
        }

        pub fn backward(self: *Self, loss: *Tensor(T)) void {
            if (loss.grad) |g| {
                g.deinit();
                self.alloc.destroy(g);
                loss.grad = null;
            }

            self.graph.dag(loss.creator.?);
            self.optimize();
            self.graph.forward();

            const grad = self.alloc.create(Tensor(T)) catch return;
            grad.* = Tensor(T).ones(self.alloc, loss.shape[0..loss.ndim], false) catch return;
            loss.grad = grad;

            self.graph.backward() catch |err| {
                std.debug.print("Error during backward: {any}\n", .{err});
            };
        }
    };
}
