const std = @import("std");
const Device = @import("../device.zig").Device;
const Graph = @import("../graph.zig").Graph;
const Tensor = @import("../tensor.zig").Tensor;
const fusion = @import("fusion.zig");
const JIT = @import("jit.zig").JIT;

pub fn Scheduler(comptime T: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        dev: *Device,
        jit: ?*JIT(T),
        graph: Graph(T),

        pub fn init(gpa: std.mem.Allocator, dev: *Device, opts: struct {
            jit: ?*JIT(T) = null,
        }) Self {
            return .{
                .gpa = gpa,
                .dev = dev,
                .jit = opts.jit,
                .graph = Graph(T).init(gpa),
            };
        }

        pub fn deinit(self: *Self) void {
            self.graph.deinit();
        }

        pub fn setJit(self: *Self, j: ?*JIT(T)) void {
            self.jit = j;
        }

        pub fn forward(self: *Self, tensor: *Tensor(T)) void {
            self.graph.dag(tensor.creator.?);
            self.getRunnableGraph().forward();
        }

        pub fn backward(self: *Self, loss: *Tensor(T)) void {
            if (loss.grad) |g| {
                g.deinit(self.gpa);
                self.gpa.destroy(g);
                loss.grad = null;
            }

            self.graph.dag(loss.creator.?);
            const g = self.getRunnableGraph();
            g.forward();

            const grad = self.gpa.create(Tensor(T)) catch return;
            grad.* = Tensor(T).ones(self.dev, loss.shape[0..loss.ndim], false) catch return;
            loss.grad = grad;

            g.backward() catch |err| {
                std.log.err("Scheduler backward error: {any}", .{err});
            };
        }

        fn getRunnableGraph(self: *Self) *Graph(T) {
            if (self.jit) |_| {
                // Apply graph fusion in-place on the fresh eager graph
                _ = fusion.optimize(T, &self.graph, self.gpa) catch {};
            }
            return &self.graph;
        }
    };
}
