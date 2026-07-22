const std = @import("std");
const assert = std.debug.assert;
const Node = @import("../node.zig").Node;
const Graph = @import("../graph.zig").Graph;
const Tensor = @import("../tensor.zig").Tensor;
const fusion = @import("fusion.zig");
const Device = @import("../device.zig").Device;

pub fn JIT(comptime T: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        cache: std.AutoHashMap(u64, *Graph(T)),

        pub fn init(gpa: std.mem.Allocator) Self {
            return .{
                .gpa = gpa,
                .cache = std.AutoHashMap(u64, *Graph(T)).init(gpa),
            };
        }

        pub fn deinit(self: *Self) void {
            var it = self.cache.valueIterator();
            while (it.next()) |compiled| {
                compiled.*.deinit();
                self.gpa.destroy(compiled.*);
            }
            self.cache.deinit();
        }

        /// Compile the given graph. Hashes the structure and caches the result.
        pub fn compile(self: *Self, graph: *Graph(T)) !*Graph(T) {
            assert(graph.nodes.items.len > 0);

            var hasher = std.hash.Fnv1a_64.init();
            for (graph.nodes.items) |node| {
                const active_tag = std.meta.activeTag(node.op.op_type);
                hasher.update(std.mem.asBytes(&active_tag));

                switch (node.op.op_type) {
                    .element_wise => |ew| hasher.update(std.mem.asBytes(&ew)),
                    .reduce => |r| hasher.update(std.mem.asBytes(&r)),
                    .linear => |l| hasher.update(std.mem.asBytes(&l)),
                    .fused => |f| hasher.update(std.mem.asBytes(&f)),
                }

                const num_inputs: u32 = @intCast(node.inputs.len);
                hasher.update(std.mem.asBytes(&num_inputs));
                hasher.update(std.mem.asBytes(&node.op.params.dim));
                hasher.update(std.mem.asBytes(&node.op.params.keepdim));
                hasher.update(std.mem.asBytes(&node.op.params.fval));
                hasher.update(std.mem.asBytes(&node.output.ndim));
                hasher.update(std.mem.sliceAsBytes(node.output.shape[0..node.output.ndim]));
            }
            const fp = hasher.final();

            if (self.cache.get(fp)) |cached| {
                return cached;
            }

            // Optimize graph structure in-place
            _ = try fusion.optimize(T, graph, self.gpa);

            const captured_nodes = try self.gpa.alloc(*Node(T), graph.nodes.items.len);
            @memcpy(captured_nodes, graph.nodes.items);

            const compiled_ptr = try self.gpa.create(Graph(T));
            compiled_ptr.* = Graph(T){
                .gpa = self.gpa,
                .nodes = std.ArrayList(*Node(T)).fromOwnedSlice(captured_nodes),
            };

            std.log.debug("Compiled graph with {d} nodes (hash={X:0>16})", .{ captured_nodes.len, fp });
            try self.cache.put(fp, compiled_ptr);
            return compiled_ptr;
        }
    };
}

test "JIT — compile and run" {
    const gpa = std.testing.allocator;
    var dev = try Device.init(.cpu);
    defer dev.deinit();

    var a = try Tensor(f32).fromData(&dev, &.{3}, &[_]f32{ 1, 2, 3 }, true);
    defer a.deinit(gpa);

    var b = try Tensor(f32).fromData(&dev, &.{3}, &[_]f32{ 4, 5, 6 }, true);
    defer b.deinit(gpa);

    const functions = @import("../ops/functions.zig");
    const c = try functions.add(f32, gpa, &a, &b);
    defer {
        if (c.creator) |node| {
            node.deinit();
            gpa.destroy(node);
        }
    }
    const d = try functions.mul(f32, gpa, c, &a);
    defer {
        if (d.creator) |node| {
            node.deinit();
            gpa.destroy(node);
        }
    }

    var graph = Graph(f32).init(gpa);
    defer graph.deinit();
    graph.dag(d.creator.?);

    var compiler = JIT(f32).init(gpa);
    defer compiler.deinit();

    var compiled = try compiler.compile(&graph);
    compiled.forward();

    try std.testing.expect(d.data != null);
    if (d.data) |data| {
        try std.testing.expectEqual(@as(f32, 5), data[0]);
        try std.testing.expectEqual(@as(f32, 14), data[1]);
        try std.testing.expectEqual(@as(f32, 27), data[2]);
    }
}
