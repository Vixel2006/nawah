const std = @import("std");
const Node = @import("../node.zig").Node;
const Graph = @import("../graph.zig").Graph;
const Tensor = @import("../tensor.zig").Tensor;
const fusion = @import("fusion.zig");
const c_api = @import("../c_api.zig");

pub fn JIT(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        cache: std.AutoHashMap(u64, *Graph(T)),

        pub fn init(alloc: std.mem.Allocator) Self {
            return .{
                .alloc = alloc,
                .cache = std.AutoHashMap(u64, *Graph(T)).init(alloc),
            };
        }

        pub fn deinit(self: *Self) void {
            var it = self.cache.valueIterator();
            while (it.next()) |compiled| {
                compiled.*.deinit();
                self.alloc.destroy(compiled.*);
            }
            self.cache.deinit();
        }

        pub fn compile(self: *Self, graph: *Graph(T)) !*Graph(T) {
            // Hash the topology and structure of the graph
            var hasher = std.hash.Fnv1a_64.init();
            for (graph.nodes.items) |node| {
                const opTag: u32 = @intCast(@intFromEnum(std.meta.activeTag(node.op.op_type)));
                hasher.update(std.mem.asBytes(&opTag));
                switch (node.op.op_type) {
                    .element_wise => |ew| {
                        var inner: u32 = @intFromEnum(ew);
                        hasher.update(std.mem.asBytes(&inner));
                    },
                    .reduce => |r| {
                        var inner: u32 = @intFromEnum(r);
                        hasher.update(std.mem.asBytes(&inner));
                    },
                    .linear => |l| {
                        var inner: u32 = @intFromEnum(l);
                        hasher.update(std.mem.asBytes(&inner));
                    },
                    .fused => |f| {
                        var inner: u32 = @intFromEnum(f);
                        hasher.update(std.mem.asBytes(&inner));
                    },
                }
                var numInputs: u32 = @intCast(node.inputs.len);
                hasher.update(std.mem.asBytes(&numInputs));
                hasher.update(std.mem.asBytes(&node.op.params.dim));
                hasher.update(std.mem.asBytes(&node.op.params.keepdim));
                hasher.update(std.mem.asBytes(&node.op.params.fval));
                hasher.update(std.mem.asBytes(&node.output.ndim));
                for (node.output.shape[0..node.output.ndim]) |dim| {
                    hasher.update(std.mem.asBytes(&dim));
                }
            }
            const fp = hasher.final();

            if (self.cache.get(fp)) |cached| {
                return cached;
            }

            // Apply JIT optimizations (like fusion) in-place
            _ = try fusion.optimize(T, graph, self.alloc);

            // Capture the optimized nodes array
            const captured_nodes = try self.alloc.alloc(*Node(T), graph.nodes.items.len);
            @memcpy(captured_nodes, graph.nodes.items);
            const node_list = std.ArrayList(*Node(T)).fromOwnedSlice(captured_nodes);
            std.debug.print("Compiled graph has {d} nodes:\n", .{node_list.items.len});
            for (node_list.items, 0..) |node, i| {
                std.debug.print("  node {d}: op_type={any}\n", .{ i, node.op.op_type });
            }

            const compiled_ptr = try self.alloc.create(Graph(T));
            compiled_ptr.* = Graph(T){
                .alloc = self.alloc,
                .nodes = node_list,
            };
            try self.cache.put(fp, compiled_ptr);
            return compiled_ptr;
        }
    };
}

const testing = std.testing;

test "JIT — compile and run" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var a = try Tensor(f32).fromData(alloc, &.{3}, &[_]f32{ 1, 2, 3 }, true);
    var b = try Tensor(f32).fromData(alloc, &.{3}, &[_]f32{ 4, 5, 6 }, true);

    const functions = @import("../ops/functions.zig");
    const c = try functions.add(f32, alloc, &a, &b);
    const d = try functions.mul(f32, alloc, c, &a);

    var graph = Graph(f32).init(alloc);
    defer graph.deinit();
    graph.dag(d.creator.?);

    var compiler = JIT(f32).init(alloc);
    defer compiler.deinit();

    var compiled = try compiler.compile(&graph);
    compiled.forward();

    try testing.expect(d.data != null);
    if (d.data) |data| {
        try testing.expectEqual(@as(f32, 5), data[0]);
        try testing.expectEqual(@as(f32, 14), data[1]);
        try testing.expectEqual(@as(f32, 27), data[2]);
    }
}
