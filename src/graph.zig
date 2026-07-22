const std = @import("std");
const assert = std.debug.assert;
const Device = @import("device.zig").Device;
const Node = @import("node.zig").Node;

pub fn Graph(comptime T: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        nodes: std.ArrayList(*Node(T)),

        pub fn init(gpa: std.mem.Allocator) Self {
            return .{ .gpa = gpa, .nodes = .empty };
        }

        pub fn deinit(self: *Self) void {
            self.nodes.deinit(self.gpa);
        }

        /// Construct the execution DAG from a root node.
        pub fn dag(self: *Self, root: *Node(T)) void {
            self.nodes.clearRetainingCapacity();
            self.resetFlags(root);
            self.topologicalSort(root);
        }

        pub fn forward(self: *Self) void {
            for (self.nodes.items) |node| {
                _ = node.forward() catch {};
            }
        }

        pub fn backward(self: *Self) !void {
            var i = self.nodes.items.len;
            while (i > 0) {
                i -= 1;
                try self.nodes.items[i].backward();
            }
        }

        /// Check if two graphs are structurally equivalent (same ops, shapes, etc.).
        pub fn eql(self: *Self, other: *Self) bool {
            if (self.nodes.items.len != other.nodes.items.len) return false;
            for (self.nodes.items, other.nodes.items) |a, b| {
                const tag_a = std.meta.activeTag(a.op.op_type);
                const tag_b = std.meta.activeTag(b.op.op_type);
                if (tag_a != tag_b) return false;
                if (a.inputs.len != b.inputs.len) return false;
                if (a.output.ndim != b.output.ndim) return false;
                if (a.output.device != b.output.device) return false;

                const ndim = a.output.ndim;
                if (!std.mem.eql(u64, a.output.shape[0..ndim], b.output.shape[0..ndim])) return false;
            }
            return true;
        }

        fn resetFlags(self: *Self, node: *Node(T)) void {
            if (!node.visited) return;
            node.visited = false;
            for (node.inputs) |input| {
                if (input.creator) |creator| {
                    self.resetFlags(creator);
                }
            }
        }

        fn topologicalSort(self: *Self, node: *Node(T)) void {
            node.visited = true;
            for (node.inputs) |input| {
                if (input.creator) |creator| {
                    if (!creator.visited) {
                        self.topologicalSort(creator);
                    }
                }
            }
            self.nodes.append(self.gpa, node) catch {};
        }
    };
}

const Tensor = @import("tensor.zig").Tensor;
const Op = @import("op.zig").Op;
const getFunction = @import("op.zig").getFunction;
const testing = std.testing;

test "Graph init and deinit" {
    const gpa = testing.allocator;
    var graph = Graph(f32).init(gpa);
    defer graph.deinit();
    try testing.expectEqual(@as(usize, 0), graph.nodes.items.len);
}

test "Graph dag — topological sort of a chain" {
    const gpa = testing.allocator;
    var dev = try Device.init(.cpu);
    defer dev.deinit();

    const func = getFunction(.{ .element_wise = .add }, 1, 0);
    const op = Op{ .op_type = .{ .element_wise = .add }, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = func };

    const t_a = try gpa.create(Tensor(f32));
    t_a.* = try Tensor(f32).zeros(&dev, &.{2}, false);
    defer {
        t_a.deinit(gpa);
        gpa.destroy(t_a);
    }

    const t_b = try gpa.create(Tensor(f32));
    t_b.* = try Tensor(f32).zeros(&dev, &.{2}, false);
    defer {
        t_b.deinit(gpa);
        gpa.destroy(t_b);
    }

    const t_c = try gpa.create(Tensor(f32));
    t_c.* = try Tensor(f32).zeros(&dev, &.{2}, false);
    defer {
        t_c.deinit(gpa);
        gpa.destroy(t_c);
    }

    const ins_a = try gpa.alloc(*Tensor(f32), 0);
    const ins_b = try gpa.alloc(*Tensor(f32), 1);
    ins_b[0] = t_a;
    const ins_c = try gpa.alloc(*Tensor(f32), 1);
    ins_c[0] = t_b;

    var node_a: Node(f32) = undefined;
    node_a.init(gpa, &dev, ins_a, t_a, op);
    defer gpa.free(ins_a);

    var node_b: Node(f32) = undefined;
    node_b.init(gpa, &dev, ins_b, t_b, op);
    defer gpa.free(ins_b);

    var node_c: Node(f32) = undefined;
    node_c.init(gpa, &dev, ins_c, t_c, op);
    defer gpa.free(ins_c);

    var graph = Graph(f32).init(gpa);
    defer graph.deinit();

    graph.dag(&node_c);
    try testing.expectEqual(@as(usize, 3), graph.nodes.items.len);
    try testing.expectEqual(&node_a, graph.nodes.items[0]);
    try testing.expectEqual(&node_b, graph.nodes.items[1]);
    try testing.expectEqual(&node_c, graph.nodes.items[2]);
}

test "Graph eql — equal graphs" {
    const gpa = testing.allocator;
    var dev = try Device.init(.cpu);
    defer dev.deinit();

    const func = getFunction(.{ .element_wise = .add }, 1, 0);
    const op = Op{ .op_type = .{ .element_wise = .add }, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = func };

    const t1 = try gpa.create(Tensor(f32));
    t1.* = try Tensor(f32).zeros(&dev, &.{ 3, 3 }, false);
    defer {
        t1.deinit(gpa);
        gpa.destroy(t1);
    }

    const t2 = try gpa.create(Tensor(f32));
    t2.* = try Tensor(f32).zeros(&dev, &.{ 3, 3 }, false);
    defer {
        t2.deinit(gpa);
        gpa.destroy(t2);
    }

    const ins1 = try gpa.alloc(*Tensor(f32), 0);
    const ins2 = try gpa.alloc(*Tensor(f32), 0);

    var node1: Node(f32) = undefined;
    node1.init(gpa, &dev, ins1, t1, op);
    defer gpa.free(ins1);

    var node2: Node(f32) = undefined;
    node2.init(gpa, &dev, ins2, t2, op);
    defer gpa.free(ins2);

    var graph_a = Graph(f32).init(gpa);
    defer graph_a.deinit();
    graph_a.dag(&node1);

    var graph_b = Graph(f32).init(gpa);
    defer graph_b.deinit();
    graph_b.dag(&node2);

    try testing.expect(graph_a.eql(&graph_b));
}

test "Graph eql — unequal graphs" {
    const gpa = testing.allocator;
    var dev = try Device.init(.cpu);
    defer dev.deinit();

    const func = getFunction(.{ .element_wise = .add }, 1, 0);
    const op = Op{ .op_type = .{ .element_wise = .add }, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = func };

    const t1 = try gpa.create(Tensor(f32));
    t1.* = try Tensor(f32).zeros(&dev, &.{2}, false);
    defer {
        t1.deinit(gpa);
        gpa.destroy(t1);
    }

    const t2 = try gpa.create(Tensor(f32));
    t2.* = try Tensor(f32).zeros(&dev, &.{3}, false);
    defer {
        t2.deinit(gpa);
        gpa.destroy(t2);
    }

    const ins1 = try gpa.alloc(*Tensor(f32), 0);
    const ins2 = try gpa.alloc(*Tensor(f32), 0);

    var node1: Node(f32) = undefined;
    node1.init(gpa, &dev, ins1, t1, op);
    defer gpa.free(ins1);

    var node2: Node(f32) = undefined;
    node2.init(gpa, &dev, ins2, t2, op);
    defer gpa.free(ins2);

    var graph_a = Graph(f32).init(gpa);
    defer graph_a.deinit();
    graph_a.dag(&node1);

    var graph_b = Graph(f32).init(gpa);
    defer graph_b.deinit();
    graph_b.dag(&node2);

    try testing.expect(!graph_a.eql(&graph_b));
}
