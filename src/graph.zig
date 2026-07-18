const std = @import("std");
const assert = std.debug.assert;

const Node = @import("node.zig").Node;

pub fn Graph(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        nodes: std.ArrayList(*Node(T)),

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .alloc = allocator, .nodes = .empty };
        }

        pub fn deinit(self: *Self) void {
            self.nodes.deinit(self.alloc);
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
            self.nodes.append(self.alloc, node) catch {};
        }

        pub fn dag(self: *Self, root: *Node(T)) void {
            self.nodes.clearRetainingCapacity();
            self.resetFlags(root);
            self.topologicalSort(root);
        }

        pub fn forward(self: *Self) void {
            for (self.nodes.items) |node| {
                _ = node.forward(self.alloc) catch {};
            }
        }

        pub fn backward(self: *Self) !void {
            var i = self.nodes.items.len;
            while (i > 0) {
                i -= 1;
                try self.nodes.items[i].backward();
            }
        }

        pub fn eql(self: *Self, other: *Self) bool {
            if (self.nodes.items.len != other.nodes.items.len) return false;
            for (self.nodes.items, other.nodes.items) |a, b| {
                if (@as(u8, @intFromEnum(std.meta.activeTag(a.op.op_type))) != @as(u8, @intFromEnum(std.meta.activeTag(b.op.op_type)))) return false;
                if (a.inputs.len != b.inputs.len) return false;
                if (a.output.ndim != b.output.ndim) return false;
                if (a.output.device != b.output.device) return false;
                for (a.output.shape[0..a.output.ndim], b.output.shape[0..b.output.ndim]) |sa, sb| {
                    if (sa != sb) return false;
                }
            }
            return true;
        }
    };
}

const Tensor = @import("tensor.zig").Tensor;
const Op = @import("op.zig").Op;
const getFunction = @import("op.zig").getFunction;
const testing = std.testing;

test "Graph init and deinit" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var graph = Graph(f32).init(alloc);
    defer graph.deinit();

    try testing.expect(graph.nodes.items.len == 0);
}

test "Graph dag — topological sort of a chain" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const func = getFunction(.{ .element_wise = .add }, 1, 0);
    const op = Op{ .op_type = .{ .element_wise = .add }, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = func };

    const t_a = try alloc.create(Tensor(f32));
    t_a.* = try Tensor(f32).zeros(alloc, &.{2}, false);
    const t_b = try alloc.create(Tensor(f32));
    t_b.* = try Tensor(f32).zeros(alloc, &.{2}, false);
    const t_c = try alloc.create(Tensor(f32));
    t_c.* = try Tensor(f32).zeros(alloc, &.{2}, false);

    const ins_a = try alloc.alloc(*Tensor(f32), 0);
    const ins_b = try alloc.alloc(*Tensor(f32), 1);
    ins_b[0] = t_a;
    const ins_c = try alloc.alloc(*Tensor(f32), 1);
    ins_c[0] = t_b;

    var node_a: Node(f32) = undefined;
    node_a.init(alloc, ins_a, t_a, op);
    var node_b: Node(f32) = undefined;
    node_b.init(alloc, ins_b, t_b, op);
    var node_c: Node(f32) = undefined;
    node_c.init(alloc, ins_c, t_c, op);

    var graph = Graph(f32).init(alloc);
    defer graph.deinit();

    graph.dag(&node_c);
    try testing.expect(graph.nodes.items.len == 3);
    try testing.expect(graph.nodes.items[0] == &node_a);
    try testing.expect(graph.nodes.items[1] == &node_b);
    try testing.expect(graph.nodes.items[2] == &node_c);
}

test "Graph eql — equal graphs" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const func = getFunction(.{ .element_wise = .add }, 1, 0);
    const op = Op{ .op_type = .{ .element_wise = .add }, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = func };

    const t1 = try alloc.create(Tensor(f32));
    t1.* = try Tensor(f32).zeros(alloc, &.{ 3, 3 }, false);
    const t2 = try alloc.create(Tensor(f32));
    t2.* = try Tensor(f32).zeros(alloc, &.{ 3, 3 }, false);

    const ins1 = try alloc.alloc(*Tensor(f32), 0);
    const ins2 = try alloc.alloc(*Tensor(f32), 0);

    var node1: Node(f32) = undefined;
    node1.init(alloc, ins1, t1, op);
    var node2: Node(f32) = undefined;
    node2.init(alloc, ins2, t2, op);

    var graph_a = Graph(f32).init(alloc);
    defer graph_a.deinit();
    graph_a.dag(&node1);

    var graph_b = Graph(f32).init(alloc);
    defer graph_b.deinit();
    graph_b.dag(&node2);

    try testing.expect(graph_a.eql(&graph_b));
}

test "Graph eql — unequal graphs" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const func = getFunction(.{ .element_wise = .add }, 1, 0);
    const op = Op{ .op_type = .{ .element_wise = .add }, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = func };

    const t1 = try alloc.create(Tensor(f32));
    t1.* = try Tensor(f32).zeros(alloc, &.{2}, false);
    const t2 = try alloc.create(Tensor(f32));
    t2.* = try Tensor(f32).zeros(alloc, &.{3}, false);

    const ins1 = try alloc.alloc(*Tensor(f32), 0);
    const ins2 = try alloc.alloc(*Tensor(f32), 0);

    var node1: Node(f32) = undefined;
    node1.init(alloc, ins1, t1, op);
    var node2: Node(f32) = undefined;
    node2.init(alloc, ins2, t2, op);

    var graph_a = Graph(f32).init(alloc);
    defer graph_a.deinit();
    graph_a.dag(&node1);

    var graph_b = Graph(f32).init(alloc);
    defer graph_b.deinit();
    graph_b.dag(&node2);

    try testing.expect(!graph_a.eql(&graph_b));
}
