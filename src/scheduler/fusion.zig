const std = @import("std");
const Node = @import("../node.zig").Node;
const Graph = @import("../graph.zig").Graph;
const Tensor = @import("../tensor.zig").Tensor;
const Op = @import("../op.zig").Op;
const getFunction = @import("../op.zig").getFunction;

pub const Pattern = enum(u32) {
    none = 0,
    matmul_relu,
    matmul_bias_relu,
};

fn Match(comptime T: type) type {
    return struct {
        pattern: Pattern,
        nodes: []*Node(T),
    };
}

fn isRelu(comptime T: type, node: *Node(T)) bool {
    return switch (node.op.op_type) {
        .element_wise => |ew| ew == .relu,
        else => false,
    };
}

fn isMatmul(comptime T: type, node: *Node(T)) bool {
    return switch (node.op.op_type) {
        .linear => |l| l == .matmul,
        else => false,
    };
}

fn isAdd(comptime T: type, node: *Node(T)) bool {
    return switch (node.op.op_type) {
        .element_wise => |ew| ew == .add,
        else => false,
    };
}

fn tryMatchEndingAt(comptime T: type, graph: *Graph(T), node_idx: usize, alloc: std.mem.Allocator) ?Match(T) {
    const node = graph.nodes.items[node_idx];
    if (!isRelu(T, node) or node.inputs.len != 1) return null;

    const producer = node.inputs[0].creator orelse return null;

    if (isMatmul(T, producer)) {
        var nodes = alloc.alloc(*Node(T), 2) catch return null;
        nodes[0] = producer;
        nodes[1] = node;
        return Match(T){ .pattern = .matmul_relu, .nodes = nodes };
    }

    if (isAdd(T, producer)) {
        const add = producer;
        if (add.inputs.len < 2) return null;
        const add_lhs = add.inputs[0].creator orelse return null;
        if (!isMatmul(T, add_lhs)) return null;
        if (add.inputs[1].ndim != 1) return null;

        var nodes = alloc.alloc(*Node(T), 3) catch return null;
        nodes[0] = add_lhs;
        nodes[1] = add;
        nodes[2] = node;
        return Match(T){ .pattern = .matmul_bias_relu, .nodes = nodes };
    }

    return null;
}

pub fn findPatterns(comptime T: type, graph: *Graph(T), alloc: std.mem.Allocator) !std.ArrayList(Match(T)) {
    var matches: std.ArrayList(Match(T)) = .empty;
    for (graph.nodes.items, 0..) |n, i| {
        if (isRelu(T, n)) {
            if (tryMatchEndingAt(T, graph, i, alloc)) |match| {
                try matches.append(alloc, match);
            }
        }
    }
    return matches;
}

fn nodeIndex(comptime T: type, graph: *Graph(T), target: *Node(T)) ?usize {
    for (graph.nodes.items, 0..) |n, i| {
        if (n == target) return i;
    }
    return null;
}

pub fn apply(comptime T: type, graph: *Graph(T), match: Match(T), alloc: std.mem.Allocator) !void {
    switch (match.pattern) {
        .matmul_relu => {
            const mm = match.nodes[0];
            const relu = match.nodes[1];

            const func = getFunction(.{ .fused = .matmul_relu }, 1, 0);
            const op = Op{ .op_type = .{ .fused = .matmul_relu }, .params = relu.op.params, .function = func };

            var fused = try alloc.create(Node(T));
            fused.init(alloc, mm.inputs, relu.output, op);

            var indices: std.ArrayList(usize) = .empty;
            defer indices.deinit(alloc);

            for (match.nodes) |mn| {
                if (nodeIndex(T, graph, mn)) |idx| try indices.append(alloc, idx);
            }
            if (indices.items.len != 2) return;

            std.mem.sort(usize, indices.items, {}, comptime std.sort.desc(usize));
            for (indices.items) |idx| _ = graph.nodes.orderedRemove(idx);

            var min: usize = indices.items[0];
            for (indices.items) |idx| {
                if (idx < min) min = idx;
            }
            try graph.nodes.insert(alloc, min, fused);
        },

        .matmul_bias_relu => {
            const mm = match.nodes[0];
            const add = match.nodes[1];
            const relu = match.nodes[2];

            const func = getFunction(.{ .fused = .matmul_relu }, 1, 0);
            const op = Op{ .op_type = .{ .fused = .matmul_relu }, .params = relu.op.params, .function = func };

            var fused_inputs = try alloc.alloc(*Tensor(T), 3);
            fused_inputs[0] = mm.inputs[0];
            fused_inputs[1] = mm.inputs[1];
            fused_inputs[2] = add.inputs[1];

            var fused = try alloc.create(Node(T));
            fused.init(alloc, fused_inputs, relu.output, op);

            var indices: std.ArrayList(usize) = .empty;
            defer indices.deinit(alloc);
            for (match.nodes) |mn| {
                if (nodeIndex(T, graph, mn)) |idx| try indices.append(alloc, idx);
            }
            if (indices.items.len != 3) return;

            std.mem.sort(usize, indices.items, {}, comptime std.sort.desc(usize));
            for (indices.items) |idx| _ = graph.nodes.orderedRemove(idx);

            var min: usize = indices.items[0];
            for (indices.items) |idx| {
                if (idx < min) min = idx;
            }
            try graph.nodes.insert(alloc, min, fused);
        },

        .none => {},
    }
}

pub fn optimize(comptime T: type, graph: *Graph(T), alloc: std.mem.Allocator) !u32 {
    var matches = try findPatterns(T, graph, alloc);
    defer {
        for (matches.items) |m| alloc.free(m.nodes);
        matches.deinit(alloc);
    }

    var applied: u32 = 0;
    for (matches.items) |m| {
        apply(T, graph, m, alloc) catch continue;
        applied += 1;
    }
    return applied;
}

const testing = std.testing;

test "fusion — find matmul+relu pattern" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const matmul_fn = getFunction(.{ .linear = .matmul }, 1, 0);
    const matmul_op = Op{ .op_type = .{ .linear = .matmul }, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = matmul_fn };

    const relu_fn = getFunction(.{ .element_wise = .relu }, 1, 0);
    const relu_op = Op{ .op_type = .{ .element_wise = .relu }, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = relu_fn };

    const t_a = try alloc.create(Tensor(f32));
    t_a.* = try Tensor(f32).zeros(alloc, &.{ 2, 2 }, false);
    const t_b = try alloc.create(Tensor(f32));
    t_b.* = try Tensor(f32).zeros(alloc, &.{ 2, 2 }, false);
    const t_mm = try alloc.create(Tensor(f32));
    t_mm.* = try Tensor(f32).zeros(alloc, &.{ 2, 2 }, false);
    const t_out = try alloc.create(Tensor(f32));
    t_out.* = try Tensor(f32).zeros(alloc, &.{ 2, 2 }, false);

    const mm_ins = try alloc.alloc(*Tensor(f32), 2);
    mm_ins[0] = t_a;
    mm_ins[1] = t_b;

    var mm_node: Node(f32) = undefined;
    mm_node.init(alloc, mm_ins, t_mm, matmul_op);

    const relu_ins = try alloc.alloc(*Tensor(f32), 1);
    relu_ins[0] = t_mm;

    var relu_node: Node(f32) = undefined;
    relu_node.init(alloc, relu_ins, t_out, relu_op);

    var graph = Graph(f32).init(alloc);
    defer graph.deinit();
    graph.dag(&relu_node);

    try testing.expect(graph.nodes.items.len == 2);

    var matches = try findPatterns(f32, &graph, alloc);
    defer {
        for (matches.items) |m| alloc.free(m.nodes);
        matches.deinit(alloc);
    }

    try testing.expect(matches.items.len == 1);
    try testing.expect(matches.items[0].pattern == .matmul_relu);
}

test "fusion — apply matmul+relu fusion" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const matmul_fn = getFunction(.{ .linear = .matmul }, 1, 0);
    const matmul_op = Op{ .op_type = .{ .linear = .matmul }, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = matmul_fn };

    const relu_fn = getFunction(.{ .element_wise = .relu }, 1, 0);
    const relu_op = Op{ .op_type = .{ .element_wise = .relu }, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = relu_fn };

    const t_a = try alloc.create(Tensor(f32));
    t_a.* = try Tensor(f32).zeros(alloc, &.{ 2, 2 }, false);
    const t_b = try alloc.create(Tensor(f32));
    t_b.* = try Tensor(f32).zeros(alloc, &.{ 2, 2 }, false);
    const t_mm = try alloc.create(Tensor(f32));
    t_mm.* = try Tensor(f32).zeros(alloc, &.{ 2, 2 }, false);
    const t_out = try alloc.create(Tensor(f32));
    t_out.* = try Tensor(f32).zeros(alloc, &.{ 2, 2 }, false);

    const mm_ins = try alloc.alloc(*Tensor(f32), 2);
    mm_ins[0] = t_a;
    mm_ins[1] = t_b;

    var mm_node: Node(f32) = undefined;
    mm_node.init(alloc, mm_ins, t_mm, matmul_op);

    const relu_ins = try alloc.alloc(*Tensor(f32), 1);
    relu_ins[0] = t_mm;

    var relu_node: Node(f32) = undefined;
    relu_node.init(alloc, relu_ins, t_out, relu_op);

    var graph = Graph(f32).init(alloc);
    defer graph.deinit();
    graph.dag(&relu_node);

    const applied = try optimize(f32, &graph, alloc);
    try testing.expect(applied == 1);
    try testing.expect(graph.nodes.items.len == 1);
}
