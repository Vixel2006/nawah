const std = @import("std");
const Device = @import("../device.zig").Device;
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

fn tryMatchEndingAt(comptime T: type, graph: *Graph(T), node_idx: usize, gpa: std.mem.Allocator) ?Match(T) {
    const node = graph.nodes.items[node_idx];
    if (!isRelu(T, node) or node.inputs.len != 1) return null;

    const producer = node.inputs[0].creator orelse return null;

    if (isMatmul(T, producer)) {
        var nodes = gpa.alloc(*Node(T), 2) catch return null;
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

        var nodes = gpa.alloc(*Node(T), 3) catch return null;
        nodes[0] = add_lhs;
        nodes[1] = add;
        nodes[2] = node;
        return Match(T){ .pattern = .matmul_bias_relu, .nodes = nodes };
    }

    return null;
}

pub fn findPatterns(comptime T: type, graph: *Graph(T), gpa: std.mem.Allocator) !std.ArrayList(Match(T)) {
    var matches: std.ArrayList(Match(T)) = .empty;
    for (graph.nodes.items, 0..) |n, i| {
        if (isRelu(T, n)) {
            if (tryMatchEndingAt(T, graph, i, gpa)) |match| {
                try matches.append(gpa, match);
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

pub fn apply(comptime T: type, graph: *Graph(T), match: Match(T), gpa: std.mem.Allocator) !void {
    switch (match.pattern) {
        .matmul_relu => {
            const mm = match.nodes[0];
            const relu = match.nodes[1];

            const func = getFunction(.{ .fused = .matmul_relu }, 1, 0);
            const op = Op{ .op_type = .{ .fused = .matmul_relu }, .params = relu.op.params, .function = func };

            var fused = try gpa.create(Node(T));
            fused.init(gpa, mm.dev, mm.inputs, relu.output, op);

            var indices: std.ArrayList(usize) = .empty;
            defer indices.deinit(gpa);

            for (match.nodes) |mn| {
                if (nodeIndex(T, graph, mn)) |idx| try indices.append(gpa, idx);
            }
            if (indices.items.len != 2) return;

            std.mem.sort(usize, indices.items, {}, comptime std.sort.desc(usize));
            for (indices.items) |idx| {
                const old = graph.nodes.orderedRemove(idx);
                if (old != mm) {
                    gpa.free(old.inputs);
                }
                gpa.destroy(old);
            }

            var min: usize = indices.items[0];
            for (indices.items) |idx| {
                if (idx < min) min = idx;
            }
            try graph.nodes.insert(gpa, min, fused);
        },

        .matmul_bias_relu => {
            const mm = match.nodes[0];
            const add = match.nodes[1];
            const relu = match.nodes[2];

            const func = getFunction(.{ .fused = .matmul_relu }, 1, 0);
            const op = Op{ .op_type = .{ .fused = .matmul_relu }, .params = relu.op.params, .function = func };

            var fused_inputs = try gpa.alloc(*Tensor(T), 3);
            fused_inputs[0] = mm.inputs[0];
            fused_inputs[1] = mm.inputs[1];
            fused_inputs[2] = add.inputs[1];

            var fused = try gpa.create(Node(T));
            fused.init(gpa, mm.dev, fused_inputs, relu.output, op);

            var indices: std.ArrayList(usize) = .empty;
            defer indices.deinit(gpa);
            for (match.nodes) |mn| {
                if (nodeIndex(T, graph, mn)) |idx| try indices.append(gpa, idx);
            }
            if (indices.items.len != 3) return;

            std.mem.sort(usize, indices.items, {}, comptime std.sort.desc(usize));
            for (indices.items) |idx| {
                const old = graph.nodes.orderedRemove(idx);
                gpa.free(old.inputs);
                gpa.destroy(old);
            }

            var min: usize = indices.items[0];
            for (indices.items) |idx| {
                if (idx < min) min = idx;
            }
            try graph.nodes.insert(gpa, min, fused);
        },

        .none => {},
    }
}

pub fn optimize(comptime T: type, graph: *Graph(T), gpa: std.mem.Allocator) !u32 {
    var matches = try findPatterns(T, graph, gpa);
    defer {
        for (matches.items) |m| gpa.free(m.nodes);
        matches.deinit(gpa);
    }

    var applied: u32 = 0;
    for (matches.items) |m| {
        apply(T, graph, m, gpa) catch continue;
        applied += 1;
    }
    return applied;
}

const testing = std.testing;

test "fusion — find matmul+relu pattern" {
    const gpa = testing.allocator;
    var dev = try Device.init(.cpu);
    defer dev.deinit();

    const matmul_fn = getFunction(.{ .linear = .matmul }, 1, 0);
    const matmul_op = Op{ .op_type = .{ .linear = .matmul }, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = matmul_fn };

    const relu_fn = getFunction(.{ .element_wise = .relu }, 1, 0);
    const relu_op = Op{ .op_type = .{ .element_wise = .relu }, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = relu_fn };

    const t_a = try gpa.create(Tensor(f32));
    t_a.* = try Tensor(f32).zeros(&dev, &.{ 2, 2 }, false);
    defer {
        t_a.deinit(gpa);
        gpa.destroy(t_a);
    }
    const t_b = try gpa.create(Tensor(f32));
    t_b.* = try Tensor(f32).zeros(&dev, &.{ 2, 2 }, false);
    defer {
        t_b.deinit(gpa);
        gpa.destroy(t_b);
    }
    const t_mm = try gpa.create(Tensor(f32));
    t_mm.* = try Tensor(f32).zeros(&dev, &.{ 2, 2 }, false);
    defer {
        t_mm.deinit(gpa);
        gpa.destroy(t_mm);
    }
    const t_out = try gpa.create(Tensor(f32));
    t_out.* = try Tensor(f32).zeros(&dev, &.{ 2, 2 }, false);
    defer {
        t_out.deinit(gpa);
        gpa.destroy(t_out);
    }

    const mm_ins = try gpa.alloc(*Tensor(f32), 2);
    mm_ins[0] = t_a;
    mm_ins[1] = t_b;

    const mm_node = try gpa.create(Node(f32));
    mm_node.init(gpa, &dev, mm_ins, t_mm, matmul_op);
    defer {
        gpa.free(mm_ins);
        gpa.destroy(mm_node);
    }

    const relu_ins = try gpa.alloc(*Tensor(f32), 1);
    relu_ins[0] = t_mm;

    const relu_node = try gpa.create(Node(f32));
    relu_node.init(gpa, &dev, relu_ins, t_out, relu_op);
    defer {
        gpa.free(relu_ins);
        gpa.destroy(relu_node);
    }

    var graph = Graph(f32).init(gpa);
    defer graph.deinit();
    graph.dag(relu_node);

    try testing.expect(graph.nodes.items.len == 2);

    var matches = try findPatterns(f32, &graph, gpa);
    defer {
        for (matches.items) |m| gpa.free(m.nodes);
        matches.deinit(gpa);
    }

    try testing.expect(matches.items.len == 1);
    try testing.expect(matches.items[0].pattern == .matmul_relu);
}

test "fusion — apply matmul+relu fusion" {
    const gpa = testing.allocator;
    var dev = try Device.init(.cpu);
    defer dev.deinit();

    const matmul_fn = getFunction(.{ .linear = .matmul }, 1, 0);
    const matmul_op = Op{ .op_type = .{ .linear = .matmul }, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = matmul_fn };

    const relu_fn = getFunction(.{ .element_wise = .relu }, 1, 0);
    const relu_op = Op{ .op_type = .{ .element_wise = .relu }, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = relu_fn };

    const t_a = try gpa.create(Tensor(f32));
    t_a.* = try Tensor(f32).zeros(&dev, &.{ 2, 2 }, false);
    defer {
        t_a.deinit(gpa);
        gpa.destroy(t_a);
    }
    const t_b = try gpa.create(Tensor(f32));
    t_b.* = try Tensor(f32).zeros(&dev, &.{ 2, 2 }, false);
    defer {
        t_b.deinit(gpa);
        gpa.destroy(t_b);
    }
    const t_mm = try gpa.create(Tensor(f32));
    t_mm.* = try Tensor(f32).zeros(&dev, &.{ 2, 2 }, false);
    defer {
        t_mm.deinit(gpa);
        gpa.destroy(t_mm);
    }
    const t_out = try gpa.create(Tensor(f32));
    t_out.* = try Tensor(f32).zeros(&dev, &.{ 2, 2 }, false);
    defer {
        t_out.deinit(gpa);
        gpa.destroy(t_out);
    }

    const mm_ins = try gpa.alloc(*Tensor(f32), 2);
    mm_ins[0] = t_a;
    mm_ins[1] = t_b;

    const mm_node = try gpa.create(Node(f32));
    mm_node.init(gpa, &dev, mm_ins, t_mm, matmul_op);

    const relu_ins = try gpa.alloc(*Tensor(f32), 1);
    relu_ins[0] = t_mm;

    const relu_node = try gpa.create(Node(f32));
    relu_node.init(gpa, &dev, relu_ins, t_out, relu_op);

    var graph = Graph(f32).init(gpa);
    defer {
        for (graph.nodes.items) |n| {
            gpa.free(n.inputs);
            gpa.destroy(n);
        }
        graph.deinit();
    }
    graph.dag(relu_node);

    const applied = try optimize(f32, &graph, gpa);
    try testing.expect(applied == 1);
    try testing.expect(graph.nodes.items.len == 1);
}
