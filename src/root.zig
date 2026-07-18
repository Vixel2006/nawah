const std = @import("std");

pub const tensor = @import("tensor.zig");
pub const node = @import("node.zig");
pub const op = @import("op.zig");
pub const graph = @import("graph.zig");
pub const c_api = @import("c_api.zig");
pub const functions = @import("ops/functions.zig");
pub const fusion = @import("scheduler/fusion.zig");
pub const scheduler = @import("scheduler/scheduler.zig");
pub const nn = @import("nn/mod.zig");

test {
    _ = @import("tensor.zig");
    _ = @import("op.zig");
    _ = @import("node.zig");
    _ = @import("graph.zig");
    _ = @import("ops/functions.zig");
    _ = @import("scheduler/fusion.zig");
    _ = @import("scheduler/scheduler.zig");
    _ = @import("scheduler/jit.zig");
    _ = @import("nn/mod.zig");
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const T = f32;
    const lr = 0.5;
    const epochs = 20000;
    const use_jit = true;

    var x = try tensor.Tensor(T).fromData(alloc, &.{ 4, 2 }, &[_]T{ 0, 0, 0, 1, 1, 0, 1, 1 }, false);
    var y = try tensor.Tensor(T).fromData(alloc, &.{ 4, 1 }, &[_]T{ 0, 1, 1, 0 }, false);

    var l1 = try nn.linear.Linear(T).init(alloc, 2, 8, .{ .seed = 42 });
    var sig1 = nn.activation.Sigmoid(T){};
    var l2 = try nn.linear.Linear(T).init(alloc, 8, 1, .{ .seed = 123 });
    var sig2 = nn.activation.Sigmoid(T){};

    var seq = nn.sequential.Sequential(T).init(alloc);
    defer seq.deinit();
    try seq.add(nn.linear.Linear(T), &l1);
    try seq.add(nn.activation.Sigmoid(T), &sig1);
    try seq.add(nn.linear.Linear(T), &l2);
    try seq.add(nn.activation.Sigmoid(T), &sig2);

    const params = try seq.parameters(alloc);
    defer alloc.free(params);

    var jit = scheduler.JIT(T).init(alloc);
    defer jit.deinit();
    var sched = scheduler.Scheduler(T).init(alloc, &jit);
    sched.setJitMode(use_jit);

    var epoch: u64 = 0;
    while (epoch < epochs) : (epoch += 1) {
        const pred = try seq.forward(&x, alloc);
        const diff = try functions.sub(T, alloc, pred, &y);
        const sq = try functions.mul(T, alloc, diff, diff);
        const loss = try functions.mean(T, alloc, sq, null, false);

        // Build the compute graph once, use for both forward and backward
        var compute = graph.Graph(T).init(alloc);
        defer compute.deinit();
        compute.dag(loss.creator.?);

        // Forward pass
        sched.forward(&compute);

        // Seed gradient (same as what Tensor.backward does internally)
        const grad = try alloc.create(tensor.Tensor(T));
        grad.* = try tensor.Tensor(T).ones(alloc, loss.shape[0..loss.ndim], false);
        loss.grad = grad;

        // Backward pass
        sched.backward(&compute);

        if (epoch < 10 or epoch % 5000 == 0) {
            std.debug.print("epoch {d}: loss = {d:.6}  (jit={})\n", .{ epoch, loss.data.?[0], use_jit });
        }

        if (epoch < 10 or epoch % 5000 == 0) {
            for (params, 0..) |p, i| {
                if (p.grad) |g| {
                    std.debug.print("  param {d}: val[0] = {d:.4}, grad[0] = {d:.4}\n", .{ i, p.data.?[0], g.data.?[0] });
                }
            }
        }

        for (params) |p| {
            if (p.grad) |g| {
                const pd = p.data.?;
                const gd = g.data.?;
                for (pd, gd) |*val, grad_val| {
                    val.* -= lr * grad_val;
                }
            }
        }

        for (params) |p| {
            if (p.grad) |g| {
                @memset(g.data.?, 0);
            }
        }
    }

    const pred = try seq.forward(&x, alloc);
    {
        var compute = graph.Graph(T).init(alloc);
        defer compute.deinit();
        compute.dag(pred.creator.?);
        sched.forward(&compute);
    }
    std.debug.print("Final predictions:\n", .{});
    for (0..4) |i| {
        const x0 = x.data.?[i * 2];
        const x1 = x.data.?[i * 2 + 1];
        std.debug.print("  {d} XOR {d} = {d:.4}  (expected {d})\n", .{ x0, x1, pred.data.?[i], y.data.?[i] });
    }
}
