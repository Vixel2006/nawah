const std = @import("std");

pub const tensor = @import("tensor.zig");
pub const node = @import("node.zig");
pub const op = @import("op.zig");
pub const graph = @import("graph.zig");
pub const c_api = @import("c_api.zig");
pub const functions = @import("ops/functions.zig");
pub const fusion = @import("scheduler/fusion.zig");
pub const scheduler = @import("scheduler/scheduler.zig");
pub const optimizer = @import("optimizer/mod.zig");
pub const optimizers = @import("optimizers/mod.zig");
pub const nn = @import("nn/mod.zig");
pub const cudaAlloc = @import("cuda_alloc.zig");

test {
    _ = @import("tensor.zig");
    _ = @import("op.zig");
    _ = @import("node.zig");
    _ = @import("graph.zig");
    _ = @import("ops/functions.zig");
    _ = @import("scheduler/fusion.zig");
    _ = @import("scheduler/scheduler.zig");
    _ = @import("scheduler/jit.zig");
    _ = @import("optimizer/mod.zig");
    _ = @import("optimizers/mod.zig");
    _ = @import("nn/mod.zig");
    _ = @import("cuda_alloc.zig");
}

fn Net(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        l1: nn.linear.Linear(T),
        sig1: nn.activation.Sigmoid(T),
        l2: nn.linear.Linear(T),
        sig2: nn.activation.Sigmoid(T),

        pub fn init(alloc: std.mem.Allocator) !Self {
            return .{
                .alloc = alloc,
                .l1 = try nn.linear.Linear(T).init(alloc, 2, 8, .{ .seed = 42 }),
                .sig1 = .{},
                .l2 = try nn.linear.Linear(T).init(alloc, 8, 1, .{ .seed = 123 }),
                .sig2 = .{},
            };
        }

        pub fn deinit(self: *Self) void {
            self.l2.deinit();
            self.l1.deinit();
        }

        pub fn call(self: *Self, x: *tensor.Tensor(T)) !*tensor.Tensor(T) {
            var out = try self.l1.call(x);
            out = try self.sig1.call(out);
            out = try self.l2.call(out);
            out = try self.sig2.call(out);
            return out;
        }

        pub fn parameters(self: *Self, allocator: std.mem.Allocator) ![]*tensor.Tensor(T) {
            const l1p = try self.l1.parameters(allocator);
            defer allocator.free(l1p);
            const l2p = try self.l2.parameters(allocator);
            defer allocator.free(l2p);

            var params = try allocator.alloc(*tensor.Tensor(T), l1p.len + l2p.len);
            @memcpy(params[0..l1p.len], l1p);
            @memcpy(params[l1p.len..], l2p);
            return params;
        }
    };
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

    var net = try Net(T).init(alloc);
    defer net.deinit();

    const params = try net.parameters(alloc);
    defer alloc.free(params);

    var jit = scheduler.JIT(T).init(alloc);
    defer jit.deinit();
    var sched = scheduler.Scheduler(T).init(alloc, .{ .jit = &jit });
    sched.setJitMode(use_jit);

    var optim = optimizers.Adam(T).init(alloc, .{ .lr = lr });
    defer optim.deinit();
    try optim.addParams(params);

    var loss_fn = nn.loss.MSELoss(T).init(.mean);

    var epoch: u64 = 0;
    while (epoch < epochs) : (epoch += 1) {
        optim.zeroGrad();

        const pred = try net.call(&x);
        const loss = try loss_fn.call(pred, &y);

        sched.backward(loss);
        try optim.step();

        if (epoch < 10 or epoch % 5000 == 0) {
            std.debug.print("epoch {d}: loss = {d:.6}  (jit={})\n", .{ epoch, loss.data.?[0], use_jit });
        }
    }

    const pred = try net.call(&x);
    sched.forward(pred);
    std.debug.print("Final predictions:\n", .{});
    for (0..4) |i| {
        const x0 = x.data.?[i * 2];
        const x1 = x.data.?[i * 2 + 1];
        std.debug.print("  {d} XOR {d} = {d:.4}  (expected {d})\n", .{ x0, x1, pred.data.?[i], y.data.?[i] });
    }
}
