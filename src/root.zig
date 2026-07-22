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
pub const alc = @import("allocator.zig");
pub const device = @import("device.zig");
pub const Device = device.Device;

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
    _ = @import("device.zig");
}

fn Net(comptime T: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        dev: *Device,
        l1: nn.linear.Linear(T),
        sig1: nn.activation.Sigmoid(T),
        l2: nn.linear.Linear(T),
        sig2: nn.activation.Sigmoid(T),

        pub fn init(gpa: std.mem.Allocator, dev: *Device) !Self {
            return .{
                .gpa = gpa,
                .dev = dev,
                .l1 = try nn.linear.Linear(T).init(gpa, dev, 2, 8, .{ .seed = 42 }),
                .sig1 = nn.activation.Sigmoid(T).init(dev),
                .l2 = try nn.linear.Linear(T).init(gpa, dev, 8, 1, .{ .seed = 123 }),
                .sig2 = nn.activation.Sigmoid(T).init(dev),
            };
        }

        pub fn deinit(self: *Self) void {
            self.l2.deinit();
            self.l1.deinit();
        }

        pub fn call(self: *Self, gpa: std.mem.Allocator, x: *tensor.Tensor(T)) !*tensor.Tensor(T) {
            var out = try self.l1.call(gpa, x);
            out = try self.sig1.call(gpa, out);
            out = try self.l2.call(gpa, out);
            out = try self.sig2.call(gpa, out);
            return out;
        }

        pub fn parameters(self: *Self, gpa: std.mem.Allocator) ![]*tensor.Tensor(T) {
            const l1p = try self.l1.parameters(gpa);
            defer gpa.free(l1p);
            const l2p = try self.l2.parameters(gpa);
            defer gpa.free(l2p);

            var params = try gpa.alloc(*tensor.Tensor(T), l1p.len + l2p.len);
            @memcpy(params[0..l1p.len], l1p);
            @memcpy(params[l1p.len..], l2p);
            return params;
        }
    };
}

pub fn main() !void {
    const gpa = std.heap.c_allocator;

    var dev = try Device.init(.cpu);
    defer dev.deinit();

    const T = f32;
    const lr = 0.5;
    const epochs = 20000;
    const use_jit = true;

    var x = try tensor.Tensor(T).fromData(&dev, &.{ 4, 2 }, &[_]T{ 0, 0, 0, 1, 1, 0, 1, 1 }, false);
    defer x.deinit(gpa);
    var y = try tensor.Tensor(T).fromData(&dev, &.{ 4, 1 }, &[_]T{ 0, 1, 1, 0 }, false);
    defer y.deinit(gpa);

    var net = try Net(T).init(gpa, &dev);
    defer net.deinit();

    const params = try net.parameters(gpa);
    defer gpa.free(params);

    if (use_jit) {
        _ = dev.jit(T);
    }

    var optim = optimizers.Adam(T).init(gpa, &dev, .{ .lr = lr });
    defer optim.deinit();
    try optim.addParams(params);

    var loss_fn = nn.loss.MSELoss(T).init(&dev, .mean);

    var epoch: u64 = 0;
    while (epoch < epochs) : (epoch += 1) {
        optim.zeroGrad();

        const pred = try net.call(gpa, &x);
        const loss = try loss_fn.call(gpa, pred, &y);

        dev.schedule(T, loss, .backward);
        try optim.step();

        if (epoch < 10 or epoch % 5000 == 0) {
            std.debug.print("epoch {d}: loss = {d:.6}  (jit={})\n", .{ epoch, loss.data.?[0], use_jit });
        }
    }

    const pred = try net.call(gpa, &x);
    dev.schedule(T, pred, .forward);
    std.debug.print("Final predictions:\n", .{});
    for (0..4) |i| {
        const x0 = x.data.?[i * 2];
        const x1 = x.data.?[i * 2 + 1];
        std.debug.print("  {d} XOR {d} = {d:.4}  (expected {d})\n", .{ x0, x1, pred.data.?[i], y.data.?[i] });
    }
}
