const std = @import("std");
const Device = @import("../device.zig").Device;
const Tensor = @import("../tensor.zig").Tensor;
const Node = @import("../node.zig").Node;
const Op = @import("../op.zig").Op;
const OpType = @import("../op.zig").OpType;
const getFunction = @import("../op.zig").getFunction;
const c_api = @import("../c_api.zig");

fn createNode(comptime T: type, gpa: std.mem.Allocator, dev: *Device, op_type: OpType, inputs: []*Tensor(T), shape: []const u64) !*Tensor(T) {
    const device = dev.kind();
    const func = getFunction(op_type, 1, device);
    const op = Op{ .op_type = op_type, .params = .{ .dim = 0, .keepdim = 0, .fval = 0 }, .function = func };

    var requires_grad = false;
    for (inputs) |inp| {
        if (inp.requires_grad) {
            requires_grad = true;
            break;
        }
    }

    const out = try gpa.create(Tensor(T));
    out.* = try Tensor(T).zeros(dev, shape, requires_grad);

    var node = try gpa.create(Node(T));
    node.init(gpa, dev, inputs, out, op);
    return out;
}

pub fn add(comptime T: type, gpa: std.mem.Allocator, lhs: *Tensor(T), rhs: *Tensor(T)) !*Tensor(T) {
    const shape = lhs.shape[0..lhs.ndim];
    const ins = try gpa.alloc(*Tensor(T), 2);
    ins[0] = lhs;
    ins[1] = rhs;
    return createNode(T, gpa, lhs.dev, .{ .element_wise = .add }, ins, shape);
}

pub fn sub(comptime T: type, gpa: std.mem.Allocator, lhs: *Tensor(T), rhs: *Tensor(T)) !*Tensor(T) {
    const shape = lhs.shape[0..lhs.ndim];
    const ins = try gpa.alloc(*Tensor(T), 2);
    ins[0] = lhs;
    ins[1] = rhs;
    return createNode(T, gpa, lhs.dev, .{ .element_wise = .sub }, ins, shape);
}

pub fn mul(comptime T: type, gpa: std.mem.Allocator, lhs: *Tensor(T), rhs: *Tensor(T)) !*Tensor(T) {
    const shape = lhs.shape[0..lhs.ndim];
    const ins = try gpa.alloc(*Tensor(T), 2);
    ins[0] = lhs;
    ins[1] = rhs;
    return createNode(T, gpa, lhs.dev, .{ .element_wise = .mul }, ins, shape);
}

pub fn div(comptime T: type, gpa: std.mem.Allocator, lhs: *Tensor(T), rhs: *Tensor(T)) !*Tensor(T) {
    const shape = lhs.shape[0..lhs.ndim];
    const ins = try gpa.alloc(*Tensor(T), 2);
    ins[0] = lhs;
    ins[1] = rhs;
    return createNode(T, gpa, lhs.dev, .{ .element_wise = .div }, ins, shape);
}

pub fn relu(comptime T: type, gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
    const shape = x.shape[0..x.ndim];
    const ins = try gpa.alloc(*Tensor(T), 1);
    ins[0] = x;
    return createNode(T, gpa, x.dev, .{ .element_wise = .relu }, ins, shape);
}

pub fn exp(comptime T: type, gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
    const shape = x.shape[0..x.ndim];
    const ins = try gpa.alloc(*Tensor(T), 1);
    ins[0] = x;
    return createNode(T, gpa, x.dev, .{ .element_wise = .exp }, ins, shape);
}

pub fn log(comptime T: type, gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
    const shape = x.shape[0..x.ndim];
    const ins = try gpa.alloc(*Tensor(T), 1);
    ins[0] = x;
    return createNode(T, gpa, x.dev, .{ .element_wise = .log }, ins, shape);
}

pub fn sin(comptime T: type, gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
    const shape = x.shape[0..x.ndim];
    const ins = try gpa.alloc(*Tensor(T), 1);
    ins[0] = x;
    return createNode(T, gpa, x.dev, .{ .element_wise = .sin }, ins, shape);
}

pub fn cos(comptime T: type, gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
    const shape = x.shape[0..x.ndim];
    const ins = try gpa.alloc(*Tensor(T), 1);
    ins[0] = x;
    return createNode(T, gpa, x.dev, .{ .element_wise = .cos }, ins, shape);
}

pub fn abs(comptime T: type, gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
    const shape = x.shape[0..x.ndim];
    const ins = try gpa.alloc(*Tensor(T), 1);
    ins[0] = x;
    return createNode(T, gpa, x.dev, .{ .element_wise = .abs }, ins, shape);
}

pub fn neg(comptime T: type, gpa: std.mem.Allocator, x: *Tensor(T)) !*Tensor(T) {
    const shape = x.shape[0..x.ndim];
    const ins = try gpa.alloc(*Tensor(T), 1);
    ins[0] = x;
    return createNode(T, gpa, x.dev, .{ .element_wise = .neg }, ins, shape);
}

pub fn matmul(comptime T: type, gpa: std.mem.Allocator, lhs: *Tensor(T), rhs: *Tensor(T)) !*Tensor(T) {
    const m = lhs.shape[0];
    const n = rhs.shape[1];
    const shape = [_]u64{ m, n };
    const ins = try gpa.alloc(*Tensor(T), 2);
    ins[0] = lhs;
    ins[1] = rhs;
    return createNode(T, gpa, lhs.dev, .{ .linear = .matmul }, ins, shape[0..]);
}

fn computeReductionShape(gpa: std.mem.Allocator, shape: []const u64, dim: ?u64, keepdim: bool) ![]u64 {
    if (dim) |d| {
        if (keepdim) {
            const out_shape = try gpa.alloc(u64, shape.len);
            @memcpy(out_shape, shape);
            out_shape[d] = 1;
            return out_shape;
        } else {
            if (shape.len <= 1) {
                const out_shape = try gpa.alloc(u64, 1);
                out_shape[0] = 1;
                return out_shape;
            }
            const out_shape = try gpa.alloc(u64, shape.len - 1);
            var idx: u64 = 0;
            for (shape, 0..) |s, i| {
                if (i == d) continue;
                out_shape[idx] = s;
                idx += 1;
            }
            return out_shape;
        }
    } else {
        const out_shape = try gpa.alloc(u64, 1);
        out_shape[0] = 1;
        return out_shape;
    }
}

pub fn sum(comptime T: type, gpa: std.mem.Allocator, x: *Tensor(T), dim: ?u64, keepdim: bool) !*Tensor(T) {
    const dev = x.dev;
    const device = dev.kind();
    const out_shape = try computeReductionShape(gpa, x.shape[0..x.ndim], dim, keepdim);
    defer gpa.free(out_shape);

    const func = getFunction(.{ .reduce = .sum }, 1, device);
    const op = Op{
        .op_type = .{ .reduce = .sum },
        .params = .{
            .dim = dim orelse (c_api.MAX_NDIM + 1),
            .keepdim = if (keepdim) 1 else 0,
            .fval = 0,
        },
        .function = func,
    };

    const out = try gpa.create(Tensor(T));
    out.* = try Tensor(T).zeros(dev, out_shape, false);

    var node = try gpa.create(Node(T));
    const ins = try gpa.alloc(*Tensor(T), 1);
    ins[0] = x;
    node.init(gpa, dev, ins, out, op);
    return out;
}

pub fn mean(comptime T: type, gpa: std.mem.Allocator, x: *Tensor(T), dim: ?u64, keepdim: bool) !*Tensor(T) {
    const dev = x.dev;
    const device = dev.kind();
    const out_shape = try computeReductionShape(gpa, x.shape[0..x.ndim], dim, keepdim);
    defer gpa.free(out_shape);

    const func = getFunction(.{ .reduce = .mean }, 1, device);
    const op = Op{
        .op_type = .{ .reduce = .mean },
        .params = .{
            .dim = dim orelse (c_api.MAX_NDIM + 1),
            .keepdim = if (keepdim) 1 else 0,
            .fval = 0,
        },
        .function = func,
    };

    const out = try gpa.create(Tensor(T));
    out.* = try Tensor(T).zeros(dev, out_shape, false);

    var node = try gpa.create(Node(T));
    const ins = try gpa.alloc(*Tensor(T), 1);
    ins[0] = x;
    node.init(gpa, dev, ins, out, op);
    return out;
}

pub fn max(comptime T: type, gpa: std.mem.Allocator, x: *Tensor(T), dim: ?u64, keepdim: bool) !*Tensor(T) {
    const dev = x.dev;
    const device = dev.kind();
    const out_shape = try computeReductionShape(gpa, x.shape[0..x.ndim], dim, keepdim);
    defer gpa.free(out_shape);

    const func = getFunction(.{ .reduce = .max }, 1, device);
    const op = Op{
        .op_type = .{ .reduce = .max },
        .params = .{
            .dim = dim orelse (c_api.MAX_NDIM + 1),
            .keepdim = if (keepdim) 1 else 0,
            .fval = 0,
        },
        .function = func,
    };

    const out = try gpa.create(Tensor(T));
    out.* = try Tensor(T).zeros(dev, out_shape, false);

    var node = try gpa.create(Node(T));
    const ins = try gpa.alloc(*Tensor(T), 1);
    ins[0] = x;
    node.init(gpa, dev, ins, out, op);
    return out;
}

pub fn min(comptime T: type, gpa: std.mem.Allocator, x: *Tensor(T), dim: ?u64, keepdim: bool) !*Tensor(T) {
    const dev = x.dev;
    const device = dev.kind();
    const out_shape = try computeReductionShape(gpa, x.shape[0..x.ndim], dim, keepdim);
    defer gpa.free(out_shape);

    const func = getFunction(.{ .reduce = .min }, 1, device);
    const op = Op{
        .op_type = .{ .reduce = .min },
        .params = .{
            .dim = dim orelse (c_api.MAX_NDIM + 1),
            .keepdim = if (keepdim) 1 else 0,
            .fval = 0,
        },
        .function = func,
    };

    const out = try gpa.create(Tensor(T));
    out.* = try Tensor(T).zeros(dev, out_shape, false);

    var node = try gpa.create(Node(T));
    const ins = try gpa.alloc(*Tensor(T), 1);
    ins[0] = x;
    node.init(gpa, dev, ins, out, op);
    return out;
}
