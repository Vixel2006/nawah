const std = @import("std");
const assert = std.debug.assert;
const UOp = @import("uop.zig").UOp;
const Graph = @import("graph.zig").Graph;
const Device = @import("device.zig").Device;
const c_api = @import("c_api.zig");

pub const MAX_NDIM = 8;

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        dev: *Device,
        ndim: u64,
        shape: [MAX_NDIM]u64,
        strides: [MAX_NDIM]u64,
        data: ?[]T = null,
        grad: ?[]T = null,
        creator: ?*UOp(T) = null,
        requires_grad: bool = false,

        pub fn init(dev: *Device, shape: []const u64, requires_grad: bool) !Self {
            assert(shape.len > 0);
            assert(shape.len <= MAX_NDIM);

            const ndim: u64 = @intCast(shape.len);
            var shape_arr: [MAX_NDIM]u64 = undefined;
            for (shape, 0..) |s, i| {
                shape_arr[i] = s;
            }
            const strides = computeStrides(&shape_arr, ndim);

            const n = numel(&shape_arr, ndim);
            const data = switch (dev.*) {
                .cpu => dev.cpu.params.alloc(n * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory,
                .cuda => dev.cuda.params.alloc(n * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory,
            };
            var grad: ?[]T = null;
            if (requires_grad) {
                const grad_ptr = switch (dev.*) {
                    .cpu => dev.cpu.params.alloc(n * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory,
                    .cuda => dev.cuda.params.alloc(n * @sizeOf(T), @alignOf(T)) orelse return error.OutOfMemory,
                };
                grad = @as([*]T, @ptrCast(@alignCast(grad_ptr)))[0..n];
                switch (dev.*) {
                    .cpu => @memset(grad.?, 0),
                    .cuda => {
                        var c_t = c_api.toCTensor(@ptrCast(grad.?.ptr), &shape_arr, &strides, ndim, requires_grad);
                        c_api.zeros_cuda(&c_t, n);
                    },
                }
            }

            return .{
                .dev = dev,
                .ndim = ndim,
                .shape = shape_arr,
                .strides = strides,
                .requires_grad = requires_grad,
                .data = @as([*]T, @ptrCast(@alignCast(data)))[0..n],
                .grad = grad,
            };
        }

        pub fn deinit(self: *Self, gpa: std.mem.Allocator) void {
            _ = gpa;
            self.grad = null;
            self.data = null;
            self.creator = null;
        }

        pub fn zeroGrad(self: *Self) void {
            if (self.grad) |g| {
                switch (self.dev.*) {
                    .cpu => @memset(g, 0),
                    .cuda => {
                        var c_t = c_api.toCTensor(@ptrCast(g.ptr), &self.shape, &self.strides, self.ndim, self.requires_grad);
                        c_api.zeros_cuda(&c_t, g.len);
                    },
                }
            }
        }



        pub fn zeros(dev: *Device, shape: []const u64, requires_grad: bool) !Self {
            const t = try init(dev, shape, requires_grad);
            const n = numel(&t.shape, t.ndim);
            switch (t.dev.*) {
                .cpu => @memset(t.data.?, 0),
                .cuda => {
                    var c_t = c_api.toCTensor(@ptrCast(t.data.?), &t.shape, &t.strides, t.ndim, t.requires_grad);
                    c_api.zeros_cuda(&c_t, n);
                },
            }
            if (requires_grad && t.grad) |g| {
                switch (t.dev.*) {
                    .cpu => @memset(g, 0),
                    .cuda => {
                        var c_t = c_api.toCTensor(@ptrCast(g.ptr), &t.shape, &t.strides, t.ndim, t.requires_grad);
                        c_api.zeros_cuda(&c_t, n);
                    },
                }
            }
            return t;
        }

        pub fn ones(dev: *Device, shape: []const u64, requires_grad: bool) !Self {
            var t = try init(dev, shape, requires_grad);
            const n = numel(&t.shape, t.ndim);
            switch (t.dev.*) {
                .cpu => @memset(t.data.?, @as(T, 1)),
                .cuda => {
                    var c_t = c_api.toCTensor(@ptrCast(t.data.?), &t.shape, &t.strides, t.ndim, t.requires_grad);
                    c_api.ones_cuda(&c_t, n);
                },
            }
            return t;
        }

        pub fn fromData(dev: *Device, shape: []const u64, data: []const T, requires_grad: bool) !Self {
            var t = try init(dev, shape, requires_grad);
            const n = numel(&t.shape, t.ndim);
            assert(data.len == n);
            switch (t.dev.*) {
                .cpu => @memcpy(t.data.?, data),
                .cuda => {
                    const cu = @import("cuda");
                    const result = cu.cuMemcpyHtoD(@intFromPtr(t.data.?.ptr), data.ptr, data.len * @sizeOf(T));
                    assert(result == cu.CUDA_SUCCESS);
                },
            }
            return t;
        }

        pub fn uniform(dev: *Device, shape: []const u64, requires_grad: bool, rng: std.Random) !Self {
            var t = try init(dev, shape, requires_grad);
            const n = numel(&t.shape, t.ndim);
            switch (t.dev.*) {
                .cpu => {
                    for (t.data.?) |*v| v.* = rng.float(T);
                },
                .cuda => {
                    const gpa = std.heap.c_allocator;
                    const host_buf = try gpa.alloc(T, n);
                    defer gpa.free(host_buf);
                    for (host_buf) |*v| v.* = rng.float(T);

                    const cu = @import("cuda");
                    const result = cu.cuMemcpyHtoD(@intFromPtr(t.data.?.ptr), host_buf.ptr, n * @sizeOf(T));
                    assert(result == cu.CUDA_SUCCESS);
                },
            }
            return t;
        }

        pub fn scalar(dev: *Device, val: T) !Self {
            return fromData(dev, &.{1}, &.{val}, false);
        }

        pub fn realize(self: *Self, gpa: std.mem.Allocator) !void {
            const creator = self.creator orelse return;
            var graph: Graph(T) = .init(gpa);
            defer graph.deinit();
            graph.build(creator);
            graph.forward();
        }

        pub fn backward(self: *Self, gpa: std.mem.Allocator) !void {
            const creator = self.creator orelse return;
            try self.realize(gpa);
            const n = numel(&self.shape, self.ndim);
            switch (self.dev.*) {
                .cpu => @memset(self.grad.?, @as(T, 1)),
                .cuda => {
                    var c_t = c_api.toCTensor(@ptrCast(self.grad.?.ptr), &self.shape, &self.strides, self.ndim, self.requires_grad);
                    c_api.ones_cuda(&c_t, n);
                },
            }
            // Zero input grads before backward pass
            var graph: Graph(T) = .init(gpa);
            defer graph.deinit();
            graph.build(creator);
            for (graph.nodes.items) |node| {
                for (node.inputs) |inp| {
                    if (inp.requires_grad && inp.grad) |g| {
                        switch (inp.dev.*) {
                            .cpu => @memset(g, 0),
                            .cuda => {
                                var c_t = c_api.toCTensor(@ptrCast(g.ptr), &inp.shape, &inp.strides, inp.ndim, inp.requires_grad);
                                c_api.zeros_cuda(&c_t, g.len);
                            },
                        }
                    }
                }
            }
            graph.backward() catch |err| {
                std.log.err("Error during backward: {any}", .{err});
            };
        }

        pub fn isContiguous(self: *const Self) bool {
            if (self.ndim == 0) return true;
            var expected: u64 = 1;
            var i: u64 = self.ndim - 1;
            while (true) {
                if (self.strides[i] != expected) return false;
                if (i == 0) break;
                expected *= self.shape[i];
                i -= 1;
            }
            return true;
        }
    };
}

fn computeStrides(shape: *const [MAX_NDIM]u64, ndim: u64) [MAX_NDIM]u64 {
    var strides: [MAX_NDIM]u64 = undefined;
    if (ndim == 0) return strides;
    strides[ndim - 1] = 1;
    var i = ndim - 1;
    while (i > 0) {
        i -= 1;
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

fn numel(shape: *const [MAX_NDIM]u64, ndim: u64) u64 {
    var n: u64 = 1;
    for (shape[0..ndim]) |dim| n *= dim;
    return n;
}

pub fn slice(comptime T: type, dev: *Device, data: ?[]T, shape: [MAX_NDIM]u64, ndim: u64) []const T {
    assert(data != null);
    const n = numel(&shape, ndim);

    return switch (dev.*) {
        .cpu => data.?[0..n],
        .cuda => {
            // Fetch CUDA device data back to a transient host buffer for viewing
            // (Must allocate transient memory via std.heap.c_allocator)
            const gpa = std.heap.c_allocator;
            const host_mem = gpa.alloc(T, n) catch @panic("OOM");
            const cu = @import("cuda");
            const result = cu.cuMemcpyDtoH(host_mem.ptr, @intFromPtr(data.?.ptr), n * @sizeOf(T));
            assert(result == cu.CUDA_SUCCESS);
            // Return as slice. Note: Caller must be careful as this is a leaked allocation for debugging.
            // We mark it thread-local or static, or return it directly. Since this is for print / test assertions:
            return host_mem;
        },
    };
}
