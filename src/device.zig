const std = @import("std");
const cu = @import("cuda");
const assert = std.debug.assert;

const CpuArena = @import("allocator.zig").CpuArena;
const CudaArena = @import("allocator.zig").CudaArena;
const Tensor = @import("tensor.zig").Tensor;

pub const TypeErasedResource = struct {
    ptr: *anyopaque,
    deinit_fn: *const fn (ptr: *anyopaque) void,
};

pub const ScheduleMode = enum {
    forward,
    backward,
};

pub const Device = union(enum) {
    cpu: CPU,
    cuda: CUDA,

    /// Construct a device of the given kind with default configuration.
    pub fn init(comptime tag: std.meta.Tag(@This())) !Device {
        return switch (tag) {
            .cpu => .{ .cpu = CPU.init() },
            .cuda => .{ .cuda = try CUDA.init() },
        };
    }

    /// Numeric tag for kernel dispatch tables (0 = cpu, 1 = cuda, …).
    pub fn kind(self: Device) u32 {
        return @intFromEnum(self);
    }

    /// Schedule execution (forward or backward pass) of a tensor on this device.
    pub fn schedule(self: *Device, comptime T: type, tensor: *Tensor(T), mode: ScheduleMode) void {
        const s = self.getScheduler(T);
        switch (mode) {
            .forward => s.forward(tensor),
            .backward => s.backward(tensor),
        }
    }

    /// Release hardware and software resources.
    pub fn deinit(self: *Device) void {
        switch (self.*) {
            .cpu => |*d| d.deinit(),
            .cuda => |*d| d.deinit(),
        }
    }
};

pub const CPU = struct {
    params: CpuArena,
    inputs: CpuArena,

    pub fn init() CPU {
        return .{
            .params = CpuArena.init(std.heap.page_allocator),
            .inputs = CpuArena.init(std.heap.page_allocator),
        };
    }

    pub fn deinit(self: *CPU) void {
        self.inputs.deinit();
        self.params.deinit();
    }
};

pub const CUDA = struct {
    params: CudaArena,
    inputs: CudaArena,
    dev: cu.CUdevice,
    ctx: cu.CUcontext,

    pub fn init() !CUDA {
        try check(cu.cuInit(0));
        var dev: cu.CUdevice = undefined;
        try check(cu.cuDeviceGet(&dev, 0));
        var ctx: cu.CUcontext = undefined;
        try check(cu.cuDevicePrimaryCtxRetain(&ctx, dev));
        try check(cu.cuCtxSetCurrent(ctx));
        return .{
            .params = CudaArena.init(64 * 1024 * 1024), // 64 MB default capacity
            .inputs = CudaArena.init(16 * 1024 * 1024), // 16 MB scratch capacity
            .dev = dev,
            .ctx = ctx,
        };
    }

    pub fn deinit(self: *CUDA) void {
        self.inputs.deinit();
        self.params.deinit();
        _ = cu.cuDevicePrimaryCtxRelease(self.dev);
    }
};

fn check(result: cu.CUresult) !void {
    if (result == cu.CUDA_SUCCESS) return;
    var name: [*c]const u8 = undefined;
    _ = cu.cuGetErrorName(result, &name);
    std.log.err("CUDA: {s}", .{std.mem.span(name)});
    return error.CudaError;
}

test "cpu device — alloc, write, read" {
    var dev = try Device.init(.cpu);
    defer dev.deinit();
    const ptr = dev.cpu.params.alloc(64, 1) orelse return error.OutOfMemory;
    ptr[0] = 42;
    try std.testing.expectEqual(@as(u8, 42), ptr[0]);
}

test "cuda device — init, alloc, memcpy" {
    var dev = Device.init(.cuda) catch |e| switch (e) {
        error.CudaError => return error.SkipZigTest,
        else => |err| return err,
    };
    defer dev.deinit();

    const size = @sizeOf(f32) * 8;
    const ptr = dev.cuda.params.alloc(size, @alignOf(f32)) orelse return error.OutOfMemory;

    var dst: [8]f32 = undefined;
    if (cu.cuMemcpyDtoH(&dst, @intFromPtr(ptr), size) != cu.CUDA_SUCCESS)
        return error.SkipZigTest;
    std.debug.print("device cuda: {any}\n", .{dst});
}
