const std = @import("std");
const cu = @import("cuda");
const assert = std.debug.assert;
const PlastAllocator = @import("allocator.zig").PlastAllocator;
const JIT = @import("scheduler/jit.zig").JIT;
const Scheduler = @import("scheduler/scheduler.zig").Scheduler;
const Tensor = @import("tensor.zig").Tensor;

pub const TypeErasedResource = struct {
    ptr: *anyopaque,
    deinit_fn: *const fn (ptr: *anyopaque) void,
};

pub const ScheduleMode = enum {
    forward,
    backward,
};

const DeviceRegistry = struct {
    jits: [2]?TypeErasedResource = .{ null, null },
    schedulers: [2]?TypeErasedResource = .{ null, null },
};

var registry = DeviceRegistry{};

pub const Device = union(enum) {
    cpu: CpuDevice,
    cuda: CudaDevice,

    /// Construct a device of the given kind with default configuration.
    pub fn init(comptime tag: std.meta.Tag(@This())) !Device {
        return switch (tag) {
            .cpu => .{ .cpu = CpuDevice.init() },
            .cuda => .{ .cuda = try CudaDevice.init() },
        };
    }

    /// Return the memory allocator for this device's address space.
    pub fn allocator(self: *Device) PlastAllocator {
        return switch (self.*) {
            .cpu => |d| d.allocator,
            .cuda => |d| d.allocator,
        };
    }

    /// Numeric tag for kernel dispatch tables (0 = cpu, 1 = cuda, …).
    pub fn kind(self: Device) u32 {
        return @intFromEnum(self);
    }

    /// Enable and return the JIT compiler for this device's scheduler.
    /// If this is not called, JIT compilation is disabled by default.
    pub fn jit(self: *Device, comptime T: type) *JIT(T) {
        const j_ptr = self.getJit(T);
        const sched = self.getScheduler(T);
        sched.setJit(j_ptr);
        return j_ptr;
    }

    /// Get or lazily initialize the JIT compiler for this device.
    pub fn getJit(self: *Device, comptime T: type) *JIT(T) {
        const k = self.kind();
        if (registry.jits[k]) |j| {
            return @ptrCast(@alignCast(j.ptr));
        }
        const gpa = std.heap.c_allocator;
        const j_ptr = gpa.create(JIT(T)) catch @panic("OOM");
        j_ptr.* = JIT(T).init(gpa);
        registry.jits[k] = .{
            .ptr = j_ptr,
            .deinit_fn = struct {
                fn deinit(ptr: *anyopaque) void {
                    const typed: *JIT(T) = @ptrCast(@alignCast(ptr));
                    typed.deinit();
                    std.heap.c_allocator.destroy(typed);
                }
            }.deinit,
        };
        return j_ptr;
    }

    /// Get or lazily initialize the scheduler for this device.
    pub fn getScheduler(self: *Device, comptime T: type) *Scheduler(T) {
        const k = self.kind();
        if (registry.schedulers[k]) |s| {
            return @ptrCast(@alignCast(s.ptr));
        }
        const gpa = std.heap.c_allocator;
        const s_ptr = gpa.create(Scheduler(T)) catch @panic("OOM");
        s_ptr.* = Scheduler(T).init(gpa, self, .{ .jit = null });
        registry.schedulers[k] = .{
            .ptr = s_ptr,
            .deinit_fn = struct {
                fn deinit(ptr: *anyopaque) void {
                    const typed: *Scheduler(T) = @ptrCast(@alignCast(ptr));
                    typed.deinit();
                    std.heap.c_allocator.destroy(typed);
                }
            }.deinit,
        };
        return s_ptr;
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
        const k = self.kind();
        if (registry.schedulers[k]) |s| {
            s.deinit_fn(s.ptr);
            registry.schedulers[k] = null;
        }
        if (registry.jits[k]) |j| {
            j.deinit_fn(j.ptr);
            registry.jits[k] = null;
        }
        switch (self.*) {
            .cpu => |*d| d.deinit(),
            .cuda => |*d| d.deinit(),
        }
    }
};

pub const CpuDevice = struct {
    allocator: PlastAllocator,

    pub fn init() CpuDevice {
        return .{
            .allocator = PlastAllocator.cpu(std.heap.page_allocator),
        };
    }

    pub fn deinit(self: *CpuDevice) void {
        _ = self;
    }
};

pub const CudaDevice = struct {
    allocator: PlastAllocator,
    dev: cu.CUdevice,
    ctx: cu.CUcontext,

    pub fn init() !CudaDevice {
        try check(cu.cuInit(0));
        var dev: cu.CUdevice = undefined;
        try check(cu.cuDeviceGet(&dev, 0));
        var ctx: cu.CUcontext = undefined;
        try check(cu.cuDevicePrimaryCtxRetain(&ctx, dev));
        try check(cu.cuCtxSetCurrent(ctx));
        return .{
            .dev = dev,
            .ctx = ctx,
            .allocator = PlastAllocator.cuda(),
        };
    }

    pub fn deinit(self: *CudaDevice) void {
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

test "cpu device — alloc, write, read, free" {
    var dev = try Device.init(.cpu);
    defer dev.deinit();
    const ptr = dev.allocator().alloc(64, 1) orelse return error.OutOfMemory;
    defer dev.allocator().free(ptr, 64, 1);
    ptr[0] = 42;
    try std.testing.expectEqual(@as(u8, 42), ptr[0]);
}

test "cpu device — get lazy resources" {
    var dev = try Device.init(.cpu);
    defer dev.deinit();

    const jit1 = dev.getJit(f32);
    const jit2 = dev.getJit(f32);
    try std.testing.expect(jit1 == jit2);

    const sched1 = dev.getScheduler(f32);
    const sched2 = dev.getScheduler(f32);
    try std.testing.expect(sched1 == sched2);
}

test "cpu device — jit enable" {
    var dev = try Device.init(.cpu);
    defer dev.deinit();

    const j = dev.jit(f32);
    const sched = dev.getScheduler(f32);
    try std.testing.expect(sched.jit == j);
}

test "cuda device — init, alloc, memcpy" {
    var dev = Device.init(.cuda) catch |e| switch (e) {
        error.CudaError => return error.SkipZigTest,
        else => |err| return err,
    };
    defer dev.deinit();

    const size = @sizeOf(f32) * 8;
    const ptr = dev.allocator().alloc(size, @alignOf(f32)) orelse return error.OutOfMemory;
    defer dev.allocator().free(ptr, size, @alignOf(f32));

    var dst: [8]f32 = undefined;
    if (cu.cuMemcpyDtoH(&dst, @intFromPtr(ptr), size) != cu.CUDA_SUCCESS)
        return error.SkipZigTest;
    std.debug.print("device cuda: {any}\n", .{dst});
}
