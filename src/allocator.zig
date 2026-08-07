const std = @import("std");
const cu = @import("cuda");
const assert = std.debug.assert;

const Device = enum { CPU, CUDA };

/// Arena allocator for CPU memory, wrapping `std.heap.ArenaAllocator`.
pub const CpuArena = struct {
    arena: std.heap.ArenaAllocator,

    pub fn init(backing: std.mem.Allocator) CpuArena {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing),
        };
    }

    pub fn deinit(self: *CpuArena) void {
        self.arena.deinit();
    }

    pub fn reset(self: *CpuArena) void {
        _ = self.arena.reset(.retain_capacity);
    }

    pub fn allocator(self: *CpuArena) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn alloc(self: *CpuArena, size: usize, alignment: usize) ?[*]u8 {
        assert(size > 0);
        assert(alignment > 0);
        assert(std.math.isPowerOfTwo(alignment));
        const log2_align: std.mem.Alignment = @enumFromInt(@ctz(alignment));
        return self.arena.allocator().rawAlloc(size, log2_align, @returnAddress());
    }
};

/// Growable arena allocator for GPU device memory.
pub const CudaArena = struct {
    head: ?*Block,
    default_capacity: usize,

    const Block = struct {
        data: [*]u8,
        capacity: usize,
        offset: usize,
        prev: ?*Block,
    };

    /// Metadata allocator — always host-side.
    const meta = std.heap.c_allocator;

    pub fn init(default_capacity: usize) CudaArena {
        assert(default_capacity > 0);
        return .{
            .head = null,
            .default_capacity = default_capacity,
        };
    }

    pub fn alloc(self: *CudaArena, size: usize, alignment: usize) ?[*]u8 {
        assert(size > 0);
        assert(alignment > 0);
        assert(std.math.isPowerOfTwo(alignment));

        if (self.head) |block| {
            const aligned_offset = alignToAddress(block.data, block.offset, alignment);
            if (aligned_offset + size <= block.capacity) {
                block.offset = aligned_offset + size;
                return block.data + aligned_offset;
            }
        }

        return self.growAndAlloc(size, alignment);
    }

    pub fn reset(self: *CudaArena) void {
        var current = self.head orelse return;

        // Walk backward, freeing every block except the tail (oldest).
        while (current.prev) |prev| {
            _ = cu.cuMemFree(@intFromPtr(current.data));
            meta.destroy(current);
            current = prev;
        }

        current.offset = 0;
        self.head = current;
    }

    pub fn deinit(self: *CudaArena) void {
        var current = self.head;
        while (current) |block| {
            const prev = block.prev;
            _ = cu.cuMemFree(@intFromPtr(block.data));
            meta.destroy(block);
            current = prev;
        }
        self.head = null;
    }

    fn growAndAlloc(self: *CudaArena, size: usize, alignment: usize) ?[*]u8 {
        // Reserve enough for the payload plus worst-case alignment waste.
        const capacity = @max(self.default_capacity, size + alignment - 1);

        var dptr: cu.CUdeviceptr = undefined;
        if (cu.cuMemAlloc(&dptr, capacity) != cu.CUDA_SUCCESS) {
            return null;
        }
        const data: [*]u8 = @ptrFromInt(dptr);

        const block = meta.create(Block) catch {
            _ = cu.cuMemFree(dptr);
            return null;
        };
        block.* = .{
            .data = data,
            .capacity = capacity,
            .offset = 0,
            .prev = self.head,
        };
        self.head = block;

        const aligned_offset = alignToAddress(data, 0, alignment);
        block.offset = aligned_offset + size;
        return data + aligned_offset;
    }

    fn alignToAddress(base: [*]u8, current_offset: usize, alignment: usize) usize {
        const addr = @intFromPtr(base) + current_offset;
        const aligned_addr = std.mem.alignForward(usize, addr, alignment);
        return aligned_addr - @intFromPtr(base);
    }
};

test "cpu alloc" {
    var arena = CpuArena.init(std.testing.allocator);
    defer arena.deinit();
    const ptr = arena.alloc(128, 8) orelse return error.OutOfMemory;
    @memset(ptr[0..128], 0xAB);
    try std.testing.expectEqual(@as(u8, 0xAB), ptr[0]);
    try std.testing.expectEqual(@as(u8, 0xAB), ptr[127]);
}

test "cpu alignment is respected" {
    var arena = CpuArena.init(std.testing.allocator);
    defer arena.deinit();
    const ptr = arena.alloc(64, 64) orelse return error.OutOfMemory;
    // The returned address must be 64-byte aligned.
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(ptr) % 64);
}

test "CpuArena bump allocation" {
    var arena = CpuArena.init(std.testing.allocator);
    defer arena.deinit();

    // Multiple small allocations should come from the same block.
    const p1 = arena.alloc(32, 8) orelse return error.OutOfMemory;
    const p2 = arena.alloc(64, 16) orelse return error.OutOfMemory;
    const p3 = arena.alloc(16, 4) orelse return error.OutOfMemory;

    // All three must be non-overlapping.
    const a1 = @intFromPtr(p1);
    const a2 = @intFromPtr(p2);
    const a3 = @intFromPtr(p3);
    try std.testing.expect(a2 >= a1 + 32);
    try std.testing.expect(a3 >= a2 + 64);

    // Alignment must be respected.
    try std.testing.expectEqual(@as(usize, 0), a2 % 16);
    try std.testing.expectEqual(@as(usize, 0), a3 % 4);

    // Write to them to verify they are writable.
    p1[0..32][0] = 1;
    p2[0..64][0] = 2;
    p3[0..16][0] = 3;

    try std.testing.expectEqual(@as(u8, 1), p1[0]);
    try std.testing.expectEqual(@as(u8, 2), p2[0]);
    try std.testing.expectEqual(@as(u8, 3), p3[0]);
}

test "CpuArena reset" {
    var arena = CpuArena.init(std.testing.allocator);
    defer arena.deinit();

    _ = arena.alloc(64, 8) orelse return error.OutOfMemory;
    arena.reset();

    const p = arena.alloc(32, 8) orelse return error.OutOfMemory;
    p[0] = 0x42;
    try std.testing.expectEqual(@as(u8, 0x42), p[0]);
}

test "CudaArena bump allocation" {
    // Attempt to initialize CUDA context. If it fails, skip the test.
    if (cu.cuInit(0) != cu.CUDA_SUCCESS) return error.SkipZigTest;
    var dev: cu.CUdevice = undefined;
    if (cu.cuDeviceGet(&dev, 0) != cu.CUDA_SUCCESS) return error.SkipZigTest;
    var ctx: cu.CUcontext = undefined;
    if (cu.cuDevicePrimaryCtxRetain(&ctx, dev) != cu.CUDA_SUCCESS) return error.SkipZigTest;
    defer _ = cu.cuDevicePrimaryCtxRelease(dev);
    if (cu.cuCtxSetCurrent(ctx) != cu.CUDA_SUCCESS) return error.SkipZigTest;

    var arena = CudaArena.init(4096);
    defer arena.deinit();

    const p1 = arena.alloc(32, 8) orelse return error.OutOfMemory;
    const p2 = arena.alloc(64, 16) orelse return error.OutOfMemory;

    // Verify pointer alignments
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(p1) % 8);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(p2) % 16);
}
