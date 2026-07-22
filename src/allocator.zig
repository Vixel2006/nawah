const std = @import("std");
const cu = @import("cuda");
const assert = std.debug.assert;

/// Unified allocator across compute backends.
///
/// Construct via `PlastAllocator.cpu()` or `PlastAllocator.cuda()`.
/// Copy-safe: holds no mutable or heap-allocated internal state.
pub const PlastAllocator = struct {
    backend: Backend,

    /// Host-memory allocator backed by any `std.mem.Allocator`.
    pub fn cpu(backing_allocator: std.mem.Allocator) PlastAllocator {
        return .{ .backend = .{ .cpu = .{ .backing = backing_allocator } } };
    }

    /// CUDA device-memory allocator.  Requires an active CUDA context.
    pub fn cuda() PlastAllocator {
        return .{ .backend = .{ .cuda = .{} } };
    }

    /// Allocate `size` bytes aligned to `alignment`.
    ///
    /// Returns a many-item pointer on success, or `null` if the backend
    /// cannot satisfy the request.
    pub fn alloc(self: PlastAllocator, size: usize, alignment: usize) ?[*]u8 {
        assert(size > 0);
        assert(alignment > 0);
        assert(std.math.isPowerOfTwo(alignment));

        return switch (self.backend) {
            .cpu => |b| b.alloc(size, alignment),
            .cuda => |b| b.alloc(size),
        };
    }

    /// Free a region previously returned by `alloc`.
    ///
    /// `size` and `alignment` must exactly match the values passed to the
    /// corresponding `alloc` call.
    pub fn free(self: PlastAllocator, ptr: [*]u8, size: usize, alignment: usize) void {
        assert(size > 0);
        assert(alignment > 0);
        assert(std.math.isPowerOfTwo(alignment));

        switch (self.backend) {
            .cpu => |b| b.free(ptr, size, alignment),
            .cuda => |b| b.free(ptr),
        }
    }

    /// Release resources held by this allocator instance.
    ///
    /// Individual allocations must still be freed via `free` before calling
    /// `deinit`.  Currently a no-op — backends carry no owned heap state —
    /// but exists as a hook for future backends that might (e.g. memory pools).
    pub fn deinit(self: PlastAllocator) void {
        _ = self;
    }
};

/// Tagged union of all supported memory backends.
///
/// To add a new backend:
///   1. Add a variant here.
///   2. Implement a struct with `alloc` and `free` matching the existing pattern.
///   3. Add arms to the `switch` in `PlastAllocator.alloc` and `.free`.
const Backend = union(enum) {
    cpu: CpuBackend,
    cuda: CudaBackend,
};

/// Delegates to any `std.mem.Allocator`.  Stored inline — no heap overhead.
const CpuBackend = struct {
    backing: std.mem.Allocator,

    fn alloc(self: CpuBackend, size: usize, alignment: usize) ?[*]u8 {
        const log2_align: std.mem.Alignment = @enumFromInt(@ctz(alignment));
        return self.backing.rawAlloc(size, log2_align, @returnAddress());
    }

    fn free(self: CpuBackend, ptr: [*]u8, size: usize, alignment: usize) void {
        const log2_align: std.mem.Alignment = @enumFromInt(@ctz(alignment));
        self.backing.rawFree(ptr[0..size], log2_align, @returnAddress());
    }
};

/// Thin wrapper over cuMemAlloc / cuMemFree.  Stateless — CUDA tracks
/// its own allocations internally via the active driver context.
const CudaBackend = struct {
    fn alloc(_: CudaBackend, size: usize) ?[*]u8 {
        var device_ptr: cu.CUdeviceptr = undefined;
        if (cu.cuMemAlloc(&device_ptr, size) != cu.CUDA_SUCCESS) return null;
        return @ptrFromInt(device_ptr);
    }

    fn free(_: CudaBackend, ptr: [*]u8) void {
        const result = cu.cuMemFree(@intFromPtr(ptr));
        // A failed free indicates a double-free or corrupt pointer — always a
        // programming error, never a recoverable condition.
        assert(result == cu.CUDA_SUCCESS);
    }
};

/// Bump allocator that sub-allocates from large blocks obtained via a
/// `PlastAllocator`.  Useful for batching many small allocations (e.g.
/// optimizer temporaries) into a few large backend allocations.
///
/// Block metadata lives on the host (via `c_allocator`) regardless of which
/// backend provides the data buffers.
///
/// Allocation is O(1) in the fast path.  There is no individual `free` —
/// call `reset` to reclaim all memory at once, or `deinit` to release
/// everything.
pub const Arena = struct {
    allocator: PlastAllocator,
    head: ?*Block,
    default_capacity: usize,

    const Block = struct {
        data: [*]u8,
        capacity: usize,
        offset: usize,
        prev: ?*Block,
    };

    /// Metadata allocator — always host-side regardless of compute backend.
    const meta = std.heap.c_allocator;

    /// Minimum alignment for block-level allocations from the backend.
    /// 16 bytes covers all scalar types up to 128-bit SIMD lanes.
    const block_alignment: usize = 16;

    pub fn init(allocator: PlastAllocator, default_capacity: usize) Arena {
        assert(default_capacity > 0);

        return .{
            .allocator = allocator,
            .head = null,
            .default_capacity = default_capacity,
        };
    }

    /// Bump-allocate `size` bytes aligned to `alignment`.
    ///
    /// Returns null only if the underlying backend allocation fails.
    pub fn alloc(self: *Arena, size: usize, alignment: usize) ?[*]u8 {
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

    /// Reset the arena: free all blocks except the oldest, rewind it to zero.
    ///
    /// This preserves a single pre-allocated block to avoid re-allocating on
    /// the next burst of allocations.
    pub fn reset(self: *Arena) void {
        var current = self.head orelse return;

        // Walk backward, freeing every block except the tail (oldest).
        while (current.prev) |prev| {
            self.allocator.free(current.data, current.capacity, block_alignment);
            meta.destroy(current);
            current = prev;
        }

        current.offset = 0;
        self.head = current;
    }

    /// Release all memory owned by this arena.
    pub fn deinit(self: *Arena) void {
        var current = self.head;
        while (current) |block| {
            const prev = block.prev;
            self.allocator.free(block.data, block.capacity, block_alignment);
            meta.destroy(block);
            current = prev;
        }
        self.head = null;
    }

    fn growAndAlloc(self: *Arena, size: usize, alignment: usize) ?[*]u8 {
        // Reserve enough for the payload plus worst-case alignment waste.
        const capacity = @max(self.default_capacity, size + alignment - 1);
        const data = self.allocator.alloc(capacity, block_alignment) orelse return null;

        const block = meta.create(Block) catch return null;
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

    /// Smallest offset ≥ `current_offset` such that `base + offset` is aligned
    /// to `alignment`.
    ///
    /// Unlike a plain `alignForward(offset, alignment)`, this accounts for the
    /// actual address of `base` — critical when the backend returns a pointer
    /// whose natural alignment is less than the requested sub-allocation
    /// alignment.
    fn alignToAddress(base: [*]u8, current_offset: usize, alignment: usize) usize {
        const addr = @intFromPtr(base) + current_offset;
        const aligned_addr = std.mem.alignForward(usize, addr, alignment);
        return aligned_addr - @intFromPtr(base);
    }
};

test "cpu alloc and free" {
    const a = PlastAllocator.cpu(std.testing.allocator);
    const ptr = a.alloc(128, 8) orelse return error.OutOfMemory;
    @memset(ptr[0..128], 0xAB);
    try std.testing.expectEqual(@as(u8, 0xAB), ptr[0]);
    try std.testing.expectEqual(@as(u8, 0xAB), ptr[127]);
    a.free(ptr, 128, 8);
}

test "cpu alignment is respected" {
    const a = PlastAllocator.cpu(std.testing.allocator);
    const ptr = a.alloc(64, 64) orelse return error.OutOfMemory;
    defer a.free(ptr, 64, 64);
    // The returned address must be 64-byte aligned.
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(ptr) % 64);
}

test "arena bump allocation" {
    const backing = PlastAllocator.cpu(std.heap.page_allocator);
    var arena = Arena.init(backing, 4096);
    defer arena.deinit();

    // Multiple small allocations should come from the same block.
    const p1 = arena.alloc(32, 8) orelse return error.OutOfMemory;
    const p2 = arena.alloc(64, 16) orelse return error.OutOfMemory;
    const p3 = arena.alloc(16, 4) orelse return error.OutOfMemory;

    // All three must be non-overlapping and from the same block.
    const a1 = @intFromPtr(p1);
    const a2 = @intFromPtr(p2);
    const a3 = @intFromPtr(p3);
    try std.testing.expect(a2 >= a1 + 32);
    try std.testing.expect(a3 >= a2 + 64);

    // Alignment must be respected.
    try std.testing.expectEqual(@as(usize, 0), a2 % 16);
    try std.testing.expectEqual(@as(usize, 0), a3 % 4);
}

test "arena reset preserves one block" {
    const backing = PlastAllocator.cpu(std.heap.page_allocator);
    var arena = Arena.init(backing, 256);
    defer arena.deinit();

    _ = arena.alloc(64, 8) orelse return error.OutOfMemory;
    arena.reset();

    // After reset the arena is empty but still has a block — next alloc is fast.
    try std.testing.expect(arena.head != null);
    try std.testing.expectEqual(@as(usize, 0), arena.head.?.offset);

    const p = arena.alloc(32, 8) orelse return error.OutOfMemory;
    p[0] = 0x42;
    try std.testing.expectEqual(@as(u8, 0x42), p[0]);
}

test "arena grows beyond default capacity" {
    const backing = PlastAllocator.cpu(std.heap.page_allocator);
    var arena = Arena.init(backing, 64);
    defer arena.deinit();

    // Request larger than default_capacity — must still succeed.
    const p = arena.alloc(256, 8) orelse return error.OutOfMemory;
    @memset(p[0..256], 0xFF);
    try std.testing.expectEqual(@as(u8, 0xFF), p[255]);
}
