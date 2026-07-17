const std = @import("std");

fn addCSources(mod: *std.Build.Module, b: *std.Build, cflags: []const []const u8) void {
    mod.addIncludePath(b.path("include"));
    mod.addCSourceFile(.{ .file = b.path("src/core/tensor.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/core/arena.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/core/arena_cpu.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/cpu_tensor_init.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/cpu_utils.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/binary_ops.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/unary_ops.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/pack.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/matmul.c"), .flags = cflags });
    mod.linkSystemLibrary("c", .{});
    mod.linkSystemLibrary("m", .{});
    mod.linkSystemLibrary("omp", .{});
}

pub fn build(b: *std.Build) void {
    const cflags = &.{ "-O3", "-march=native", "-fopenmp" };

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = b.graph.host,
    });
    addCSources(exe_mod, b, cflags);

    const exe = b.addExecutable(.{
        .name = "tensor",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    const run_exe = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the framework");
    run_step.dependOn(&run_exe.step);

    const test_step = b.step("test", "Run unit tests");

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = b.graph.host,
    });
    addCSources(test_mod, b, cflags);

    const unit_tests = b.addTest(.{
        .root_module = test_mod,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_unit_tests.step);
}
