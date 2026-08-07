const std = @import("std");

fn addCSources(mod: *std.Build.Module, b: *std.Build, cflags: []const []const u8) void {
    mod.addIncludePath(b.path("include"));
    mod.addCSourceFile(.{ .file = b.path("src/core/tensor.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/core/arena.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/core/arena_cpu.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/core/op.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/cpu_tensor_init.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/cpu_utils.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/binary_ops.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/unary_ops.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/pack.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/matmul.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/reduce_ops.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/shape_ops.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/conv2d.c"), .flags = cflags });
    mod.addCSourceFile(.{ .file = b.path("src/kernels/cpu/fused_ops.c"), .flags = cflags });
    mod.linkSystemLibrary("c", .{});
    mod.linkSystemLibrary("m", .{});
    mod.linkSystemLibrary("omp", .{});
}

const cuda_objs = &[_][]const u8{
    "src/core/arena_cuda.cu.o",
    "src/kernels/cuda/binary_ops.cu.o",
    "src/kernels/cuda/conv2d.cu.o",
    "src/kernels/cuda/conv_relu.cu.o",
    "src/kernels/cuda/cuda_iter_init.cu.o",
    "src/kernels/cuda/cuda_pack.cu.o",
    "src/kernels/cuda/cuda_tensor_init.cu.o",
    "src/kernels/cuda/matmul.cu.o",
    "src/kernels/cuda/matmul_bias_relu.cu.o",
    "src/kernels/cuda/matmul_relu.cu.o",
    "src/kernels/cuda/reduce_ops.cu.o",
    "src/kernels/cuda/shape_ops_cuda.cu.o",
    "src/kernels/cuda/unary_ops.cu.o",
    "src/optimizers/cuda/adam.cu.o",
    "src/optimizers/cuda/adamw.cu.o",
    "src/optimizers/cuda/sgd.cu.o",
    "src/optimizers/cuda/zero_grad.cu.o",
};

pub fn build(b: *std.Build) void {
    const cflags = &.{ "-O3", "-march=native", "-fopenmp" };

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = b.graph.host,
        .link_libc = true,
    });
    addCSources(exe_mod, b, cflags);

    const cuda_bindings = b.addTranslateC(.{
        .root_source_file = b.path("../../../../opt/cuda/include/cuda.h"),
        .target = b.graph.host,
        .optimize = .Debug,
    });

    cuda_bindings.addIncludePath(b.path("../../../../opt/cuda/include"));
    cuda_bindings.defineCMacroRaw("__NV_NO_VECTOR_DEPRECATION_DIAG");
    cuda_bindings.linkSystemLibrary("cuda", .{});

    const cuda_module = cuda_bindings.createModule();
    exe_mod.addImport("cuda", cuda_module);

    const exe = b.addExecutable(.{
        .name = "tensor",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    // Link CUDA runtime and pre-compiled kernels to executable
    exe_mod.addLibraryPath(.{ .cwd_relative = "/opt/cuda/lib64" });
    exe_mod.linkSystemLibrary("cudart", .{});
    exe_mod.linkSystemLibrary("c++", .{});
    for (cuda_objs) |obj| {
        exe_mod.addObjectFile(b.path(obj));
    }

    const run_exe = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the framework");
    run_step.dependOn(&run_exe.step);

    const test_step = b.step("test", "Run unit tests");

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = b.graph.host,
    });
    addCSources(test_mod, b, cflags);
    test_mod.addImport("cuda", cuda_module);

    const unit_tests = b.addTest(.{
        .root_module = test_mod,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_unit_tests.step);

    // Link CUDA runtime and pre-compiled kernels to tests
    test_mod.addLibraryPath(.{ .cwd_relative = "/opt/cuda/lib64" });
    test_mod.linkSystemLibrary("cudart", .{});
    test_mod.linkSystemLibrary("c++", .{});
    for (cuda_objs) |obj| {
        test_mod.addObjectFile(b.path(obj));
    }
}
