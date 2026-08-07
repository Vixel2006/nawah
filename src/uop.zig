const std = @import("std");
const assert = std.debug.assert;
const cu = @import("cuda");
const Tensor = @import("tensor.zig").Tensor;
const Op = @import("op.zig").Op;
const ops = @import("ops/mod.zig");
const c_api = @import("c_api.zig");
const Device = @import("device.zig").Device;

pub fn UOp(comptime T: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        dev: *Device,
        inputs: []*Tensor(T),
        output: *Tensor(T),
        op: Op,
        visited: bool = false,

        pub fn init(
            self: *Self,
            gpa: std.mem.Allocator,
            dev: *Device,
            inputs: []*Tensor(T),
            output: *Tensor(T),
            op: Op,
        ) void {
            self.* = .{
                .gpa = gpa,
                .dev = dev,
                .inputs = inputs,
                .output = output,
                .op = op,
            };
            output.creator = self;
        }

        pub fn deinit(self: *Self) void {
            self.gpa.free(self.inputs);
            self.output.deinit(self.gpa);
            self.gpa.destroy(self.output);
        }

        fn launchCudaKernel(
            func: ?*const anyopaque,
            grid_size: u32,
            block_size: u32,
            args: []const ?*anyopaque,
        ) !void {
            const grid_dim = c_api.Dim3{ .x = grid_size, .y = 1, .z = 1 };
            const block_dim = c_api.Dim3{ .x = block_size, .y = 1, .z = 1 };
            const rc = c_api.cudaLaunchKernel(func, grid_dim, block_dim, args.ptr, 0, null);
            if (rc != 0) return error.CudaError;
            const sync_res = cu.cuCtxSynchronize();
            if (sync_res != cu.CUDA_SUCCESS) return error.CudaError;
        }

        fn copyToDevice(self: *Self, host_slice: []const u64) !?*anyopaque {
            const size = host_slice.len * @sizeOf(u64);
            const ptr = self.dev.cuda.params.alloc(size, @alignOf(u64)) orelse return error.OutOfMemory;
            const res = cu.cuMemcpyHtoD(@intFromPtr(ptr), host_slice.ptr, size);
            if (res != cu.CUDA_SUCCESS) return error.CudaError;
            return ptr;
        }

        pub fn forward(self: *Self) !*Tensor(T) {
            assert(self.output.data != null);
            const dev = self.dev;

            switch (self.op.op_type) {
                .element_wise => |ew| {
                    const a = self.inputs[0];
                    const out = self.output;
                    if (self.inputs.len == 2) {
                        const b = self.inputs[1];
                        switch (dev.*) {
                            .cpu => {
                                const is_contig =
                                    a.isContiguous() and b.isContiguous() and out.isContiguous() and
                                    std.mem.eql(u64, a.shape[0..a.ndim], b.shape[0..b.ndim]);
                                if (is_contig) {
                                    const func = switch (ew) {
                                        .add => &c_api.add_cpu_forward_float_contig_kernel,
                                        .sub => &c_api.sub_cpu_forward_float_contig_kernel,
                                        .mul => &c_api.mul_cpu_forward_float_contig_kernel,
                                        .div => &c_api.div_cpu_forward_float_contig_kernel,
                                        else => unreachable,
                                    };
                                    func(
                                        @ptrCast(a.data.?.ptr),
                                        @ptrCast(b.data.?.ptr),
                                        @ptrCast(out.data.?.ptr),
                                        out.data.?.len,
                                    );
                                } else {
                                    const func = switch (ew) {
                                        .add => &c_api.add_cpu_forward_float_kernel,
                                        .sub => &c_api.sub_cpu_forward_float_kernel,
                                        .mul => &c_api.mul_cpu_forward_float_kernel,
                                        .div => &c_api.div_cpu_forward_float_kernel,
                                        else => unreachable,
                                    };
                                    func(
                                        @ptrCast(a.data.?.ptr),
                                        &a.strides,
                                        &a.shape,
                                        a.ndim,
                                        @ptrCast(b.data.?.ptr),
                                        &b.strides,
                                        &b.shape,
                                        b.ndim,
                                        @ptrCast(out.data.?.ptr),
                                        &out.strides,
                                        &out.shape,
                                        out.ndim,
                                    );
                                }
                            },
                            .cuda => {
                                const is_contig =
                                    a.isContiguous() and b.isContiguous() and out.isContiguous() and
                                    std.mem.eql(u64, a.shape[0..a.ndim], b.shape[0..b.ndim]);
                                const num_elements = out.data.?.len;
                                if (is_contig) {
                                    const host_symbol = switch (ew) {
                                        .add => &c_api.add_cuda_forward_float_contig_kernel,
                                        .sub => &c_api.sub_cuda_forward_float_contig_kernel,
                                        .mul => &c_api.mul_cuda_forward_float_contig_kernel,
                                        .div => &c_api.div_cuda_forward_float_contig_kernel,
                                        else => unreachable,
                                    };

                                    const block_size: u32 = 256;
                                    const grid_size: u32 =
                                        @intCast((num_elements + block_size - 1) / block_size);

                                    var arg_a = a.data.?.ptr;
                                    var arg_b = b.data.?.ptr;
                                    var arg_c = out.data.?.ptr;
                                    var arg_num = num_elements;
                                    const args = [_]?*anyopaque{
                                        @ptrCast(&arg_a),
                                        @ptrCast(&arg_b),
                                        @ptrCast(&arg_c),
                                        @ptrCast(&arg_num),
                                    };
                                    try launchCudaKernel(host_symbol, grid_size, block_size, &args);
                                } else {
                                    const host_symbol = switch (ew) {
                                        .add => &c_api.add_cuda_forward_float_non_contig_kernel,
                                        .sub => &c_api.sub_cuda_forward_float_non_contig_kernel,
                                        .mul => &c_api.mul_cuda_forward_float_non_contig_kernel,
                                        .div => &c_api.div_cuda_forward_float_non_contig_kernel,
                                        else => unreachable,
                                    };

                                    const block_size: u32 = 256;
                                    const grid_size: u32 =
                                        @intCast((num_elements + block_size - 1) / block_size);

                                    var dev_a_strides = try copyToDevice(self, a.strides[0..a.ndim]);
                                    var dev_a_shape = try copyToDevice(self, a.shape[0..a.ndim]);
                                    var dev_b_strides = try copyToDevice(self, b.strides[0..b.ndim]);
                                    var dev_b_shape = try copyToDevice(self, b.shape[0..b.ndim]);
                                    var dev_c_strides = try copyToDevice(self, out.strides[0..out.ndim]);
                                    var dev_c_shape = try copyToDevice(self, out.shape[0..out.ndim]);

                                    var arg_a = a.data.?.ptr;
                                    var arg_b = b.data.?.ptr;
                                    var arg_c = out.data.?.ptr;
                                    var arg_ndim_a = a.ndim;
                                    var arg_ndim_b = b.ndim;
                                    var arg_ndim_c = out.ndim;
                                    var arg_num = num_elements;

                                    const args = [_]?*anyopaque{
                                        @ptrCast(&arg_a),   @ptrCast(&dev_a_strides), @ptrCast(&dev_a_shape), @ptrCast(&arg_ndim_a),
                                        @ptrCast(&arg_b),   @ptrCast(&dev_b_strides), @ptrCast(&dev_b_shape), @ptrCast(&arg_ndim_b),
                                        @ptrCast(&arg_c),   @ptrCast(&dev_c_strides), @ptrCast(&dev_c_shape), @ptrCast(&arg_ndim_c),
                                        @ptrCast(&arg_num),
                                    };
                                    try launchCudaKernel(host_symbol, grid_size, block_size, &args);
                                }
                            },
                        }
                    } else {
                        // Unary Op
                        switch (dev.*) {
                            .cpu => {
                                const is_contig = a.isContiguous() and out.isContiguous();
                                if (ew == .relu) {
                                    const alpha: f32 = @floatCast(self.op.params.fval);
                                    if (is_contig) {
                                        c_api.leaky_relu_cpu_forward_float_contig_kernel(
                                            @ptrCast(a.data.?.ptr),
                                            @ptrCast(out.data.?.ptr),
                                            out.data.?.len,
                                            alpha,
                                        );
                                    } else {
                                        c_api.leaky_relu_cpu_forward_float_non_contig_kernel(
                                            @ptrCast(a.data.?.ptr),
                                            &a.strides,
                                            @ptrCast(out.data.?.ptr),
                                            &out.strides,
                                            &a.shape,
                                            a.ndim,
                                            out.data.?.len,
                                            alpha,
                                        );
                                    }
                                } else {
                                    const is_contig_un = a.isContiguous() and out.isContiguous();
                                    if (is_contig_un) {
                                        const func = switch (ew) {
                                            .exp => &c_api.exp_cpu_forward_float_contig_kernel,
                                            .log => &c_api.log_cpu_forward_float_contig_kernel,
                                            .sin => &c_api.sin_cpu_forward_float_contig_kernel,
                                            .cos => &c_api.cos_cpu_forward_float_contig_kernel,
                                            .abs => &c_api.abs_cpu_forward_float_contig_kernel,
                                            .neg => &c_api.neg_cpu_forward_float_contig_kernel,
                                            else => unreachable,
                                        };
                                        func(
                                            @ptrCast(a.data.?.ptr),
                                            @ptrCast(out.data.?.ptr),
                                            out.data.?.len,
                                        );
                                    } else {
                                        const func = switch (ew) {
                                            .exp => &c_api.exp_cpu_forward_float_non_contig_kernel,
                                            .log => &c_api.log_cpu_forward_float_non_contig_kernel,
                                            .sin => &c_api.sin_cpu_forward_float_non_contig_kernel,
                                            .cos => &c_api.cos_cpu_forward_float_non_contig_kernel,
                                            .abs => &c_api.abs_cpu_forward_float_non_contig_kernel,
                                            .neg => &c_api.neg_cpu_forward_float_non_contig_kernel,
                                            else => unreachable,
                                        };
                                        func(
                                            @ptrCast(a.data.?.ptr),
                                            &a.strides,
                                            @ptrCast(out.data.?.ptr),
                                            &out.strides,
                                            &a.shape,
                                            a.ndim,
                                            out.data.?.len,
                                        );
                                    }
                                }
                            },
                            .cuda => {
                                const is_contig = a.isContiguous() and out.isContiguous();
                                const num_elements = out.data.?.len;
                                const block_size: u32 = 256;
                                const grid_size: u32 = @intCast((num_elements + block_size - 1) / block_size);

                                if (ew == .relu) {
                                    var alpha: f32 = @floatCast(self.op.params.fval);
                                    if (is_contig) {
                                        const host_symbol = &c_api.leaky_relu_cuda_forward_float_contig_kernel;

                                        var arg_a = a.data.?.ptr;
                                        var arg_c = out.data.?.ptr;
                                        var arg_num = num_elements;
                                        const args = [_]?*anyopaque{ @ptrCast(&arg_a), @ptrCast(&arg_c), @ptrCast(&arg_num), @ptrCast(&alpha) };
                                        try launchCudaKernel(host_symbol, grid_size, block_size, &args);
                                    } else {
                                        const host_symbol = &c_api.leaky_relu_cuda_forward_float_non_contig_kernel;

                                        var dev_a_strides = try copyToDevice(self, a.strides[0..a.ndim]);
                                        var dev_c_strides = try copyToDevice(self, out.strides[0..out.ndim]);
                                        var dev_shape = try copyToDevice(self, a.shape[0..a.ndim]);

                                        var arg_a = a.data.?.ptr;
                                        var arg_c = out.data.?.ptr;
                                        var arg_ndim = a.ndim;
                                        var arg_num = num_elements;

                                        const args = [_]?*anyopaque{
                                            @ptrCast(&arg_a),     @ptrCast(&dev_a_strides),
                                            @ptrCast(&arg_c),     @ptrCast(&dev_c_strides),
                                            @ptrCast(&dev_shape), @ptrCast(&arg_ndim),
                                            @ptrCast(&arg_num),   @ptrCast(&alpha),
                                        };
                                        try launchCudaKernel(host_symbol, grid_size, block_size, &args);
                                    }
                                } else {
                                    if (is_contig) {
                                        const host_symbol = switch (ew) {
                                            .exp => &c_api.exp_cuda_forward_float_contig_kernel,
                                            .log => &c_api.log_cuda_forward_float_contig_kernel,
                                            .sin => &c_api.sin_cuda_forward_float_contig_kernel,
                                            .cos => &c_api.cos_cuda_forward_float_contig_kernel,
                                            .abs => &c_api.abs_cuda_forward_float_contig_kernel,
                                            .neg => &c_api.neg_cuda_forward_float_contig_kernel,
                                            else => unreachable,
                                        };

                                        var arg_a = a.data.?.ptr;
                                        var arg_c = out.data.?.ptr;
                                        var arg_num = num_elements;
                                        const args = [_]?*anyopaque{ @ptrCast(&arg_a), @ptrCast(&arg_c), @ptrCast(&arg_num) };
                                        try launchCudaKernel(host_symbol, grid_size, block_size, &args);
                                    } else {
                                        const host_symbol = switch (ew) {
                                            .exp => &c_api.exp_cuda_forward_float_non_contig_kernel,
                                            .log => &c_api.log_cuda_forward_float_non_contig_kernel,
                                            .sin => &c_api.sin_cuda_forward_float_non_contig_kernel,
                                            .cos => &c_api.cos_cuda_forward_float_non_contig_kernel,
                                            .abs => &c_api.abs_cuda_forward_float_non_contig_kernel,
                                            .neg => &c_api.neg_cuda_forward_float_non_contig_kernel,
                                            else => unreachable,
                                        };

                                        var dev_a_strides = try copyToDevice(self, a.strides[0..a.ndim]);
                                        var dev_c_strides = try copyToDevice(self, out.strides[0..out.ndim]);
                                        var dev_shape = try copyToDevice(self, a.shape[0..a.ndim]);

                                        var arg_a = a.data.?.ptr;
                                        var arg_c = out.data.?.ptr;
                                        var arg_ndim = a.ndim;
                                        var arg_num = num_elements;

                                        const args = [_]?*anyopaque{
                                            @ptrCast(&arg_a),     @ptrCast(&dev_a_strides),
                                            @ptrCast(&arg_c),     @ptrCast(&dev_c_strides),
                                            @ptrCast(&dev_shape), @ptrCast(&arg_ndim),
                                            @ptrCast(&arg_num),
                                        };
                                        try launchCudaKernel(host_symbol, grid_size, block_size, &args);
                                    }
                                }
                            },
                        }
                    }
                },
                .reduce => |r| {
                    const a = self.inputs[0];
                    const out = self.output;
                    const dim = self.op.params.dim;
                    const keepdim = self.op.params.keepdim != 0;

                    switch (dev.*) {
                        .cpu => {
                            if (dim == c_api.MAX_NDIM + 1) {
                                if (a.isContiguous()) {
                                    const num_elements = a.data.?.len;
                                    switch (r) {
                                        .sum => c_api.sum_cpu_forward_float_contig_kernel(@ptrCast(a.data.?.ptr), @ptrCast(out.data.?.ptr), num_elements),
                                        .mean => c_api.mean_cpu_forward_float_contig_kernel(@ptrCast(a.data.?.ptr), @ptrCast(out.data.?.ptr), num_elements),
                                        .max => c_api.max_cpu_forward_float_contig_kernel(@ptrCast(a.data.?.ptr), @ptrCast(out.data.?.ptr), num_elements),
                                        .min => c_api.min_cpu_forward_float_contig_kernel(@ptrCast(a.data.?.ptr), @ptrCast(out.data.?.ptr), num_elements),
                                    }
                                } else {
                                    const num_elements = a.data.?.len;
                                    switch (r) {
                                        .sum => c_api.sum_cpu_forward_float_non_contig_kernel(@ptrCast(a.data.?.ptr), &a.strides, &a.shape, a.ndim, num_elements, @ptrCast(out.data.?.ptr)),
                                        .mean => c_api.mean_cpu_forward_float_non_contig_kernel(@ptrCast(a.data.?.ptr), &a.strides, &a.shape, a.ndim, num_elements, @ptrCast(out.data.?.ptr)),
                                        .max => c_api.max_cpu_forward_float_non_contig_kernel(@ptrCast(a.data.?.ptr), &a.strides, &a.shape, a.ndim, num_elements, @ptrCast(out.data.?.ptr)),
                                        .min => c_api.min_cpu_forward_float_non_contig_kernel(@ptrCast(a.data.?.ptr), &a.strides, &a.shape, a.ndim, num_elements, @ptrCast(out.data.?.ptr)),
                                    }
                                }
                            } else {
                                switch (r) {
                                    .sum => c_api.sum_cpu_forward_float_dim_kernel(
                                        @ptrCast(a.data.?.ptr),
                                        &a.strides,
                                        &a.shape,
                                        a.ndim,
                                        @ptrCast(out.data.?.ptr),
                                        &out.strides,
                                        &out.shape,
                                        out.ndim,
                                        dim,
                                        keepdim,
                                    ),
                                    .mean => c_api.mean_cpu_forward_float_dim_kernel(
                                        @ptrCast(a.data.?.ptr),
                                        &a.strides,
                                        &a.shape,
                                        a.ndim,
                                        @ptrCast(out.data.?.ptr),
                                        &out.strides,
                                        &out.shape,
                                        out.ndim,
                                        dim,
                                        keepdim,
                                    ),
                                    .max => c_api.max_cpu_forward_float_dim_kernel(
                                        @ptrCast(a.data.?.ptr),
                                        &a.strides,
                                        &a.shape,
                                        a.ndim,
                                        @ptrCast(out.data.?.ptr),
                                        &out.strides,
                                        &out.shape,
                                        out.ndim,
                                        dim,
                                        keepdim,
                                    ),
                                    .min => c_api.min_cpu_forward_float_dim_kernel(
                                        @ptrCast(a.data.?.ptr),
                                        &a.strides,
                                        &a.shape,
                                        a.ndim,
                                        @ptrCast(out.data.?.ptr),
                                        &out.strides,
                                        &out.shape,
                                        out.ndim,
                                        dim,
                                        keepdim,
                                    ),
                                }
                            }
                        },
                        .cuda => {
                            const is_contig = a.isContiguous();
                            switch (r) {
                                .sum => c_api.sum_cuda_forward_direct(a.data.?.ptr, &a.shape, &a.strides, a.ndim, is_contig, out.data.?.ptr, &out.shape, &out.strides, out.ndim, dim, keepdim),
                                .mean => c_api.mean_cuda_forward_direct(a.data.?.ptr, &a.shape, &a.strides, a.ndim, is_contig, out.data.?.ptr, &out.shape, &out.strides, out.ndim, dim, keepdim),
                                .max => c_api.max_cuda_forward_direct(a.data.?.ptr, &a.shape, &a.strides, a.ndim, is_contig, out.data.?.ptr, &out.shape, &out.strides, out.ndim, dim, keepdim),
                                .min => c_api.min_cuda_forward_direct(a.data.?.ptr, &a.shape, &a.strides, a.ndim, is_contig, out.data.?.ptr, &out.shape, &out.strides, out.ndim, dim, keepdim),
                            }
                        },
                    }
                },
                .linear => |l| {
                    const a = self.inputs[0];
                    const b = self.inputs[1];
                    const out = self.output;
                    _ = l;

                    switch (dev.*) {
                        .cpu => {
                            c_api.matmul_cpu_forward_direct(
                                @ptrCast(a.data.?.ptr),
                                &a.shape,
                                &a.strides,
                                a.ndim,
                                @ptrCast(b.data.?.ptr),
                                &b.shape,
                                &b.strides,
                                b.ndim,
                                @ptrCast(out.data.?.ptr),
                                &out.shape,
                                &out.strides,
                                out.ndim,
                            );
                        },
                        .cuda => {
                            c_api.matmul_cuda_forward_direct(
                                a.data.?.ptr,
                                &a.shape,
                                &a.strides,
                                a.ndim,
                                b.data.?.ptr,
                                &b.shape,
                                &b.strides,
                                b.ndim,
                                out.data.?.ptr,
                                &out.shape,
                                &out.strides,
                                out.ndim,
                            );
                        },
                    }
                },
                .fused => |f| {
                    const a = self.inputs[0];
                    const b = self.inputs[1];
                    const out = self.output;
                    const alpha: f32 = @floatCast(self.op.params.fval);
                    _ = f;

                    switch (dev.*) {
                        .cpu => {
                            c_api.matmul_relu_cpu_forward_direct(
                                @ptrCast(a.data.?.ptr),
                                &a.shape,
                                &a.strides,
                                a.ndim,
                                @ptrCast(b.data.?.ptr),
                                &b.shape,
                                &b.strides,
                                b.ndim,
                                @ptrCast(out.data.?.ptr),
                                &out.shape,
                                &out.strides,
                                out.ndim,
                                alpha,
                            );
                        },
                        .cuda => {
                            c_api.matmul_relu_cuda_forward_direct(
                                a.data.?.ptr,
                                &a.shape,
                                &a.strides,
                                a.ndim,
                                b.data.?.ptr,
                                &b.shape,
                                &b.strides,
                                b.ndim,
                                out.data.?.ptr,
                                &out.shape,
                                &out.strides,
                                out.ndim,
                                alpha,
                            );
                        },
                    }
                },
            }
            return self.output;
        }

        pub fn backward(self: *Self) !void {
            const g = self.output.grad orelse return;
            assert(self.output.data != null);
            const dev = self.dev;

            switch (self.op.op_type) {
                .element_wise => |ew| {
                    const a = self.inputs[0];
                    const out = self.output;
                    const dout = g;
                    const da = a.grad;

                    if (self.inputs.len == 2) {
                        const b = self.inputs[1];
                        const db = b.grad;

                        switch (dev.*) {
                            .cpu => {
                                const is_contig = a.isContiguous() and b.isContiguous() and out.isContiguous() and std.mem.eql(u64, a.shape[0..a.ndim], b.shape[0..b.ndim]);
                                if (is_contig) {
                                    const func = switch (ew) {
                                        .add => &c_api.add_cpu_backward_float_contig_kernel,
                                        .sub => &c_api.sub_cpu_backward_float_contig_kernel,
                                        .mul => &c_api.mul_cpu_backward_float_contig_kernel,
                                        .div => &c_api.div_cpu_backward_float_contig_kernel,
                                        else => unreachable,
                                    };
                                    func(
                                        @ptrCast(dout.ptr),
                                        @ptrCast(a.data.?.ptr),
                                        @ptrCast(b.data.?.ptr),
                                        if (da) |v| @ptrCast(v.ptr) else null,
                                        if (db) |v| @ptrCast(v.ptr) else null,
                                        out.data.?.len,
                                    );
                                } else {
                                    const func = switch (ew) {
                                        .add => &c_api.add_cpu_backward_float_kernel,
                                        .sub => &c_api.sub_cpu_backward_float_kernel,
                                        .mul => &c_api.mul_cpu_backward_float_kernel,
                                        .div => &c_api.div_cpu_backward_float_kernel,
                                        else => unreachable,
                                    };
                                    func(
                                        @ptrCast(dout.ptr),
                                        &out.strides,
                                        &out.shape,
                                        out.ndim,
                                        if (da) |v| @ptrCast(v.ptr) else null,
                                        &a.strides,
                                        &a.shape,
                                        a.ndim,
                                        if (db) |v| @ptrCast(v.ptr) else null,
                                        &b.strides,
                                        &b.shape,
                                        b.ndim,
                                        @ptrCast(a.data.?.ptr),
                                        &a.strides,
                                        @ptrCast(b.data.?.ptr),
                                        &b.strides,
                                    );
                                }
                            },
                            .cuda => {
                                const is_contig = a.isContiguous() and b.isContiguous() and out.isContiguous() and std.mem.eql(u64, a.shape[0..a.ndim], b.shape[0..b.ndim]);
                                const num_elements = out.data.?.len;
                                var null_ptr: ?*anyopaque = null;

                                if (is_contig) {
                                    const host_symbol = switch (ew) {
                                        .add => &c_api.add_cuda_backward_float_contig_kernel,
                                        .sub => &c_api.sub_cuda_backward_float_contig_kernel,
                                        .mul => &c_api.mul_cuda_backward_float_contig_kernel,
                                        .div => &c_api.div_cuda_backward_float_contig_kernel,
                                        else => unreachable,
                                    };

                                    const block_size: u32 = 256;
                                    const grid_size: u32 = @intCast((num_elements + block_size - 1) / block_size);

                                    var arg_dout = dout.ptr;
                                    var arg_a = a.data.?.ptr;
                                    var arg_b = b.data.?.ptr;
                                    var arg_da = if (da) |v| v.ptr else null;
                                    var arg_db = if (db) |v| v.ptr else null;
                                    var arg_num = num_elements;

                                    const args = [_]?*anyopaque{
                                        @ptrCast(&arg_dout),                                        @ptrCast(&arg_a),                                           @ptrCast(&arg_b),
                                        if (da != null) @ptrCast(&arg_da) else @ptrCast(&null_ptr), if (db != null) @ptrCast(&arg_db) else @ptrCast(&null_ptr), @ptrCast(&arg_num),
                                    };
                                    try launchCudaKernel(host_symbol, grid_size, block_size, &args);
                                } else {
                                    const host_symbol = switch (ew) {
                                        .add => &c_api.add_cuda_backward_float_non_contig_kernel,
                                        .sub => &c_api.sub_cuda_backward_float_non_contig_kernel,
                                        .mul => &c_api.mul_cuda_backward_float_non_contig_kernel,
                                        .div => &c_api.div_cuda_backward_float_non_contig_kernel,
                                        else => unreachable,
                                    };

                                    const block_size: u32 = 256;
                                    const grid_size: u32 = @intCast((num_elements + block_size - 1) / block_size);

                                    var dev_dout_strides = try copyToDevice(self, out.strides[0..out.ndim]);
                                    var dev_dout_shape = try copyToDevice(self, out.shape[0..out.ndim]);
                                    var dev_a_strides = try copyToDevice(self, a.strides[0..a.ndim]);
                                    var dev_a_shape = try copyToDevice(self, a.shape[0..a.ndim]);
                                    var dev_b_strides = try copyToDevice(self, b.strides[0..b.ndim]);
                                    var dev_b_shape = try copyToDevice(self, b.shape[0..b.ndim]);
                                    var dev_da_strides = try copyToDevice(self, a.strides[0..a.ndim]);
                                    var dev_db_strides = try copyToDevice(self, b.strides[0..b.ndim]);

                                    var arg_dout = dout.ptr;
                                    var arg_dout_ndim = out.ndim;
                                    var arg_a = a.data.?.ptr;
                                    var arg_a_ndim = a.ndim;
                                    var arg_b = b.data.?.ptr;
                                    var arg_b_ndim = b.ndim;
                                    var arg_da = if (da) |v| v.ptr else null;
                                    var arg_db = if (db) |v| v.ptr else null;
                                    var arg_num = num_elements;

                                    const args = [_]?*anyopaque{
                                        @ptrCast(&arg_dout),                                        @ptrCast(&dev_dout_strides), @ptrCast(&dev_dout_shape),                                  @ptrCast(&arg_dout_ndim),
                                        @ptrCast(&arg_a),                                           @ptrCast(&dev_a_strides),    @ptrCast(&dev_a_shape),                                     @ptrCast(&arg_a_ndim),
                                        @ptrCast(&arg_b),                                           @ptrCast(&dev_b_strides),    @ptrCast(&dev_b_shape),                                     @ptrCast(&arg_b_ndim),
                                        if (da != null) @ptrCast(&arg_da) else @ptrCast(&null_ptr), @ptrCast(&dev_da_strides),   if (db != null) @ptrCast(&arg_db) else @ptrCast(&null_ptr), @ptrCast(&dev_db_strides),
                                        @ptrCast(&arg_num),
                                    };
                                    try launchCudaKernel(host_symbol, grid_size, block_size, &args);
                                }
                            },
                        }
                    } else {
                        // Unary backward
                        switch (dev.*) {
                            .cpu => {
                                const is_contig = a.isContiguous() and out.isContiguous();
                                if (ew == .relu) {
                                    const alpha: f32 = @floatCast(self.op.params.fval);
                                    if (is_contig) {
                                        c_api.leaky_relu_cpu_backward_float_contig_kernel(
                                            @ptrCast(dout.ptr),
                                            @ptrCast(a.data.?.ptr),
                                            if (da) |v| @ptrCast(v.ptr) else null,
                                            out.data.?.len,
                                            alpha,
                                        );
                                    } else {
                                        c_api.leaky_relu_cpu_backward_float_non_contig_kernel(
                                            @ptrCast(dout.ptr),
                                            &out.strides,
                                            @ptrCast(a.data.?.ptr),
                                            &a.strides,
                                            if (da) |v| @ptrCast(v.ptr) else null,
                                            &a.strides,
                                            &a.shape,
                                            a.ndim,
                                            out.data.?.len,
                                            alpha,
                                        );
                                    }
                                } else {
                                    if (is_contig) {
                                        const func = switch (ew) {
                                            .exp => &c_api.exp_cpu_backward_float_contig_kernel,
                                            .log => &c_api.log_cpu_backward_float_contig_kernel,
                                            .sin => &c_api.sin_cpu_backward_float_contig_kernel,
                                            .cos => &c_api.cos_cpu_backward_float_contig_kernel,
                                            .abs => &c_api.abs_cpu_backward_float_contig_kernel,
                                            .neg => &c_api.neg_cpu_backward_float_contig_kernel,
                                            else => unreachable,
                                        };
                                        func(
                                            @ptrCast(dout.ptr),
                                            @ptrCast(a.data.?.ptr),
                                            if (da) |v| @ptrCast(v.ptr) else null,
                                            out.data.?.len,
                                        );
                                    } else {
                                        const func = switch (ew) {
                                            .exp => &c_api.exp_cpu_backward_float_non_contig_kernel,
                                            .log => &c_api.log_cpu_backward_float_non_contig_kernel,
                                            .sin => &c_api.sin_cpu_backward_float_non_contig_kernel,
                                            .cos => &c_api.cos_cpu_backward_float_non_contig_kernel,
                                            .abs => &c_api.abs_cpu_backward_float_non_contig_kernel,
                                            .neg => &c_api.neg_cpu_backward_float_non_contig_kernel,
                                            else => unreachable,
                                        };
                                        func(
                                            @ptrCast(dout.ptr),
                                            &out.strides,
                                            @ptrCast(a.data.?.ptr),
                                            &a.strides,
                                            if (da) |v| @ptrCast(v.ptr) else null,
                                            &a.strides,
                                            &a.shape,
                                            a.ndim,
                                            out.data.?.len,
                                        );
                                    }
                                }
                            },
                            .cuda => {
                                const is_contig = a.isContiguous() and out.isContiguous();
                                const num_elements = out.data.?.len;
                                const block_size: u32 = 256;
                                const grid_size: u32 = @intCast((num_elements + block_size - 1) / block_size);
                                var null_ptr: ?*anyopaque = null;

                                if (ew == .relu) {
                                    var alpha: f32 = @floatCast(self.op.params.fval);
                                    if (is_contig) {
                                        const host_symbol = &c_api.leaky_relu_cuda_backward_float_contig_kernel;

                                        var arg_dout = dout.ptr;
                                        var arg_a = a.data.?.ptr;
                                        var arg_da = if (da) |v| v.ptr else null;
                                        var arg_num = num_elements;
                                        const args = [_]?*anyopaque{ @ptrCast(&arg_dout), @ptrCast(&arg_a), if (da != null) @ptrCast(&arg_da) else @ptrCast(&null_ptr), @ptrCast(&arg_num), @ptrCast(&alpha) };
                                        try launchCudaKernel(host_symbol, grid_size, block_size, &args);
                                    } else {
                                        const host_symbol = &c_api.leaky_relu_cuda_backward_float_non_contig_kernel;

                                        var dev_dout_strides = try copyToDevice(self, out.strides[0..out.ndim]);
                                        var dev_a_strides = try copyToDevice(self, a.strides[0..a.ndim]);
                                        var dev_da_strides = try copyToDevice(self, a.strides[0..a.ndim]);
                                        var dev_shape = try copyToDevice(self, a.shape[0..a.ndim]);

                                        var arg_dout = dout.ptr;
                                        var arg_a = a.data.?.ptr;
                                        var arg_da = if (da) |v| v.ptr else null;
                                        var arg_ndim = a.ndim;
                                        var arg_num = num_elements;

                                        const args = [_]?*anyopaque{
                                            @ptrCast(&arg_dout),                                        @ptrCast(&dev_dout_strides),
                                            @ptrCast(&arg_a),                                           @ptrCast(&dev_a_strides),
                                            if (da != null) @ptrCast(&arg_da) else @ptrCast(&null_ptr), @ptrCast(&dev_da_strides),
                                            @ptrCast(&dev_shape),                                       @ptrCast(&arg_ndim),
                                            @ptrCast(&arg_num),                                         @ptrCast(&alpha),
                                        };
                                        try launchCudaKernel(host_symbol, grid_size, block_size, &args);
                                    }
                                } else {
                                    if (is_contig) {
                                        const host_symbol = switch (ew) {
                                            .exp => &c_api.exp_cuda_backward_float_contig_kernel,
                                            .log => &c_api.log_cuda_backward_float_contig_kernel,
                                            .sin => &c_api.sin_cuda_backward_float_contig_kernel,
                                            .cos => &c_api.cos_cuda_backward_float_contig_kernel,
                                            .abs => &c_api.abs_cuda_backward_float_contig_kernel,
                                            .neg => &c_api.neg_cuda_backward_float_contig_kernel,
                                            else => unreachable,
                                        };

                                        var arg_dout = dout.ptr;
                                        var arg_a = a.data.?.ptr;
                                        var arg_da = if (da) |v| v.ptr else null;
                                        var arg_num = num_elements;
                                        const args = [_]?*anyopaque{ @ptrCast(&arg_dout), @ptrCast(&arg_a), if (da != null) @ptrCast(&arg_da) else @ptrCast(&null_ptr), @ptrCast(&arg_num) };
                                        try launchCudaKernel(host_symbol, grid_size, block_size, &args);
                                    } else {
                                        const host_symbol = switch (ew) {
                                            .exp => &c_api.exp_cuda_backward_float_non_contig_kernel,
                                            .log => &c_api.log_cuda_backward_float_non_contig_kernel,
                                            .sin => &c_api.sin_cuda_backward_float_non_contig_kernel,
                                            .cos => &c_api.cos_cuda_backward_float_non_contig_kernel,
                                            .abs => &c_api.abs_cuda_backward_float_non_contig_kernel,
                                            .neg => &c_api.neg_cuda_backward_float_non_contig_kernel,
                                            else => unreachable,
                                        };

                                        var dev_dout_strides = try copyToDevice(self, out.strides[0..out.ndim]);
                                        var dev_a_strides = try copyToDevice(self, a.strides[0..a.ndim]);
                                        var dev_da_strides = try copyToDevice(self, a.strides[0..a.ndim]);
                                        var dev_shape = try copyToDevice(self, a.shape[0..a.ndim]);

                                        var arg_dout = dout.ptr;
                                        var arg_a = a.data.?.ptr;
                                        var arg_da = if (da) |v| v.ptr else null;
                                        var arg_ndim = a.ndim;
                                        var arg_num = num_elements;

                                        const args = [_]?*anyopaque{
                                            @ptrCast(&arg_dout),                                        @ptrCast(&dev_dout_strides),
                                            @ptrCast(&arg_a),                                           @ptrCast(&dev_a_strides),
                                            if (da != null) @ptrCast(&arg_da) else @ptrCast(&null_ptr), @ptrCast(&dev_da_strides),
                                            @ptrCast(&dev_shape),                                       @ptrCast(&arg_ndim),
                                            @ptrCast(&arg_num),
                                        };
                                        try launchCudaKernel(host_symbol, grid_size, block_size, &args);
                                    }
                                }
                            },
                        }
                    }
                },
                .reduce => |r| {
                    const a = self.inputs[0];
                    const out = self.output;
                    const dout = g;
                    const da = a.grad orelse return;
                    const dim = self.op.params.dim;
                    const keepdim = self.op.params.keepdim != 0;

                    switch (dev.*) {
                        .cpu => {
                            if (dim == c_api.MAX_NDIM + 1) {
                                if (a.isContiguous()) {
                                    const num_elements = a.data.?.len;
                                    switch (r) {
                                        .sum => c_api.sum_cpu_backward_float_contig_kernel(@ptrCast(dout.ptr), @ptrCast(a.data.?.ptr), @ptrCast(da.ptr), num_elements),
                                        .mean => c_api.mean_cpu_backward_float_contig_kernel(@ptrCast(dout.ptr), @ptrCast(a.data.?.ptr), @ptrCast(da.ptr), num_elements),
                                        .max => c_api.max_cpu_backward_float_contig_kernel(@ptrCast(dout.ptr), @ptrCast(a.data.?.ptr), @ptrCast(da.ptr), num_elements),
                                        .min => c_api.min_cpu_backward_float_contig_kernel(@ptrCast(dout.ptr), @ptrCast(a.data.?.ptr), @ptrCast(da.ptr), num_elements),
                                    }
                                } else {
                                    const num_elements = a.data.?.len;
                                    switch (r) {
                                        .sum => c_api.sum_cpu_backward_float_non_contig_kernel(@ptrCast(dout.ptr), @ptrCast(da.ptr), &a.strides, &a.shape, a.ndim, num_elements),
                                        .mean => c_api.mean_cpu_backward_float_non_contig_kernel(@ptrCast(dout.ptr), @ptrCast(da.ptr), &a.strides, &a.shape, a.ndim, num_elements),
                                        .max => c_api.max_cpu_backward_float_non_contig_kernel(@ptrCast(a.data.?.ptr), &a.strides, @ptrCast(out.data.?.ptr), @ptrCast(dout.ptr), @ptrCast(da.ptr), &a.strides, &a.shape, a.ndim, num_elements),
                                        .min => c_api.min_cpu_backward_float_non_contig_kernel(@ptrCast(a.data.?.ptr), &a.strides, @ptrCast(out.data.?.ptr), @ptrCast(dout.ptr), @ptrCast(da.ptr), &a.strides, &a.shape, a.ndim, num_elements),
                                    }
                                }
                            } else {
                                switch (r) {
                                    .sum => c_api.sum_cpu_backward_float_dim_kernel(
                                        @ptrCast(a.data.?.ptr),
                                        &a.strides,
                                        &a.shape,
                                        a.ndim,
                                        @ptrCast(out.data.?.ptr),
                                        &out.strides,
                                        &out.shape,
                                        out.ndim,
                                        @ptrCast(dout.ptr),
                                        @ptrCast(da.ptr),
                                        &a.strides,
                                        dim,
                                        keepdim,
                                    ),
                                    .mean => c_api.mean_cpu_backward_float_dim_kernel(
                                        @ptrCast(a.data.?.ptr),
                                        &a.strides,
                                        &a.shape,
                                        a.ndim,
                                        @ptrCast(out.data.?.ptr),
                                        &out.strides,
                                        &out.shape,
                                        out.ndim,
                                        @ptrCast(dout.ptr),
                                        @ptrCast(da.ptr),
                                        &a.strides,
                                        dim,
                                        keepdim,
                                    ),
                                    .max => c_api.max_cpu_backward_float_dim_kernel(
                                        @ptrCast(a.data.?.ptr),
                                        &a.strides,
                                        &a.shape,
                                        a.ndim,
                                        @ptrCast(out.data.?.ptr),
                                        &out.strides,
                                        &out.shape,
                                        out.ndim,
                                        @ptrCast(dout.ptr),
                                        @ptrCast(da.ptr),
                                        &a.strides,
                                        dim,
                                        keepdim,
                                    ),
                                    .min => c_api.min_cpu_backward_float_dim_kernel(
                                        @ptrCast(a.data.?.ptr),
                                        &a.strides,
                                        &a.shape,
                                        a.ndim,
                                        @ptrCast(out.data.?.ptr),
                                        &out.strides,
                                        &out.shape,
                                        out.ndim,
                                        @ptrCast(dout.ptr),
                                        @ptrCast(da.ptr),
                                        &a.strides,
                                        dim,
                                        keepdim,
                                    ),
                                }
                            }
                        },
                        .cuda => {
                            const da_is_contig = a.isContiguous();
                            switch (r) {
                                .sum => c_api.sum_mean_cuda_backward_direct(dout.ptr, &out.shape, &out.strides, out.ndim, da.ptr, &a.shape, &a.strides, a.ndim, da_is_contig, dim, keepdim, false),
                                .mean => c_api.sum_mean_cuda_backward_direct(dout.ptr, &out.shape, &out.strides, out.ndim, da.ptr, &a.shape, &a.strides, a.ndim, da_is_contig, dim, keepdim, true),
                                .max => c_api.max_min_cuda_backward_direct(dout.ptr, &out.shape, &out.strides, out.ndim, da.ptr, &a.shape, &a.strides, a.ndim, da_is_contig, a.data.?.ptr, out.data.?.ptr, dim, keepdim),
                                .min => c_api.max_min_cuda_backward_direct(dout.ptr, &out.shape, &out.strides, out.ndim, da.ptr, &a.shape, &a.strides, a.ndim, da_is_contig, a.data.?.ptr, out.data.?.ptr, dim, keepdim),
                            }
                        },
                    }
                },
                .linear => {
                    const a = self.inputs[0];
                    const b = self.inputs[1];
                    const out = self.output;
                    const dout = g;
                    const da = a.grad;
                    const db = b.grad;

                    switch (dev.*) {
                        .cpu => {
                            c_api.matmul_cpu_backward_direct(
                                @ptrCast(a.data.?.ptr),
                                &a.shape,
                                &a.strides,
                                a.ndim,
                                if (da) |v| @ptrCast(v.ptr) else null,
                                &a.strides,
                                da != null,
                                @ptrCast(b.data.?.ptr),
                                &b.shape,
                                &b.strides,
                                b.ndim,
                                if (db) |v| @ptrCast(v.ptr) else null,
                                &b.strides,
                                db != null,
                                @ptrCast(dout.ptr),
                                &out.shape,
                                &out.strides,
                                out.ndim,
                            );
                        },
                        .cuda => {
                            c_api.matmul_cuda_backward_direct(
                                a.data.?.ptr,
                                &a.shape,
                                &a.strides,
                                a.ndim,
                                if (da) |v| v.ptr else null,
                                &a.strides,
                                da != null,
                                b.data.?.ptr,
                                &b.shape,
                                &b.strides,
                                b.ndim,
                                if (db) |v| v.ptr else null,
                                &b.strides,
                                db != null,
                                dout.ptr,
                                &out.shape,
                                &out.strides,
                                out.ndim,
                            );
                        },
                    }
                },
                .fused => {
                    const a = self.inputs[0];
                    const b = self.inputs[1];
                    const out = self.output;
                    const dout = g;
                    const da = a.grad;
                    const db = b.grad;
                    const alpha: f32 = @floatCast(self.op.params.fval);

                    switch (dev.*) {
                        .cpu => {
                            c_api.matmul_relu_cpu_backward_direct(
                                @ptrCast(a.data.?.ptr),
                                &a.shape,
                                &a.strides,
                                a.ndim,
                                if (da) |v| @ptrCast(v.ptr) else null,
                                &a.strides,
                                da != null,
                                @ptrCast(b.data.?.ptr),
                                &b.shape,
                                &b.strides,
                                b.ndim,
                                if (db) |v| @ptrCast(v.ptr) else null,
                                &b.strides,
                                db != null,
                                @ptrCast(out.data.?.ptr),
                                &out.shape,
                                &out.strides,
                                out.ndim,
                                @ptrCast(dout.ptr),
                                &out.shape,
                                &out.strides,
                                out.ndim,
                                alpha,
                            );
                        },
                        .cuda => {
                            c_api.matmul_relu_cuda_backward_direct(
                                a.data.?.ptr,
                                &a.shape,
                                &a.strides,
                                a.ndim,
                                if (da) |v| v.ptr else null,
                                &a.strides,
                                da != null,
                                b.data.?.ptr,
                                &b.shape,
                                &b.strides,
                                b.ndim,
                                if (db) |v| v.ptr else null,
                                &b.strides,
                                db != null,
                                out.data.?.ptr,
                                &out.shape,
                                &out.strides,
                                out.ndim,
                                dout.ptr,
                                &out.shape,
                                &out.strides,
                                out.ndim,
                                alpha,
                            );
                        },
                    }
                },
            }
        }
    };
}
