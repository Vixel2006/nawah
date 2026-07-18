const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Op = @import("op.zig").Op;
const c_api = @import("c_api.zig");

pub fn Node(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: std.mem.Allocator,
        inputs: []*Tensor(T),
        output: *Tensor(T),
        op: Op,
        visited: bool = false,

        pub fn init(self: *Self, alloc: std.mem.Allocator, inputs: []*Tensor(T), output: *Tensor(T), op: Op) void {
            self.* = .{ .alloc = alloc, .inputs = inputs, .output = output, .op = op };
            output.creator = self;
        }

        pub fn deinit(self: *Self) void {
            self.alloc.free(self.inputs);
            self.output.deinit();
            self.alloc.destroy(self.output);
        }

        pub fn forward(self: *Self, allocator: std.mem.Allocator) !*Tensor(T) {
            var c_out = c_api.toCTensor(@ptrCast(self.output.data.?), &self.output.shape, &self.output.strides, self.output.ndim, self.output.requires_grad);
            
            const c_inputs = try allocator.alloc(?*const c_api.C_Tensor, self.inputs.len);
            defer allocator.free(c_inputs);
            const c_inputs_tensors = try allocator.alloc(c_api.C_Tensor, self.inputs.len);
            defer allocator.free(c_inputs_tensors);

            for (self.inputs, 0..) |inp, i| {
                c_inputs_tensors[i] = c_api.toCTensor(@ptrCast(inp.data.?), &inp.shape, &inp.strides, inp.ndim, inp.requires_grad);
                c_inputs[i] = &c_inputs_tensors[i];
            }
            self.op.function.forward(@ptrCast(c_inputs.ptr), &c_out, self.op.params);
            return self.output;
        }

        pub fn backward(self: *Self) !void {
            if (self.output.grad) |g| {
                var c_out = c_api.toCTensor(@ptrCast(self.output.data.?), &self.output.shape, &self.output.strides, self.output.ndim, self.output.requires_grad);
                var c_out_grad = c_api.toCTensor(@ptrCast(g.data.?), &g.shape, &g.strides, g.ndim, g.requires_grad);
                c_out.grad = &c_out_grad;

                const c_inputs = try self.alloc.alloc(?*const c_api.C_Tensor, self.inputs.len);
                defer self.alloc.free(c_inputs);
                const c_inputs_tensors = try self.alloc.alloc(c_api.C_Tensor, self.inputs.len);
                defer self.alloc.free(c_inputs_tensors);
                const c_inputs_grads = try self.alloc.alloc(c_api.C_Tensor, self.inputs.len);
                defer self.alloc.free(c_inputs_grads);

                for (self.inputs, 0..) |inp, i| {
                    if (inp.requires_grad and inp.grad == null) {
                        const ig = try self.alloc.create(Tensor(T));
                        ig.* = try Tensor(T).zeros(self.alloc, inp.shape[0..inp.ndim], false);
                        inp.grad = ig;
                    }
                    c_inputs_tensors[i] = c_api.toCTensor(@ptrCast(inp.data.?), &inp.shape, &inp.strides, inp.ndim, inp.requires_grad);
                    if (inp.grad) |ig| {
                        c_inputs_grads[i] = c_api.toCTensor(@ptrCast(ig.data.?), &ig.shape, &ig.strides, ig.ndim, ig.requires_grad);
                        c_inputs_tensors[i].grad = &c_inputs_grads[i];
                    }
                    c_inputs[i] = &c_inputs_tensors[i];
                }
                self.op.function.backward(@ptrCast(c_inputs.ptr), &c_out, self.op.params);
            }
        }
    };
}
