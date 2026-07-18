const c_api = @import("c_api.zig");
const ops = @import("ops/mod.zig");
const Function = @import("function.zig").Function;

pub const OpType = union(enum) {
    element_wise: ops.element_wise.ElementWise,
    reduce: ops.reduce.Reduce,
    linear: ops.linear.Linear,
};

pub const Op = struct {
    op_type: OpType,
    params: c_api.KernelParams,
    function: Function,
};

pub fn getFunction(op_type: OpType, dtype: u32, device: u32) Function {
    return switch (op_type) {
        .element_wise => |ew| ops.element_wise.getElementWiseFn(ew, dtype, device),
        .reduce => |r| ops.reduce.getReduceFn(r, dtype, device),
        .linear => |l| ops.linear.getLinearFn(l, dtype, device),
    };
}
