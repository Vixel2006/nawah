const c_api = @import("c_api.zig");

pub const Function = struct {
    forward: *const fn (inputs: [*c]?*const c_api.C_Tensor, output: *c_api.C_Tensor, params: c_api.KernelParams) callconv(.c) void,
    backward: *const fn (inputs: [*c]?*const c_api.C_Tensor, output: *const c_api.C_Tensor, params: c_api.KernelParams) callconv(.c) void,
};
