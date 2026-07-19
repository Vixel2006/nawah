pub const sgd = @import("sgd.zig");
pub const adam = @import("adam.zig");
pub const adamw = @import("adamw.zig");
pub const rmsprop = @import("rmsprop.zig");

pub const SGD = sgd.SGD;
pub const Adam = adam.Adam;
pub const AdamW = adamw.AdamW;
pub const RMSprop = rmsprop.RMSprop;

test {
    _ = sgd;
    _ = adam;
    _ = adamw;
    _ = rmsprop;
}
