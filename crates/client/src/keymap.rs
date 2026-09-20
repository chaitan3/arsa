//! Re-export: the evdev <-> qnum table lives in the proto crate (shared with
//! the server's input sinks — QEMU D-Bus consumes qnums natively, the uinput
//! sink needs the evdev code back).
pub use rdma_display_proto::keymap::linux_to_qnum;
