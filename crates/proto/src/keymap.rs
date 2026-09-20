//! Linux evdev keycode <-> QEMU "key number" (qnum) translation table.
//!
//! Generated from keycodemapdb `keymaps.csv` — the same source QEMU uses
//! to generate `qemu_input_map_linux_to_qnum` (the "Xfree86 KBD XT"
//! column = the qnum encoding: "xtkbd + special re-encoding of high
//! bit"). Sorted by Linux keycode.
//!
//! The wire protocol (`InputEvent::Key`) carries qnums; the client translates
//! from Wayland's evdev codes with [`linux_to_qnum`], and input sinks
//! translate back with [`qnum_to_linux`] (QEMU D-Bus speaks qnum natively;
//! the uinput sink needs the evdev code again).

pub const LINUX_TO_QNUM: &[(u16, u16)] = &[
    (1, 0x1), // KEY_ESC
    (2, 0x2), // KEY_1
    (3, 0x3), // KEY_2
    (4, 0x4), // KEY_3
    (5, 0x5), // KEY_4
    (6, 0x6), // KEY_5
    (7, 0x7), // KEY_6
    (8, 0x8), // KEY_7
    (9, 0x9), // KEY_8
    (10, 0xa), // KEY_9
    (11, 0xb), // KEY_0
    (12, 0xc), // KEY_MINUS
    (13, 0xd), // KEY_EQUAL
    (14, 0xe), // KEY_BACKSPACE
    (15, 0xf), // KEY_TAB
    (16, 0x10), // KEY_Q
    (17, 0x11), // KEY_W
    (18, 0x12), // KEY_E
    (19, 0x13), // KEY_R
    (20, 0x14), // KEY_T
    (21, 0x15), // KEY_Y
    (22, 0x16), // KEY_U
    (23, 0x17), // KEY_I
    (24, 0x18), // KEY_O
    (25, 0x19), // KEY_P
    (26, 0x1a), // KEY_LEFTBRACE
    (27, 0x1b), // KEY_RIGHTBRACE
    (28, 0x1c), // KEY_ENTER
    (29, 0x1d), // KEY_LEFTCTRL
    (30, 0x1e), // KEY_A
    (31, 0x1f), // KEY_S
    (32, 0x20), // KEY_D
    (33, 0x21), // KEY_F
    (34, 0x22), // KEY_G
    (35, 0x23), // KEY_H
    (36, 0x24), // KEY_J
    (37, 0x25), // KEY_K
    (38, 0x26), // KEY_L
    (39, 0x27), // KEY_SEMICOLON
    (40, 0x28), // KEY_APOSTROPHE
    (41, 0x29), // KEY_GRAVE
    (42, 0x2a), // KEY_SHIFT
    (43, 0x2b), // KEY_BACKSLASH
    (44, 0x2c), // KEY_Z
    (45, 0x2d), // KEY_X
    (46, 0x2e), // KEY_C
    (47, 0x2f), // KEY_V
    (48, 0x30), // KEY_B
    (49, 0x31), // KEY_N
    (50, 0x32), // KEY_M
    (51, 0x33), // KEY_COMMA
    (52, 0x34), // KEY_DOT
    (53, 0x35), // KEY_SLASH
    (54, 0x36), // KEY_RIGHTSHIFT
    (55, 0x37), // KEY_KPASTERISK
    (56, 0x38), // KEY_LEFTALT
    (57, 0x39), // KEY_SPACE
    (58, 0x3a), // KEY_CAPSLOCK
    (59, 0x3b), // KEY_F1
    (60, 0x3c), // KEY_F2
    (61, 0x3d), // KEY_F3
    (62, 0x3e), // KEY_F4
    (63, 0x3f), // KEY_F5
    (64, 0x40), // KEY_F6
    (65, 0x41), // KEY_F7
    (66, 0x42), // KEY_F8
    (67, 0x43), // KEY_F9
    (68, 0x44), // KEY_F10
    (69, 0x45), // KEY_NUMLOCK
    (70, 0x46), // KEY_SCROLLLOCK
    (71, 0x47), // KEY_KP7
    (72, 0x48), // KEY_KP8
    (73, 0x49), // KEY_KP9
    (74, 0x4a), // KEY_KPMINUS
    (75, 0x4b), // KEY_KP4
    (76, 0x4c), // KEY_KP5
    (77, 0x4d), // KEY_KP6
    (78, 0x4e), // KEY_KPPLUS
    (79, 0x4f), // KEY_KP1
    (80, 0x50), // KEY_KP2
    (81, 0x51), // KEY_KP3
    (82, 0x52), // KEY_KP0
    (83, 0x53), // KEY_KPDOT
    (86, 0x56), // KEY_102ND
    (87, 0x57), // KEY_F11
    (88, 0x58), // KEY_F12
    (89, 0x73), // KEY_RO
    (92, 0x79), // KEY_HENKAN
    (93, 0x70), // KEY_KATAKANAHIRAGANA
    (94, 0x7b), // KEY_MUHENKAN
    (96, 0x64), // KEY_KPENTER
    (97, 0x65), // KEY_RIGHTCTRL
    (98, 0x68), // KEY_KPSLASH
    (99, 0x67), // KEY_SYSRQ
    (100, 0x69), // KEY_RIGHTALT
    (102, 0x59), // KEY_HOME
    (103, 0x5a), // KEY_UP
    (104, 0x5b), // KEY_PAGEUP
    (105, 0x5c), // KEY_LEFT
    (106, 0x5e), // KEY_RIGHT
    (107, 0x5f), // KEY_END
    (108, 0x60), // KEY_DOWN
    (109, 0x61), // KEY_PAGEDOWN
    (110, 0x62), // KEY_INSERT
    (111, 0x63), // KEY_DELETE
    (117, 0x76), // KEY_KPEQUAL
    (119, 0x66), // KEY_PAUSE
    (122, 0x72), // KEY_HANGEUL
    (123, 0x71), // KEY_HANJA
    (124, 0x7d), // KEY_YEN
    (125, 0x6b), // KEY_LEFTMETA
    (126, 0x6c), // KEY_RIGHTMETA
    (127, 0x6d), // KEY_COMPOSE
    (183, 0x6e), // KEY_F13
    (184, 0x6f), // KEY_F14
    (185, 0x70), // KEY_F15
    (186, 0x71), // KEY_F16
    (187, 0x72), // KEY_F17
];

/// Translate a Linux evdev keycode (as delivered by `wl_keyboard::key`)
/// into the QEMU "key number" expected by
/// `org.qemu.Display1.Keyboard.Press/Release`.
pub fn linux_to_qnum(evdev: u32) -> Option<u32> {
    LINUX_TO_QNUM
        .binary_search_by_key(&(evdev as u16), |&(k, _)| k)
        .ok()
        .map(|i| LINUX_TO_QNUM[i].1 as u32)
}

/// Inverse of [`linux_to_qnum`]: a wire qnum back to the Linux evdev keycode
/// (used by the uinput input sink, which injects evdev events into the host).
/// First mapping wins on the (practically empty) collision set.
pub fn qnum_to_linux(qnum: u32) -> Option<u32> {
    LINUX_TO_QNUM
        .iter()
        .find(|&&(_, q)| q == qnum as u16)
        .map(|&(k, _)| k as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        // linux -> qnum must invert exactly; qnum -> linux is first-wins
        // (two evdev keys can share a qnum — e.g. 0x70 is both
        // KATAKANAHIRAGANA and F15 in the XT mapping)
        for &(linux, qnum) in LINUX_TO_QNUM {
            assert_eq!(qnum_to_linux(qnum as u32).and_then(linux_to_qnum), Some(qnum as u32));
            assert_eq!(linux_to_qnum(linux as u32), Some(qnum as u32));
        }
        assert_eq!(linux_to_qnum(30), Some(0x1e)); // KEY_A
        assert_eq!(qnum_to_linux(0x1e), Some(30));
        assert_eq!(qnum_to_linux(0x1e5), None);
    }
}
