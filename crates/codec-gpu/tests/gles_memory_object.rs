//! Probes for the server's strict zero-copy output path.
//!
//! The lossless GPU codec writes its coded stream into a GL shader-storage
//! buffer. For the NIC to read that stream directly, the storage must be a
//! registered dma-buf. There are two ways to get there on the server's GLES
//! context:
//!
//! * `GL_EXT_memory_object_fd` — import the dma-buf as a GL memory object and
//!   create the SSBO from it (keeps the codec's SSBO shaders);
//! * `EGL_EXT_image_dma_buf_import` + `glEGLImageTargetTexture2DOES` — import
//!   the dma-buf as a texture and have the codec write it as an image.
//!
//! These tests report which one works on the actual driver. They are ignored
//! by default:
//!
//! ```text
//! cargo test -p rdma-display-codec-gpu --test gles_memory_object \
//!     -- --ignored --nocapture
//! ```
//!
//! `ARSA_DRM_NODE` selects the DRM node used for the GBM source.

use std::ffi::{c_void, CString};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use rdma_display_egl::{FnGetProcAddress, SharedContext};

const GL_EXTENSIONS: u32 = 0x1F03;
const GL_SHADER_STORAGE_BUFFER: u32 = 0x90D2;
const GL_TEXTURE_2D: u32 = 0x0DE1;
const GL_RGBA8: u32 = 0x8058;
const GL_READ_WRITE: u32 = 0x88BA;
const GL_HANDLE_TYPE_OPAQUE_FD_EXT: u32 = 0x9586;
const GL_MAP_READ_BIT: u32 = 0x0001;

// EGL_EXT_image_dma_buf_import attribute list (same values the codec uses).
const EGL_NONE: i32 = 0x3038;
const EGL_WIDTH: i32 = 0x3057;
const EGL_HEIGHT: i32 = 0x3056;
const EGL_LINUX_DMA_BUF_EXT: u32 = 0x3270;
const EGL_LINUX_DRM_FOURCC_EXT: i32 = 0x3271;
const EGL_DMA_BUF_PLANE0_FD_EXT: i32 = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: i32 = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: i32 = 0x3274;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: i32 = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: i32 = 0x3444;

type GlGetString = unsafe extern "system" fn(u32) -> *const u8;
type GlGetError = unsafe extern "system" fn() -> u32;
type GlGenBuffers = unsafe extern "system" fn(i32, *mut u32);
type GlBindBuffer = unsafe extern "system" fn(u32, u32);
type GlBufferStorageMem = unsafe extern "system" fn(u32, isize, u32, u64);
type GlBufferSubData = unsafe extern "system" fn(u32, isize, isize, *const c_void);
type GlMapBufferRange = unsafe extern "system" fn(u32, isize, isize, u32) -> *mut c_void;
type GlUnmapBuffer = unsafe extern "system" fn(u32) -> u8;
type GlDeleteBuffers = unsafe extern "system" fn(i32, *const u32);
type GlCreateMemoryObjects = unsafe extern "system" fn(i32, *mut u32);
type GlImportMemoryFd = unsafe extern "system" fn(u32, u64, u32, i32);
type GlDeleteMemoryObjects = unsafe extern "system" fn(i32, *const u32);
type GlGenTextures = unsafe extern "system" fn(i32, *mut u32);
type GlBindTexture = unsafe extern "system" fn(u32, u32);
type GlDeleteTextures = unsafe extern "system" fn(i32, *const u32);
type GlEglImageTargetTexture = unsafe extern "system" fn(u32, *mut c_void);
type GlBindImageTexture = unsafe extern "system" fn(u32, u32, i32, u8, i32, u32, u32);

fn sym<T: Copy>(get: FnGetProcAddress, name: &str) -> Option<T> {
    let cname = CString::new(name).ok()?;
    let p = unsafe { get(cname.as_ptr() as *const u8) };
    if p.is_null() {
        return None;
    }
    Some(unsafe { std::mem::transmute_copy::<*const c_void, T>(&p) })
}

#[repr(C)]
struct DmaHeapAlloc {
    len: u64,
    fd_flags: u32,
    heap_flags: u32,
    fd: u32,
}

const DMA_HEAP_IOCTL_ALLOC: libc::c_ulong = 0xc018_4800;

/// Allocate `len` bytes from `/dev/dma_heap/system`.
fn dma_heap_alloc(len: usize) -> Option<OwnedFd> {
    let path = CString::new("/dev/dma_heap/system").ok()?;
    let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
    if fd < 0 {
        return None;
    }
    let mut data = DmaHeapAlloc {
        len: len as u64,
        fd_flags: (libc::O_RDWR | libc::O_CLOEXEC) as u32,
        heap_flags: 0,
        fd: 0,
    };
    let rc = unsafe { libc::ioctl(fd, DMA_HEAP_IOCTL_ALLOC, &mut data) };
    unsafe { libc::close(fd) };
    if rc != 0 || data.fd == 0 {
        return None;
    }
    Some(unsafe { OwnedFd::from_raw_fd(data.fd as i32) })
}

/// A linear GBM bo exported as a dma-buf, with the layout an EGLImage needs.
struct GbmBuf {
    fd: OwnedFd,
    offset: u64,
    stride: usize,
    width: u32,
    height: u32,
    modifier: u64,
    size: usize,
}

type GbmCreateDevice = unsafe extern "C" fn(i32) -> *mut c_void;
type GbmDeviceDestroy = unsafe extern "C" fn(*mut c_void);
type GbmBoCreateMods =
    unsafe extern "C" fn(*mut c_void, u32, u32, u32, *const u64, u32) -> *mut c_void;
type GbmBoCreatePlain = unsafe extern "C" fn(*mut c_void, u32, u32, u32, u32) -> *mut c_void;
type GbmBoGetFd = unsafe extern "C" fn(*mut c_void) -> i32;
type GbmBoGetStride = unsafe extern "C" fn(*mut c_void) -> u32;
type GbmBoGetHeight = unsafe extern "C" fn(*mut c_void) -> u32;
type GbmBoGetOffset = unsafe extern "C" fn(*mut c_void, i32) -> u32;
type GbmBoGetModifier = unsafe extern "C" fn(*mut c_void) -> u64;
type GbmBoDestroy = unsafe extern "C" fn(*mut c_void);

fn gbm_buffer(width: u32, height: u32) -> Option<GbmBuf> {
    let node = std::env::var("ARSA_DRM_NODE").ok().or_else(|| {
        ["/dev/dri/renderD128", "/dev/dri/renderD129", "/dev/dri/card0"]
            .iter()
            .find(|p| {
                std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(p)
                    .is_ok()
            })
            .map(|s| s.to_string())
    })?;
    let drm = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&node)
        .ok()?;
    let lib = unsafe { libloading::Library::new("libgbm.so.1") }.ok()?;
    unsafe {
        let create: GbmCreateDevice = *lib.get(b"gbm_create_device").ok()?;
        let dev_destroy: GbmDeviceDestroy = *lib.get(b"gbm_device_destroy").ok()?;
        let create_mods: Option<GbmBoCreateMods> =
            lib.get(b"gbm_bo_create_with_modifiers").ok().map(|s| *s);
        let create_plain: GbmBoCreatePlain = *lib.get(b"gbm_bo_create").ok()?;
        let get_fd: GbmBoGetFd = *lib.get(b"gbm_bo_get_fd").ok()?;
        let get_stride: GbmBoGetStride = *lib.get(b"gbm_bo_get_stride").ok()?;
        let get_height: GbmBoGetHeight = *lib.get(b"gbm_bo_get_height").ok()?;
        let get_offset: GbmBoGetOffset = *lib.get(b"gbm_bo_get_offset").ok()?;
        let get_modifier: GbmBoGetModifier = *lib.get(b"gbm_bo_get_modifier").ok()?;
        let bo_destroy: GbmBoDestroy = *lib.get(b"gbm_bo_destroy").ok()?;

        let dev = create(drm.as_raw_fd());
        if dev.is_null() {
            return None;
        }
        let mods = [0u64];
        let mut bo = std::ptr::null_mut();
        if let Some(f) = create_mods {
            bo = f(dev, width, height, 0x3432_5258, mods.as_ptr(), 1);
        }
        if bo.is_null() {
            bo = create_plain(dev, width, height, 0x3432_5258, 16 | 4);
        }
        if bo.is_null() {
            dev_destroy(dev);
            return None;
        }
        let modifier = get_modifier(bo);
        let stride = get_stride(bo) as usize;
        let height = get_height(bo);
        let offset = u64::from(get_offset(bo, 0));
        let raw = get_fd(bo);
        bo_destroy(bo);
        dev_destroy(dev);
        if raw < 0 || modifier != 0 {
            return None;
        }
        let fd = OwnedFd::from_raw_fd(raw);
        Some(GbmBuf {
            fd,
            offset,
            stride,
            width,
            height,
            modifier,
            size: stride * height as usize,
        })
    }
}

fn print_gl_info(get_string: GlGetString) {
    for (label, name) in [("vendor", 0x1F00u32), ("renderer", 0x1F01), ("version", 0x1F02)] {
        let p = unsafe { get_string(name) };
        if !p.is_null() {
            let s = unsafe { std::ffi::CStr::from_ptr(p as *const i8) }.to_string_lossy();
            eprintln!("GL {label}: {s}");
        }
    }
}

fn print_extensions(get_string: GlGetString) -> Option<String> {
    let raw = unsafe { get_string(GL_EXTENSIONS) };
    if raw.is_null() {
        return None;
    }
    Some(
        unsafe { std::ffi::CStr::from_ptr(raw as *const i8) }
            .to_string_lossy()
            .into_owned(),
    )
}

fn clear_gl_errors(get_error: GlGetError) {
    unsafe {
        while get_error() != 0 {}
    }
}

#[test]
#[ignore = "requires an EGL device and a dma-buf source; run with --ignored --nocapture"]
fn gles_memory_object_fd_imports_a_dmabuf() {
    let sh = match SharedContext::new(None) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skip: no EGL device ({e:#})");
            return;
        }
    };
    sh.make_current().expect("make the shared EGL context current");

    let get_proc = sh.get_proc_address();
    let get_string: GlGetString = match sym(get_proc, "glGetString") {
        Some(f) => f,
        None => {
            eprintln!("skip: glGetString unavailable");
            return;
        }
    };
    let get_error: GlGetError = match sym(get_proc, "glGetError") {
        Some(f) => f,
        None => {
            eprintln!("skip: glGetError unavailable");
            return;
        }
    };
    print_gl_info(get_string);

    let exts = match print_extensions(get_string) {
        Some(e) => e,
        None => {
            eprintln!("skip: glGetString(GL_EXTENSIONS) returned null");
            return;
        }
    };
    let has = |name: &str| exts.split_whitespace().any(|e| e == name);
    eprintln!(
        "GL extensions: memory_object={} memory_object_fd={} buffer_storage={}",
        has("GL_EXT_memory_object"),
        has("GL_EXT_memory_object_fd"),
        has("GL_EXT_buffer_storage")
    );
    if !has("GL_EXT_memory_object") || !has("GL_EXT_memory_object_fd") {
        eprintln!(
            "skip: GL_EXT_memory_object(_fd) is absent; \
             use the EGLImage path instead"
        );
        return;
    }

    let create_memory: GlCreateMemoryObjects =
        sym(get_proc, "glCreateMemoryObjectsEXT").expect("glCreateMemoryObjectsEXT");
    let import_fd: GlImportMemoryFd =
        sym(get_proc, "glImportMemoryFdEXT").expect("glImportMemoryFdEXT");
    let buffer_storage_mem: GlBufferStorageMem =
        sym(get_proc, "glBufferStorageMemEXT").expect("glBufferStorageMemEXT");
    let delete_memory: GlDeleteMemoryObjects =
        sym(get_proc, "glDeleteMemoryObjectsEXT").expect("glDeleteMemoryObjectsEXT");
    let gen_buffers: GlGenBuffers = sym(get_proc, "glGenBuffers").expect("glGenBuffers");
    let bind_buffer: GlBindBuffer = sym(get_proc, "glBindBuffer").expect("glBindBuffer");
    let buffer_sub_data: GlBufferSubData =
        sym(get_proc, "glBufferSubData").expect("glBufferSubData");
    let map_buffer: GlMapBufferRange =
        sym(get_proc, "glMapBufferRange").expect("glMapBufferRange");
    let unmap_buffer: GlUnmapBuffer = sym(get_proc, "glUnmapBuffer").expect("glUnmapBuffer");
    let delete_buffers: GlDeleteBuffers =
        sym(get_proc, "glDeleteBuffers").expect("glDeleteBuffers");

    const SIZE: usize = 1 << 20; // 1 MiB
    // Prefer GBM: that is what the server allocates its staging slots from.
    let (source, fd, size) = if let Some(b) = gbm_buffer(1024, 256) {
        ("gbm", b.fd, b.size)
    } else if let Some(fd) = dma_heap_alloc(SIZE) {
        ("dma-heap", fd, SIZE)
    } else {
        eprintln!(
            "skip: no dma-buf source \
             (/dev/dma_heap/system and GBM both unavailable)"
        );
        return;
    };
    eprintln!(
        "dma-buf: source={source} fd={} size={size}",
        fd.as_raw_fd()
    );

    // Drain earlier errors, then make the failure point unambiguous.
    clear_gl_errors(get_error);
    let mut mem = 0u32;
    unsafe { create_memory(1, &mut mem) };
    let err = unsafe { get_error() };
    eprintln!("glCreateMemoryObjectsEXT: mem={mem} err=0x{err:04x}");
    if mem == 0 || err != 0 {
        panic!("glCreateMemoryObjectsEXT failed: mem={mem} err=0x{err:04x}");
    }

    // glImportMemoryFdEXT consumes the fd on success; dup it for GL.
    let gl_fd = unsafe { libc::dup(fd.as_raw_fd()) };
    assert!(gl_fd >= 0, "dup dma-buf fd failed");
    clear_gl_errors(get_error);
    unsafe { import_fd(mem, size as u64, GL_HANDLE_TYPE_OPAQUE_FD_EXT, gl_fd) };
    let err = unsafe { get_error() };
    if err != 0 {
        panic!("glImportMemoryFdEXT failed for {source}: 0x{err:04x}");
    }

    let mut buf = 0u32;
    unsafe { gen_buffers(1, &mut buf) };
    unsafe { bind_buffer(GL_SHADER_STORAGE_BUFFER, buf) };
    clear_gl_errors(get_error);
    unsafe { buffer_storage_mem(GL_SHADER_STORAGE_BUFFER, size as isize, mem, 0) };
    let err = unsafe { get_error() };
    if err != 0 {
        panic!("glBufferStorageMemEXT failed: 0x{err:04x}");
    }
    eprintln!("OK: imported the dma-buf as a GL memory object and created an SSBO");

    let pattern: [u8; 16] = *b"arsa-dmabuf-gl!!";
    unsafe {
        buffer_sub_data(
            GL_SHADER_STORAGE_BUFFER,
            0,
            pattern.len() as isize,
            pattern.as_ptr() as *const c_void,
        )
    };
    let p = unsafe {
        map_buffer(
            GL_SHADER_STORAGE_BUFFER,
            0,
            pattern.len() as isize,
            GL_MAP_READ_BIT,
        )
    };
    if p.is_null() {
        panic!("glMapBufferRange on the imported SSBO returned null");
    }
    let got = unsafe { std::slice::from_raw_parts(p as *const u8, pattern.len()) };
    assert_eq!(got, &pattern, "SSBO contents differ from what was written");
    unsafe { unmap_buffer(GL_SHADER_STORAGE_BUFFER) };
    eprintln!("OK: the imported SSBO is readable/writable (pattern verified)");

    unsafe {
        delete_buffers(1, &buf);
        delete_memory(1, &mem);
    }
}

#[test]
#[ignore = "requires an EGL device and a GBM render node; run with --ignored --nocapture"]
fn egl_image_imports_a_linear_gbm_dmabuf() {
    let sh = match SharedContext::new(None) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skip: no EGL device ({e:#})");
            return;
        }
    };
    sh.make_current().expect("make the shared EGL context current");

    let create = match sh.create_image_fn() {
        Some(f) => f,
        None => {
            eprintln!("skip: EGL_EXT_image_dma_buf_import is absent");
            return;
        }
    };
    let destroy = sh.destroy_image_fn();
    let egl_get_error = sh.get_error_fn();

    let get_proc = sh.get_proc_address();
    let get_error: GlGetError = match sym(get_proc, "glGetError") {
        Some(f) => f,
        None => {
            eprintln!("skip: glGetError unavailable");
            return;
        }
    };
    let gen_textures: GlGenTextures = sym(get_proc, "glGenTextures").expect("glGenTextures");
    let bind_texture: GlBindTexture = sym(get_proc, "glBindTexture").expect("glBindTexture");
    let delete_textures: GlDeleteTextures =
        sym(get_proc, "glDeleteTextures").expect("glDeleteTextures");
    let image_target: GlEglImageTargetTexture = match sym(get_proc, "glEGLImageTargetTexture2DOES")
    {
        Some(f) => f,
        None => {
            eprintln!("skip: glEGLImageTargetTexture2DOES is absent");
            return;
        }
    };
    let bind_image: GlBindImageTexture =
        sym(get_proc, "glBindImageTexture").expect("glBindImageTexture");

    let b = match gbm_buffer(1024, 256) {
        Some(b) => b,
        None => {
            eprintln!("skip: no linear GBM bo (check /dev/dri access or ARSA_DRM_NODE)");
            return;
        }
    };
    eprintln!(
        "gbm: fd={} offset={} stride={} {}x{} modifier={:#x} size={}",
        b.fd.as_raw_fd(),
        b.offset,
        b.stride,
        b.width,
        b.height,
        b.modifier,
        b.size
    );

    let attrs: [i32; 17] = [
        EGL_WIDTH,
        b.width as i32,
        EGL_HEIGHT,
        b.height as i32,
        EGL_LINUX_DRM_FOURCC_EXT,
        0x3432_5258, // XR24
        EGL_DMA_BUF_PLANE0_FD_EXT,
        b.fd.as_raw_fd(),
        EGL_DMA_BUF_PLANE0_OFFSET_EXT,
        b.offset as i32,
        EGL_DMA_BUF_PLANE0_PITCH_EXT,
        b.stride as i32,
        EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
        (b.modifier & 0xffff_ffff) as i32,
        EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
        (b.modifier >> 32) as i32,
        EGL_NONE,
    ];
    let img = unsafe {
        create(
            sh.dpy(),
            std::ptr::null_mut(),
            EGL_LINUX_DMA_BUF_EXT,
            std::ptr::null_mut(),
            attrs.as_ptr(),
        )
    };
    if img.is_null() {
        let e = unsafe { egl_get_error() };
        panic!("eglCreateImageKHR(linear GBM dma-buf) failed: 0x{e:04x}");
    }
    eprintln!("OK: eglCreateImageKHR accepted the linear GBM dma-buf");

    clear_gl_errors(get_error);
    let mut tex = 0u32;
    unsafe { gen_textures(1, &mut tex) };
    unsafe { bind_texture(GL_TEXTURE_2D, tex) };
    unsafe { image_target(GL_TEXTURE_2D, img) };
    let err = unsafe { get_error() };
    if err != 0 {
        panic!("glEGLImageTargetTexture2DOES(GL_TEXTURE_2D) failed: 0x{err:04x}");
    }
    eprintln!("OK: bound the EGLImage as GL_TEXTURE_2D");

    clear_gl_errors(get_error);
    unsafe { bind_image(0, tex, 0, 0, 0, GL_READ_WRITE, GL_RGBA8) };
    let err = unsafe { get_error() };
    if err != 0 {
        panic!("glBindImageTexture(READ_WRITE) failed: 0x{err:04x}");
    }
    eprintln!("OK: bound the image for shader writes (imageStore)");

    unsafe {
        delete_textures(1, &tex);
        if let Some(d) = destroy {
            (d)(sh.dpy(), img);
        }
    }
}
