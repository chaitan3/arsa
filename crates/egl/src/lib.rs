//! A shared, surfaceless EGL/GLES context.
//!
//! The server needs one GPU context used by two consumers: the capture
//! materializer (`rdma-display-capture`) imports scanout dmabufs and reads
//! them back, and the lossless encoder (`rdma-display-codec-gpu`) imports the
//! same dmabufs and runs compute shaders. Historically each created its own
//! EGL context, which measured ~49 ms (context) + ~18 ms (load/display) per
//! context on this stack — and per client connect. Creating a single context
//! and sharing it removes the second context's cost entirely.
//!
//! Both consumers use GLES 3.1 and only ever touch GL on the server thread, so
//! one current context is sufficient. `SharedContext` owns the dlopened
//! libEGL, the EGLDisplay, and the EGLContext, and exposes the display/context
//! handles plus `make_current` for the consumers' own EGL entry points
//! (`eglCreateImageKHR`, sync fences, ...).
//!
//! libEGL is dlopened at runtime (house style), so a binary that never builds
//! a context still runs on hosts without the GPU stack.

use std::ffi::{c_void, CStr, CString};

use anyhow::{bail, Context, Result};

pub type EglHandle = *mut c_void;
pub type FnGetProcAddress = unsafe extern "system" fn(*const u8) -> *const c_void;
/// `eglCreateImageKHR`/`eglCreateImage` (dmabuf import); absent without
/// `EGL_EXT_image_dma_buf_import`.
pub type FnCreateImage =
    unsafe extern "system" fn(EglHandle, EglHandle, u32, EglHandle, *const i32) -> EglHandle;
pub type FnDestroyImage = unsafe extern "system" fn(EglHandle, EglHandle) -> u32;
pub type FnGetError = unsafe extern "system" fn() -> u32;

type FnGetPlatformDisplay = unsafe extern "system" fn(u32, EglHandle, *const i64) -> EglHandle;
type FnQueryDevices = unsafe extern "system" fn(i32, *mut EglHandle, *mut i32) -> i32;
type FnQueryDeviceString = unsafe extern "system" fn(EglHandle, i32) -> *const u8;
type FnInitialize = unsafe extern "system" fn(EglHandle, *mut i32, *mut i32) -> u32;
type FnBindApi = unsafe extern "system" fn(u32) -> u32;
type FnChooseConfig = unsafe extern "system" fn(EglHandle, *const i32, *mut EglHandle, i32, *mut i32) -> u32;
type FnCreateContext = unsafe extern "system" fn(EglHandle, EglHandle, EglHandle, *const i32) -> EglHandle;
type FnMakeCurrent = unsafe extern "system" fn(EglHandle, EglHandle, EglHandle, EglHandle) -> u32;
type FnCreatePbuffer = unsafe extern "system" fn(EglHandle, EglHandle, *const i32) -> EglHandle;
type FnQueryString = unsafe extern "system" fn(EglHandle, i32) -> *const u8;

const EGL_PLATFORM_DEVICE_EXT: u32 = 0x313F;
const EGL_DRM_DEVICE_FILE_EXT: i32 = 0x3233;
const EGL_OPENGL_ES_API: u32 = 0x30A0;
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_PBUFFER_BIT: i32 = 0x0001;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_OPENGL_ES3_BIT: i32 = 0x0040;
const EGL_NONE: i32 = 0x3038;
const EGL_CONTEXT_CLIENT_VERSION: i32 = 0x3098;
const EGL_PBUFFER_WIDTH: i32 = 0x3050;
const EGL_PBUFFER_HEIGHT: i32 = 0x3051;
const EGL_EXTENSIONS: i32 = 0x3055;

/// One surfaceless GLES 3.x context on an EGL device, shareable by the
/// materializer and the GPU codec.
pub struct SharedContext {
    // keep libEGL loaded for the lifetime of the display/context
    _lib: libloading::Library,
    dpy: EglHandle,
    ctx: EglHandle,
    /// pbuffer surface, or null when the driver supports surfaceless contexts
    pbuf: EglHandle,
    make_current: FnMakeCurrent,
    get_error: FnGetError,
    query_string: FnQueryString,
    get_proc_address: FnGetProcAddress,
    create_image: Option<FnCreateImage>,
    destroy_image: Option<FnDestroyImage>,
}

// The server touches GL only from its capture thread; the Arc exists so the
// materializer and codec can co-own the context, not to move it across threads.
unsafe impl Send for SharedContext {}
unsafe impl Sync for SharedContext {}

unsafe fn load_sym<T: Copy>(
    get: FnGetProcAddress,
    lib: &libloading::Library,
    names: &[&str],
) -> Result<T> {
    for name in names {
        let c = CString::new(*name).unwrap();
        let mut p = (get)(c.as_ptr() as *const u8);
        if p.is_null() {
            if let Ok(s) = lib.get::<*mut c_void>(name.as_bytes()) {
                p = *s as *const c_void;
            }
        }
        if !p.is_null() {
            return Ok(std::mem::transmute_copy::<*const c_void, T>(&p));
        }
    }
    bail!("missing EGL symbol {names:?}")
}

impl SharedContext {
    /// Create a surfaceless GLES 3.x context. `rendernode` selects the EGL
    /// device by its `/dev/dri/renderD*` path; when `None`, a render node is
    /// preferred and the first device is the fallback.
    pub fn new(rendernode: Option<&str>) -> Result<Self> {
        unsafe {
            let lib = libloading::Library::new("libEGL.so.1")
                .or_else(|_| libloading::Library::new("libEGL.so"))
                .context("dlopen libEGL")?;
            let get_proc_address: FnGetProcAddress = *lib
                .get::<FnGetProcAddress>(b"eglGetProcAddress")
                .context("eglGetProcAddress")?;
            let get_error: FnGetError = load_sym(get_proc_address, &lib, &["eglGetError"])?;
            let query_string: FnQueryString = load_sym(get_proc_address, &lib, &["eglQueryString"])?;
            // dmabuf import is optional: absent means the texture-direct path
            // is disabled and callers fall back to readback + upload.
            let create_image = load_sym::<FnCreateImage>(
                get_proc_address,
                &lib,
                &["eglCreateImageKHR", "eglCreateImage"],
            )
            .ok();
            let destroy_image = load_sym::<FnDestroyImage>(
                get_proc_address,
                &lib,
                &["eglDestroyImageKHR", "eglDestroyImage"],
            )
            .ok();
            let get_platform_display: FnGetPlatformDisplay =
                load_sym(get_proc_address, &lib, &["eglGetPlatformDisplayEXT"])?;
            let query_devices: FnQueryDevices =
                load_sym(get_proc_address, &lib, &["eglQueryDevicesEXT"])?;
            let query_device_string: FnQueryDeviceString =
                load_sym(get_proc_address, &lib, &["eglQueryDeviceStringEXT"])?;
            let initialize: FnInitialize = load_sym(get_proc_address, &lib, &["eglInitialize"])?;
            let bind_api: FnBindApi = load_sym(get_proc_address, &lib, &["eglBindAPI"])?;
            let choose_config: FnChooseConfig =
                load_sym(get_proc_address, &lib, &["eglChooseConfig"])?;
            let create_context: FnCreateContext =
                load_sym(get_proc_address, &lib, &["eglCreateContext"])?;
            let make_current: FnMakeCurrent =
                load_sym(get_proc_address, &lib, &["eglMakeCurrent"])?;
            let create_pbuffer: FnCreatePbuffer =
                load_sym(get_proc_address, &lib, &["eglCreatePbufferSurface"])?;

            // device selection: explicit path > first render node > first device
            let mut devs: [EglHandle; 16] = [std::ptr::null_mut(); 16];
            let mut n = 0i32;
            query_devices(16, devs.as_mut_ptr(), &mut n);
            if n < 1 {
                bail!("eglQueryDevicesEXT: no devices");
            }
            let mut dpy: EglHandle = std::ptr::null_mut();
            let mut render_fallback: EglHandle = std::ptr::null_mut();
            let mut first: EglHandle = std::ptr::null_mut();
            for d in &devs[..n as usize] {
                let path = query_device_string(*d, EGL_DRM_DEVICE_FILE_EXT);
                let path = if path.is_null() {
                    None
                } else {
                    Some(CStr::from_ptr(path as *const i8).to_string_lossy().into_owned())
                };
                let cand = get_platform_display(EGL_PLATFORM_DEVICE_EXT, *d, std::ptr::null());
                if cand.is_null() {
                    continue;
                }
                if first.is_null() {
                    first = cand;
                }
                match (&path, rendernode) {
                    (Some(p), Some(want)) if p == want => {
                        dpy = cand;
                        break;
                    }
                    (Some(p), _) if p.contains("renderD") && render_fallback.is_null() => {
                        render_fallback = cand;
                    }
                    _ => {}
                }
            }
            if dpy.is_null() {
                dpy = if !render_fallback.is_null() { render_fallback } else { first };
            }
            if dpy.is_null() {
                bail!("no EGL device display");
            }

            let (mut major, mut minor) = (0i32, 0i32);
            if initialize(dpy, &mut major, &mut minor) == 0 {
                bail!("eglInitialize failed (err {:#06x})", get_error());
            }
            log::debug!(
                "[egl] EGL {major}.{minor} on {:?}",
                rendernode.unwrap_or("default device")
            );
            if bind_api(EGL_OPENGL_ES_API) == 0 {
                bail!("eglBindAPI(OpenGL ES) failed");
            }
            // extension advertisement is useful for diagnosing dmabuf import
            let exts = query_string(dpy, EGL_EXTENSIONS);
            if !exts.is_null() {
                let list = CStr::from_ptr(exts as *const i8).to_string_lossy().into_owned();
                for name in [
                    "EGL_KHR_surfaceless_context",
                    "EGL_EXT_image_dma_buf_import",
                    "EGL_EXT_image_dma_buf_import_modifiers",
                    "EGL_ANDROID_native_fence_sync",
                ] {
                    let have = list.split(' ').any(|e| e == name);
                    log::debug!("[egl]   {} {}", if have { "yes" } else { "-- " }, name);
                }
            }

            let cfg_attr = [
                EGL_SURFACE_TYPE,
                EGL_PBUFFER_BIT,
                EGL_RENDERABLE_TYPE,
                EGL_OPENGL_ES3_BIT,
                EGL_NONE,
            ];
            let mut cfg: EglHandle = std::ptr::null_mut();
            let mut ncfg = 0i32;
            if choose_config(dpy, cfg_attr.as_ptr(), &mut cfg, 1, &mut ncfg) == 0 || ncfg < 1 {
                bail!("eglChooseConfig: no GLES3-capable config");
            }
            let ctx_attr = [EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE];
            let ctx = create_context(dpy, cfg, std::ptr::null_mut(), ctx_attr.as_ptr());
            if ctx.is_null() {
                bail!("eglCreateContext failed (err {:#06x})", get_error());
            }
            let pbuf = if make_current(dpy, std::ptr::null_mut(), std::ptr::null_mut(), ctx) != 0 {
                log::debug!("[egl] using surfaceless context (EGL_KHR_surfaceless_context)");
                std::ptr::null_mut()
            } else {
                let pb = [EGL_PBUFFER_WIDTH, 1, EGL_PBUFFER_HEIGHT, 1, EGL_NONE];
                let surf = create_pbuffer(dpy, cfg, pb.as_ptr());
                if surf.is_null() {
                    bail!(
                        "eglCreatePbufferSurface failed (err {:#06x}) and surfaceless unsupported",
                        get_error()
                    );
                }
                if make_current(dpy, surf, surf, ctx) == 0 {
                    bail!("eglMakeCurrent(pbuffer) failed (err {:#06x})", get_error());
                }
                log::debug!("[egl] using pbuffer context");
                surf
            };
            log::debug!("[egl] GLES 3.x shared context ready");

            Ok(Self {
                _lib: lib,
                dpy,
                ctx,
                pbuf,
                make_current,
                get_error,
                query_string,
                get_proc_address,
                create_image,
                destroy_image,
            })
        }
    }

    /// Make this context current on the calling thread.
    pub fn make_current(&self) -> Result<()> {
        if unsafe { (self.make_current)(self.dpy, self.pbuf, self.pbuf, self.ctx) } == 0 {
            bail!("eglMakeCurrent(shared context) failed (err {:#06x})", self.get_error());
        }
        Ok(())
    }

    pub fn dpy(&self) -> EglHandle {
        self.dpy
    }

    pub fn ctx(&self) -> EglHandle {
        self.ctx
    }

    pub fn pbuf(&self) -> EglHandle {
        self.pbuf
    }

    pub fn get_error(&self) -> i32 {
        unsafe { (self.get_error)() as i32 }
    }

    pub fn query_string(&self, name: i32) -> *const u8 {
        unsafe { (self.query_string)(self.dpy, name) }
    }

    pub fn get_proc_address(&self) -> FnGetProcAddress {
        self.get_proc_address
    }

    /// `eglCreateImageKHR`, or `None` when dmabuf import is unavailable.
    pub fn create_image_fn(&self) -> Option<FnCreateImage> {
        self.create_image
    }

    pub fn destroy_image_fn(&self) -> Option<FnDestroyImage> {
        self.destroy_image
    }

    pub fn get_error_fn(&self) -> FnGetError {
        self.get_error
    }
}
