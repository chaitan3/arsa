//! EGL/GL frame materializer: import scanout dmabufs on the rendernode, blit
//! to a linear RGBA8 target, read back into the RDMA staging buffer.
//!
//! Default copy path (`--materialize egl`). Uses a surfaceless GLES 3.1
//! context on the rendernode's EGL device. libEGL is dlopened at runtime, so
//! the binary also runs on hosts without the GPU stack.
//!
//! Import: the scanout dmabuf is imported with its explicit modifier via
//! eglCreateImageKHR and bound as GL_TEXTURE_EXTERNAL_OES (foreign dmabufs
//! require the OES target on NVIDIA). Readback: glReadPixels(GL_BGRA) writes
//! the wire bytes straight into the caller's staging slot (pinned host
//! memory) — the GPU's copy performs detile + format normalize in one pass;
//! no CPU swizzle, no PBO. Orientation: empirically the OES sampling puts
//! readback row order correct as-is (no extra flip).

use std::ffi::CString;
use std::os::fd::AsRawFd;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};

use rdma_display_proto::MatTimes;

use crate::{CaptureEvent, DmabufScanout, FrameFormat, FrameMaterializer};

// --- EGL constants ----------------------------------------------------------
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_PBUFFER_BIT: i32 = 0x0001;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_OPENGL_BIT: i32 = 0x0008;
const EGL_OPENGL_API: u32 = 0x30A2;
const GL_TEXTURE_EXTERNAL_OES: u32 = 0x8D65;
const EGL_NONE: i32 = 0x3038;
const EGL_CONTEXT_MAJOR_VERSION: i32 = 0x3098;
const EGL_CONTEXT_MINOR_VERSION: i32 = 0x30FB;
const EGL_PBUFFER_WIDTH: i32 = 0x3050;
const EGL_PBUFFER_HEIGHT: i32 = 0x3051;
const EGL_PLATFORM_DEVICE_EXT: i32 = 0x313F;
const EGL_DRM_DEVICE_FILE_EXT: i32 = 0x3233;
const EGL_LINUX_DMA_BUF_EXT: u32 = 0x3270;
const EGL_LINUX_DRM_FOURCC_EXT: i32 = 0x3271;
const EGL_DMA_BUF_PLANE0_FD_EXT: i32 = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: i32 = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: i32 = 0x3274;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: i32 = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: i32 = 0x3444;
const EGL_WIDTH: i32 = 0x3057;
const EGL_HEIGHT: i32 = 0x3056;

// --- GL constants -----------------------------------------------------------
const GL_TEXTURE_2D: u32 = 0x0DE1;
const GL_RGBA8: u32 = 0x8058;
const GL_RGBA: u32 = 0x1908;
const GL_UNSIGNED_BYTE: u32 = 0x1401;
const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
const GL_TEXTURE_WRAP_S: u32 = 0x2802;
const GL_TEXTURE_WRAP_T: u32 = 0x2803;
const GL_NEAREST: u32 = 0x2600;
const GL_CLAMP_TO_EDGE: i32 = 0x812F as i32;
const GL_FRAMEBUFFER: u32 = 0x8D40;
const GL_COLOR_ATTACHMENT0: u32 = 0x8CE0;
const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;
const GL_PIXEL_PACK_BUFFER: u32 = 0x88EB;
// readback format: driver emits bytes B,G,R,X — the wire format (XR24) —
// so the GPU's copy also performs the channel normalize; no CPU swizzle
const GL_BGRA: u32 = 0x80E1;
const GL_PACK_ROW_LENGTH: u32 = 0x0D02;
const GL_TRIANGLES: u32 = 0x0004;
const GL_VERTEX_SHADER: u32 = 0x8B31;
const GL_FRAGMENT_SHADER: u32 = 0x8B30;
const GL_COMPILE_STATUS: u32 = 0x8B81;
const GL_LINK_STATUS: u32 = 0x8B82;

type EglHandle = *mut core::ffi::c_void;

fn get_sym(lib: &libloading::Library, name: &str) -> Result<*mut core::ffi::c_void> {
    unsafe {
        let s: libloading::Symbol<*mut core::ffi::c_void> =
            lib.get(name.as_bytes()).with_context(|| format!("missing symbol {name}"))?;
        Ok(*s)
    }
}

// --- libEGL bindings --------------------------------------------------------

type FnGetPlatformDisplay = unsafe extern "system" fn(i32, EglHandle, *const i64) -> EglHandle;
type FnQueryDevices = unsafe extern "system" fn(i32, *mut EglHandle, *mut i32) -> i32;
type FnQueryDeviceString = unsafe extern "system" fn(EglHandle, i32) -> *const u8;
type FnInitialize = unsafe extern "system" fn(EglHandle, *mut i32, *mut i32) -> u32;
type FnBindApi = unsafe extern "system" fn(u32) -> u32;
type FnChooseConfig =
    unsafe extern "system" fn(EglHandle, *const i32, *mut EglHandle, i32, *mut i32) -> u32;
type FnCreateContext =
    unsafe extern "system" fn(EglHandle, EglHandle, EglHandle, *const i32) -> EglHandle;
type FnCreatePbuffer = unsafe extern "system" fn(EglHandle, EglHandle, *const i32) -> EglHandle;
type FnMakeCurrent = unsafe extern "system" fn(EglHandle, EglHandle, EglHandle, EglHandle) -> u32;
type FnGetError = unsafe extern "system" fn() -> i32;
type FnCreateImage =
    unsafe extern "system" fn(EglHandle, EglHandle, u32, EglHandle, *const i32) -> EglHandle;
type FnQueryString = unsafe extern "system" fn(EglHandle, i32) -> *const u8;
type FnGetProcAddress = unsafe extern "system" fn(*const i8) -> *const core::ffi::c_void;

const EGL_EXTENSIONS: i32 = 0x3055;

pub struct Egl {
    pub lib: Arc<libloading::Library>,
    pub get_platform_display: FnGetPlatformDisplay,
    pub query_devices: FnQueryDevices,
    pub query_device_string: FnQueryDeviceString,
    pub initialize: FnInitialize,
    pub bind_api: FnBindApi,
    pub choose_config: FnChooseConfig,
    pub create_context: FnCreateContext,
    pub create_pbuffer: FnCreatePbuffer,
    pub make_current: FnMakeCurrent,
    pub get_error: FnGetError,
    pub create_image_khr: FnCreateImage,
    pub destroy_image: unsafe extern "system" fn(EglHandle, EglHandle) -> u32,
    pub get_proc_address: FnGetProcAddress,
    pub query_string: FnQueryString,
}

impl Egl {
    pub fn load() -> Result<Self> {
        let lib = unsafe {
            libloading::Library::new("libEGL.so.1")
                .or_else(|_| libloading::Library::new("libEGL.so"))
                .context("dlopen libEGL")?
        };
        // eglGetError + eglGetProcAddress are core API and always exported;
        // everything else (including EGL 1.5 platform functions) goes through
        // eglGetProcAddress with a dlsym fallback.
        let get_error = unsafe {
            std::mem::transmute::<*mut core::ffi::c_void, FnGetError>(get_sym(&lib, "eglGetError")?)
        };
        let bind_api = unsafe {
            std::mem::transmute::<*mut core::ffi::c_void, FnBindApi>(get_sym(&lib, "eglBindAPI")?)
        };
        let get_proc_address = unsafe {
            std::mem::transmute::<*mut core::ffi::c_void, FnGetProcAddress>(get_sym(
                &lib,
                "eglGetProcAddress",
            )?)
        };
        macro_rules! load {
            ($name:literal) => {
                load_ext(&get_proc_address, &lib, $name)?
            };
        }
        Ok(Egl {
            get_platform_display: load!("eglGetPlatformDisplayEXT"),
            query_devices: load!("eglQueryDevicesEXT"),
            query_device_string: load!("eglQueryDeviceStringEXT"),
            initialize: load!("eglInitialize"),
            bind_api,
            choose_config: load!("eglChooseConfig"),
            create_context: load!("eglCreateContext"),
            create_pbuffer: load!("eglCreatePbufferSurface"),
            make_current: load!("eglMakeCurrent"),
            get_error,
            create_image_khr: load!("eglCreateImageKHR"),
            destroy_image: load!("eglDestroyImageKHR"),
            query_string: load!("eglQueryString"),
            get_proc_address,
            lib: Arc::new(lib),
        })
    }

    fn get_error(&self) -> i32 {
        unsafe { (self.get_error)() }
    }
}

/// Load `name` via dlsym, falling back to eglGetProcAddress (some EGL entry
/// points are not exported by the shared object at all).
fn load_sym_proc(
    proc_addr: &FnGetProcAddress,
    lib: &libloading::Library,
    name: &str,
) -> *const core::ffi::c_void {
    let c = CString::new(name).unwrap();
    let p = unsafe { (*proc_addr)(c.as_ptr()) };
    if !p.is_null() {
        return p;
    }
    unsafe {
        libloading::Library::get(lib, name.as_bytes())
            .map(|s: libloading::Symbol<*mut core::ffi::c_void>| *s as *const core::ffi::c_void)
            .unwrap_or(std::ptr::null())
    }
}

fn load_ext<T: Copy>(
    proc_addr: &FnGetProcAddress,
    lib: &libloading::Library,
    name: &str,
) -> Result<T> {
    let p = load_sym_proc(proc_addr, lib, name);
    if p.is_null() {
        bail!("missing symbol {name}");
    }
    unsafe { Ok(std::mem::transmute_copy::<*const core::ffi::c_void, T>(&p)) }
}

// --- GL bindings ------------------------------------------------------------
fn gl_proc_ptr(egl: &Egl, name: &str) -> Result<*const core::ffi::c_void> {
    let c = CString::new(name).unwrap();
    let p = unsafe { (egl.get_proc_address)(c.as_ptr()) };
    if p.is_null() {
        bail!("GL function {name} unavailable");
    }
    Ok(p)
}


struct Gl {
    create_shader: unsafe extern "system" fn(u32) -> u32,
    shader_source: unsafe extern "system" fn(u32, i32, *const *const i8, *const i32),
    compile_shader: unsafe extern "system" fn(u32),
    get_shaderiv: unsafe extern "system" fn(u32, u32, *mut i32),
    get_shader_info_log: unsafe extern "system" fn(u32, i32, *mut i32, *mut u8),
    create_program: unsafe extern "system" fn() -> u32,
    attach_shader: unsafe extern "system" fn(u32, u32),
    link_program: unsafe extern "system" fn(u32),
    get_programiv: unsafe extern "system" fn(u32, u32, *mut i32),
    gen_textures: unsafe extern "system" fn(i32, *mut u32),
    bind_texture: unsafe extern "system" fn(u32, u32),
    tex_parameteri: unsafe extern "system" fn(u32, u32, i32),
    tex_image_2d: unsafe extern "system" fn(
        u32,
        i32,
        i32,
        i32,
        i32,
        i32,
        u32,
        u32,
        *const core::ffi::c_void,
    ),
    gen_framebuffers: unsafe extern "system" fn(i32, *mut u32),
    bind_framebuffer: unsafe extern "system" fn(u32, u32),
    framebuffer_texture_2d: unsafe extern "system" fn(u32, u32, u32, u32, i32),
    check_framebuffer_status: unsafe extern "system" fn(u32) -> u32,
    delete_framebuffers: unsafe extern "system" fn(i32, *const u32),
    bind_buffer: unsafe extern "system" fn(u32, u32),
    pixel_storei: unsafe extern "system" fn(u32, i32),
    read_pixels: unsafe extern "system" fn(i32, i32, i32, i32, u32, u32, *mut core::ffi::c_void),
    use_program: unsafe extern "system" fn(u32),
    draw_arrays: unsafe extern "system" fn(u32, i32, i32),
    viewport: unsafe extern "system" fn(i32, i32, i32, i32),
    delete_textures: unsafe extern "system" fn(i32, *const u32),
    finish: unsafe extern "system" fn(),
    get_error: unsafe extern "system" fn() -> u32,
    image_target_texture: unsafe extern "system" fn(u32, EglHandle),
}


impl Gl {
    fn load(egl: &Egl) -> Result<Self> {
        macro_rules! load {
            ($name:literal) => {
                unsafe {
                    std::mem::transmute::<*const core::ffi::c_void, _>(gl_proc_ptr(egl, $name)?)
                }
            };
        }
        let gl = Gl {
            create_shader: load!("glCreateShader"),
            shader_source: load!("glShaderSource"),
            compile_shader: load!("glCompileShader"),
            get_shaderiv: load!("glGetShaderiv"),
            get_shader_info_log: load!("glGetShaderInfoLog"),
            create_program: load!("glCreateProgram"),
            attach_shader: load!("glAttachShader"),
            link_program: load!("glLinkProgram"),
            get_programiv: load!("glGetProgramiv"),
            gen_textures: load!("glGenTextures"),
            bind_texture: load!("glBindTexture"),
            tex_parameteri: load!("glTexParameteri"),
            tex_image_2d: load!("glTexImage2D"),
            gen_framebuffers: load!("glGenFramebuffers"),
            delete_framebuffers: load!("glDeleteFramebuffers"),
            bind_framebuffer: load!("glBindFramebuffer"),
            framebuffer_texture_2d: load!("glFramebufferTexture2D"),
            check_framebuffer_status: load!("glCheckFramebufferStatus"),
            bind_buffer: load!("glBindBuffer"),
            pixel_storei: load!("glPixelStorei"),
            read_pixels: load!("glReadPixels"),
            use_program: load!("glUseProgram"),
            draw_arrays: load!("glDrawArrays"),
            viewport: load!("glViewport"),
            delete_textures: load!("glDeleteTextures"),
            finish: load!("glFinish"),
            get_error: load!("glGetError"),
            image_target_texture: load!("glEGLImageTargetTexture2DOES"),
        };
        Ok(gl)
    }
}

const VS_SRC: &str = r#"#version 310 es
out vec2 v_uv;
void main() {
    // full-screen triangle from gl_VertexID, no VBO needed
    float x = float((gl_VertexID == 1) ? 3 : -1);
    float y = float((gl_VertexID == 2) ? 3 : -1);
    v_uv = vec2(x, y) * 0.5 + 0.5;
    gl_Position = vec4(vec2(x, y), 0.0, 1.0);
}
"#;

const FS_SRC: &str = r#"#version 310 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
uniform samplerExternalOES tex;
in vec2 v_uv;
out vec4 frag;
void main() { frag = texture(tex, v_uv); }
"#;

// EGL/GL objects are bound to the materializer's own thread (all GL calls
// happen inside the RDMA sender loop, which owns the materializer).
unsafe impl Send for EglMaterializer {}

/// Single-plane, 4-byte-per-pixel RGB formats the EGL blit can normalize.
///
/// The blit samples the imported `GL_TEXTURE_EXTERNAL_OES` texture and writes
/// an RGBA8 target, so the driver handles channel order and bit depth — we only
/// require that EGL can import the dmabuf and that it is 4 bytes/pixel (the
/// wire format is always XR24 regardless). Two families qualify:
///
///   * 8888 (8 bits/channel): AR24 AB24 RA24 BA24 XR24 XB24 RX24 BX24
///   * 2101010 / 1010102 (10 bits/channel): AR30 AB30 RA30 BA30
///     XR30 XB30 RX30 BX30
///
/// The 10-bit formats appear when the compositor switches the primary plane to
/// a wide-gamut/HDR scanout — e.g. KWin while mpv plays HDR video. An
/// `XR24`-only whitelist made that switch a fatal error and stopped capture.
fn is_supported_rgb_fourcc(fourcc: u32) -> bool {
    const SUPPORTED: &[u32] = &[
        // 32bpp 8888: ARGB ABGR RGBA BGRA XRGB XBGR RGBX BGRX
        0x3432_5241,
        0x3432_4241,
        0x3432_4152,
        0x3432_4142,
        0x3432_5258,
        0x3432_4258,
        0x3432_5852,
        0x3432_5842,
        // 32bpp 10-bit 2101010/1010102: the same eight orders ('?30')
        0x3033_5241,
        0x3033_4241,
        0x3033_4152,
        0x3033_4142,
        0x3033_5258,
        0x3033_4258,
        0x3033_5852,
        0x3033_5842,
    ];
    SUPPORTED.contains(&fourcc)
}

pub struct EglMaterializer {
    egl: Egl,
    gl: Gl,
    dpy: EglHandle,
    /// keep-alive handles: the current context and the pbuffer surface (if
    /// surfaceless was unavailable). Never read directly, but dropping the
    /// handles here would not destroy the EGL objects anyway.
    #[allow(dead_code)]
    ctx: EglHandle,
    #[allow(dead_code)]
    pbuf: EglHandle,
    prog: u32,
    linear_tex: u32,
    fbo: u32,
    fmt: Option<FrameFormat>,
    /// (EGLImage, GL texture) of the scanout
    img: Option<(EglHandle, u32)>,
    dirty: bool,
    /// GPU stage times of the most recent materialization (µs); sent to the
    /// client in the per-frame Stamp for overlay attribution
    last_mat: MatTimes,
    /// zero-copy staging (PBO mode): n persistent-mapped PIXEL_PACK buffers;
    /// the server registers their pages as RDMA staging slots so the GL
    /// readback DMA fills exactly what the NIC sends. None = direct readback
    /// into the caller's buffer (the legacy path).
    stage: Option<StageSlots>,
    /// pipelined per-slot work: the EGL fence of the in-flight readback
    stage_work: Vec<Option<StageWork>>,
}

/// Persistent-mapped staging PBOs + the EGL fence that orders the readback
/// DMA before the RDMA engine reads the same pages.
struct StageSlots {
    pbos: Vec<u32>,
    /// persistent CPU/GPU-visible maps (== the addresses the server registered)
    maps: Vec<*mut core::ffi::c_void>,
    frame_bytes: usize,
    /// EGL fence sync (the GL glFenceSync path is broken on this driver)
    fence: Option<(
        FnCreateSyncKhr,
        FnClientWaitSyncKhr,
        FnDestroySyncKhr,
    )>,
    /// capability symbols (kept for the PBO lifecycle: Drop deletes the buffers)
    syms: BenchSyms,
}

/// One issued frame: the readback into this slot is in flight, ordered by
/// the stored EGL fence; completed by finish_frame (fence wait).
struct StageWork {
    fence: EglHandle,
    blit_us: u64,
    read_us: u64,
}

impl Drop for StageSlots {
    fn drop(&mut self) {
        // best-effort GL cleanup: the context may or may not be current at
        // drop time; drivers generally ignore invalid-context deletes
        unsafe {
            for p in &self.pbos {
                (self.syms.delete_buffers)(1, p);
            }
        }
    }
}

impl EglMaterializer {
    pub fn new(rendernode: Option<&str>) -> Result<Self> {
        let egl = Egl::load()?;
        let dpy = device_display(&egl, rendernode)?;

        let mut major = 0;
        let mut minor = 0;
        if unsafe { (egl.initialize)(dpy, &mut major, &mut minor) } == 0 {
            bail!("eglInitialize failed (err {:#06x})", egl.get_error());
        }
        eprintln!("[egl] EGL {}.{} on {:?}", major, minor, rendernode.unwrap_or("default device"));
        if unsafe { (egl.bind_api)(EGL_OPENGL_API) } == 0 {
            bail!("eglBindAPI(OpenGL) failed");
        }
        // log the extensions we care about for debugging
        unsafe {
            let exts = (egl.query_string)(dpy, EGL_EXTENSIONS);
            if !exts.is_null() {
                let list = std::ffi::CStr::from_ptr(exts as *const i8)
                    .to_string_lossy()
                    .into_owned();
                let wanted = [
                    "EGL_KHR_surfaceless_context",
                    "EGL_EXT_platform_device",
                    "EGL_KHR_image_base",
                    "EGL_EXT_image_dma_buf_import",
                    "EGL_EXT_image_dma_buf_import_modifiers",
                    "EGL_ANDROID_native_fence_sync",
                ];
                for name in wanted {
                    let have = list.split(' ').any(|e| e == name);
                    eprintln!("[egl]   {} {}", if have { "yes" } else { "-- " }, name);
                }
            }
        }
        let cfg_attribs =
            [EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE];
        let mut cfg: EglHandle = std::ptr::null_mut();
        let mut n = 0;
        if unsafe { (egl.choose_config)(dpy, cfg_attribs.as_ptr(), &mut cfg, 1, &mut n) } == 0
            || n < 1
        {
            bail!("eglChooseConfig: no OpenGL-capable config");
        }
        let ctx_attribs = [EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE];
        let ctx = unsafe { (egl.create_context)(dpy, cfg, std::ptr::null_mut(), ctx_attribs.as_ptr()) };
        if ctx.is_null() {
            bail!(
                "eglCreateContext failed (version 3.1, err {:#06x})",
                egl.get_error()
            );
        }
        let pb = [EGL_PBUFFER_WIDTH, 1, EGL_PBUFFER_HEIGHT, 1, EGL_NONE];
        // Prefer a surfaceless context (no surface needed at all); fall back
        // to a 1x1 pbuffer if the driver doesn't support it.
        let surfaceless = unsafe { (egl.make_current)(dpy, std::ptr::null_mut(), std::ptr::null_mut(), ctx) } != 0;
        let pbuf = if surfaceless {
            eprintln!("[egl] using surfaceless context (EGL_KHR_surfaceless_context)");
            std::ptr::null_mut()
        } else {
            let pb_surf = unsafe { (egl.create_pbuffer)(dpy, cfg, pb.as_ptr()) };
            if pb_surf.is_null() {
                bail!(
                    "eglCreatePbufferSurface failed (err {:#06x}) and surfaceless context unsupported",
                    egl.get_error()
                );
            }
            if unsafe { (egl.make_current)(dpy, pb_surf, pb_surf, ctx) } == 0 {
                bail!("eglMakeCurrent(pbuffer) failed (err {:#06x})", egl.get_error());
            }
            eprintln!("[egl] using pbuffer context");
            pb_surf
        };
        eprintln!("[egl] GL ES 3.1 context ready");

        let gl = Gl::load(&egl)?;
        let mut mat = EglMaterializer {
            egl,
            gl,
            dpy,
            ctx,
            pbuf,
            prog: 0,
            linear_tex: 0,
            fbo: 0,
            fmt: None,
            img: None,
            dirty: false,
            last_mat: MatTimes::ZERO,
            stage: None,
            stage_work: Vec::new(),
        };
        mat.build_program()?;
        Ok(mat)
    }

    fn build_program(&mut self) -> Result<()> {
        unsafe {
            let compile = |gl: &Gl, src: &str, ty: u32| -> Result<u32> {
                let sh = (gl.create_shader)(ty);
                let c = CString::new(src.as_bytes())?;
                (gl.shader_source)(sh, 1, (&[c.as_ptr()] as *const [*const i8]) as *const *const i8, std::ptr::null());
                (gl.compile_shader)(sh);
                let mut ok = 0;
                (gl.get_shaderiv)(sh, GL_COMPILE_STATUS, &mut ok);
                if ok == 0 {
                    let mut log = [0u8; 1024];
                    let mut n = 0;
                    (gl.get_shader_info_log)(sh, 1024, &mut n, log.as_mut_ptr());
                    bail!(
                        "shader compile failed: {}",
                        String::from_utf8_lossy(&log[..n as usize])
                    );
                }
                Ok(sh)
            };
            let vs = compile(&self.gl, VS_SRC, GL_VERTEX_SHADER)?;
            let fs = compile(&self.gl, FS_SRC, GL_FRAGMENT_SHADER)?;
            let prog = (self.gl.create_program)();
            (self.gl.attach_shader)(prog, vs);
            (self.gl.attach_shader)(prog, fs);
            (self.gl.link_program)(prog);
            let mut ok = 0;
            (self.gl.get_programiv)(prog, GL_LINK_STATUS, &mut ok);
            if ok == 0 {
                bail!("GL program link failed");
            }
            self.prog = prog;
        }
        Ok(())
    }

    /// (re)create the linear RGBA8 target and the persistent readback buffer
    fn build_targets(&mut self, fmt: &FrameFormat) -> Result<()> {
        // Only rebuild when the format actually changed: QEMU re-announces
        // the scanout on every buffer alternation, and rebuilding here
        // allocates + frees an 8 MB texture per update (churn even when the
        // delete is correct).
        if let Some(f) = &self.fmt {
            if f.width == fmt.width && f.height == fmt.height && f.stride == fmt.stride {
                return Ok(());
            }
        }
        let w = fmt.width as i32;
        let h = fmt.height as i32;
        unsafe {
            // linear target texture
            let mut tex = 0u32;
            (self.gl.gen_textures)(1, &mut tex);
            (self.gl.bind_texture)(GL_TEXTURE_2D, tex);
            (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as i32);
            (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as i32);
            (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE as i32);
            (self.gl.tex_image_2d)(
                GL_TEXTURE_2D, 0, GL_RGBA8 as i32, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                std::ptr::null(),
            );
            let mut fbo = 0u32;
            (self.gl.gen_framebuffers)(1, &mut fbo);
            (self.gl.bind_framebuffer)(GL_FRAMEBUFFER, fbo);
            (self.gl.framebuffer_texture_2d)(
                GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0,
            );
            if (self.gl.check_framebuffer_status)(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE {
                bail!("linear FBO incomplete");
            }
            (self.gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);

            // delete the previous FBO *and* texture: a deleted texture stays
            // alive while an FBO attachment still references it — skipping the
            // FBO delete leaked one 8 MB texture + FBO per scanout re-announce
            // (the VRAM leak)
            if self.fbo != 0 {
                (self.gl.delete_framebuffers)(1, &[self.fbo] as *const u32);
            }
            (self.gl.bind_texture)(GL_TEXTURE_2D, 0);
            if self.linear_tex != 0 {
                (self.gl.delete_textures)(1, &[self.linear_tex] as *const u32);
            }
            self.linear_tex = tex;
            self.fbo = fbo;
        }
        Ok(())
    }

    fn import_dmabuf(&mut self, d: &DmabufScanout) -> Result<()> {
        let fmt = &d.format;
        if d.fds.len() != 1 || fmt.num_planes != 1 {
            bail!("multi-plane dmabuf not supported yet (planes={})", d.fds.len());
        }
        // all supported single-plane 4bpp RGB variants (see
        // is_supported_rgb_fourcc); the blit normalizes memory order and bit
        // depth, and readback always produces XR24 (BGRX) bytes
        if !is_supported_rgb_fourcc(fmt.fourcc) {
            bail!("EGL materializer: unsupported fourcc {:#010x}", fmt.fourcc);
        }
        // destroy the previous import: unbind BOTH targets the blit touches
        // (the OES binding is what the blit leaves bound — a deleted texture
        // stays alive while any binding still references it)
        if let Some((img, tex)) = self.img.take() {
            unsafe {
                (self.gl.bind_texture)(GL_TEXTURE_EXTERNAL_OES, 0);
                (self.gl.bind_texture)(GL_TEXTURE_2D, 0);
                (self.gl.delete_textures)(1, &[tex] as *const u32);
                (self.egl.destroy_image)(self.dpy, img);
            }
        }

        let fd = d.fds[0].as_raw_fd();
        let modifier = d.format.modifier;

        // base attributes (per-plane group), always present
        let base: Vec<i32> = vec![
            EGL_WIDTH, d.format.width as i32,
            EGL_HEIGHT, d.format.height as i32,
            EGL_LINUX_DRM_FOURCC_EXT, d.format.fourcc as i32,
            EGL_DMA_BUF_PLANE0_FD_EXT, fd,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT, d.offsets[0] as i32,
            EGL_DMA_BUF_PLANE0_PITCH_EXT, d.strides[0] as i32,
        ];

        // import with the explicit modifier (KHR entry point + EGLint list) —
        // verified path on NVIDIA EGL 1.5
        let mut a: Vec<i32> = base.clone();
        a.extend_from_slice(&[
            EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (modifier & 0xffff_ffff) as i32,
            EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, (modifier >> 32) as i32,
            EGL_NONE,
        ]);
        let img = unsafe {
            (self.egl.create_image_khr)(
                self.dpy, std::ptr::null_mut(), EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(), a.as_ptr(),
            )
        };
        if img.is_null() {
            let err = self.egl.get_error();
            bail!("eglCreateImageKHR(dmabuf, modifier {modifier:#x}) failed (err {err:#06x})");
        }

        unsafe {
            let mut tex = 0u32;
            (self.gl.gen_textures)(1, &mut tex);
            // foreign dmabufs on NVIDIA bind as EXTERNAL_OES (GLES)
            (self.gl.bind_texture)(GL_TEXTURE_EXTERNAL_OES, tex);
            (self.gl.image_target_texture)(GL_TEXTURE_EXTERNAL_OES, img);
            let gl_err = (self.gl.get_error)();
            if gl_err != 0 {
                eprintln!(
                    "[egl] WARNING: image_target_texture(OES) glError={:#06x}",
                    gl_err
                );
            }
            self.img = Some((img, tex));
        }
        eprintln!(
            "[egl] imported dmabuf {}x{} fourcc={:#010x} modifier={:#x}",
            d.format.width, d.format.height, d.format.fourcc, d.format.modifier
        );
        Ok(())
    }
}


fn device_display(egl: &Egl, rendernode: Option<&str>) -> Result<EglHandle> {
    unsafe {
        let mut devices: [EglHandle; 16] = [std::ptr::null_mut(); 16];
        let mut count = 0i32;
        (egl.query_devices)(16, devices.as_mut_ptr(), &mut count);
        if count == 0 {
            bail!(
                "eglQueryDevicesEXT: no devices (requires EGL_EXT_device_enumeration + EGL_EXT_device_drm)"
            );
        }
        for i in 0..count as usize {
            let dev = devices[i];
            let file = (egl.query_device_string)(dev, EGL_DRM_DEVICE_FILE_EXT);
            let path = (!file.is_null())
                .then(|| {
                    std::ffi::CStr::from_ptr(file as *const i8).to_string_lossy().into_owned()
                })
                .unwrap_or_default();
            eprintln!("[egl] device[{i}]: {path}");
            if let Some(want) = rendernode {
                if path != want {
                    continue;
                }
            }
            let dpy = (egl.get_platform_display)(EGL_PLATFORM_DEVICE_EXT, dev, std::ptr::null());
            if !dpy.is_null() {
                return Ok(dpy);
            }
        }
        bail!("no EGL device matching rendernode {rendernode:?}")
    }
}

impl crate::FrameMaterializer for EglMaterializer {
    fn name(&self) -> &'static str {
        "egl"
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }


    fn prepare(&mut self, fmt: &FrameFormat, dmabuf: Option<&DmabufScanout>) -> Result<()> {
        // all supported single-plane 4bpp RGB variants (see
        // is_supported_rgb_fourcc); the blit normalizes memory order and bit
        // depth, and readback always produces XR24 (BGRX) bytes
        if !is_supported_rgb_fourcc(fmt.fourcc) {
            bail!("EGL materializer: unsupported fourcc {:#010x}", fmt.fourcc);
        }
        if fmt.stride % 4 != 0 {
            bail!(
                "EGL materializer: stride {} not a multiple of 4 (pixel rows)",
                fmt.stride
            );
        }
        self.build_targets(fmt)?;
        if let Some(d) = dmabuf {
            self.import_dmabuf(d)?;
        }
        self.fmt = Some(fmt.clone());
        self.dirty = true;
        Ok(())
    }

    /// Zero-copy staging: allocate `n` persistent-mapped PBOs of
    /// `frame_bytes` each; the returned pointers are registered by the server
    /// as RDMA staging slots. Idempotent: re-binds on reconnect if already
    /// sized correctly (the PBOs outlive sessions). None = fall back to
    /// direct readback (missing ARB_buffer_storage or EGL fence).
    fn bind_staging(&mut self, n: usize, frame_bytes: usize) -> Option<Vec<(*mut u8, usize)>> {
        if let Some(s) = &self.stage {
            if s.pbos.len() == n && s.frame_bytes == frame_bytes {
                return Some(
                    s.maps
                        .iter()
                        .map(|m| (*m as *mut u8, s.frame_bytes))
                        .collect(),
                );
            }
            eprintln!(
                "[egl] staging rebind mismatch (have {} slots of {} bytes, want {n} × {frame_bytes})",
                s.pbos.len(),
                s.frame_bytes
            );
            return None;
        }
        let syms = match BenchSyms::load(&self.egl) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[egl] PBO staging unavailable: {e}");
                return None;
            }
        };
        let Some(bufstorage) = syms.buffer_storage else {
            eprintln!("[egl] PBO staging unavailable: no glBufferStorage (ARB_buffer_storage)");
            return None;
        };
        let fence = match (syms.create_sync, syms.client_wait_sync, syms.destroy_sync) {
            (Some(a), Some(b), Some(c)) => Some((a, b, c)),
            _ => {
                eprintln!("[egl] no EGL_KHR_fence_sync — staging falls back to glFinish sync");
                None
            }
        };
        let mut pbos = Vec::with_capacity(n);
        let mut maps = Vec::with_capacity(n);
        unsafe {
            for _ in 0..n {
                let mut pbo = 0u32;
                (syms.gen_buffers)(1, &mut pbo);
                (self.gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, pbo);
                (bufstorage)(
                    GL_PIXEL_PACK_BUFFER,
                    frame_bytes as isize,
                    std::ptr::null(),
                    GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT,
                );
                if (self.gl.get_error)() != 0 {
                    eprintln!("[egl] PBO staging: glBufferStorage failed");
                    (self.gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, 0);
                    for p in &pbos {
                        (syms.delete_buffers)(1, p);
                    }
                    return None;
                }
                let map = (syms.map_buffer_range)(
                    GL_PIXEL_PACK_BUFFER,
                    0,
                    frame_bytes as isize,
                    GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT,
                );
                (self.gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, 0);
                if map.is_null() {
                    eprintln!("[egl] PBO staging: persistent map failed");
                    for p in &pbos {
                        (syms.delete_buffers)(1, p);
                    }
                    return None;
                }
                pbos.push(pbo);
                maps.push(map);
            }
        }
        eprintln!(
            "[egl] PBO staging: {n} × {frame_bytes}B persistent-mapped (zero-copy send path)"
        );
        self.stage = Some(StageSlots {
            pbos,
            maps,
            frame_bytes,
            fence,
            syms,
        });
        self.stage_work = (0..n).map(|_| None).collect();
        Some(
            self.stage
                .as_ref()
                .unwrap()
                .maps
                .iter()
                .map(|m| (*m as *mut u8, frame_bytes))
                .collect(),
        )
    }

    fn copy_frame(&mut self, out: &mut [u8]) -> Result<(bool, MatTimes)> {
        if !self.dirty {
            return Ok((false, MatTimes::ZERO));
        }
        let fmt = self.fmt.as_ref().ok_or_else(|| anyhow!("no scanout"))?;
        let (_, tex) = self.img.as_ref().ok_or_else(|| {
            anyhow!("no imported dmabuf (plain scanout needs --materialize plain)")
        })?;
        // staging selection: when the caller's buffer IS one of our PBO maps
        // (zero-copy mode, registered server-side), read into that PBO and
        // order the DMA with an EGL fence; otherwise the legacy direct path.
        let out_ptr = out.as_ptr() as *mut core::ffi::c_void;
        let stage_i = self.stage.as_ref().and_then(|s| {
            s.maps.iter().position(|m| *m == out_ptr)
        });
        unsafe {
            // shader blit into the linear target, then read it back
            (self.gl.use_program)(self.prog);
            (self.gl.bind_framebuffer)(GL_FRAMEBUFFER, self.fbo);
            (self.gl.viewport)(0, 0, fmt.width as i32, fmt.height as i32);
            (self.gl.bind_texture)(GL_TEXTURE_EXTERNAL_OES, *tex);
            let t0 = std::time::Instant::now();
            (self.gl.draw_arrays)(GL_TRIANGLES, 0, 3);
            // wall-clock split (coarse): blit = CPU call latency (the draw is
            // queued, not awaited); read = everything GPU-side from "draw
            // queued" to "pixels in host RAM"; fin = fence wait (PBO) or
            // residual glFinish (direct)
            let blit = t0.elapsed();

            // GL_BGRA readback means the bytes land in wire order (B,G,R,X)
            // — the GPU's copy does the channel normalize, the CPU touches
            // nothing but the DMA later. Two destinations:
            //   * PBO staging (zero-copy): readPixels with offset 0 into the
            //     bound PIXEL_PACK buffer (== the server's registered RDMA
            //     slot); an EGL fence orders the DMA before the NIC reads.
            //   * direct: the caller's pinned buffer; synchronous glFinish
            //     (v1 path; QEMU Fence-based sync comes later).
            (self.gl.get_error)(); // clear sticky error; post-read check stays attributable
            (self.gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, 0);
            (self.gl.pixel_storei)(GL_PACK_ROW_LENGTH, (fmt.stride / 4) as i32);
            let t_read0 = std::time::Instant::now();
            let read_target = match (stage_i, &self.stage) {
                (Some(i), Some(stage)) => Some((i, stage.pbos[i])),
                _ => None,
            };
            match read_target {
                Some((i, pbo)) => {
                    (self.gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, pbo);
                    (self.gl.read_pixels)(
                        0,
                        0,
                        fmt.width as i32,
                        fmt.height as i32,
                        GL_BGRA,
                        GL_UNSIGNED_BYTE,
                        std::ptr::null_mut(), // offset 0 into the bound PBO
                    );
                    let read = t_read0.elapsed(); // read call latency (async)
                    // order the DMA before the RDMA engine reads the PBO:
                    // EGL fence (fail-soft to glFinish, which orders the
                    // whole pipeline — the DMA is already in flight)
                    let fin = match &self.stage.as_ref().unwrap().fence {
                        Some((create, wait, destroy)) => {
                            let t1 = std::time::Instant::now();
                            let fence = (create)(
                                self.dpy,
                                EGL_SYNC_FENCE_KHR as i32,
                                std::ptr::null(),
                            );
                            if fence.is_null() {
                                (self.gl.finish)();
                                t1.elapsed()
                            } else {
                                let r = (wait)(
                                    self.dpy,
                                    fence,
                                    EGL_SYNC_FLUSH_COMMANDS_BIT_KHR as i32,
                                    1_000_000_000,
                                );
                                (destroy)(self.dpy, fence);
                                if r != EGL_CONDITION_SATISFIED_KHR as i32 {
                                    // fence failed: order with a full sync
                                    (self.gl.finish)();
                                }
                                t1.elapsed()
                            }
                        }
                        None => {
                            let t1 = std::time::Instant::now();
                            (self.gl.finish)();
                            t1.elapsed()
                        }
                    };
                    (self.gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, 0);
                    let _ = i;
                    self.last_mat = MatTimes {
                        blit_us: blit.as_micros() as u64,
                        read_us: read.as_micros() as u64,
                        fin_us: fin.as_micros() as u64,
                    };
                }
                None => {
                    (self.gl.read_pixels)(
                        0,
                        0,
                        fmt.width as i32,
                        fmt.height as i32,
                        GL_BGRA,
                        GL_UNSIGNED_BYTE,
                        out.as_mut_ptr() as *mut core::ffi::c_void,
                    );
                    let t_after_read = t0.elapsed(); // draw queued + readback DMA done
                    (self.gl.finish)(); // v1 sync; QEMU Fence-based sync comes later
                    let total = t0.elapsed();
                    self.last_mat = MatTimes {
                        blit_us: blit.as_micros() as u64,
                        read_us: (t_after_read - blit).as_micros() as u64,
                        fin_us: (total - t_after_read).as_micros() as u64,
                    };
                }
            }
            let gl_err = (self.gl.get_error)();
            if gl_err != 0 {
                bail!("GL readback failed 0x{gl_err:04x}");
            }
        }
        self.dirty = false;
        Ok((true, self.last_mat))
    }

    fn issue_frame(&mut self, slot: usize, out: &mut [u8]) -> Result<bool> {
        if !self.dirty {
            return Ok(false);
        }
        // PBO staging required for the pipelined path
        let (stage, work) = match (&self.stage, self.stage_work.get_mut(slot)) {
            (Some(s), Some(w)) => (s, w),
            _ => bail!("issue_frame: no PBO staging slot {slot}"),
        };
        debug_assert_eq!(stage.maps[slot], out.as_ptr() as *mut core::ffi::c_void);
        let (create, _wait, _destroy) = match stage.fence {
            Some(f) => f,
            None => bail!("issue_frame: no EGL fence sync"),
        };
        let fmt = self.fmt.as_ref().ok_or_else(|| anyhow!("no scanout"))?;
        let (_, tex) = self.img.as_ref().ok_or_else(|| anyhow!("no imported dmabuf"))?;
        unsafe {
            (self.gl.use_program)(self.prog);
            (self.gl.bind_framebuffer)(GL_FRAMEBUFFER, self.fbo);
            (self.gl.viewport)(0, 0, fmt.width as i32, fmt.height as i32);
            (self.gl.bind_texture)(GL_TEXTURE_EXTERNAL_OES, *tex);
            let t0 = std::time::Instant::now();
            (self.gl.draw_arrays)(GL_TRIANGLES, 0, 3);
            let blit = t0.elapsed();
            (self.gl.get_error)();
            (self.gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, 0);
            (self.gl.pixel_storei)(GL_PACK_ROW_LENGTH, (fmt.stride / 4) as i32);
            (self.gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, stage.pbos[slot]);
            let t_read0 = std::time::Instant::now();
            (self.gl.read_pixels)(
                0,
                0,
                fmt.width as i32,
                fmt.height as i32,
                GL_BGRA,
                GL_UNSIGNED_BYTE,
                std::ptr::null_mut(), // offset 0 into the bound PBO
            );
            let read = t_read0.elapsed();
            let fence = (create)(self.dpy, EGL_SYNC_FENCE_KHR as i32, std::ptr::null());
            if fence.is_null() {
                bail!("issue_frame: eglCreateSyncKHR returned null");
            }
            *work = Some(StageWork {
                fence,
                blit_us: blit.as_micros() as u64,
                read_us: read.as_micros() as u64,
            });
            let gl_err = (self.gl.get_error)();
            (self.gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, 0);
            if gl_err != 0 {
                bail!("issue_frame: GL readback failed 0x{gl_err:04x}");
            }
        }
        self.dirty = false;
        Ok(true)
    }

    fn finish_frame(&mut self, slot: usize, _out: &mut [u8]) -> Result<(bool, MatTimes)> {
        let work = match self.stage_work.get_mut(slot).and_then(|w| w.take()) {
            Some(w) => w,
            None => return Ok((false, MatTimes::ZERO)),
        };
        let (create, wait, destroy) = match self.stage.as_ref().unwrap().fence {
            Some(f) => f,
            None => bail!("finish_frame: no EGL fence sync"),
        };
        unsafe {
            let t1 = std::time::Instant::now();
            let r = (wait)(self.dpy, work.fence, EGL_SYNC_FLUSH_COMMANDS_BIT_KHR as i32, 2_000_000_000);
            (destroy)(self.dpy, work.fence);
            if r == EGL_TIMEOUT_EXPIRED_KHR as i32 {
                bail!("finish_frame: EGL fence timed out — readback DMA never completed");
            }
            if r != EGL_CONDITION_SATISFIED_KHR as i32 {
                bail!("finish_frame: EGL fence wait failed ({r:#06x})");
            }
            self.last_mat = MatTimes {
                blit_us: work.blit_us,
                read_us: work.read_us,
                fin_us: t1.elapsed().as_micros() as u64,
            };
        }
        let _ = create;
        Ok((true, self.last_mat))
    }

    fn drop_frames(&mut self) {
        if let Some(stage) = &self.stage {
            let (_create, _wait, destroy) = match stage.fence {
                Some(f) => f,
                None => return,
            };
            for w in self.stage_work.iter_mut() {
                if let Some(work) = w.take() {
                    unsafe {
                        (destroy)(self.dpy, work.fence);
                    }
                }
            }
        }
    }

    fn handle_event(&mut self, ev: CaptureEvent) -> Result<bool> {
        match ev {
            CaptureEvent::DmabufFrame(..) => {
                self.dirty = true;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn effective_format(&self) -> Option<FrameFormat> {
        self.fmt.clone()
    }
}

// ============================================================================
// Readback benchmark (readbench bin) — compare the ways to move the guest
// dmabuf into host memory, all sharing one EGL context:
//
//   v0  oes-blit-direct  production path via EglMaterializer itself (OES
//                        import + shader blit + glReadPixels + glFinish)
//   A1  tex2d-fbo        dmabuf imported as renderable GL_TEXTURE_2D, FBO
//                        attached directly, no blit, glReadPixels + glFinish
//   A2  renderbuf        dmabuf imported as a renderbuffer
//                        (glEGLImageTargetRenderbufferStorageOES), FBO read
//   B   gettexsub        glGetTextureSubImage straight from the imported
//                        texture — no FBO, no draw at all
//   C   pbo-fence        async readback: persistent-mapped PIXEL_PACK buffer
//                        (triple), EGL fence sync (eglCreateSyncKHR
//                        EGL_SYNC_FENCE_KHR + eglClientWaitSyncKHR); measured
//                        both serial (latency) and pipelined (throughput).
//                        Falls back to per-frame mapped PBO if
//                        ARB_buffer_storage is unavailable.
//
// Every variant ends with a full sync (glFinish / fence wait), so reported
// time = "CPU issue → bytes in host memory". Data validity: each variant is
// checksum-compared against the baseline read (as-is or vertically flipped —
// variants that read the guest buffer without the production blit come out
//                        v0. GL_BGRA everywhere: the wire byte order.
// Fence note: glFenceSync is core GL 3.2 and this driver rejects it on a 3.1
// context (WAIT_FAILED 0x911c measured); EGL_KHR_fence_sync via EGL entry
// points is the working primitive.
// ============================================================================

const GL_RENDERBUFFER: u32 = 0x8D41;
const GL_MAP_READ_BIT: u32 = 0x0001;
const GL_MAP_PERSISTENT_BIT: u32 = 0x0040;
const GL_MAP_COHERENT_BIT: u32 = 0x0080;
const GL_STREAM_READ: u32 = 0x88E2;
// EGL fence sync (EGL_KHR_fence_sync): the correct fence primitive on this
// stack — glFenceSync is core GL 3.2 and this driver rejects it on a 3.1
// context (WAIT_FAILED), while the EGL path is advertised
// (EGL_ANDROID_native_fence_sync present ⇒ KHR fence supported).
// Wait returns (per eglext.h): EGL_FALSE=0 (error), EGL_TIMEOUT_EXPIRED_KHR
// 0x30F5, EGL_CONDITION_SATISFIED_KHR 0x30F6 — no ALREADY_SIGNALED for fences.
const EGL_SYNC_FENCE_KHR: u32 = 0x30F9;
const EGL_SYNC_FLUSH_COMMANDS_BIT_KHR: u32 = 0x0001;
const EGL_TIMEOUT_EXPIRED_KHR: u32 = 0x30F5;
const EGL_CONDITION_SATISFIED_KHR: u32 = 0x30F6;

type FnMapBufferRange =
    unsafe extern "system" fn(u32, isize, isize, u32) -> *mut core::ffi::c_void;
type FnBufferStorage = unsafe extern "system" fn(u32, isize, *const core::ffi::c_void, u32);
type FnBufferData = unsafe extern "system" fn(u32, isize, *const core::ffi::c_void, u32);
type FnUnmapBuffer = unsafe extern "system" fn(u32) -> u32;
type FnGetTexImage =
    unsafe extern "system" fn(u32, i32, u32, u32, i32, *mut core::ffi::c_void);
type FnGetTextureSubImage = unsafe extern "system" fn(
    u32, i32, i32, i32, i32, i32, i32, i32, u32, u32, i32, *mut core::ffi::c_void,
);
type FnGenBuffers = unsafe extern "system" fn(i32, *mut u32);
type FnDeleteBuffers = unsafe extern "system" fn(i32, *const u32);
type FnGenRenderbuffers = unsafe extern "system" fn(i32, *mut u32);
type FnBindRenderbuffer = unsafe extern "system" fn(u32, u32);
type FnFramebufferRenderbuffer = unsafe extern "system" fn(u32, u32, u32, u32);
type FnEglImageTargetRenderbuffer = unsafe extern "system" fn(u32, EglHandle);
type FnCreateSyncKhr = unsafe extern "system" fn(EglHandle, i32, *const i32) -> EglHandle;
type FnClientWaitSyncKhr =
    unsafe extern "system" fn(EglHandle, EglHandle, i32, u64) -> i32;
type FnDestroySyncKhr = unsafe extern "system" fn(EglHandle, EglHandle) -> u32;

pub struct ReadBenchOpts {
    pub iterations: usize,
    pub warmup: usize,
    pub rendernode: Option<String>,
    /// keep the guest repainting during the bench (pointer churn): measures
    /// the contended readback — the production condition. Checksum
    /// validation is disabled under load.
    pub load: bool,
    /// 0 = static mode (fixed buffer, N iterations per variant). N > 0 =
    /// event-driven: follow the capture stream like the production server
    /// (re-import on Scanout, read on every frame event) for N frames —
    /// implies pointer churn. Measures v0 + C per event.
    pub events: usize,
}

/// one timed iteration: issue (call latency of the enqueuing GL calls) and
/// sync (the blocking wait that guarantees the bytes are in host memory)
#[derive(Clone, Copy)]
struct BSample {
    issue_us: u64,
    sync_us: u64,
}

impl BSample {
    fn total(&self) -> u64 {
        self.issue_us + self.sync_us
    }
}

/// dlsym(libGL.so.1) first, eglGetProcAddress fallback — the lessons from the
/// fence attempt: eglGetProc hands out stubs for core symbols on this driver,
/// while 4.x extension entry points may only exist via eglGetProc. Core-GL
/// symbols are hard requirements (bail); capability symbols stay Option.
struct BenchSyms {
    /// kept alive for the process lifetime: the dlsym'd GL entry points stay
    /// valid regardless, but keeping the handle documents the dependency
    #[allow(dead_code)]
    gl_lib: Option<libloading::Library>,
    // core GL — required
    map_buffer_range: FnMapBufferRange,
    buffer_data: FnBufferData,
    unmap_buffer: FnUnmapBuffer,
    gen_buffers: FnGenBuffers,
    delete_buffers: FnDeleteBuffers,
    gen_renderbuffers: FnGenRenderbuffers,
    bind_renderbuffer: FnBindRenderbuffer,
    framebuffer_renderbuffer: FnFramebufferRenderbuffer,
    // capability-gated
    buffer_storage: Option<FnBufferStorage>,
    get_tex_image: Option<FnGetTexImage>,
    get_texture_sub_image: Option<FnGetTextureSubImage>,
    egl_image_target_renderbuffer: Option<FnEglImageTargetRenderbuffer>,
    // EGL fence sync (loaded via eglGetProcAddress — EGL entry points are
    // exactly what it exists for)
    create_sync: Option<FnCreateSyncKhr>,
    client_wait_sync: Option<FnClientWaitSyncKhr>,
    destroy_sync: Option<FnDestroySyncKhr>,
}

impl BenchSyms {
    fn load(egl: &Egl) -> Result<Self> {
        let gl_lib = unsafe {
            libloading::Library::new("libGL.so.1")
                .or_else(|_| libloading::Library::new("libGL.so"))
                .ok()
        };
        fn sym_ptr(
            egl: &Egl,
            gl_lib: &Option<libloading::Library>,
            name: &str,
        ) -> *const core::ffi::c_void {
            if let Some(lib) = gl_lib {
                if let Ok(s) = unsafe {
                    libloading::Library::get::<*mut core::ffi::c_void>(lib, name.as_bytes())
                } {
                    let p = *s as *const core::ffi::c_void;
                    if !p.is_null() {
                        return p;
                    }
                }
            }
            let c = std::ffi::CString::new(name).unwrap();
            let p = unsafe { (egl.get_proc_address)(c.as_ptr()) };
            if p.is_null() {
                std::ptr::null()
            } else {
                p
            }
        }
        macro_rules! sym {
            ($name:literal) => {{
                let p = sym_ptr(egl, &gl_lib, $name);
                (!p.is_null()).then(|| unsafe {
                    std::mem::transmute_copy::<*const core::ffi::c_void, _>(&p)
                })
            }};
        }
        macro_rules! req {
            ($name:literal) => {
                sym!($name).with_context(|| format!("missing GL symbol {}", $name))?
            };
        }
        Ok(BenchSyms {
            map_buffer_range: req!("glMapBufferRange"),
            buffer_data: req!("glBufferData"),
            unmap_buffer: req!("glUnmapBuffer"),
            gen_buffers: req!("glGenBuffers"),
            delete_buffers: req!("glDeleteBuffers"),
            gen_renderbuffers: req!("glGenRenderbuffers"),
            bind_renderbuffer: req!("glBindRenderbuffer"),
            framebuffer_renderbuffer: req!("glFramebufferRenderbuffer"),
            buffer_storage: sym!("glBufferStorage"),
            get_tex_image: sym!("glGetTexImage"),
            get_texture_sub_image: sym!("glGetTextureSubImage"),
            egl_image_target_renderbuffer: sym!("glEGLImageTargetRenderbufferStorageOES"),
            create_sync: sym!("eglCreateSyncKHR"),
            client_wait_sync: sym!("eglClientWaitSyncKHR"),
            destroy_sync: sym!("eglDestroySyncKHR"),
            gl_lib,
        })
    }
}

/// (base_sum, base_flipped_sum) — validation targets for the variants
type VCheck = (u64, u64);

// ---- ibverbs MR probe: can the verbs layer register the PBO's pages? -------
// If ibv_reg_mr accepts the persistent-mapped PBO region, the RDMA engine may
// be able to read it directly (zero-copy C: GPU DMA → PBO → NIC). Full proof
// needs the NIC actually transferring — this probe only proves pinning works.
const IBV_ACCESS_LOCAL_WRITE: u32 = 1;
const IBV_ACCESS_REMOTE_READ: u32 = 4;

type FnIbvGetDeviceList = unsafe extern "system" fn(*mut i32) -> *mut *mut core::ffi::c_void;
type FnIbvFreeDeviceList = unsafe extern "system" fn(*mut *mut core::ffi::c_void);
type FnIbvGetDeviceName =
    unsafe extern "system" fn(*mut core::ffi::c_void) -> *const u8;
type FnIbvOpenDevice =
    unsafe extern "system" fn(*mut core::ffi::c_void) -> *mut core::ffi::c_void;
type FnIbvAllocPd =
    unsafe extern "system" fn(*mut core::ffi::c_void) -> *mut core::ffi::c_void;
type FnIbvRegMr = unsafe extern "system" fn(
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    usize,
    u32,
) -> *mut core::ffi::c_void;
type FnIbvDeregMr = unsafe extern "system" fn(*mut core::ffi::c_void) -> i32;
type FnIbvDeallocPd = unsafe extern "system" fn(*mut core::ffi::c_void) -> i32;
type FnIbvCloseDevice = unsafe extern "system" fn(*mut core::ffi::c_void) -> i32;

/// Registers one memory range with the first working RDMA device; all verbs
/// resources are released on drop.
struct MrProbe {
    lib: libloading::Library,
    devs: *mut *mut core::ffi::c_void,
    ctx: *mut core::ffi::c_void,
    pd: *mut core::ffi::c_void,
    mr: *mut core::ffi::c_void,
}

impl MrProbe {
    unsafe fn register(addr: *mut core::ffi::c_void, size: usize) -> Result<MrProbe> {
        let lib = libloading::Library::new("libibverbs.so.1")
            .or_else(|_| libloading::Library::new("libibverbs.so"))
            .context("dlopen libibverbs (MR probe)")?;
        macro_rules! dlsym {
            ($name:literal) => {
                unsafe {
                    std::mem::transmute_copy::<*mut core::ffi::c_void, _>(&get_sym(
                        &lib,
                        $name,
                    )?)
                }
            };
        }
        let get_device_list: FnIbvGetDeviceList = dlsym!("ibv_get_device_list");
        let free_device_list: FnIbvFreeDeviceList = dlsym!("ibv_free_device_list");
        let get_device_name: FnIbvGetDeviceName = dlsym!("ibv_get_device_name");
        let open_device: FnIbvOpenDevice = dlsym!("ibv_open_device");
        let alloc_pd: FnIbvAllocPd = dlsym!("ibv_alloc_pd");
        let reg_mr: FnIbvRegMr = dlsym!("ibv_reg_mr");
        let dealloc_pd: FnIbvDeallocPd = dlsym!("ibv_dealloc_pd");
        let close_device: FnIbvCloseDevice = dlsym!("ibv_close_device");
        let mut n = 0i32;
        let devs = get_device_list(&mut n);
        if devs.is_null() || n < 1 {
            if !devs.is_null() {
                free_device_list(devs);
            }
            bail!("no RDMA devices found (ibv_get_device_list)");
        }
        let mut ok: Option<MrProbe> = None;
        for i in 0..n as usize {
            let dev = *devs.add(i);
            let name = if dev.is_null() {
                "?".to_string()
            } else {
                let p = get_device_name(dev);
                if p.is_null() {
                    "?".into()
                } else {
                    std::ffi::CStr::from_ptr(p as *const i8)
                        .to_string_lossy()
                        .into_owned()
                }
            };
            let ctx = open_device(dev);
            if ctx.is_null() {
                eprintln!("[bench] MR probe: open_device({name}) failed");
                continue;
            }
            let pd = alloc_pd(ctx);
            if pd.is_null() {
                eprintln!("[bench] MR probe: alloc_pd({name}) failed");
                close_device(ctx);
                continue;
            }
            let mr = reg_mr(pd, addr, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
            if mr.is_null() {
                let e = std::io::Error::last_os_error();
                eprintln!("[bench] MR probe: reg_mr({name}) failed: {e}");
                dealloc_pd(pd);
                close_device(ctx);
                continue;
            }
            eprintln!("[bench] MR probe: registered PBO pages on {name} (lkey usable)");
            ok = Some(MrProbe { lib, devs, ctx, pd, mr });
            break;
        }
        match ok {
            Some(p) => Ok(p),
            None => {
                free_device_list(devs);
                bail!("ibv_reg_mr over the PBO pages failed on all RDMA devices")
            }
        }
    }
}

impl Drop for MrProbe {
    fn drop(&mut self) {
        unsafe {
            macro_rules! dlsym {
                ($name:literal) => {
                    std::mem::transmute::<*mut core::ffi::c_void, _>(get_sym(&self.lib, $name).unwrap())
                };
            }
            let dereg_mr: FnIbvDeregMr = dlsym!("ibv_dereg_mr");
            let dealloc_pd: FnIbvDeallocPd = dlsym!("ibv_dealloc_pd");
            let close_device: FnIbvCloseDevice = dlsym!("ibv_close_device");
            let free_device_list: FnIbvFreeDeviceList = dlsym!("ibv_free_device_list");
            dereg_mr(self.mr);
            dealloc_pd(self.pd);
            close_device(self.ctx);
            free_device_list(self.devs);
        }
    }
}

/// Validation label: under load the guest frame content changes while the
/// bench runs, so cross-variant checksums are meaningless.
fn bench_data_label(out: &[u8], chk: &VCheck, load: bool) -> &'static str {
    if load {
        "load-chk-off"
    } else {
        bench_validate(out, chk)
    }
}

/// Create the shared EGLImage for the scanout dmabuf (same attribute list as
/// the production import).
unsafe fn bench_egl_image(egl: &Egl, dpy: EglHandle, d: &DmabufScanout) -> Result<EglHandle> {
    let fmt = &d.format;
    if d.fds.len() != 1 || fmt.num_planes != 1 {
        bail!("multi-plane dmabuf not supported (planes={})", d.fds.len());
    }
    let modifier = fmt.modifier;
    let mut a: Vec<i32> = vec![
        EGL_WIDTH, fmt.width as i32,
        EGL_HEIGHT, fmt.height as i32,
        EGL_LINUX_DRM_FOURCC_EXT, fmt.fourcc as i32,
        EGL_DMA_BUF_PLANE0_FD_EXT, d.fds[0].as_raw_fd(),
        EGL_DMA_BUF_PLANE0_OFFSET_EXT, d.offsets[0] as i32,
        EGL_DMA_BUF_PLANE0_PITCH_EXT, d.strides[0] as i32,
        EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (modifier & 0xffff_ffff) as i32,
        EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, (modifier >> 32) as i32,
        EGL_NONE,
    ];
    let img = (egl.create_image_khr)(
        dpy,
        std::ptr::null_mut(),
        EGL_LINUX_DMA_BUF_EXT,
        std::ptr::null_mut(),
        a.as_mut_ptr(),
    );
    if img.is_null() {
        bail!("eglCreateImageKHR failed (err {:#06x})", egl.get_error());
    }
    Ok(img)
}

fn bench_checksum(buf: &[u8]) -> u64 {
    let mut acc = 0u64;
    for c in buf.chunks_exact(8) {
        acc = acc.wrapping_add(u64::from_le_bytes(c.try_into().unwrap()));
    }
    acc
}

/// checksum of the buffer's rows in reversed order — the expected content for
/// variants that read the guest dmabuf without the production blit
fn bench_checksum_flipped(buf: &[u8], stride: usize, height: usize) -> u64 {
    let mut acc = 0u64;
    for y in 0..height {
        let row = &buf[(height - 1 - y) * stride..(height - y) * stride];
        for c in row.chunks_exact(8) {
            acc = acc.wrapping_add(u64::from_le_bytes(c.try_into().unwrap()));
        }
    }
    acc
}

fn bench_validate(out: &[u8], chk: &VCheck) -> &'static str {
    let s = bench_checksum(out);
    if s == chk.0 {
        "as-baseline"
    } else if s == chk.1 {
        "v-flipped"
    } else {
        "MISMATCH"
    }
}

fn bench_stats(samples: &[BSample]) -> (u64, u64, u64, u64, f64) {
    let mut issue: Vec<u64> = samples.iter().map(|s| s.issue_us).collect();
    let mut sync: Vec<u64> = samples.iter().map(|s| s.sync_us).collect();
    let mut total: Vec<u64> = samples.iter().map(|s| s.total()).collect();
    issue.sort_unstable();
    sync.sort_unstable();
    total.sort_unstable();
    let med = |v: &[u64]| v[v.len() / 2];
    let mean = samples.iter().map(|s| s.total() as f64).sum::<f64>() / samples.len() as f64;
    (med(&issue), med(&sync), med(&total), total[0], mean)
}

fn bench_print(
    name: &str,
    note: &str,
    samples: &[BSample],
    size: usize,
    data_ok: Option<&str>,
    pipe_us_per_frame: Option<f64>,
) {
    let pipe = pipe_us_per_frame
        .map(|p| format!("  pipelined {p:7.0} µs/f"))
        .unwrap_or_default();
    if samples.is_empty() {
        println!("{name:<22} UNAVAILABLE  ({note}){pipe}");
        return;
    }
    let (mi, ms, mt, mint, mean) = bench_stats(samples);
    let gbs = size as f64 / mt as f64 / 1000.0; // B/µs = MB/ms = GB/s
    let data = data_ok.unwrap_or("n/a");
    println!(
        "{name:<22} issue {mi:5} µs  sync {ms:5} µs  total {mt:5} µs  min {mint:5} µs  \
         mean {mean:7.0} µs  {gbs:4.2} GB/s  {data:<11}{pipe}",
    );
    if !note.is_empty() {
        println!("{:<22}   note: {note}", "");
    }
}

/// Shared body for the direct-read variants: caller has bound the FBO and set
/// GL_PACK_ROW_LENGTH; measures glReadPixels call latency (issue) + glFinish
/// (sync) and checks GL errors.
unsafe fn bench_direct_read(
    gl: &Gl,
    out: *mut core::ffi::c_void,
    w: i32,
    h: i32,
) -> Result<BSample> {
    (gl.get_error)(); // clear sticky error so the post-read check is attributable
    let t0 = std::time::Instant::now();
    (gl.read_pixels)(0, 0, w, h, GL_BGRA, GL_UNSIGNED_BYTE, out);
    let issue = t0.elapsed();
    let t1 = std::time::Instant::now();
    (gl.finish)();
    let sync = t1.elapsed();
    let err = (gl.get_error)();
    if err != 0 {
        bail!("glReadPixels failed 0x{err:04x}");
    }
    Ok(BSample {
        issue_us: issue.as_micros() as u64,
        sync_us: sync.as_micros() as u64,
    })
}

/// The production draw: OES-sampled blit into the linear target (used by C).
unsafe fn bench_blit_into_linear(
    gl: &Gl,
    mat: &EglMaterializer,
    w: i32,
    h: i32,
) -> Result<()> {
    (gl.use_program)(mat.prog);
    (gl.bind_framebuffer)(GL_FRAMEBUFFER, mat.fbo);
    (gl.viewport)(0, 0, w, h);
    if let Some((_, tex)) = mat.img.as_ref() {
        (gl.bind_texture)(GL_TEXTURE_EXTERNAL_OES, *tex);
    }
    (gl.draw_arrays)(GL_TRIANGLES, 0, 3);
    let err = (gl.get_error)();
    if err != 0 {
        bail!("blit draw failed 0x{err:04x}");
    }
    Ok(())
}

/// A1: import the dmabuf as a renderable GL_TEXTURE_2D, FBO attached directly
/// (no blit), glReadPixels + glFinish.
unsafe fn run_variant_tex2d(
    gl: &Gl,
    egl: &Egl,
    dpy: EglHandle,
    d: &DmabufScanout,
    w: i32,
    h: i32,
    row_len: i32,
    size: usize,
    chk: &VCheck,
    opts: &ReadBenchOpts,
) {
    let name = "A1 tex2d-fbo";
    let img = match bench_egl_image(egl, dpy, d) {
        Ok(i) => i,
        Err(e) => {
            bench_print(name, &format!("import failed: {e}"), &[], size, None, None);
            return;
        }
    };
    let mut tex = 0u32;
    let mut fbo = 0u32;
    (gl.gen_textures)(1, &mut tex);
    (gl.bind_texture)(GL_TEXTURE_2D, tex);
    (gl.image_target_texture)(GL_TEXTURE_2D, img);
    let err = (gl.get_error)();
    if err != 0 {
        bench_print(
            name,
            &format!("EGLImage import as GL_TEXTURE_2D rejected 0x{err:04x}"),
            &[],
            size,
            None,
            None,
        );
        (gl.bind_texture)(GL_TEXTURE_2D, 0);
        (gl.delete_textures)(1, &[tex] as *const u32);
        (egl.destroy_image)(dpy, img);
        return;
    }
    (gl.gen_framebuffers)(1, &mut fbo);
    (gl.bind_framebuffer)(GL_FRAMEBUFFER, fbo);
    (gl.framebuffer_texture_2d)(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    let st = (gl.check_framebuffer_status)(GL_FRAMEBUFFER);
    if st != GL_FRAMEBUFFER_COMPLETE {
        bench_print(
            name,
            &format!("dmabuf texture as FBO attachment incomplete 0x{st:04x}"),
            &[],
            size,
            None,
            None,
        );
        (gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
        (gl.delete_framebuffers)(1, &[fbo] as *const u32);
        (gl.bind_texture)(GL_TEXTURE_2D, 0);
        (gl.delete_textures)(1, &[tex] as *const u32);
        (egl.destroy_image)(dpy, img);
        return;
    }
    (gl.pixel_storei)(GL_PACK_ROW_LENGTH, row_len);
    let mut out = vec![0u8; size];
    for _ in 0..opts.warmup {
        if let Err(e) = bench_direct_read(gl, out.as_mut_ptr() as *mut _, w, h) {
            bench_print(name, &format!("read failed: {e}"), &[], size, None, None);
            (gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
            (gl.delete_framebuffers)(1, &[fbo] as *const u32);
            (gl.bind_texture)(GL_TEXTURE_2D, 0);
            (gl.delete_textures)(1, &[tex] as *const u32);
            (egl.destroy_image)(dpy, img);
            return;
        }
    }
    let mut samples = Vec::new();
    for _ in 0..opts.iterations {
        match bench_direct_read(gl, out.as_mut_ptr() as *mut _, w, h) {
            Ok(s) => samples.push(s),
            Err(e) => {
                bench_print(name, &format!("read failed: {e}"), &[], size, None, None);
                (gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
                (gl.delete_framebuffers)(1, &[fbo] as *const u32);
                (gl.bind_texture)(GL_TEXTURE_2D, 0);
                (gl.delete_textures)(1, &[tex] as *const u32);
                (egl.destroy_image)(dpy, img);
                return;
            }
        }
    }
    let data = bench_data_label(&out, chk, opts.load);
    bench_print(name, "", &samples, size, Some(data), None);
    (gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
    (gl.delete_framebuffers)(1, &[fbo] as *const u32);
    (gl.bind_texture)(GL_TEXTURE_2D, 0);
    (gl.delete_textures)(1, &[tex] as *const u32);
    (egl.destroy_image)(dpy, img);
}

/// A2: import the dmabuf as a renderbuffer
/// (glEGLImageTargetRenderbufferStorageOES), FBO read.
unsafe fn run_variant_renderbuf(
    gl: &Gl,
    egl: &Egl,
    dpy: EglHandle,
    syms: &BenchSyms,
    d: &DmabufScanout,
    w: i32,
    h: i32,
    row_len: i32,
    size: usize,
    chk: &VCheck,
    opts: &ReadBenchOpts,
) {
    let name = "A2 renderbuf";
    let Some(target_rb) = syms.egl_image_target_renderbuffer else {
        bench_print(name, "no glEGLImageTargetRenderbufferStorageOES", &[], size, None, None);
        return;
    };
    let img = match bench_egl_image(egl, dpy, d) {
        Ok(i) => i,
        Err(e) => {
            bench_print(name, &format!("import failed: {e}"), &[], size, None, None);
            return;
        }
    };
    let mut rb = 0u32;
    let mut fbo = 0u32;
    (syms.gen_renderbuffers)(1, &mut rb);
    (syms.bind_renderbuffer)(GL_RENDERBUFFER, rb);
    target_rb(GL_RENDERBUFFER, img);
    let err = (gl.get_error)();
    if err != 0 {
        bench_print(
            name,
            &format!("renderbuffer import rejected 0x{err:04x}"),
            &[],
            size,
            None,
            None,
        );
        (syms.bind_renderbuffer)(GL_RENDERBUFFER, 0);
        (egl.destroy_image)(dpy, img);
        return;
    }
    (gl.gen_framebuffers)(1, &mut fbo);
    (gl.bind_framebuffer)(GL_FRAMEBUFFER, fbo);
    (syms.framebuffer_renderbuffer)(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    let st = (gl.check_framebuffer_status)(GL_FRAMEBUFFER);
    if st != GL_FRAMEBUFFER_COMPLETE {
        bench_print(
            name,
            &format!("renderbuffer FBO incomplete 0x{st:04x}"),
            &[],
            size,
            None,
            None,
        );
        (gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
        (gl.delete_framebuffers)(1, &[fbo] as *const u32);
        (syms.bind_renderbuffer)(GL_RENDERBUFFER, 0);
        (egl.destroy_image)(dpy, img);
        return;
    }
    (gl.pixel_storei)(GL_PACK_ROW_LENGTH, row_len);
    let mut out = vec![0u8; size];
    for _ in 0..opts.warmup {
        if let Err(e) = bench_direct_read(gl, out.as_mut_ptr() as *mut _, w, h) {
            bench_print(name, &format!("read failed: {e}"), &[], size, None, None);
            (gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
            (gl.delete_framebuffers)(1, &[fbo] as *const u32);
            (syms.bind_renderbuffer)(GL_RENDERBUFFER, 0);
            (egl.destroy_image)(dpy, img);
            return;
        }
    }
    let mut samples = Vec::new();
    for _ in 0..opts.iterations {
        match bench_direct_read(gl, out.as_mut_ptr() as *mut _, w, h) {
            Ok(s) => samples.push(s),
            Err(e) => {
                bench_print(name, &format!("read failed: {e}"), &[], size, None, None);
                (gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
                (gl.delete_framebuffers)(1, &[fbo] as *const u32);
                (syms.bind_renderbuffer)(GL_RENDERBUFFER, 0);
                (egl.destroy_image)(dpy, img);
                return;
            }
        }
    }
    let data = bench_data_label(&out, chk, opts.load);
    bench_print(name, "", &samples, size, Some(data), None);
    (gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
    (gl.delete_framebuffers)(1, &[fbo] as *const u32);
    (syms.bind_renderbuffer)(GL_RENDERBUFFER, 0);
    (egl.destroy_image)(dpy, img);
}

/// B: glGetTextureSubImage straight from the imported (2D) texture — no FBO,
/// no draw. Falls back to glGetTexImage if the 4.5 entry point is missing.
unsafe fn run_variant_gettexsub(
    gl: &Gl,
    egl: &Egl,
    dpy: EglHandle,
    syms: &BenchSyms,
    d: &DmabufScanout,
    w: i32,
    h: i32,
    row_len: i32,
    size: usize,
    chk: &VCheck,
    opts: &ReadBenchOpts,
) {
    let name = "B  gettexsub";
    let getter = syms.get_texture_sub_image;
    let getter_ti = syms.get_tex_image;
    if getter.is_none() && getter_ti.is_none() {
        bench_print(name, "no texture-readback GL entry point", &[], size, None, None);
        return;
    }
    let note = if getter.is_none() {
        "glGetTextureSubImage missing; using glGetTexImage"
    } else {
        ""
    };
    let img = match bench_egl_image(egl, dpy, d) {
        Ok(i) => i,
        Err(e) => {
            bench_print(name, &format!("import failed: {e}"), &[], size, None, None);
            return;
        }
    };
    let mut tex = 0u32;
    (gl.gen_textures)(1, &mut tex);
    (gl.bind_texture)(GL_TEXTURE_2D, tex);
    (gl.image_target_texture)(GL_TEXTURE_2D, img);
    let err = (gl.get_error)();
    if err != 0 {
        bench_print(
            name,
            &format!("needs a renderable GL_TEXTURE_2D import; rejected 0x{err:04x}"),
            &[],
            size,
            None,
            None,
        );
        (gl.bind_texture)(GL_TEXTURE_2D, 0);
        (gl.delete_textures)(1, &[tex] as *const u32);
        (egl.destroy_image)(dpy, img);
        return;
    }
    (gl.pixel_storei)(GL_PACK_ROW_LENGTH, row_len);
    let mut out = vec![0u8; size];
    let one_read = |out: &mut Vec<u8>| -> Result<BSample> {
        (gl.get_error)();
        let t0 = std::time::Instant::now();
        if let Some(f) = getter {
            f(tex, 0, 0, 0, 0, w, h, 1, GL_BGRA, GL_UNSIGNED_BYTE, size as i32,
              out.as_mut_ptr() as *mut _);
        } else {
            // glGetTexImage(target, level, format, type, pixels) — legacy
            // desktop path; only tried if dlsym found the symbol
            getter_ti.unwrap()(
                GL_TEXTURE_2D,
                0,
                GL_BGRA,
                GL_UNSIGNED_BYTE,
                size as i32,
                out.as_mut_ptr() as *mut _,
            );
        }
        let issue = t0.elapsed();
        let t1 = std::time::Instant::now();
        (gl.finish)();
        let sync = t1.elapsed();
        let err = (gl.get_error)();
        if err != 0 {
            bail!("texture readback failed 0x{err:04x}");
        }
        Ok(BSample {
            issue_us: issue.as_micros() as u64,
            sync_us: sync.as_micros() as u64,
        })
    };
    for _ in 0..opts.warmup {
        if let Err(e) = one_read(&mut out) {
            bench_print(name, &format!("{e}"), &[], size, None, None);
            (gl.bind_texture)(GL_TEXTURE_2D, 0);
            (gl.delete_textures)(1, &[tex] as *const u32);
            (egl.destroy_image)(dpy, img);
            return;
        }
    }
    let mut samples = Vec::new();
    for _ in 0..opts.iterations {
        match one_read(&mut out) {
            Ok(s) => samples.push(s),
            Err(e) => {
                bench_print(name, &format!("{e}"), &[], size, None, None);
                (gl.bind_texture)(GL_TEXTURE_2D, 0);
                (gl.delete_textures)(1, &[tex] as *const u32);
                (egl.destroy_image)(dpy, img);
                return;
            }
        }
    }
    let data = bench_data_label(&out, chk, opts.load);
    bench_print(name, note, &samples, size, Some(data), None);
    (gl.bind_texture)(GL_TEXTURE_2D, 0);
    (gl.delete_textures)(1, &[tex] as *const u32);
    (egl.destroy_image)(dpy, img);
}

/// PBO setup for the C variant (shared by static and event-driven modes).
/// Returns (pbos, persistent mapped pointers or nulls, persistent?).
unsafe fn bench_pbo_setup(
    gl: &Gl,
    syms: &BenchSyms,
    size: usize,
) -> Result<([u32; 3], [*mut core::ffi::c_void; 3], bool)> {
    let mut pbos = [0u32; 3];
    let mut maps: [*mut core::ffi::c_void; 3] = [std::ptr::null_mut(); 3];
    if let Some(bufstorage) = syms.buffer_storage {
        (syms.gen_buffers)(3, pbos.as_mut_ptr());
        for (i, &pbo) in pbos.iter().enumerate() {
            (gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, pbo);
            (bufstorage)(
                GL_PIXEL_PACK_BUFFER,
                size as isize,
                std::ptr::null(),
                GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT,
            );
            let err = (gl.get_error)();
            if err != 0 {
                (gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, 0);
                (syms.delete_buffers)(3, pbos.as_ptr());
                bail!("glBufferStorage failed 0x{err:04x}");
            }
            maps[i] = (syms.map_buffer_range)(
                GL_PIXEL_PACK_BUFFER,
                0,
                size as isize,
                GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT,
            );
            if maps[i].is_null() {
                (gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, 0);
                (syms.delete_buffers)(3, pbos.as_ptr());
                bail!("persistent map failed");
            }
        }
        (gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, 0);
        Ok((pbos, maps, true))
    } else {
        (syms.gen_buffers)(1, pbos.as_mut_ptr());
        (gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, pbos[0]);
        (syms.buffer_data)(
            GL_PIXEL_PACK_BUFFER,
            size as isize,
            std::ptr::null(),
            GL_STREAM_READ,
        );
        (gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, 0);
        Ok((pbos, maps, false))
    }
}

/// one async read into `pbo`: issue (read call) + EGL fence wait. Returns
/// (issue_us, sync_us). Fences are created+deleted per call (bench only).
unsafe fn bench_pbo_iter(
    gl: &Gl,
    dpy: EglHandle,
    create_sync: FnCreateSyncKhr,
    wait_sync: FnClientWaitSyncKhr,
    destroy_sync: FnDestroySyncKhr,
    pbo: u32,
    w: i32,
    h: i32,
    row_len: i32,
) -> Result<(u64, u64)> {
    (gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, pbo);
    (gl.pixel_storei)(GL_PACK_ROW_LENGTH, row_len);
    (gl.get_error)();
    let t0 = std::time::Instant::now();
    (gl.read_pixels)(0, 0, w, h, GL_BGRA, GL_UNSIGNED_BYTE, std::ptr::null_mut());
    let issue = t0.elapsed();
    let fence = (create_sync)(dpy, EGL_SYNC_FENCE_KHR as i32, std::ptr::null());
    if fence.is_null() {
        bail!("eglCreateSyncKHR returned null");
    }
    let t1 = std::time::Instant::now();
    let r = (wait_sync)(dpy, fence, EGL_SYNC_FLUSH_COMMANDS_BIT_KHR as i32, 1_000_000_000);
    let sync = t1.elapsed();
    (destroy_sync)(dpy, fence);
    let err = (gl.get_error)();
    (gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, 0);
    if err != 0 {
        bail!("glReadPixels(PBO) failed 0x{err:04x}");
    }
    if r == EGL_TIMEOUT_EXPIRED_KHR as i32 {
        bail!("EGL fence timed out — readback DMA never completed");
    }
    if r != EGL_CONDITION_SATISFIED_KHR as i32 {
        bail!("EGL fence wait failed ({r:#06x})");
    }
    Ok((issue.as_micros() as u64, sync.as_micros() as u64))
}

/// C: async readback into a PIXEL_PACK buffer with fence sync. Persistent
/// mapping (ARB_buffer_storage) when available; per-frame mapped otherwise.
/// Latency = serial per-frame issue+fence-wait; pipelined = 3 frames issued
/// back-to-back, then waited — the overlap the mat∥send split would exploit.
/// Runs on the production linear FBO (same draw as the baseline) so it
/// isolates the transport difference against v0.
unsafe fn run_variant_pbo(
    gl: &Gl,
    syms: &BenchSyms,
    mat: &EglMaterializer,
    dpy: EglHandle,
    w: i32,
    h: i32,
    row_len: i32,
    size: usize,
    chk: &VCheck,
    opts: &ReadBenchOpts,
) {
    let name = "C  pbo-fence";
    // fence primitive: EGL_KHR_fence_sync. glFenceSync was measured broken on
    // this driver (WAIT_FAILED on a 3.1 context); the EGL entry points are the
    // reliable path (EGL_ANDROID_native_fence_sync advertised).
    let (create_sync, wait_sync, destroy_sync) = match (
        syms.create_sync,
        syms.client_wait_sync,
        syms.destroy_sync,
    ) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => {
            bench_print(
                name,
                "no EGL_KHR_fence_sync (eglCreateSyncKHR/eglClientWaitSyncKHR)",
                &[],
                size,
                None,
                None,
            );
            return;
        }
    };
    if let Err(e) = bench_blit_into_linear(gl, mat, w, h) {
        bench_print(name, &format!("blit failed: {e}"), &[], size, None, None);
        return;
    }
    let (pbos, maps, persistent) = match bench_pbo_setup(gl, syms, size) {
        Ok(x) => x,
        Err(e) => {
            bench_print(name, &format!("{e}"), &[], size, None, None);
            return;
        }
    };
    let maps: [*mut core::ffi::c_void; 3] = maps;

    // -- serial latency: per-iteration issue + fence wait, round-robin slots
    let mut samples = Vec::new();
    for it in 0..opts.iterations {
        match bench_pbo_iter(
            gl, dpy, create_sync, wait_sync, destroy_sync,
            pbos[it % 3], w, h, row_len,
        ) {
            Ok((is, ss)) => samples.push(BSample { issue_us: is, sync_us: ss }),
            Err(e) => {
                bench_print(name, &format!("{e}"), &[], size, None, None);
                (syms.delete_buffers)(3, pbos.as_ptr());
                return;
            }
        }
    }

    // zero-copy probe: can the verbs layer register the PBO's pages? Only
    // meaningful for the persistent path (stable address for the lifetime).
    // Binding lives to the end of this function: it holds the verbs handles.
    let _probe_keepalive = if persistent {
        match MrProbe::register(maps[0], size) {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("[bench] MR probe: {e}");
                None
            }
        }
    } else {
        None
    };

    // validate the last-written slot
    let last = (opts.iterations - 1) % 3;
    let data;
    if persistent {
        let slice = std::slice::from_raw_parts(maps[last] as *const u8, size);
        data = bench_data_label(slice, chk, opts.load);
    } else {
        let mut scratch = vec![0u8; size];
        (gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, pbos[last]);
        let p = (syms.map_buffer_range)(GL_PIXEL_PACK_BUFFER, 0, size as isize, GL_MAP_READ_BIT);
        scratch.copy_from_slice(std::slice::from_raw_parts(p as *const u8, size));
        (syms.unmap_buffer)(GL_PIXEL_PACK_BUFFER);
        (gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, 0);
        data = bench_data_label(&scratch, chk, opts.load);
    }

    // -- pipelined throughput: 3 frames issued back-to-back, then all waits
    let rounds = (opts.iterations / 3).max(3);
    let mut total_us = 0u64;
    let mut pipe_err: Option<String> = None;
    'pipe: for _ in 0..rounds {
        let t0 = std::time::Instant::now();
        let mut fences = [std::ptr::null_mut(); 3];
        for i in 0..3 {
            (gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, pbos[i]);
            (gl.pixel_storei)(GL_PACK_ROW_LENGTH, row_len);
            (gl.read_pixels)(0, 0, w, h, GL_BGRA, GL_UNSIGNED_BYTE, std::ptr::null_mut());
            fences[i] = (create_sync)(dpy, EGL_SYNC_FENCE_KHR as i32, std::ptr::null());
            if fences[i].is_null() {
                pipe_err = Some("eglCreateSyncKHR returned null (pipelined)".into());
                break 'pipe;
            }
        }
        (gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, 0);
        for f in fences {
            let r = (wait_sync)(dpy, f, EGL_SYNC_FLUSH_COMMANDS_BIT_KHR as i32, 5_000_000_000);
            (destroy_sync)(dpy, f);
            if r != EGL_CONDITION_SATISFIED_KHR as i32 {
                pipe_err = Some(format!("pipelined EGL fence wait failed ({r:#06x})"));
                break 'pipe;
            }
        }
        total_us += t0.elapsed().as_micros() as u64;
    }
    let per_frame = pipe_err.is_none().then(|| total_us as f64 / (rounds * 3) as f64);
    let mut note = if persistent {
        ""
    } else {
        "no ARB_buffer_storage: per-frame map + memcpy needed before RDMA"
    }
    .to_string();
    if let Some(e) = pipe_err {
        note.push_str("; ");
        note.push_str(&e);
    }
    bench_print(name, &note, &samples, size, Some(data), per_frame);
    (syms.delete_buffers)(3, pbos.as_ptr());
}

/// Event-driven mode: follow the capture stream exactly like the production
/// server — re-import on every Scanout (QEMU alternates dmabufs!), read
/// immediately on every frame event. Pointer churn generates the events.
/// Measures v0 (production path) and C (PBO + EGL fence) per frame on FRESH
/// content, C cross-validated against v0 read of the same instant.
unsafe fn bench_event_mode(
    rx: std::sync::mpsc::Receiver<CaptureEvent>,
    mat: &mut EglMaterializer,
    syms: &BenchSyms,
    dpy: EglHandle,
    fmt: &FrameFormat,
    size: usize,
    opts: &ReadBenchOpts,
) -> Result<()> {
    let (w, h) = (fmt.width as i32, fmt.height as i32);
    let row_len = (fmt.stride / 4) as i32;

    let (pbos, maps, persistent) = match bench_pbo_setup(&mat.gl, syms, size) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("[bench] C event setup failed: {e}");
            return Ok(());
        }
    };
    let _probe_keepalive = if persistent {
        match MrProbe::register(maps[0], size) {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("[bench] MR probe: {e}");
                None
            }
        }
    } else {
        None
    };
    let (create_sync, wait_sync, destroy_sync) = match (
        syms.create_sync,
        syms.client_wait_sync,
        syms.destroy_sync,
    ) {
        (Some(a), Some(b), Some(c)) => (Some(a), Some(b), Some(c)),
        _ => {
            eprintln!("[bench] no EGL_KHR_fence_sync — event mode runs v0 only");
            (None, None, None)
        }
    };

    let mut v0s: Vec<BSample> = Vec::new();
    let mut cv: Vec<BSample> = Vec::new();
    let mut mismatches = 0usize;
    let mut frames = 0usize;
    let mut scratch = vec![0u8; size];
    while frames < opts.events {
        let ev = match rx.recv() {
            Ok(e) => e,
            Err(_) => bail!("capture source ended"),
        };
        match ev {
            CaptureEvent::Scanout(d) => {
                if d.format.frame_bytes() != size as u32 {
                    bail!(
                        "scanout size changed ({} frame bytes, expected {size}) — \
                         event bench does not support resize; restart",
                        d.format.frame_bytes()
                    );
                }
                mat.prepare(&d.format, Some(&d))?;
            }
            CaptureEvent::DmabufFrame(..) => {
                // v0: the production path, on the fresh frame
                mat.mark_dirty();
                let (_, mt) = mat.copy_frame(&mut scratch)?;
                v0s.push(BSample {
                    issue_us: mt.blit_us,
                    sync_us: mt.read_us + mt.fin_us,
                });
                let v0_sum = bench_checksum(&scratch);
                // C: PBO read of the same content
                if create_sync.is_some() {
                    let (cs, ws, ds) =
                        (create_sync.unwrap(), wait_sync.unwrap(), destroy_sync.unwrap());
                    if let Err(e) = bench_blit_into_linear(&mat.gl, mat, w, h) {
                        eprintln!("[bench] C blit failed: {e}");
                        break;
                    }
                    match bench_pbo_iter(
                        &mat.gl, dpy, cs, ws, ds,
                        pbos[frames % 3], w, h, row_len,
                    ) {
                        Ok((is, ss)) => cv.push(BSample { issue_us: is, sync_us: ss }),
                        Err(e) => {
                            eprintln!("[bench] C event iteration failed: {e}");
                            break;
                        }
                    }
                    if persistent {
                        let slice =
                            std::slice::from_raw_parts(maps[frames % 3] as *const u8, size);
                        if bench_checksum(slice) != v0_sum {
                            mismatches += 1;
                        }
                    }
                }
                frames += 1;
            }
            CaptureEvent::PlainScanout(..) | CaptureEvent::PlainUpdate(..) => {
                bail!("event bench needs dmabuf scanouts (guest produced plain)")
            }
            CaptureEvent::Disabled => eprintln!("[bench] scanout disabled — waiting…"),
        }
    }
    bench_print(
        "v0 event-driven",
        "re-import per Scanout + read per frame event (production flow)",
        &v0s,
        size,
        None,
        None,
    );
    bench_print(
        "C  pbo-fence (event)",
        "same frames as v0 (PBO + EGL fence)",
        &cv,
        size,
        Some(if mismatches == 0 { "matches-v0" } else { "MISMATCH" }),
        None,
    );
    (syms.delete_buffers)(3, pbos.as_ptr());
    println!("[bench] done");
    Ok(())
}

pub fn run_readback_bench(
    d: &DmabufScanout,
    opts: &ReadBenchOpts,
    input_tx: Option<std::sync::mpsc::Sender<rdma_display_proto::InputEvent>>,
    rx: std::sync::mpsc::Receiver<CaptureEvent>,
) -> Result<()> {
    let fmt = d.format.clone();
    let size = fmt.frame_bytes() as usize;
    let (w, h) = (fmt.width as i32, fmt.height as i32);
    let row_len = (fmt.stride / 4) as i32;
    eprintln!(
        "[bench] scanout {}x{} stride {} fourcc {:#010x} modifier {:#x}; {} iters + {} warmup{}{}",
        fmt.width, fmt.height, fmt.stride, fmt.fourcc, fmt.modifier, opts.iterations, opts.warmup,
        if opts.load { "; load mode: guest pointer churn" } else { "" },
        if opts.events > 0 { format!("; event-driven: {} frames", opts.events) } else { "".to_string() }
    );

    // load driver: bounce the guest pointer at ~120 Hz so the guest keeps
    // repainting the scanout dmabuf while the bench reads it — the contended
    // condition the production server actually runs in (also the event
    // source for --events mode)
    let _load = if opts.load || opts.events > 0 {
        input_tx.map(|tx| {
            let w = fmt.width as i32;
            let h = fmt.height as i32;
            std::thread::spawn(move || {
                let (mut x, mut y) = (w / 2, h / 2);
                let (mut dx, mut dy) = (7i32, 5i32);
                loop {
                    x = (x + dx).clamp(10, w - 10);
                    y = (y + dy).clamp(10, h - 10);
                    if x == 10 || x == w - 10 {
                        dx = -dx;
                    }
                    if y == 10 || y == h - 10 {
                        dy = -dy;
                    }
                    if tx
                        .send(rdma_display_proto::InputEvent::Abs {
                            x: x as u32,
                            y: y as u32,
                        })
                        .is_err()
                    {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(8));
                }
            })
        })
    } else {
        None
    };

    // production materializer provides the shared EGL context, the OES import,
    // the blit program and the linear FBO (used by the baseline and by C)
    let mut mat = EglMaterializer::new(opts.rendernode.as_deref())?;
    mat.prepare(&fmt, Some(d))?;
    let syms = BenchSyms::load(&mat.egl)?;

    if opts.events > 0 {
        // event-driven: consume the capture stream like production does
        let dpy = mat.dpy;
        return unsafe { bench_event_mode(rx, &mut mat, &syms, dpy, &fmt, size, opts) };
    }

    // static mode: the caller's remaining events are drained in the
    // background so the channel doesn't grow unboundedly
    let drainer = std::thread::spawn(move || {
        while rx.recv().is_ok() {}
    });

    // ---- baseline: production path (OES + blit + direct read) --------------
    let mut base = vec![0u8; size];
    for _ in 0..opts.warmup {
        mat.mark_dirty();
        mat.copy_frame(&mut base)?;
    }
    let mut samples = Vec::new();
    for _ in 0..opts.iterations {
        mat.mark_dirty();
        let (_, mt) = mat.copy_frame(&mut base)?;
        samples.push(BSample {
            issue_us: mt.blit_us,
            sync_us: mt.read_us + mt.fin_us,
        });
    }
    let chk: VCheck = (
        bench_checksum(&base),
        bench_checksum_flipped(&base, fmt.stride as usize, fmt.height as usize),
    );
    bench_print(
        "v0 oes-blit-direct",
        "production path (issue = blit call; sync = readback + finish)",
        &samples,
        size,
        Some("baseline"),
        None,
    );

    // shared context pieces for the variants (immutable borrows from here on)
    let gl = &mat.gl;
    let egl = &mat.egl;
    let dpy = mat.dpy;

    // ---- A/B/C variants ----------------------------------------------------
    unsafe {
        run_variant_tex2d(gl, egl, dpy, d, w, h, row_len, size, &chk, opts);
        run_variant_renderbuf(gl, egl, dpy, &syms, d, w, h, row_len, size, &chk, opts);
        run_variant_gettexsub(gl, egl, dpy, &syms, d, w, h, row_len, size, &chk, opts);
        run_variant_pbo(gl, &syms, &mat, dpy, w, h, row_len, size, &chk, opts);
    }

    drainer.join().ok();
    println!("[bench] done");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::is_supported_rgb_fourcc;

    #[test]
    fn supported_rgb_fourccs() {
        // KWin's normal desktop scanout
        assert!(is_supported_rgb_fourcc(0x3432_5258)); // XR24
        // ...and the 10-bit wide-gamut plane it switches to for HDR video
        // (the format that used to stop capture)
        assert!(is_supported_rgb_fourcc(0x3033_4241)); // AB30
        assert!(is_supported_rgb_fourcc(0x3033_5241)); // AR30
        // a single-plane packed-YUV format must NOT be treated as RGB
        assert!(!is_supported_rgb_fourcc(0x5655_5958)); // XYUV
        assert!(!is_supported_rgb_fourcc(0x3231_564e)); // NV12
    }
}
