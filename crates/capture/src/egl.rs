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

use crate::{CaptureEvent, DmabufScanout, FrameFormat};

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
        let w = fmt.width as i32;
        let h = fmt.height as i32;
        let size = fmt.frame_bytes() as isize;
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

            if self.linear_tex != 0 {
                (self.gl.delete_textures)(1, &[self.linear_tex] as *const u32);
            }
            self.linear_tex = tex;
            self.fbo = fbo;
            let _ = size; // staging is written by glReadPixels directly now
        }
        Ok(())
    }

    fn import_dmabuf(&mut self, d: &DmabufScanout) -> Result<()> {
        let fmt = &d.format;
        if d.fds.len() != 1 || fmt.num_planes != 1 {
            bail!("multi-plane dmabuf not supported yet (planes={})", d.fds.len());
        }
        // all single-plane 4bpp RGB variants: XRGB('XR24'), XBGR('XB24'),
        // ARGB('AR24'), ABGR('AB24') — the blit normalizes memory order, and
        // readback always produces XR24 (BGRX) bytes
        match fmt.fourcc {
            0x3432_5258 | 0x3432_4258 | 0x3432_5241 | 0x3432_4241 => {}
            other => bail!("EGL materializer: unsupported fourcc {other:#010x}"),
        }
        // destroy the previous import
        if let Some((img, tex)) = self.img.take() {
            unsafe {
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
        // all single-plane 4bpp RGB variants: XRGB('XR24'), XBGR('XB24'),
        // ARGB('AR24'), ABGR('AB24') — the blit normalizes memory order, and
        // readback always produces XR24 (BGRX) bytes
        match fmt.fourcc {
            0x3432_5258 | 0x3432_4258 | 0x3432_5241 | 0x3432_4241 => {}
            other => bail!("EGL materializer: unsupported fourcc {other:#010x}"),
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

    fn copy_frame(&mut self, out: &mut [u8]) -> Result<(bool, MatTimes)> {
        if !self.dirty {
            return Ok((false, MatTimes::ZERO));
        }
        let fmt = self.fmt.as_ref().ok_or_else(|| anyhow!("no scanout"))?;
        let (_, tex) = self.img.as_ref().ok_or_else(|| {
            anyhow!("no imported dmabuf (plain scanout needs --materialize plain)")
        })?;
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
            // queued" to "pixels in host RAM"; fin = residual glFinish
            let blit = t0.elapsed();

            // Read the linear target directly into the caller's buffer (the
            // RDMA staging slot: pinned host memory). GL_BGRA readback means
            // the bytes land in wire order (B,G,R,X) — the GPU's copy does
            // the channel normalize, the CPU touches nothing but the DMA
            // later. No PBO: the pipeline is synchronous per frame already
            // (glFinish below), so the pack buffer bought only an extra hop.
            // clear any sticky GL error so the post-readback check below is
            // attributable to the readback itself (instrumentation must not
            // inject errors into the pipeline's error rail)
            (self.gl.get_error)();
            (self.gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, 0); // client-array dest
            (self.gl.pixel_storei)(GL_PACK_ROW_LENGTH, (fmt.stride / 4) as i32);
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
            let gl_err = (self.gl.get_error)();
            if gl_err != 0 {
                bail!(
                    "glReadPixels(GL_BGRA) failed 0x{gl_err:04x} — driver rejects \
                    BGRA readback into client memory; this hardware is unmeasured"
                );
            }
            self.last_mat = MatTimes {
                blit_us: blit.as_micros() as u64,
                read_us: (t_after_read - blit).as_micros() as u64,
                fin_us: (total - t_after_read).as_micros() as u64,
            };
        }
        self.dirty = false;
        Ok((true, self.last_mat))
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

