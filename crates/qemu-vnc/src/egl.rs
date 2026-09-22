//! Minimal EGL dmabuf downloader for the vendored qemu-vnc bridge.
//!
//! Trimmed from rdma-display's capture crate (the proven NVIDIA recipe):
//! dlopen libEGL/GLES, a surfaceless (or 1x1 pbuffer) GLES 3.1 context on the
//! render node, import the QEMU scanout dmabuf with its explicit modifier as
//! GL_TEXTURE_EXTERNAL_OES, blit into a linear RGBA8 target, and read the
//! frame back with glReadPixels(GL_BGRA) — bytes land in B,G,R,X order,
//! exactly the pixman X8R8G8B8 layout the bridge already sends to VNC
//! clients. The OES sampling lands row order correct as-is for these
//! dmabufs (y0_top=false, verified in production); no flip.
//!
//! Thread model: EGL/GL objects are used with make_current()/done_current()
//! around each download, so the caller (zbus's connection executor thread)
//! may hop threads between callbacks; context acquisition is serialized by
//! the object server's per-interface dispatch.

use std::ffi::CString;

use anyhow::{anyhow, bail, Context, Result};

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
const EGL_EXTENSIONS: i32 = 0x3055;

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
// readback format: driver emits bytes B,G,R,X — the client-visible XRGB order
const GL_BGRA: u32 = 0x80E1;
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
type FnDestroyImage = unsafe extern "system" fn(EglHandle, EglHandle) -> u32;
type FnQueryString = unsafe extern "system" fn(EglHandle, i32) -> *const u8;
type FnGetProcAddress = unsafe extern "system" fn(*const i8) -> *const core::ffi::c_void;

struct Egl {
    /// the dlopen handle: MUST outlive every function pointer below (they
    /// point into libEGL; dropping it would dlclose the driver and turn each
    /// call into a jump into unmapped memory)
    _lib: std::sync::Arc<libloading::Library>,
    get_platform_display: FnGetPlatformDisplay,
    query_devices: FnQueryDevices,
    query_device_string: FnQueryDeviceString,
    initialize: FnInitialize,
    bind_api: FnBindApi,
    choose_config: FnChooseConfig,
    create_context: FnCreateContext,
    create_pbuffer: FnCreatePbuffer,
    make_current: FnMakeCurrent,
    get_error: FnGetError,
    create_image_khr: FnCreateImage,
    destroy_image: FnDestroyImage,
    query_string: FnQueryString,
    get_proc_address: FnGetProcAddress,
}

impl Egl {
    fn load() -> Result<Self> {
        let lib = unsafe {
            libloading::Library::new("libEGL.so.1")
                .or_else(|_| libloading::Library::new("libEGL.so"))
                .context("dlopen libEGL")?
        };
        // eglGetError/eglBindAPI/eglGetProcAddress are core exports; the rest
        // goes through eglGetProcAddress with a dlsym fallback.
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
            _lib: std::sync::Arc::new(lib),
        })
    }
}

/// Load `name` via eglGetProcAddress with a dlsym fallback (some EGL entry
/// points are not exported by the shared object at all).
fn load_ext<T: Copy>(
    proc_addr: &FnGetProcAddress,
    lib: &libloading::Library,
    name: &str,
) -> Result<T> {
    let c = CString::new(name).unwrap();
    let mut p = unsafe { (*proc_addr)(c.as_ptr()) };
    if p.is_null() {
        p = unsafe {
            libloading::Library::get(lib, name.as_bytes())
                .map(|s: libloading::Symbol<*mut core::ffi::c_void>| *s as *const core::ffi::c_void)
                .unwrap_or(std::ptr::null())
        };
    }
    if p.is_null() {
        bail!("missing symbol {name}");
    }
    unsafe { Ok(std::mem::transmute_copy::<*const core::ffi::c_void, T>(&p)) }
}

// --- GL bindings ------------------------------------------------------------

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
    delete_textures: unsafe extern "system" fn(i32, *const u32),
    read_pixels: unsafe extern "system" fn(i32, i32, i32, i32, u32, u32, *mut core::ffi::c_void),
    use_program: unsafe extern "system" fn(u32),
    draw_arrays: unsafe extern "system" fn(u32, i32, i32),
    viewport: unsafe extern "system" fn(i32, i32, i32, i32),
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
        Ok(Gl {
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
            delete_framebuffers: load!("glDeleteFramebuffers"),
            delete_textures: load!("glDeleteTextures"),
            read_pixels: load!("glReadPixels"),
            use_program: load!("glUseProgram"),
            draw_arrays: load!("glDrawArrays"),
            viewport: load!("glViewport"),
            get_error: load!("glGetError"),
            image_target_texture: load!("glEGLImageTargetTexture2DOES"),
        })
    }
}

fn gl_proc_ptr(egl: &Egl, name: &str) -> Result<*const core::ffi::c_void> {
    let c = CString::new(name).unwrap();
    let p = unsafe { (egl.get_proc_address)(c.as_ptr()) };
    if p.is_null() {
        bail!("GL function {name} unavailable");
    }
    Ok(p)
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

/// Dmabuf parameters of one scanout announcement (subset we consume).
#[derive(Debug, Clone, Copy)]
pub struct DmabufFmt {
    pub width: u32,
    pub height: u32,
    pub fourcc: u32,
    pub modifier: u64,
    pub stride: u32,
}

/// Imports QEMU scanout dmabufs and downloads them to host memory.
///
/// The context is acquired per download and released after, so the object
/// server may dispatch callbacks on different executor threads. Object
/// destruction on release is fine: the context is kept alive, only the
/// current-binding moves.
pub struct Grabber {
    egl: Egl,
    gl: Gl,
    dpy: EglHandle,
    ctx: EglHandle,
    /// 1x1 pbuffer, only when surfaceless context is unsupported
    pbuf: EglHandle,
    prog: u32,
    linear_tex: u32,
    fbo: u32,
    /// (EGLImage, GL texture) of the most recent import
    img: Option<(EglHandle, u32)>,
    fmt: Option<DmabufFmt>,
}

// bound to whichever thread holds it during a download
unsafe impl Send for Grabber {}

impl Grabber {
    pub fn new(rendernode: Option<&str>) -> Result<Self> {
        let egl = Egl::load()?;
        let dpy = device_display(&egl, rendernode)?;

        let mut major = 0;
        let mut minor = 0;
        if unsafe { (egl.initialize)(dpy, &mut major, &mut minor) } == 0 {
            bail!("eglInitialize failed (err {:#06x})", unsafe { (egl.get_error)() });
        }
        eprintln!(
            "[egl] EGL {}.{} on {:?}",
            major,
            minor,
            rendernode.unwrap_or("default device")
        );
        if unsafe { (egl.bind_api)(EGL_OPENGL_API) } == 0 {
            bail!("eglBindAPI(OpenGL) failed");
        }
        unsafe {
            let exts = (egl.query_string)(dpy, EGL_EXTENSIONS);
            if !exts.is_null() {
                let list = std::ffi::CStr::from_ptr(exts as *const i8).to_string_lossy();
                let wanted = [
                    "EGL_KHR_surfaceless_context",
                    "EGL_EXT_platform_device",
                    "EGL_KHR_image_base",
                    "EGL_EXT_image_dma_buf_import",
                    "EGL_EXT_image_dma_buf_import_modifiers",
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
        let ctx =
            unsafe { (egl.create_context)(dpy, cfg, std::ptr::null_mut(), ctx_attribs.as_ptr()) };
        if ctx.is_null() {
            bail!(
                "eglCreateContext failed (version 3.1, err {:#06x})",
                unsafe { (egl.get_error)() }
            );
        }
        let pb = [EGL_PBUFFER_WIDTH, 1, EGL_PBUFFER_HEIGHT, 1, EGL_NONE];
        // Probe for a surfaceless context, then RELEASE: downloads make the
        // context current on whichever executor thread runs the callback, so
        // it must not stay bound to the construction thread.
        let surfaceless =
            unsafe { (egl.make_current)(dpy, std::ptr::null_mut(), std::ptr::null_mut(), ctx) }
                != 0;
        unsafe {
            (egl.make_current)(dpy, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        }
        let pbuf = if surfaceless {
            eprintln!("[egl] using surfaceless context (EGL_KHR_surfaceless_context)");
            std::ptr::null_mut()
        } else {
            let pb_surf = unsafe { (egl.create_pbuffer)(dpy, cfg, pb.as_ptr()) };
            if pb_surf.is_null() {
                bail!("eglCreatePbufferSurface failed and surfaceless context unsupported");
            }
            // verify the pbuffer can be made current, then release (downloads
            // bind/unbind per callback on the executor thread)
            if unsafe { (egl.make_current)(dpy, pb_surf, pb_surf, ctx) } == 0 {
                bail!("eglMakeCurrent(pbuffer) failed");
            }
            unsafe {
                (egl.make_current)(dpy, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
            }
            eprintln!("[egl] using pbuffer context");
            pb_surf
        };
        eprintln!("[egl] GL ES 3.1 context ready");

        let gl = Gl::load(&egl)?;
        let mut grabber = Grabber {
            egl,
            gl,
            dpy,
            ctx,
            pbuf,
            prog: 0,
            linear_tex: 0,
            fbo: 0,
            img: None,
            fmt: None,
        };
        // the shader program needs a current context: acquire on this thread,
        // build, release (downloads re-acquire on the executor thread)
        grabber.acquire_current()?;
        let r = grabber.build_program();
        grabber.done_current();
        r?;
        Ok(grabber)
    }

    fn build_program(&mut self) -> Result<()> {
        unsafe {
            let compile = |src: &str, ty: u32| -> Result<u32> {
                let sh = (self.gl.create_shader)(ty);
                let c = CString::new(src.as_bytes())?;
                (self.gl.shader_source)(
                    sh,
                    1,
                    (&[c.as_ptr()] as *const [*const i8]) as *const *const i8,
                    std::ptr::null(),
                );
                (self.gl.compile_shader)(sh);
                let mut ok = 0;
                (self.gl.get_shaderiv)(sh, GL_COMPILE_STATUS, &mut ok);
                if ok == 0 {
                    let mut log = [0u8; 1024];
                    let mut n = 0;
                    (self.gl.get_shader_info_log)(sh, 1024, &mut n, log.as_mut_ptr());
                    bail!(
                        "shader compile failed: {}",
                        String::from_utf8_lossy(&log[..n as usize])
                    );
                }
                Ok(sh)
            };
            let vs = compile(VS_SRC, GL_VERTEX_SHADER)?;
            let fs = compile(FS_SRC, GL_FRAGMENT_SHADER)?;
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

    /// (re)create the linear RGBA8 target. Only on format change — rebuilding
    /// per update would churn an 8 MB texture per flip.
    fn build_targets(&mut self, w: u32, h: u32) -> Result<()> {
        if let Some(f) = &self.fmt {
            if f.width == w && f.height == h {
                return Ok(());
            }
        }
        unsafe {
            let mut tex = 0u32;
            (self.gl.gen_textures)(1, &mut tex);
            (self.gl.bind_texture)(GL_TEXTURE_2D, tex);
            (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as i32);
            (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as i32);
            (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE as i32);
            (self.gl.tex_image_2d)(
                GL_TEXTURE_2D, 0, GL_RGBA8 as i32, w as i32, h as i32, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, std::ptr::null(),
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

            // delete the previous FBO *before* the previous texture: a deleted
            // texture stays alive while an FBO attachment still references it
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

    /// Import one dmabuf (destroying the previous import — QEMU alternates
    /// scanout buffers, so every flip is a new import).
    fn import_one(&mut self, fds: &[i32; 4], fmt: &DmabufFmt) -> Result<()> {
        // the fd is valid for the duration of the D-Bus method call; the
        // EGLImage takes its own reference, so the buffer outlives the fd
        let fd = fds[0];
        if fd < 0 {
            bail!("dmabuf scanout without plane fd");
        }
        // single-plane 4-byte RGB: the 8888 family and the 10-bit
        // 2101010/1010102 family (compositors switch the primary plane to the
        // latter for wide-gamut/HDR content); the blit normalizes channel
        // order and depth, so only 4 bytes/pixel matters
        match fmt.fourcc {
            0x3432_5241 | 0x3432_4241 | 0x3432_4152 | 0x3432_4142
            | 0x3432_5258 | 0x3432_4258 | 0x3432_5852 | 0x3432_5842
            | 0x3033_5241 | 0x3033_4241 | 0x3033_4152 | 0x3033_4142
            | 0x3033_5258 | 0x3033_4258 | 0x3033_5852 | 0x3033_5842 => {}
            other => bail!("unsupported fourcc {other:#010x}"),
        }
        // destroy the previous import: unbind BOTH targets the blit touches
        if let Some((img, tex)) = self.img.take() {
            unsafe {
                (self.gl.bind_texture)(GL_TEXTURE_EXTERNAL_OES, 0);
                (self.gl.bind_texture)(GL_TEXTURE_2D, 0);
                (self.gl.delete_textures)(1, &[tex] as *const u32);
                (self.egl.destroy_image)(self.dpy, img);
            }
        }
        let a: Vec<i32> = vec![
            EGL_WIDTH,
            fmt.width as i32,
            EGL_HEIGHT,
            fmt.height as i32,
            EGL_LINUX_DRM_FOURCC_EXT,
            fmt.fourcc as i32,
            EGL_DMA_BUF_PLANE0_FD_EXT,
            fd,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT,
            0,
            EGL_DMA_BUF_PLANE0_PITCH_EXT,
            fmt.stride as i32,
            EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
            (fmt.modifier & 0xffff_ffff) as i32,
            EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
            (fmt.modifier >> 32) as i32,
            EGL_NONE,
        ];
        let img = unsafe {
            (self.egl.create_image_khr)(
                self.dpy,
                std::ptr::null_mut(),
                EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(),
                a.as_ptr(),
            )
        };
        if img.is_null() {
            let err = unsafe { (self.egl.get_error)() };
            bail!("eglCreateImageKHR(dmabuf, modifier {:#x}) failed (err {:#06x})", fmt.modifier, err);
        }
        unsafe {
            let mut tex = 0u32;
            (self.gl.gen_textures)(1, &mut tex);
            // foreign dmabufs on NVIDIA bind as EXTERNAL_OES (GLES)
            (self.gl.bind_texture)(GL_TEXTURE_EXTERNAL_OES, tex);
            (self.gl.image_target_texture)(GL_TEXTURE_EXTERNAL_OES, img);
            let gl_err = (self.gl.get_error)();
            if gl_err != 0 {
                eprintln!("[egl] WARNING: image_target_texture(OES) glError={:#06x}", gl_err);
            }
            self.img = Some((img, tex));
        }
        eprintln!(
            "[egl] imported dmabuf {}x{} fourcc={:#010x} modifier={:#x}",
            fmt.width, fmt.height, fmt.fourcc, fmt.modifier
        );
        Ok(())
    }

    /// Blit + readback: returns tight w*4*h bytes in B,G,R,X (XRGB) order.
    fn blit_readback(&mut self, fmt: &DmabufFmt) -> Result<Vec<u8>> {
        let (_, tex) = self.img.as_ref().ok_or_else(|| anyhow!("no imported dmabuf"))?;
        let mut buf = vec![0u8; fmt.width as usize * 4 * fmt.height as usize];
        unsafe {
            (self.gl.use_program)(self.prog);
            (self.gl.bind_framebuffer)(GL_FRAMEBUFFER, self.fbo);
            (self.gl.viewport)(0, 0, fmt.width as i32, fmt.height as i32);
            (self.gl.bind_texture)(GL_TEXTURE_EXTERNAL_OES, *tex);
            (self.gl.draw_arrays)(GL_TRIANGLES, 0, 3);
            // readPixels is synchronous per spec (the draw is flushed and the
            // data delivered before it returns)
            (self.gl.read_pixels)(
                0,
                0,
                fmt.width as i32,
                fmt.height as i32,
                GL_BGRA,
                GL_UNSIGNED_BYTE,
                buf.as_mut_ptr() as *mut core::ffi::c_void,
            );
            let gl_err = (self.gl.get_error)();
            if gl_err != 0 {
                bail!("GL readback failed 0x{gl_err:04x}");
            }
        }
        Ok(buf)
    }

    /// Import a dmabuf scanout announcement. QEMU alternates scanout
    /// buffers, so every scanout is a new import (the previous one is
    /// destroyed). Stores the format for subsequent downloads.
    pub fn import(
        &mut self,
        fds: &[i32; 4],
        num_planes: u32,
        width: u32,
        height: u32,
        fourcc: u32,
        modifier: u64,
        stride: u32,
    ) -> Result<()> {
        if num_planes != 1 {
            bail!("multi-plane dmabuf not supported (planes={num_planes})");
        }
        let fmt = DmabufFmt { width, height, fourcc, modifier, stride };
        self.acquire_current()?;
        let res = (|| -> Result<()> {
            self.build_targets(fmt.width, fmt.height)?;
            self.import_one(fds, &fmt)
        })();
        self.done_current();
        res?;
        self.fmt = Some(fmt);
        Ok(())
    }

    /// Blit + readback of the currently imported dmabuf.
    /// Returns (tight bytes, width, height) in B,G,R,X (XRGB) order.
    pub fn download(&mut self) -> Result<(Vec<u8>, u32, u32)> {
        let Some(fmt) = self.fmt else {
            bail!("no imported dmabuf (scanout_dmabuf must arrive first)");
        };
        self.acquire_current()?;
        let res = self.blit_readback(&fmt).map(|buf| (buf, fmt.width, fmt.height));
        self.done_current();
        res
    }

    /// Acquire the context on the calling thread. zbus's executor may hop
    /// threads between callbacks; the previous user always releases, so a
    /// re-bind is legal and cheap.
    fn acquire_current(&self) -> Result<()> {
        let surf = if self.pbuf.is_null() { std::ptr::null_mut() } else { self.pbuf };
        if unsafe { (self.egl.make_current)(self.dpy, surf, surf, self.ctx) } == 0 {
            let err = unsafe { (self.egl.get_error)() };
            bail!("eglMakeCurrent failed (err {:#06x})", err);
        }
        Ok(())
    }

    fn done_current(&self) {
        unsafe {
            (self.egl.make_current)(self.dpy, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        }
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
