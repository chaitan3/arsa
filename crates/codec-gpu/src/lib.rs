//! GPU (GLES 3.1 compute) encoder/decoder for the rdma-display lossless
//! frame codec. Produces and consumes exactly the byte format defined by
//! `rdma-display-codec`, so a GPU-coded frame is decodable by the CPU
//! reference decoder and vice versa.
//!
//! Shape matches the production materializer: the frame lives in a GPU buffer,
//! tiles are compared against a persistent reference, and only changed tiles
//! are compacted into the coded stream. In the server this reads the linear
//! RGBA8 texture the readback blits into; here we use plain SSBOs so the code
//! is self-contained and testable headlessly.

use std::ffi::{c_void, CStr, CString};

use anyhow::{bail, Context, Result};
use rdma_display_codec::{Geometry, Header, BPP, HEADER_LEN, INDEX_WORDS};

const EGL_PLATFORM_DEVICE_EXT: u32 = 0x313F;
const EGL_DRM_DEVICE_FILE_EXT: i32 = 0x3233;
const EGL_OPENGL_ES_API: u32 = 0x30A0;
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_PBUFFER_BIT: i32 = 0x0001;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_OPENGL_ES3_BIT: i32 = 0x0040;
const EGL_NONE: i32 = 0x3038;
const EGL_CONTEXT_CLIENT_VERSION: i32 = 0x3098;

const GL_SHADER_STORAGE_BUFFER: u32 = 0x90D2;
const GL_MAP_READ_BIT: u32 = 0x0001;
const GL_MAP_WRITE_BIT: u32 = 0x0002;
const GL_MAP_PERSISTENT_BIT: u32 = 0x0040;
const GL_MAP_COHERENT_BIT: u32 = 0x0080;
const GL_DYNAMIC_DRAW: u32 = 0x88E8;
const GL_COMPUTE_SHADER: u32 = 0x91B9;
const GL_COMPILE_STATUS: u32 = 0x8B81;
const GL_LINK_STATUS: u32 = 0x8B82;
const GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT: u32 = 0x4000;
const GL_SHADER_STORAGE_BARRIER_BIT: u32 = 0x0000_2000;
/// Repetitions used by the read-bandwidth probe (`memprobe`) so the fixed
/// per-dispatch launch/finish latency does not dominate a 31 MiB read.
const PROBE_REPS: u32 = 8;
const GL_RENDERER: u32 = 0x1F01;
const GL_VERSION: u32 = 0x1F02;

// --- texture-direct input: dmabuf -> external texture -> RGBA8 image --------
// EGL_EXT_image_dma_buf_import(_modifiers) attribute list
const EGL_WIDTH: i32 = 0x3057;
const EGL_HEIGHT: i32 = 0x3056;
const EGL_LINUX_DMA_BUF_EXT: u32 = 0x3270;
const EGL_LINUX_DRM_FOURCC_EXT: i32 = 0x3271;
const EGL_DMA_BUF_PLANE0_FD_EXT: i32 = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: i32 = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: i32 = 0x3274;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: i32 = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: i32 = 0x3444;
// EGL_KHR_gl_texture_2D_image (benchmark hook: wrap a GL texture as an image)
const EGL_GL_TEXTURE_2D_KHR: u32 = 0x30B1;
const EGL_IMAGE_PRESERVED_KHR: i32 = 0x30D2;
const EGL_TRUE: i32 = 1;

const GL_TEXTURE_EXTERNAL_OES: u32 = 0x8D65;
const GL_TEXTURE0: u32 = 0x84C0;
const GL_TEXTURE_WRAP_S: u32 = 0x2802;
const GL_TEXTURE_WRAP_T: u32 = 0x2803;
const GL_CLAMP_TO_EDGE: u32 = 0x812F;
const GL_TRIANGLES: u32 = 0x0004;
const GL_READ_ONLY: u32 = 0x88B8;
const GL_FRAGMENT_SHADER: u32 = 0x8B30;
const GL_VERTEX_SHADER: u32 = 0x8B31;
const GL_FRAMEBUFFER_BARRIER_BIT: u32 = 0x0000_0400;

type Handle = *mut c_void;
type FnGetProc = unsafe extern "system" fn(*const u8) -> *const c_void;

type FnGetPlatformDisplay = unsafe extern "system" fn(u32, Handle, *const i64) -> Handle;
type FnQueryDevices = unsafe extern "system" fn(i32, *mut Handle, *mut i32) -> i32;
type FnQueryDeviceString = unsafe extern "system" fn(Handle, i32) -> *const u8;
type FnInitialize = unsafe extern "system" fn(Handle, *mut i32, *mut i32) -> u32;
type FnBindApi = unsafe extern "system" fn(u32) -> u32;
type FnChooseConfig = unsafe extern "system" fn(Handle, *const i32, *mut Handle, i32, *mut i32) -> u32;
type FnCreateContext = unsafe extern "system" fn(Handle, Handle, Handle, *const i32) -> Handle;
type FnMakeCurrent = unsafe extern "system" fn(Handle, Handle, Handle, Handle) -> u32;
type FnCreateImageKhr =
    unsafe extern "system" fn(Handle, Handle, u32, Handle, *const i32) -> Handle;
type FnDestroyImageKhr = unsafe extern "system" fn(Handle, Handle) -> u32;
type FnEglGetError = unsafe extern "system" fn() -> u32;
type FnImageTargetTexture = unsafe extern "system" fn(u32, *const c_void);

type FnGetString = unsafe extern "system" fn(u32) -> *const u8;
type FnGetError = unsafe extern "system" fn() -> u32;
type FnFinish = unsafe extern "system" fn();
type FnGenBuffers = unsafe extern "system" fn(i32, *mut u32);
type FnBindBuffer = unsafe extern "system" fn(u32, u32);
type FnBufferData = unsafe extern "system" fn(u32, isize, *const c_void, u32);
type FnBufferSubData = unsafe extern "system" fn(u32, isize, isize, *const c_void);
type FnBufferStorage = unsafe extern "system" fn(u32, isize, *const c_void, u32);
type FnMapBufferRange = unsafe extern "system" fn(u32, isize, isize, u32) -> *mut c_void;
type FnUnmapBuffer = unsafe extern "system" fn(u32) -> u32;
type FnBindBufferBase = unsafe extern "system" fn(u32, u32, u32);
type FnMemoryBarrier = unsafe extern "system" fn(u32);
type FnCreateShader = unsafe extern "system" fn(u32) -> u32;
type FnShaderSource = unsafe extern "system" fn(u32, i32, *const *const u8, *const i32);
type FnCompileShader = unsafe extern "system" fn(u32);
type FnGetShaderiv = unsafe extern "system" fn(u32, u32, *mut i32);
type FnGetShaderInfoLog = unsafe extern "system" fn(u32, i32, *mut i32, *mut u8);
type FnCreateProgram = unsafe extern "system" fn() -> u32;
type FnAttachShader = unsafe extern "system" fn(u32, u32);
type FnLinkProgram = unsafe extern "system" fn(u32);
type FnGetProgramiv = unsafe extern "system" fn(u32, u32, *mut i32);
type FnGetProgramInfoLog = unsafe extern "system" fn(u32, i32, *mut i32, *mut u8);
type FnUseProgram = unsafe extern "system" fn(u32);
type FnGetUniformLocation = unsafe extern "system" fn(u32, *const u8) -> i32;
type FnUniform1ui = unsafe extern "system" fn(i32, u32);
type FnDispatchCompute = unsafe extern "system" fn(u32, u32, u32);

/// dlsym/eglGetProcAddress a symbol, trying each candidate name.
unsafe fn load_sym<T: Copy>(
    get: FnGetProc,
    lib: Option<&libloading::Library>,
    names: &[&str],
) -> Result<T> {
    for name in names {
        let c = CString::new(*name).unwrap();
        let mut p = (get)(c.as_ptr() as *const u8);
        if p.is_null() {
            if let Some(l) = lib {
                if let Ok(s) = l.get::<*mut c_void>(name.as_bytes()) {
                    p = *s as *const c_void;
                }
            }
        }
        if !p.is_null() {
            return Ok(std::mem::transmute_copy::<*const c_void, T>(&p));
        }
    }
    bail!("missing symbol {names:?}")
}

struct Gl {
    get_string: FnGetString,
    get_error: FnGetError,
    finish: FnFinish,
    gen_buffers: FnGenBuffers,
    bind_buffer: FnBindBuffer,
    buffer_data: FnBufferData,
    buffer_sub_data: FnBufferSubData,
    buffer_storage: FnBufferStorage,
    map_buffer_range: FnMapBufferRange,
    unmap_buffer: FnUnmapBuffer,
    bind_buffer_base: FnBindBufferBase,
    memory_barrier: FnMemoryBarrier,
    create_shader: FnCreateShader,
    shader_source: FnShaderSource,
    compile_shader: FnCompileShader,
    get_shaderiv: FnGetShaderiv,
    get_shader_info_log: FnGetShaderInfoLog,
    create_program: FnCreateProgram,
    attach_shader: FnAttachShader,
    link_program: FnLinkProgram,
    get_programiv: FnGetProgramiv,
    get_program_info_log: FnGetProgramInfoLog,
    use_program: FnUseProgram,
    get_uniform_location: FnGetUniformLocation,
    uniform1ui: FnUniform1ui,
    dispatch_compute: FnDispatchCompute,
    bind_image_texture: FnBindImageTexture,
    gen_textures: unsafe extern "system" fn(i32, *mut u32),
    bind_texture: unsafe extern "system" fn(u32, u32),
    tex_parameteri: unsafe extern "system" fn(u32, u32, i32),
    tex_storage_2d: unsafe extern "system" fn(u32, i32, u32, i32, i32),
    gen_framebuffers: unsafe extern "system" fn(i32, *mut u32),
    bind_framebuffer: unsafe extern "system" fn(u32, u32),
    framebuffer_texture_2d: unsafe extern "system" fn(u32, u32, u32, u32, i32),
    check_framebuffer_status: unsafe extern "system" fn(u32) -> u32,
    delete_framebuffers: unsafe extern "system" fn(i32, *const u32),
    read_pixels: unsafe extern "system" fn(i32, i32, i32, i32, u32, u32, *mut c_void),
    // texture-direct input + blit
    draw_arrays: unsafe extern "system" fn(u32, i32, i32),
    viewport: unsafe extern "system" fn(i32, i32, i32, i32),
    uniform1i: unsafe extern "system" fn(i32, i32),
    active_texture: unsafe extern "system" fn(u32),
    delete_textures: unsafe extern "system" fn(i32, *const u32),
    tex_sub_image_2d:
        unsafe extern "system" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void),
    /// `glEGLImageTargetTexture2DOES` (optional: absent without external-image
    /// support; the texture-direct path is then unavailable)
    image_target_texture: Option<FnImageTargetTexture>,
}

type FnBindImageTexture = unsafe extern "system" fn(u32, u32, i32, u8, i32, u32, u32);

const GL_TEXTURE_2D: u32 = 0x0DE1;
const GL_RGBA: u32 = 0x1908;
const GL_UNSIGNED_BYTE: u32 = 0x1401;
const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
const GL_NEAREST: u32 = 0x2600;
const GL_FRAMEBUFFER: u32 = 0x8D40;
const GL_COLOR_ATTACHMENT0: u32 = 0x8CE0;
const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;
const GL_RGBA8: u32 = 0x8058;
const GL_READ_WRITE: u32 = 0x88BA;
const GL_SHADER_IMAGE_ACCESS_BARRIER_BIT: u32 = 0x0000_0020;
const GL_TEXTURE_FETCH_BARRIER_BIT: u32 = 0x0000_0008;
const GL_TEXTURE_UPDATE_BARRIER_BIT: u32 = 0x0000_0100;

impl Gl {
    unsafe fn load(get: FnGetProc, lib: Option<&libloading::Library>) -> Result<Self> {
        macro_rules! s {
            ($($n:literal),+ $(,)?) => { load_sym(get, lib, &[$($n),+])? };
        }
        Ok(Gl {
            get_string: s!("glGetString"),
            get_error: s!("glGetError"),
            finish: s!("glFinish"),
            gen_buffers: s!("glGenBuffers"),
            bind_buffer: s!("glBindBuffer"),
            buffer_data: s!("glBufferData"),
            buffer_sub_data: s!("glBufferSubData"),
            buffer_storage: s!("glBufferStorageEXT", "glBufferStorage"),
            map_buffer_range: s!("glMapBufferRange"),
            unmap_buffer: s!("glUnmapBuffer"),
            bind_buffer_base: s!("glBindBufferBase"),
            memory_barrier: s!("glMemoryBarrier"),
            create_shader: s!("glCreateShader"),
            shader_source: s!("glShaderSource"),
            compile_shader: s!("glCompileShader"),
            get_shaderiv: s!("glGetShaderiv"),
            get_shader_info_log: s!("glGetShaderInfoLog"),
            create_program: s!("glCreateProgram"),
            attach_shader: s!("glAttachShader"),
            link_program: s!("glLinkProgram"),
            get_programiv: s!("glGetProgramiv"),
            get_program_info_log: s!("glGetProgramInfoLog"),
            use_program: s!("glUseProgram"),
            get_uniform_location: s!("glGetUniformLocation"),
            uniform1ui: s!("glUniform1ui"),
            dispatch_compute: s!("glDispatchCompute"),
            bind_image_texture: s!("glBindImageTexture"),
            gen_textures: s!("glGenTextures"),
            bind_texture: s!("glBindTexture"),
            tex_parameteri: s!("glTexParameteri"),
            tex_storage_2d: s!("glTexStorage2D"),
            gen_framebuffers: s!("glGenFramebuffers"),
            bind_framebuffer: s!("glBindFramebuffer"),
            framebuffer_texture_2d: s!("glFramebufferTexture2D"),
            check_framebuffer_status: s!("glCheckFramebufferStatus"),
            delete_framebuffers: s!("glDeleteFramebuffers"),
            read_pixels: s!("glReadPixels"),
            draw_arrays: s!("glDrawArrays"),
            viewport: s!("glViewport"),
            uniform1i: s!("glUniform1i"),
            active_texture: s!("glActiveTexture"),
            delete_textures: s!("glDeleteTextures"),
            tex_sub_image_2d: s!("glTexSubImage2D"),
            image_target_texture: load_sym::<FnImageTargetTexture>(
                get,
                lib,
                &["glEGLImageTargetTexture2DOES"],
            )
            .ok(),
        })
    }
}

/// One scanout dmabuf to encode directly (no readback, no re-upload). The fd
/// is only borrowed for the duration of [`GpuCodec::set_scanout`].
#[derive(Clone, Copy, Debug)]
pub struct DmabufImage {
    /// stable buffer id (DRM fb_id): the fds are re-created each frame, so the
    /// import cache keys on this rather than the fd
    pub key: u64,
    pub fd: i32,
    pub offset: u32,
    pub stride: u32,
    pub fourcc: u32,
    pub modifier: u64,
    pub width: u32,
    pub height: u32,
}

/// A cached dmabuf import (EGLImage + texture target).
struct SrcImport {
    key: (u64, u32, u32, u32, u64, u32, u32),
    img: Handle,
    tex: u32,
    linear: bool,
}

/// GPU codec instance: owns the EGL context and the SSBOs.
pub struct GpuCodec {
    geom: Geometry,
    gl: Gl,
    _egl: libloading::Library,
    _gl_lib: Option<libloading::Library>,
    dpy: Handle,
    ctx: Handle,
    make_current: FnMakeCurrent,
    create_image: Option<FnCreateImageKhr>,
    destroy_image: Option<FnDestroyImageKhr>,
    egl_get_error: Option<FnEglGetError>,
    enc_prog: u32,
    dec_prog: u32,
    /// blit program + program that encodes from the source image
    blit_prog: u32,
    enc_tex_prog: u32,
    /// program that samples the imported external texture directly in compute
    /// (no blit / intermediate image); 0 = unsupported by the driver
    enc_ext_prog: u32,
    /// program that reads the source as a normal `sampler2D` (texelFetch); used
    /// when the scanout is linear and can bind to `GL_TEXTURE_2D`
    enc_fetch_prog: u32,
    /// read-bandwidth probe programs + scratch output (diagnostic; see
    /// [`GpuCodec::memprobe`])
    probe_buf_prog: u32,
    probe_ext_prog: u32,
    probe_2d_prog: u32,
    probe_out: u32,
    /// source bound as `GL_TEXTURE_2D` (linear scanout) or 0
    direct_img: u32,
    cur: u32,
    reference: u32,
    coded: u32,
    counter: u32,
    /// texture-direct source: external texture (dmabuf) + RGBA8 image the blit
    /// normalizes into; `has_scanout` gates that path
    src_tex: u32,
    src_img: u32,
    src_fbo: u32,
    src_egl_image: Handle,
    /// test hook only: the `GL_TEXTURE_2D` behind a benchmark EGLImage
    src_tex2d: u32,
    /// force binding a tiled scanout to GL_TEXTURE_2D (experiment; the driver
    /// normally rejects non-linear modifiers on GL_TEXTURE_2D)
    force_2d: bool,
    has_scanout: bool,
    /// cached dmabuf imports, keyed so a repeated scanout buffer isn't
    /// re-imported (the import enqueues ~3 ms of GPU work for 4K — measured)
    src_cache: Vec<SrcImport>,
    cache_hits: u64,
    cache_misses: u64,
    coded_map: *mut u8,
    coded_len: usize,
    counter_map: *mut u32,
    index_cap: usize,
    payload_base_words: u32,
    frame_bytes: usize,
    stride_px: u32,
    /// number of tiles in the most recent `encode` (for `update_reference`)
    last_count: usize,
}

impl GpuCodec {
    pub fn new(geom: Geometry) -> Result<Self> {
        geom.validate()?;
        unsafe {
            let egl = libloading::Library::new("libEGL.so.1")
                .or_else(|_| libloading::Library::new("libEGL.so"))
                .context("dlopen libEGL")?;

            let get_proc: FnGetProc = *egl
                .get::<FnGetProc>(b"eglGetProcAddress")
                .context("eglGetProcAddress")?;
            let initialize: FnInitialize = load_sym(get_proc, Some(&egl), &["eglInitialize"])?;
            let bind_api: FnBindApi = load_sym(get_proc, Some(&egl), &["eglBindAPI"])?;
            let choose_config: FnChooseConfig =
                load_sym(get_proc, Some(&egl), &["eglChooseConfig"])?;
            let create_context: FnCreateContext =
                load_sym(get_proc, Some(&egl), &["eglCreateContext"])?;
            let make_current: FnMakeCurrent =
                load_sym(get_proc, Some(&egl), &["eglMakeCurrent"])?;
            let get_platform_display: FnGetPlatformDisplay =
                load_sym(get_proc, Some(&egl), &["eglGetPlatformDisplayEXT"])?;
            let query_devices: FnQueryDevices =
                load_sym(get_proc, Some(&egl), &["eglQueryDevicesEXT"])?;
            let query_device_string: FnQueryDeviceString =
                load_sym(get_proc, Some(&egl), &["eglQueryDeviceStringEXT"])?;
            // dmabuf image import is optional: if absent, the texture-direct
            // path is disabled and the caller falls back to upload+encode.
            let create_image = load_sym::<FnCreateImageKhr>(
                get_proc,
                Some(&egl),
                &["eglCreateImageKHR", "eglCreateImage"],
            )
            .ok();
            let destroy_image = load_sym::<FnDestroyImageKhr>(
                get_proc,
                Some(&egl),
                &["eglDestroyImageKHR", "eglDestroyImage"],
            )
            .ok();
            let egl_get_error =
                load_sym::<FnEglGetError>(get_proc, Some(&egl), &["eglGetError"]).ok();

            let mut devs: [Handle; 16] = [std::ptr::null_mut(); 16];
            let mut n = 0i32;
            query_devices(16, devs.as_mut_ptr(), &mut n);
            if n < 1 {
                bail!("eglQueryDevicesEXT: no devices");
            }
            let mut dpy = std::ptr::null_mut();
            let mut fallback: Handle = std::ptr::null_mut();
            for d in &devs[..n as usize] {
                let path = query_device_string(*d, EGL_DRM_DEVICE_FILE_EXT);
                let cand = get_platform_display(EGL_PLATFORM_DEVICE_EXT, *d, std::ptr::null());
                if cand.is_null() {
                    continue;
                }
                if !path.is_null() {
                    let s = CStr::from_ptr(path as *const i8).to_string_lossy();
                    if s.contains("renderD128") {
                        dpy = cand;
                        break;
                    }
                }
                if fallback.is_null() {
                    fallback = cand;
                }
            }
            if dpy.is_null() {
                dpy = fallback;
            }
            if dpy.is_null() {
                bail!("no EGL device display");
            }

            let (mut major, mut minor) = (0i32, 0i32);
            if initialize(dpy, &mut major, &mut minor) == 0 {
                bail!("eglInitialize failed");
            }
            if bind_api(EGL_OPENGL_ES_API) == 0 {
                bail!("eglBindAPI(ES) failed");
            }
            let cfg_attr = [
                EGL_SURFACE_TYPE,
                EGL_PBUFFER_BIT,
                EGL_RENDERABLE_TYPE,
                EGL_OPENGL_ES3_BIT,
                EGL_NONE,
            ];
            let mut cfg: Handle = std::ptr::null_mut();
            let mut ncfg = 0i32;
            if choose_config(dpy, cfg_attr.as_ptr(), &mut cfg, 1, &mut ncfg) == 0 || ncfg < 1 {
                bail!("eglChooseConfig failed");
            }
            let ctx_attr = [EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE];
            let ctx = create_context(dpy, cfg, std::ptr::null_mut(), ctx_attr.as_ptr());
            if ctx.is_null() {
                bail!("eglCreateContext failed");
            }
            if make_current(dpy, std::ptr::null_mut(), std::ptr::null_mut(), ctx) == 0 {
                bail!("eglMakeCurrent(surfaceless) failed");
            }

            let gl_lib = libloading::Library::new("libGLESv2.so.2")
                .or_else(|_| libloading::Library::new("libGLESv2.so"))
                .ok();
            let gl = Gl::load(get_proc, gl_lib.as_ref())?;

            let ver = (gl.get_string)(GL_VERSION);
            let ren = (gl.get_string)(GL_RENDERER);
            eprintln!(
                "[gpu] {} | {}",
                CStr::from_ptr(ver as *const i8).to_string_lossy(),
                CStr::from_ptr(ren as *const i8).to_string_lossy()
            );

            let enc_prog = build_program(&gl, &enc_src())?;
            let dec_prog = build_program(&gl, &dec_src())?;
            let blit_prog = build_graphics_program(&gl, BLIT_VS, BLIT_FS)?;
            let enc_tex_prog = build_program(&gl, &enc_tex_src())?;
            // Sampling an imported external texture from a compute shader is
            // the fast path (no blit). Not all drivers allow it; if the shader
            // does not compile we keep the blit + image path.
            // RDMA_LOSSLESS_SRC=blit forces the fragment-blit path (A/B probe:
            // a tiled scanout may sample faster in a fragment shader than in
            // compute).
            let force_blit = std::env::var("RDMA_LOSSLESS_SRC")
                .map(|v| v == "blit")
                .unwrap_or(false);
            let force_2d = std::env::var("RDMA_LOSSLESS_SRC")
                .map(|v| v == "2d")
                .unwrap_or(false);
            let enc_ext_prog = if force_blit {
                eprintln!("[gpu] RDMA_LOSSLESS_SRC=blit: forcing the blit + image encode path");
                0
            } else {
                match build_program(&gl, &enc_ext_src()) {
                    Ok(p) => {
                        eprintln!("[gpu] external-texture compute encode available (no blit)");
                        p
                    }
                    Err(e) => {
                        eprintln!("[gpu] external-texture compute encode unavailable ({e:#}); using blit path");
                        0
                    }
                }
            };
            let enc_fetch_prog = build_program(&gl, &enc_fetch_src())?;
            let probe_buf_prog = build_program(&gl, PROBE_BUF_SRC)?;
            let probe_2d_prog = build_program(&gl, PROBE_2D_SRC)?;
            let probe_ext_prog = build_program(&gl, PROBE_EXT_SRC).unwrap_or(0);

            let frame_bytes = geom.frame_bytes();
            let index_cap = geom.tile_count();
            let coded_len = geom.max_coded_len();
            let payload_base_words = ((HEADER_LEN + index_cap * INDEX_WORDS * 4) / 4) as u32;

            let mut buf = [0u32; 4];
            (gl.gen_buffers)(4, buf.as_mut_ptr());
            let (cur, reference, coded, counter) = (buf[0], buf[1], buf[2], buf[3]);

            // cur + reference
            (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, cur);
            (gl.buffer_data)(GL_SHADER_STORAGE_BUFFER, frame_bytes as isize, std::ptr::null(), GL_DYNAMIC_DRAW);
            (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, reference);
            (gl.buffer_data)(GL_SHADER_STORAGE_BUFFER, frame_bytes as isize, std::ptr::null(), GL_DYNAMIC_DRAW);
            (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, counter);
            (gl.buffer_storage)(
                GL_SHADER_STORAGE_BUFFER,
                8,
                std::ptr::null(),
                GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT,
            );

            (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, coded);
            (gl.buffer_storage)(
                GL_SHADER_STORAGE_BUFFER,
                coded_len as isize,
                std::ptr::null(),
                GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT,
            );
            let err = (gl.get_error)();
            if err != 0 {
                bail!("glBufferStorage (EXT_buffer_storage) failed: 0x{err:04x}");
            }
            let coded_map = (gl.map_buffer_range)(
                GL_SHADER_STORAGE_BUFFER,
                0,
                coded_len as isize,
                GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT,
            );
            if coded_map.is_null() {
                bail!("persistent map of the coded buffer failed");
            }
            (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, counter);
            let counter_map = (gl.map_buffer_range)(
                GL_SHADER_STORAGE_BUFFER,
                0,
                8,
                GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT,
            );
            if counter_map.is_null() {
                bail!("persistent map of the counter failed");
            }

            (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 0, cur);
            (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 1, reference);
            (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 2, coded);
            (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 3, counter);

            // probe scratch: one uint the read-bandwidth probes may write
            let mut pbuf = [0u32; 1];
            (gl.gen_buffers)(1, pbuf.as_mut_ptr());
            let probe_out = pbuf[0];
            (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, probe_out);
            (gl.buffer_data)(GL_SHADER_STORAGE_BUFFER, 4, std::ptr::null(), GL_DYNAMIC_DRAW);

            // texture-direct source image: immutable RGBA8 storage so it can
            // be both an FBO color attachment (the blit) and a bound image
            // (the encode compute reads it)
            let mut texs = [0u32; 1];
            (gl.gen_textures)(1, texs.as_mut_ptr());
            let src_img = texs[0];
            (gl.bind_texture)(GL_TEXTURE_2D, src_img);
            (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as i32);
            (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as i32);
            (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE as i32);
            (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE as i32);
            (gl.tex_storage_2d)(
                GL_TEXTURE_2D,
                1,
                GL_RGBA8,
                geom.width as i32,
                geom.height as i32,
            );
            let mut fbos = [0u32; 1];
            (gl.gen_framebuffers)(1, fbos.as_mut_ptr());
            let src_fbo = fbos[0];
            (gl.bind_framebuffer)(GL_FRAMEBUFFER, src_fbo);
            (gl.framebuffer_texture_2d)(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                src_img,
                0,
            );
            let st = (gl.check_framebuffer_status)(GL_FRAMEBUFFER);
            if st != GL_FRAMEBUFFER_COMPLETE {
                bail!("gpu codec source FBO incomplete: 0x{st:04x}");
            }
            (gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
            (gl.bind_texture)(GL_TEXTURE_2D, 0);

            Ok(Self {
                geom,
                gl,
                _egl: egl,
                _gl_lib: gl_lib,
                dpy,
                ctx,
                make_current,
                create_image,
                destroy_image,
                egl_get_error,
                enc_prog,
                dec_prog,
                blit_prog,
                enc_tex_prog,
                enc_ext_prog,
                enc_fetch_prog,
                probe_buf_prog,
                probe_ext_prog,
                probe_2d_prog,
                probe_out,
                direct_img: 0,
                cur,
                reference,
                coded,
                counter,
                src_tex: 0,
                src_img,
                src_fbo,
                src_egl_image: std::ptr::null_mut(),
                src_tex2d: 0,
                has_scanout: false,
                force_2d,
                src_cache: Vec::new(),
                cache_hits: 0,
                cache_misses: 0,
                coded_map: coded_map as *mut u8,
                coded_len,
                counter_map: counter_map as *mut u32,
                index_cap,
                payload_base_words,
                frame_bytes,
                stride_px: geom.stride / BPP,
                last_count: 0,
            })
        }
    }

    pub fn geometry(&self) -> Geometry {
        self.geom
    }

    /// Make this codec's EGL context current on the calling thread. Required
    /// before `upload`/`encode`/`decode` when another GL context (e.g. the
    /// capture materializer) may have been current.
    pub fn activate(&self) -> Result<()> {
        if unsafe { (self.make_current)(self.dpy, std::ptr::null_mut(), std::ptr::null_mut(), self.ctx) }
            == 0
        {
            bail!("eglMakeCurrent(gpu codec) failed");
        }
        Ok(())
    }

    /// Bring the encode reference up to date with the frame just encoded, so
    /// the next `encode` can delta against it. Runs the decode pass over the
    /// tiles that were written (all of them for a full frame).
    pub fn update_reference(&self) -> Result<()> {
        self.decode(self.last_count)
    }

    /// Test/bench hook: overwrite `words` at word offset `at` in the reference
    /// buffer. Writing a word inside a tile makes the next delta encode see that
    /// tile as changed — a way to simulate "the source changed here" without
    /// having to write the (possibly tiled) source itself.
    pub fn debug_poke_reference(&self, at_word: usize, words: &[u32]) {
        unsafe {
            (self.gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, self.reference);
            (self.gl.buffer_sub_data)(
                GL_SHADER_STORAGE_BUFFER,
                (at_word * 4) as isize,
                (words.len() * 4) as isize,
                words.as_ptr() as *const c_void,
            );
        }
    }

    pub fn upload(&self, frame: &[u8]) -> Result<()> {
        assert_eq!(frame.len(), self.frame_bytes);
        unsafe {
            (self.gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, self.cur);
            (self.gl.buffer_sub_data)(
                GL_SHADER_STORAGE_BUFFER,
                0,
                frame.len() as isize,
                frame.as_ptr() as *const c_void,
            );
        }
        Ok(())
    }

    /// Whether a scanout is imported for texture-direct encoding.
    pub fn has_scanout(&self) -> bool {
        self.has_scanout
    }

    /// (import-cache hits, misses) since construction.
    pub fn cache_stats(&self) -> (u64, u64) {
        (self.cache_hits, self.cache_misses)
    }

    /// Tiles changed by the most recent `encode`/`encode_tex` (0 = idle delta).
    pub fn last_count(&self) -> usize {
        self.last_count
    }

    /// Import a scanout dmabuf for direct encoding, caching the import so a
    /// previously seen buffer is not re-imported. `eglCreateImageKHR` +
    /// `glEGLImageTargetTexture2DOES` enqueue ~3 ms of GPU work for 4K on this
    /// driver (measured), so re-importing every frame dominates the encode. A
    /// linear scanout binds as `GL_TEXTURE_2D` (texelFetch, fast); a tiled one
    /// as `GL_TEXTURE_EXTERNAL_OES`. Makes the codec's EGL context current.
    pub fn set_scanout(&mut self, d: &DmabufImage) -> Result<()> {
        self.activate()?;
        let key = (d.key, d.offset, d.stride, d.fourcc, d.modifier, d.width, d.height);
        // key == 0 means "no stable buffer id" (e.g. QEMU D-Bus scanouts):
        // re-import every frame rather than risk a stale cache entry
        let cacheable = d.key != 0;
        if cacheable {
            if let Some(e) = self.src_cache.iter().find(|e| e.key == key) {
                self.src_egl_image = e.img;
                self.src_tex = e.tex;
                self.direct_img = if e.linear { e.tex } else { 0 };
                self.has_scanout = true;
                self.cache_hits += 1;
                return Ok(());
            }
        }
        let create = self
            .create_image
            .context("no eglCreateImageKHR (EGL_EXT_image_dma_buf_import absent)")?;
        let destroy = self.destroy_image.context("no eglDestroyImageKHR")?;
        let itt = self
            .gl
            .image_target_texture
            .context("no glEGLImageTargetTexture2DOES")?;

        let attrs: [i32; 17] = [
            EGL_WIDTH,
            d.width as i32,
            EGL_HEIGHT,
            d.height as i32,
            EGL_LINUX_DRM_FOURCC_EXT,
            d.fourcc as i32,
            EGL_DMA_BUF_PLANE0_FD_EXT,
            d.fd,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT,
            d.offset as i32,
            EGL_DMA_BUF_PLANE0_PITCH_EXT,
            d.stride as i32,
            EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
            (d.modifier & 0xffff_ffff) as i32,
            EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
            (d.modifier >> 32) as i32,
            EGL_NONE,
        ];
        let img = unsafe {
            (create)(
                self.dpy,
                std::ptr::null_mut(),
                EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(),
                attrs.as_ptr(),
            )
        };
        if img.is_null() {
            let e = self.egl_get_error.map(|f| unsafe { f() }).unwrap_or(0);
            bail!("eglCreateImageKHR(dmabuf) failed (err {e:#06x})");
        }

        unsafe {
            let mut t = [0u32; 1];
            (self.gl.gen_textures)(1, t.as_mut_ptr());
            let tex = t[0];
            // A LINEAR scanout binds as a normal GL_TEXTURE_2D, which the
            // encode reads with texelFetch (~0.2 ms for 4K). Tiled modifiers
            // normally require GL_TEXTURE_EXTERNAL_OES; RDMA_LOSSLESS_SRC=2d
            // forces the GL_TEXTURE_2D attempt (the driver usually rejects it,
            // in which case we fall back to the external sampler).
            let try_2d = d.modifier == 0 || self.force_2d;
            let mut target = if try_2d {
                GL_TEXTURE_2D
            } else {
                GL_TEXTURE_EXTERNAL_OES
            };
            (self.gl.bind_texture)(target, tex);
            (itt)(target, img);
            (self.gl.tex_parameteri)(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST as i32);
            (self.gl.tex_parameteri)(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST as i32);
            let mut err = (self.gl.get_error)();
            // If GL_TEXTURE_2D can't take this EGLImage (linear dma-bufs
            // usually can, but not always; tiled ones only under =2d), fall
            // back to the external sampler.
            if err != 0 && target == GL_TEXTURE_2D {
                eprintln!(
                    "[gpu] EGLImage -> GL_TEXTURE_2D bind failed (glError {err:#06x}, modifier {:#x}); using the external sampler",
                    d.modifier
                );
                target = GL_TEXTURE_EXTERNAL_OES;
                (self.gl.bind_texture)(target, tex);
                (itt)(target, img);
                (self.gl.tex_parameteri)(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST as i32);
                (self.gl.tex_parameteri)(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST as i32);
                err = (self.gl.get_error)();
            }
            if err != 0 {
                (destroy)(self.dpy, img);
                bail!("EGLImage -> texture {target:#x} glError 0x{err:04x}");
            }
            if target == GL_TEXTURE_EXTERNAL_OES {
                (self.gl.tex_parameteri)(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE as i32);
                (self.gl.tex_parameteri)(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE as i32);
            }
            self.src_tex = tex;
            self.direct_img = if target == GL_TEXTURE_2D { tex } else { 0 };
        }
        self.src_egl_image = img;
        self.has_scanout = true;
        self.cache_misses += 1;
        // keep the import cached (bounded) so a repeat of this buffer is a
        // lookup, not a re-import; evict the oldest non-current entry
        let tex = self.src_tex;
        if cacheable {
            self.src_cache.push(SrcImport {
                key,
                img,
                tex,
                linear: self.direct_img != 0,
            });
        }
        while self.src_cache.len() > 4 {
            match self.src_cache.iter().position(|e| e.img != self.src_egl_image) {
                Some(i) => {
                    let e = self.src_cache.remove(i);
                    unsafe {
                        if e.tex != 0 {
                            let t = [e.tex];
                            (self.gl.delete_textures)(1, t.as_ptr());
                        }
                        (destroy)(self.dpy, e.img);
                    }
                }
                None => break,
            }
        }
        Ok(())
    }

    /// Test/benchmark hook: make the encode source a `GL_TEXTURE_2D` built from
    /// `frame` (wire bytes), wrapped as an EGLImage external texture via
    /// `EGL_KHR_gl_texture_2D_image`. No dmabuf needed, so the external-sample
    /// shader (`encode_ext`) can be validated and timed headlessly. The
    /// sampling cost is optimistic vs a tiled/compressed dmabuf, but the
    /// shader's memory/instruction behaviour is the same.
    pub fn set_scanout_test_frame(&mut self, frame: &[u8], linear: bool) -> Result<()> {
        assert_eq!(frame.len(), self.frame_bytes);
        self.activate()?;
        let create = self
            .create_image
            .context("no eglCreateImageKHR (EGL_KHR_gl_texture_2D_image absent)")?;
        let destroy = self.destroy_image.context("no eglDestroyImageKHR")?;
        let itt = self
            .gl
            .image_target_texture
            .context("no glEGLImageTargetTexture2DOES")?;

        // release the previous import (external texture, image, 2D texture)
        if !self.src_egl_image.is_null() {
            unsafe {
                (self.gl.bind_texture)(GL_TEXTURE_EXTERNAL_OES, 0);
                (self.gl.bind_texture)(GL_TEXTURE_2D, 0);
                if self.src_tex != 0 {
                    let t = [self.src_tex];
                    (self.gl.delete_textures)(1, t.as_ptr());
                }
                (destroy)(self.dpy, self.src_egl_image);
                if self.src_tex2d != 0 {
                    let t = [self.src_tex2d];
                    (self.gl.delete_textures)(1, t.as_ptr());
                }
            }
            self.src_egl_image = std::ptr::null_mut();
            self.src_tex = 0;
            self.src_tex2d = 0;
            self.has_scanout = false;
        }

        let (w, h) = (self.geom.width as i32, self.geom.height as i32);
        // Model what the *sampler* sees. A production external image is a
        // dma-buf whose fourcc the external sampler interprets as real colour
        // (R=R); a GL-texture-backed EGLImage, as used here, returns the
        // texture's raw RGBA instead. So for the external path upload the wire
        // bytes with R and B swapped (the colour order the dma-buf sampler
        // reports). The linear/2D path keeps the raw byte order, because a
        // dma-buf bound to GL_TEXTURE_2D is read as bytes by texelFetch.
        let swapped: Vec<u8>;
        let upload: &[u8] = if linear {
            frame
        } else {
            swapped = frame
                .chunks_exact(4)
                .flat_map(|p| [p[2], p[1], p[0], p[3]])
                .collect();
            &swapped
        };
        let tex2d = unsafe {
            let mut t = [0u32; 1];
            (self.gl.gen_textures)(1, t.as_mut_ptr());
            (self.gl.bind_texture)(GL_TEXTURE_2D, t[0]);
            (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as i32);
            (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as i32);
            (self.gl.tex_storage_2d)(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);
            (self.gl.tex_sub_image_2d)(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                w,
                h,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                upload.as_ptr() as *const c_void,
            );
            // finish the upload outside the caller's timed region
            (self.gl.finish)();
            t[0]
        };
        let attrs = [EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE];
        let img = unsafe {
            (create)(
                self.dpy,
                self.ctx,
                EGL_GL_TEXTURE_2D_KHR,
                tex2d as usize as Handle,
                attrs.as_ptr(),
            )
        };
        if img.is_null() {
            let e = self.egl_get_error.map(|f| unsafe { f() }).unwrap_or(0);
            bail!("eglCreateImageKHR(GL_TEXTURE_2D) failed (err {e:#06x})");
        }
        unsafe {
            let mut t = [0u32; 1];
            (self.gl.gen_textures)(1, t.as_mut_ptr());
            let target = if linear {
                GL_TEXTURE_2D
            } else {
                GL_TEXTURE_EXTERNAL_OES
            };
            (self.gl.bind_texture)(target, t[0]);
            (itt)(target, img);
            (self.gl.tex_parameteri)(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST as i32);
            (self.gl.tex_parameteri)(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST as i32);
            let err = (self.gl.get_error)();
            if err != 0 {
                bail!("test external texture glError 0x{err:04x}");
            }
            self.src_tex = t[0];
            self.direct_img = if linear { t[0] } else { 0 };
        }
        self.src_egl_image = img;
        self.src_tex2d = tex2d;
        self.has_scanout = true;
        Ok(())
    }

    /// Blit the imported external texture into the source image, swizzled so
    /// the image's `(R,G,B,A)` equals the wire bytes `(B,G,R,A)` — exactly what
    /// the materializer's `glReadPixels(GL_BGRA)` of its linear target yields.
    /// The encode compute then reconstructs the same 32-bit words via
    /// `packUnorm4x8(imageLoad(...))`. GPU→GPU only.
    pub fn blit_source(&self) -> Result<()> {
        if !self.has_scanout {
            bail!("blit_source: no scanout imported");
        }
        unsafe {
            (self.gl.use_program)(self.blit_prog);
            (self.gl.viewport)(0, 0, self.geom.width as i32, self.geom.height as i32);
            (self.gl.bind_framebuffer)(GL_FRAMEBUFFER, self.src_fbo);
            (self.gl.active_texture)(GL_TEXTURE0);
            (self.gl.bind_texture)(GL_TEXTURE_EXTERNAL_OES, self.src_tex);
            let c = CString::new("tex").unwrap();
            let loc = (self.gl.get_uniform_location)(self.blit_prog, c.as_ptr() as *const u8);
            if loc >= 0 {
                (self.gl.uniform1i)(loc, 0);
            }
            (self.gl.draw_arrays)(GL_TRIANGLES, 0, 3);
            (self.gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
            (self.gl.memory_barrier)(
                GL_FRAMEBUFFER_BARRIER_BIT
                    | GL_TEXTURE_FETCH_BARRIER_BIT
                    | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
            );
            // diagnostic: finishing here attributes the blit's *GPU* time to
            // `blit` instead of letting the encode's finish absorb it. Cheap
            // (the encode depends on the blit anyway).
            (self.gl.finish)();
            let err = (self.gl.get_error)();
            if err != 0 {
                bail!("source blit glError 0x{err:04x}");
            }
        }
        Ok(())
    }

    /// Encode directly from the source image (after `blit_source`, or after
    /// `upload_source_image` in tests); no `cur` upload. Binds the image at
    /// unit 0 and runs the texture-input compute.
    pub fn encode_tex(&mut self, full: bool) -> Result<usize> {
        if !self.has_scanout && self.src_img == 0 {
            bail!("encode_tex: no source image");
        }
        unsafe {
            (self.gl.bind_image_texture)(0, self.src_img, 0, 0, 0, GL_READ_ONLY, GL_RGBA8);
            let e = (self.gl.get_error)();
            if e != 0 {
                bail!("encode_tex: bind_image_texture glError 0x{e:04x}");
            }
        }
        let prog = self.enc_tex_prog;
        let n = self.encode_with(prog, full, true)?;
        unsafe {
            let e = (self.gl.get_error)();
            if e != 0 {
                bail!("encode_tex: glError 0x{e:04x} after dispatch");
            }
        }
        Ok(n)
    }

    /// True when the driver can sample the imported external texture directly
    /// in compute, so no blit is needed.
    pub fn has_ext_encode(&self) -> bool {
        self.enc_ext_prog != 0
    }

    /// Encode straight from the imported external texture — no blit, no
    /// intermediate RGBA8 image. Requires `set_scanout`.
    pub fn encode_ext(&mut self, full: bool) -> Result<usize> {
        if !self.has_scanout {
            bail!("encode_ext: no scanout imported");
        }
        if self.enc_ext_prog == 0 {
            bail!("encode_ext: external-texture compute encode unavailable");
        }
        unsafe {
            (self.gl.get_error)(); // clear any stale error
            (self.gl.use_program)(self.enc_ext_prog);
            (self.gl.active_texture)(GL_TEXTURE0);
            (self.gl.bind_texture)(GL_TEXTURE_EXTERNAL_OES, self.src_tex);
            let c = CString::new("u_src").unwrap();
            let loc = (self.gl.get_uniform_location)(self.enc_ext_prog, c.as_ptr() as *const u8);
            if loc >= 0 {
                (self.gl.uniform1i)(loc, 0);
            }
            let e = (self.gl.get_error)();
            if e != 0 {
                bail!("encode_ext: bind external texture glError 0x{e:04x}");
            }
        }
        let prog = self.enc_ext_prog;
        let n = self.encode_with(prog, full, true)?;
        unsafe {
            let e = (self.gl.get_error)();
            if e != 0 {
                bail!("encode_ext: glError 0x{e:04x} after dispatch");
            }
        }
        Ok(n)
    }

    /// True when a linear scanout is bound as a normal `GL_TEXTURE_2D` source.
    pub fn has_fetch_encode(&self) -> bool {
        self.direct_img != 0
    }

    /// Encode reading the source as a normal `sampler2D` (`texelFetch`) — the
    /// fast path for a linear scanout whose EGLImage binds to `GL_TEXTURE_2D`.
    pub fn encode_fetch(&mut self, full: bool) -> Result<usize> {
        if self.direct_img == 0 {
            bail!("encode_fetch: no GL_TEXTURE_2D source");
        }
        unsafe {
            (self.gl.get_error)();
            (self.gl.use_program)(self.enc_fetch_prog);
            (self.gl.active_texture)(GL_TEXTURE0);
            (self.gl.bind_texture)(GL_TEXTURE_2D, self.direct_img);
            let c = CString::new("u_src").unwrap();
            let loc = (self.gl.get_uniform_location)(self.enc_fetch_prog, c.as_ptr() as *const u8);
            if loc >= 0 {
                (self.gl.uniform1i)(loc, 0);
            }
            let e = (self.gl.get_error)();
            if e != 0 {
                bail!("encode_fetch: bind source glError 0x{e:04x}");
            }
        }
        let prog = self.enc_fetch_prog;
        let n = self.encode_with(prog, full, true)?;
        unsafe {
            let e = (self.gl.get_error)();
            if e != 0 {
                bail!("encode_fetch: glError 0x{e:04x} after dispatch");
            }
        }
        Ok(n)
    }

    /// Test/benchmark hook: make the source a plain immutable `GL_TEXTURE_2D`
    /// holding `frame` (wire bytes), for `encode_fetch`. This is what a linear
    /// dmabuf EGLImage bound to `GL_TEXTURE_2D` provides in production.
    pub fn set_source_texture_frame(&mut self, frame: &[u8]) -> Result<()> {
        assert_eq!(frame.len(), self.frame_bytes);
        self.activate()?;
        let (w, h) = (self.geom.width as i32, self.geom.height as i32);
        unsafe {
            if self.direct_img == 0 {
                let mut t = [0u32; 1];
                (self.gl.gen_textures)(1, t.as_mut_ptr());
                self.direct_img = t[0];
                (self.gl.bind_texture)(GL_TEXTURE_2D, self.direct_img);
                (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as i32);
                (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as i32);
                (self.gl.tex_storage_2d)(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);
            }
            (self.gl.bind_texture)(GL_TEXTURE_2D, self.direct_img);
            (self.gl.tex_sub_image_2d)(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                w,
                h,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                frame.as_ptr() as *const c_void,
            );
            // finish the upload OUTSIDE the caller's timed region, so the
            // benchmark measures the encode, not the 33 MB host->GPU copy
            (self.gl.finish)();
            let e = (self.gl.get_error)();
            if e != 0 {
                bail!("set_source_texture_frame glError 0x{e:04x}");
            }
        }
        self.has_scanout = true;
        Ok(())
    }

    /// Test hook: fill the source image from tightly-packed wire bytes
    /// (`(R,G,B,A) == (wire0,wire1,wire2,wire3)`), mirroring the blit output.
    pub fn upload_source_image(&self, frame: &[u8]) -> Result<()> {
        assert_eq!(frame.len(), self.frame_bytes);
        unsafe {
            (self.gl.bind_texture)(GL_TEXTURE_2D, self.src_img);
            (self.gl.tex_sub_image_2d)(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                self.geom.width as i32,
                self.geom.height as i32,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                frame.as_ptr() as *const c_void,
            );
            (self.gl.memory_barrier)(GL_TEXTURE_FETCH_BARRIER_BIT);
            (self.gl.finish)();
        }
        Ok(())
    }

    /// Encode the uploaded frame; `full` forces a full frame. Returns the
    /// coded length.
    pub fn encode(&mut self, full: bool) -> Result<usize> {
        let prog = self.enc_prog;
        self.encode_with(prog, full, false)
    }

    /// Shared encode body. `prog` is either the SSBO-input (`cur`) encoder or
    /// the image-input (`src_img`) encoder. `update_ref` makes the shader
    /// advance the reference in-pass (the scanout paths); the SSBO path leaves
    /// that to `decode`, which also validates the decode.
    fn encode_with(&mut self, prog: u32, full: bool, update_ref: bool) -> Result<usize> {
        let g = self.geom;
        let h = Header {
            codec: rdma_display_codec::CODEC_LOSSLESS,
            version: rdma_display_codec::VERSION,
            delta: !full,
            has_crc: false,
            tile_log2: g.tile_log2,
            width: g.width,
            height: g.height,
            stride: g.stride,
            count: 0,
            index_capacity: self.index_cap as u32,
            crc32: 0,
            payload_words: 0,
        };
        unsafe {
            h.write(std::slice::from_raw_parts_mut(self.coded_map, HEADER_LEN));
            *self.counter_map = 0;
            *self.counter_map.add(1) = 0;
            (self.gl.memory_barrier)(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
            (self.gl.use_program)(prog);
            let u = |name: &str| {
                let c = CString::new(name).unwrap();
                (self.gl.get_uniform_location)(prog, c.as_ptr() as *const u8)
            };
            (self.gl.uniform1ui)(u("u_width"), g.width);
            (self.gl.uniform1ui)(u("u_height"), g.height);
            (self.gl.uniform1ui)(u("u_stride_px"), self.stride_px);
            (self.gl.uniform1ui)(u("u_tiles_x"), g.tiles_x() as u32);
            (self.gl.uniform1ui)(u("u_full"), full as u32);
            (self.gl.uniform1ui)(u("u_pbase_words"), self.payload_base_words);
            (self.gl.uniform1ui)(u("u_update_ref"), update_ref as u32);
            (self.gl.dispatch_compute)(g.tile_count() as u32, 1, 1);
            (self.gl.memory_barrier)(GL_SHADER_STORAGE_BARRIER_BIT);
            (self.gl.finish)();
        }
        let count = unsafe { *self.counter_map } as usize;
        let payload_words = unsafe { *self.counter_map.add(1) } as usize;
        if count > self.index_cap {
            bail!("shader produced {count} tiles > capacity {}", self.index_cap);
        }
        self.last_count = count;
        let mut patched = h;
        patched.count = count as u32;
        patched.payload_words = payload_words as u32;
        unsafe {
            patched.write(std::slice::from_raw_parts_mut(self.coded_map, HEADER_LEN));
        }
        let end = HEADER_LEN + self.index_cap * INDEX_WORDS * 4 + payload_words * 4;
        Ok(end)
    }

    /// Decode the last coded frame into the persistent reference buffer.
    pub fn decode(&self, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let g = self.geom;
        unsafe {
            (self.gl.memory_barrier)(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
            (self.gl.use_program)(self.dec_prog);
            let u = |name: &str| {
                let c = CString::new(name).unwrap();
                (self.gl.get_uniform_location)(self.dec_prog, c.as_ptr() as *const u8)
            };
            (self.gl.uniform1ui)(u("u_width"), g.width);
            (self.gl.uniform1ui)(u("u_height"), g.height);
            (self.gl.uniform1ui)(u("u_stride_px"), self.stride_px);
            (self.gl.uniform1ui)(u("u_tiles_x"), g.tiles_x() as u32);
            (self.gl.uniform1ui)(u("u_count"), count as u32);
            (self.gl.uniform1ui)(u("u_pbase_words"), self.payload_base_words);
            (self.gl.dispatch_compute)(count as u32, 1, 1);
            (self.gl.memory_barrier)(GL_SHADER_STORAGE_BARRIER_BIT);
            (self.gl.finish)();
        }
        Ok(())
    }

    /// Wait for all submitted GPU work to finish.
    pub fn sync(&self) {
        unsafe { (self.gl.finish)() };
    }

    /// Read-bandwidth diagnostic: streams the whole imported scanout and each
    /// of the codec's buffers once and prints GB/s. VRAM reads run at hundreds
    /// of GB/s; a buffer that is host-visible/GTT reads at PCIe speed (~8 GB/s
    /// on Gen3 x8). This is how we tell where the scanout and the reference
    /// actually live, rather than assuming.
    pub fn memprobe(&mut self) -> Result<()> {
        let report = |name: &str, bytes: usize, d: std::time::Duration| {
            let s = d.as_secs_f64().max(1e-9);
            eprintln!(
                "[memprobe] {name:<9} {:>4} MiB read: {:>8.3} ms = {:>6.1} GB/s",
                bytes / (1024 * 1024),
                d.as_secs_f64() * 1e3,
                bytes as f64 / s / 1e9,
            );
        };
        if self.has_scanout {
            let px = (self.geom.width as usize) * (self.geom.height as usize) * 4;
            report("scanout", px, self.probe_scanout()?);
        } else {
            eprintln!("[memprobe] scanout   (none imported)");
        }
        report("reference", self.frame_bytes, self.probe_buf(self.reference, self.frame_bytes / 4)?);
        report("cur", self.frame_bytes, self.probe_buf(self.cur, self.frame_bytes / 4)?);
        report("coded", self.coded_len, self.probe_buf(self.coded, self.coded_len / 4)?);
        // The number that actually matters: run the real encode dispatch, in the
        // same conditions as the probes above. With the reference already
        // matching (in-pass write on the previous frame) this is a pure
        // detection pass, so any gap versus `scanout` is the shader itself, not
        // the read.
        if self.has_scanout {
            unsafe { (self.gl.finish)() };
            let t = std::time::Instant::now();
            for _ in 0..PROBE_REPS {
                self.encode_ext(false)?;
            }
            let d = t.elapsed() / PROBE_REPS;
            // a detection pass reads the scanout AND the reference
            report("encode", self.frame_bytes * 2, d);
        }
        Ok(())
    }

    /// GPU time to stream `words` uints of `buf` once (bindings 4/5).
    fn probe_buf(&self, buf: u32, words: usize) -> Result<std::time::Duration> {
        unsafe {
            (self.gl.use_program)(self.probe_buf_prog);
            (self.gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 4, buf);
            (self.gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 5, self.probe_out);
            let c = CString::new("u_words").unwrap();
            let loc = (self.gl.get_uniform_location)(self.probe_buf_prog, c.as_ptr() as *const u8);
            if loc >= 0 {
                (self.gl.uniform1ui)(loc, words as u32);
            }
            (self.gl.finish)();
            let t = std::time::Instant::now();
            // several back-to-back dispatches: the per-dispatch launch/finish
            // latency is fixed, so amortizing it is the difference between
            // measuring the launch and measuring the memory
            for _ in 0..PROBE_REPS {
                (self.gl.dispatch_compute)(((words as u32) + 255) / 256, 1, 1);
            }
            (self.gl.memory_barrier)(GL_SHADER_STORAGE_BARRIER_BIT);
            (self.gl.finish)();
            Ok(t.elapsed() / PROBE_REPS)
        }
    }

    /// GPU time to stream the whole imported scanout once, sampling it exactly
    /// like the encoder (external sampler or texelFetch).
    fn probe_scanout(&self) -> Result<std::time::Duration> {
        if !self.has_scanout {
            bail!("probe_scanout: no scanout imported");
        }
        let prog = if self.direct_img != 0 { self.probe_2d_prog } else { self.probe_ext_prog };
        if prog == 0 {
            bail!("probe_scanout: no source probe program");
        }
        let px = (self.geom.width as u32) * (self.geom.height as u32);
        unsafe {
            (self.gl.use_program)(prog);
            (self.gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 5, self.probe_out);
            (self.gl.active_texture)(GL_TEXTURE0);
            if self.direct_img != 0 {
                (self.gl.bind_texture)(GL_TEXTURE_2D, self.direct_img);
            } else {
                (self.gl.bind_texture)(GL_TEXTURE_EXTERNAL_OES, self.src_tex);
            }
            let u = |n: &str| {
                let c = CString::new(n).unwrap();
                (self.gl.get_uniform_location)(prog, c.as_ptr() as *const u8)
            };
            let l = u("u_src");
            if l >= 0 {
                (self.gl.uniform1i)(l, 0);
            }
            let l = u("u_width");
            if l >= 0 {
                (self.gl.uniform1ui)(l, self.geom.width);
            }
            let l = u("u_height");
            if l >= 0 {
                (self.gl.uniform1ui)(l, self.geom.height);
            }
            (self.gl.finish)();
            let t = std::time::Instant::now();
            for _ in 0..PROBE_REPS {
                (self.gl.dispatch_compute)((px + 255) / 256, 1, 1);
            }
            (self.gl.memory_barrier)(GL_SHADER_STORAGE_BARRIER_BIT);
            (self.gl.finish)();
            Ok(t.elapsed() / PROBE_REPS)
        }
    }

    /// Read the decoded reference frame back to host memory (validation only).
    pub fn read_frame(&self) -> Result<Vec<u8>> {
        self.read_buf(self.reference)
    }

    /// Read the uploaded input buffer back (validation only).
    pub fn read_cur(&self) -> Result<Vec<u8>> {
        self.read_buf(self.cur)
    }

    fn read_buf(&self, id: u32) -> Result<Vec<u8>> {
        const GL_COPY_READ_BUFFER: u32 = 0x8F36;
        let mut v = vec![0u8; self.frame_bytes];
        unsafe {
            (self.gl.get_error)();
            (self.gl.bind_buffer)(GL_COPY_READ_BUFFER, id);
            let p = (self.gl.map_buffer_range)(
                GL_COPY_READ_BUFFER,
                0,
                self.frame_bytes as isize,
                GL_MAP_READ_BIT,
            );
            if p.is_null() {
                bail!("glMapBufferRange(read) returned null");
            }
            std::ptr::copy_nonoverlapping(p as *const u8, v.as_mut_ptr(), self.frame_bytes);
            (self.gl.unmap_buffer)(GL_COPY_READ_BUFFER);
        }
        Ok(v)
    }

    pub fn coded(&self, len: usize) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.coded_map, len) }
    }

    pub fn coded_capacity(&self) -> usize {
        self.coded_len
    }

    /// Reset the reference so the next encode is full.
    pub fn reset(&mut self) {
        // nothing to do: the next encode is requested with full=true
    }
}

impl Drop for GpuCodec {
    fn drop(&mut self) {
        let _ = (
            self.dpy,
            self.ctx,
            self.cur,
            self.reference,
            self.coded,
            self.counter,
        );
    }
}

/// Decode-only GPU codec: attaches to the caller's current GLES 3.1 context
/// and decodes the coded tiles into a `GL_RGBA8` texture in that context
/// (which the presenter then draws). The client uses this so a lossless delta
/// updates the on-screen texture in place — no full-frame CPU decode and no
/// full-frame upload.
pub struct GpuVideoDecoder {
    geom: Geometry,
    gl: Gl,
    _gl_lib: Option<libloading::Library>,
    _egl_lib: Option<libloading::Library>,
    dec_prog: u32,
    _coded: u32,
    coded_map: *mut u8,
    coded_len: usize,
    payload_base_words: u32,
    stride_px: u32,
    index_cap: usize,
}

impl GpuVideoDecoder {
    /// `get_proc` must resolve GL entry points for the current context (e.g.
    /// `eglGetProcAddress`); the context stays current for the decoder's life.
    pub fn attach(
        get_proc: FnGetProc,
        gl_lib: Option<libloading::Library>,
        geom: Geometry,
    ) -> Result<Self> {
        geom.validate()?;
        let gl = unsafe { Gl::load(get_proc, gl_lib.as_ref())? };
        let dec_prog = build_program(&gl, &dec_tex_src())?;
        let coded_len = geom.max_coded_len();
        let index_cap = geom.tile_count();
        let payload_base_words = ((HEADER_LEN + index_cap * INDEX_WORDS * 4) / 4) as u32;
        unsafe {
            let mut buf = [0u32; 1];
            (gl.gen_buffers)(1, buf.as_mut_ptr());
            let coded = buf[0];
            (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, coded);
            (gl.buffer_storage)(
                GL_SHADER_STORAGE_BUFFER,
                coded_len as isize,
                std::ptr::null(),
                GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT,
            );
            let err = (gl.get_error)();
            if err != 0 {
                bail!("glBufferStorage (gpu decoder) failed: 0x{err:04x}");
            }
            let coded_map = (gl.map_buffer_range)(
                GL_SHADER_STORAGE_BUFFER,
                0,
                coded_len as isize,
                GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT,
            );
            if coded_map.is_null() {
                bail!("persistent map of the gpu decoder buffer failed");
            }
            (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 4, coded);
            eprintln!("[gpu-dec] attached ({}x{})", geom.width, geom.height);
            Ok(Self {
                geom,
                gl,
                _gl_lib: gl_lib,
                _egl_lib: None,
                dec_prog,
                _coded: coded,
                coded_map: coded_map as *mut u8,
                coded_len,
                payload_base_words,
                stride_px: geom.stride / BPP,
                index_cap,
            })
        }
    }

    pub fn geometry(&self) -> Geometry {
        self.geom
    }

    /// Attach to the caller's current GLES 3.1 context, getting GL entry
    /// points from libEGL's `eglGetProcAddress`. Convenience for callers that
    /// don't already hold a proc-address function (the client's `eglwin`).
    pub fn attach_current(geom: Geometry) -> Result<Self> {
        let egl = match unsafe { libloading::Library::new("libEGL.so.1") } {
            Ok(l) => l,
            Err(_) => unsafe { libloading::Library::new("libEGL.so") }
                .context("dlopen libEGL")?,
        };
        let get_proc: FnGetProc = unsafe {
            *egl
                .get::<FnGetProc>(b"eglGetProcAddress")
                .context("eglGetProcAddress")?
        };
        let mut d = Self::attach(get_proc, None, geom)?;
        d._egl_lib = Some(egl);
        Ok(d)
    }

    /// Create the `GL_RGBA8` texture this decoder writes into (and the
    /// presenter draws). Contents are defined by the first decode.
    pub fn create_texture(&self, w: u32, h: u32) -> Result<u32> {
        let mut tex = 0u32;
        unsafe {
            (self.gl.gen_textures)(1, &mut tex);
            (self.gl.bind_texture)(GL_TEXTURE_2D, tex);
            (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as i32);
            (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as i32);
            // immutable storage: required for image load/store in GLES 3.1
            (self.gl.tex_storage_2d)(GL_TEXTURE_2D, 1, GL_RGBA8, w as i32, h as i32);
            let err = (self.gl.get_error)();
            (self.gl.bind_texture)(GL_TEXTURE_2D, 0);
            if err != 0 {
                bail!("decode texture glError 0x{err:04x}");
            }
        }
        Ok(tex)
    }

    /// Read a decoded texture back to host bytes (validation/harness only).
    pub fn read_texture(&self, tex: u32, out: &mut [u8]) -> Result<()> {
        let need = (self.geom.width as usize) * (self.geom.height as usize) * 4;
        if out.len() < need {
            bail!("read_texture buffer {} < {need}", out.len());
        }
        unsafe {
            let mut fbo = 0u32;
            (self.gl.gen_framebuffers)(1, &mut fbo);
            (self.gl.bind_framebuffer)(GL_FRAMEBUFFER, fbo);
            (self.gl.framebuffer_texture_2d)(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                tex,
                0,
            );
            let st = (self.gl.check_framebuffer_status)(GL_FRAMEBUFFER);
            if st != GL_FRAMEBUFFER_COMPLETE {
                bail!("decode FBO incomplete 0x{st:04x}");
            }
            (self.gl.read_pixels)(
                0,
                0,
                self.geom.width as i32,
                self.geom.height as i32,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                out.as_mut_ptr() as *mut c_void,
            );
            (self.gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
            (self.gl.delete_framebuffers)(1, &fbo);
        }
        Ok(())
    }

    /// Copy one coded frame into the decoder's buffer. The buffer is
    /// persistently mapped, so this is a memcpy + a mapped-buffer barrier.
    pub fn upload(&self, coded: &[u8]) -> Result<()> {
        if coded.len() > self.coded_len {
            bail!(
                "coded frame {} > decoder capacity {}",
                coded.len(),
                self.coded_len
            );
        }
        unsafe {
            std::ptr::copy_nonoverlapping(coded.as_ptr(), self.coded_map, coded.len());
            (self.gl.memory_barrier)(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
        }
        Ok(())
    }

    /// Decode the uploaded frame into `tex` (a `GL_RGBA8` texture in the
    /// current context), patching only the tiles listed in the frame. The
    /// fragment shader that samples `tex` must run after this; the image +
    /// texture-fetch barriers are issued here.
    pub fn decode_to_image(&self, tex: u32, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        if count > self.index_cap {
            bail!("decoder count {count} > capacity {}", self.index_cap);
        }
        let g = self.geom;
        unsafe {
            (self.gl.bind_image_texture)(0, tex, 0, 0, 0, GL_READ_WRITE, GL_RGBA8);
            (self.gl.use_program)(self.dec_prog);
            let u = |name: &str| {
                let c = CString::new(name).unwrap();
                (self.gl.get_uniform_location)(self.dec_prog, c.as_ptr() as *const u8)
            };
            (self.gl.uniform1ui)(u("u_width"), g.width);
            (self.gl.uniform1ui)(u("u_height"), g.height);
            (self.gl.uniform1ui)(u("u_stride_px"), self.stride_px);
            (self.gl.uniform1ui)(u("u_tiles_x"), g.tiles_x() as u32);
            (self.gl.uniform1ui)(u("u_count"), count as u32);
            (self.gl.uniform1ui)(u("u_pbase_words"), self.payload_base_words);
            (self.gl.dispatch_compute)(count as u32, 1, 1);
            // Make the image stores visible to the presenter's sampler. The
            // spec bit for imageStore -> texture() is GL_TEXTURE_FETCH_BARRIER_BIT
            // (and GL_SHADER_IMAGE_ACCESS_BARRIER_BIT for the next decode's
            // imageLoad of the previous frame). Mesa/Intel additionally keeps
            // the image-store result in a write cache until the image is
            // unbound, so the presenter would sample stale texels (a persistent
            // cursor trail on the in-place lossless texture). Cover all four
            // consumers and unbind the image before the caller samples it.
            (self.gl.memory_barrier)(
                GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
                    | GL_TEXTURE_FETCH_BARRIER_BIT
                    | GL_TEXTURE_UPDATE_BARRIER_BIT
                    | GL_FRAMEBUFFER_BARRIER_BIT,
            );
            (self.gl.bind_image_texture)(0, 0, 0, 0, 0, GL_READ_WRITE, GL_RGBA8);
            let err = (self.gl.get_error)();
            if err != 0 {
                bail!("decode_to_image glError 0x{err:04x}");
            }
            (self.gl.finish)();
        }
        Ok(())
    }
}

fn build_program(gl: &Gl, src: &str) -> Result<u32> {
    unsafe {
        let sh = (gl.create_shader)(GL_COMPUTE_SHADER);
        let c = CString::new(src.as_bytes()).unwrap();
        (gl.shader_source)(
            sh,
            1,
            (&[c.as_ptr() as *const u8] as *const [*const u8]) as *const *const u8,
            std::ptr::null(),
        );
        (gl.compile_shader)(sh);
        let mut ok = 0;
        (gl.get_shaderiv)(sh, GL_COMPILE_STATUS, &mut ok);
        if ok == 0 {
            let mut log = [0u8; 4096];
            let mut n = 0;
            (gl.get_shader_info_log)(sh, 4096, &mut n, log.as_mut_ptr());
            if std::env::var("RDMA_SHADER_DUMP").is_ok() {
                for (i, l) in src.lines().enumerate() {
                    eprintln!("{:>4}: {l}", i + 1);
                }
            }
            bail!(
                "compute shader compile failed: {}",
                String::from_utf8_lossy(&log[..n as usize])
            );
        }
        let prog = (gl.create_program)();
        (gl.attach_shader)(prog, sh);
        (gl.link_program)(prog);
        let mut ok = 0;
        (gl.get_programiv)(prog, GL_LINK_STATUS, &mut ok);
        if ok == 0 {
            let mut log = [0u8; 4096];
            let mut n = 0;
            (gl.get_program_info_log)(prog, 4096, &mut n, log.as_mut_ptr());
            bail!(
                "program link failed: {}",
                String::from_utf8_lossy(&log[..n as usize])
            );
        }
        Ok(prog)
    }
}

/// Compile+link a vertex+fragment program (the dmabuf blit).
fn build_graphics_program(gl: &Gl, vs_src: &str, fs_src: &str) -> Result<u32> {
    unsafe {
        let compile = |ty: u32, src: &str, what: &str| -> Result<u32> {
            let sh = (gl.create_shader)(ty);
            let c = CString::new(src.as_bytes()).unwrap();
            (gl.shader_source)(
                sh,
                1,
                (&[c.as_ptr() as *const u8] as *const [*const u8]) as *const *const u8,
                std::ptr::null(),
            );
            (gl.compile_shader)(sh);
            let mut ok = 0;
            (gl.get_shaderiv)(sh, GL_COMPILE_STATUS, &mut ok);
            if ok == 0 {
                let mut log = [0u8; 4096];
                let mut n = 0;
                (gl.get_shader_info_log)(sh, 4096, &mut n, log.as_mut_ptr());
                bail!(
                    "{what} shader compile failed: {}",
                    String::from_utf8_lossy(&log[..n as usize])
                );
            }
            Ok(sh)
        };
        let vs = compile(GL_VERTEX_SHADER, vs_src, "vertex")?;
        let fs = compile(GL_FRAGMENT_SHADER, fs_src, "fragment")?;
        let prog = (gl.create_program)();
        (gl.attach_shader)(prog, vs);
        (gl.attach_shader)(prog, fs);
        (gl.link_program)(prog);
        let mut ok = 0;
        (gl.get_programiv)(prog, GL_LINK_STATUS, &mut ok);
        if ok == 0 {
            let mut log = [0u8; 4096];
            let mut n = 0;
            (gl.get_program_info_log)(prog, 4096, &mut n, log.as_mut_ptr());
            bail!(
                "blit program link failed: {}",
                String::from_utf8_lossy(&log[..n as usize])
            );
        }
        Ok(prog)
    }
}

/// Full-screen triangle for the scanout blit; the `.bgra` in the fragment
/// shader makes the source image's R,G,B,A equal the wire bytes B,G,R,A
/// (matching the materializer's `glReadPixels(GL_BGRA)` readback), so the
/// encoder reconstructs identical 32-bit words.
const BLIT_VS: &str = r#"#version 310 es
out vec2 v_uv;
void main() {
    float x = float((gl_VertexID == 1) ? 3 : -1);
    float y = float((gl_VertexID == 2) ? 3 : -1);
    v_uv = vec2(x, y) * 0.5 + 0.5;
    gl_Position = vec4(vec2(x, y), 0.0, 1.0);
}
"#;

const BLIT_FS: &str = r#"#version 310 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
uniform samplerExternalOES tex;
in vec2 v_uv;
out vec4 frag;
void main() { frag = texture(tex, v_uv).bgra; }
"#;

/// Shared lossless-encode compute body (v2). The four encoders differ only in
/// how they produce the current source word via `word(x, y)` (SSBO buffer,
/// RGBA8 image, external sampler, or `sampler2D`). A tile picks the cheapest
/// of `RAW` (packed valid pixels), `REF` (runs of `cur ^ ref`) and `DPCM`
/// (runs of the horizontal predictor); `REF` is never used on a full frame.
/// Payloads are packed densely by an atomic bump allocator, so each tile
/// records its own offset in the index entry. The mode choice and token layout
/// must match `rdma_display_codec::tile_plan` exactly, or the coded total and
/// the CPU/GPU cross-decode tests disagree.
const ENC_PRELUDE: &str = r#"
layout(std430, binding = 1) buffer Ref { uint ref[]; };
layout(std430, binding = 2) writeonly buffer Out { uint outb[]; };
layout(std430, binding = 3) buffer Cnt { uint count; uint alloc; };
const uint TILE = 64u;
const uint TW = TILE * TILE;
shared uint s_runs_r[256];
shared uint s_runs_d[256];
shared uint s_roff[256];
shared uint s_mode;
shared uint s_off;
shared uint s_changed;
"#;

const ENC_BODY: &str = r#"
void main() {
    uint tid = gl_LocalInvocationID.x;
    uint tile_id = gl_WorkGroupID.x;
    uint tx = tile_id % u_tiles_x;
    uint ty = tile_id / u_tiles_x;
    uint x0 = tx * TILE, y0 = ty * TILE;
    uint vw = min(TILE, u_width - x0);
    uint vh = min(TILE, u_height - y0);

    bool changed = (u_full != 0u);
    if (!changed) {
        uint acc = 0u;
        for (uint i = tid; i < TW; i += 256u) {
            uint lx = i % TILE, ly = i / TILE;
            if (lx >= vw || ly >= vh) continue;
            uint gi = (y0 + ly) * u_stride_px + (x0 + lx);
            acc |= (word(x0 + lx, y0 + ly) ^ ref[gi]);
        }
        changed = (acc != 0u);
    }
    if (tid == 0u) s_changed = 0u;
    barrier();
    if (changed) atomicOr(s_changed, 1u);
    barrier();
    if (s_changed == 0u) return;

    if (tid < vh) {
        uint y = tid;
        uint runs_r = 0u, runs_d = 0u;
        uint prev_r = 0u, prev_d = 0u, pcur = 0u;
        for (uint px = 0u; px < vw; ++px) {
            uint w = word(x0 + px, y0 + y);
            uint rf = ref[(y0 + y) * u_stride_px + (x0 + px)];
            uint r = w ^ rf;
            uint d = w - ((px == 0u) ? 0u : pcur);
            if (px == 0u || r != prev_r) { runs_r++; prev_r = r; }
            if (px == 0u || d != prev_d) { runs_d++; prev_d = d; }
            pcur = w;
        }
        s_runs_r[y] = runs_r;
        s_runs_d[y] = runs_d;
    }
    barrier();

    if (tid == 0u) {
        uint tr = 0u, td = 0u;
        for (uint y = 0u; y < vh; ++y) { tr += s_runs_r[y]; td += s_runs_d[y]; }
        uint mode = 0u;
        uint best = vw * vh;
        if (u_full == 0u && (vh + 2u * tr) < best) { best = vh + 2u * tr; mode = 1u; }
        if ((vh + 2u * td) < best) { best = vh + 2u * td; mode = 2u; }
        s_mode = mode;
        uint off = atomicAdd(alloc, (mode == 0u) ? (vw * vh) : best);
        s_off = off;
        if (mode != 0u) {
            uint acc = 0u;
            for (uint y = 0u; y < vh; ++y) {
                s_roff[y] = acc;
                acc += (mode == 1u) ? s_runs_r[y] : s_runs_d[y];
            }
        }
        uint k = atomicAdd(count, 1u);
        uint e = 12u + k * 3u;
        outb[e] = tile_id;
        outb[e + 1u] = off;
        outb[e + 2u] = (best & 0xffffu) | (mode << 16u);
    }
    barrier();

    uint mode = s_mode;
    uint base = u_pbase_words + s_off;
    if (mode == 0u) {
        for (uint i = tid; i < vw * vh; i += 256u) {
            uint lx = i % vw, ly = i / vw;
            outb[base + i] = word(x0 + lx, y0 + ly);
        }
    } else if (tid < vh) {
        uint y = tid;
        outb[base + y] = (mode == 1u) ? s_runs_r[y] : s_runs_d[y];
        uint tok = base + vh + 2u * s_roff[y];
        uint lr = 0u, lv = 0u, pcur = 0u;
        for (uint i = 0u; i < vw; ++i) {
            uint w = word(x0 + i, y0 + y);
            uint rf = ref[(y0 + y) * u_stride_px + (x0 + i)];
            uint res = (mode == 1u) ? (w ^ rf) : (w - ((i == 0u) ? 0u : pcur));
            if (i == 0u) { lv = res; lr = 1u; }
            else if (res == lv) { lr++; }
            else { outb[tok] = lr; outb[tok + 1u] = lv; tok += 2u; lv = res; lr = 1u; }
            pcur = w;
        }
        outb[tok] = lr;
        outb[tok + 1u] = lv;
    }
    barrier();

    // advance the reference in-pass for the next delta (the texture/external/
    // fetch paths); the SSBO path leaves this to `decode`, which validates the
    // decode at the same time
    if (u_update_ref != 0u) {
        for (uint i = tid; i < TW; i += 256u) {
            uint lx = i % TILE, ly = i / TILE;
            if (lx < vw && ly < vh) {
                uint gi = (y0 + ly) * u_stride_px + (x0 + lx);
                ref[gi] = word(x0 + lx, y0 + ly);
            }
        }
    }
}
"#;

fn enc_shader(extension: &str, source_decl: &str, word_fn: &str) -> String {
    format!(
        "#version 310 es\n{extension}precision highp int;\nprecision highp float;\n\
layout(local_size_x = 256) in;\n\
uniform uint u_width, u_height, u_stride_px, u_tiles_x, u_full, u_pbase_words, u_update_ref;\n\
{source_decl}\n{ENC_PRELUDE}\n{word_fn}\n{ENC_BODY}\n"
    )
}

/// SSBO-input encode (the CPU-equivalent reference path; also `encode`).
fn enc_src() -> String {
    enc_shader(
        "",
        "layout(std430, binding = 0) readonly buffer Cur { uint cur[]; };",
        "uint word(uint x, uint y) { return cur[y * u_stride_px + x]; }",
    )
}

/// Encode directly from the RGBA8 source image (after the blit).
fn enc_tex_src() -> String {
    enc_shader(
        "",
        "layout(rgba8, binding = 0) readonly uniform highp image2D u_src;",
        "uint word(uint x, uint y) { return packUnorm4x8(imageLoad(u_src, ivec2(int(x), int(y)))); }",
    )
}

/// Encode straight from the imported external texture (no blit).
fn enc_ext_src() -> String {
    enc_shader(
        "#extension GL_OES_EGL_image_external_essl3 : require\n",
        "uniform highp samplerExternalOES u_src;",
        "uint word(uint x, uint y) {\n    vec2 uv = (vec2(float(x), float(y)) + 0.5) / vec2(float(u_width), float(u_height));\n    return packUnorm4x8(texture(u_src, uv).bgra);\n}",
    )
}

/// Encode from a linear scanout bound to `GL_TEXTURE_2D` (`texelFetch`).
fn enc_fetch_src() -> String {
    enc_shader(
        "",
        "uniform highp sampler2D u_src;",
        "uint word(uint x, uint y) { return packUnorm4x8(texelFetch(u_src, ivec2(int(x), int(y)), 0)); }",
    )
}

/// Read-bandwidth probe: stream `u_words` uints from the buffer at binding 4.
/// The load is kept live by a data-dependent store, so the compiler cannot
/// drop it. Reads run at hundreds of GB/s from VRAM but at PCIe speed (~8 GB/s
/// on Gen3 x8) from host-visible/GTT memory, so this is how we tell where a
/// surface actually lives.
const PROBE_BUF_SRC: &str = r#"#version 310 es
precision highp int;
layout(local_size_x = 256) in;
uniform uint u_words;
layout(std430, binding = 4) readonly buffer Src { uint s[]; };
layout(std430, binding = 5) writeonly buffer Out { uint o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= u_words) return;
    uint v = s[i];
    if ((v ^ (i * 2654435761u)) == 0xdeadbeefu) o[0] = i;
}
"#;

/// Same probe for the imported scanout, sampled exactly like the encoder does.
const PROBE_EXT_SRC: &str = r#"#version 310 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp int;
precision highp float;
layout(local_size_x = 256) in;
uniform uint u_width, u_height;
uniform highp samplerExternalOES u_src;
layout(std430, binding = 5) writeonly buffer Out { uint o[]; };
uint word(uint x, uint y) {
    vec2 uv = (vec2(float(x), float(y)) + 0.5) / vec2(float(u_width), float(u_height));
    return packUnorm4x8(texture(u_src, uv));
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= u_width * u_height) return;
    uint w = word(i % u_width, i / u_width);
    if ((w ^ (i * 2654435761u)) == 0xdeadbeefu) o[0] = i;
}
"#;

/// Same probe for a linear scanout bound to `GL_TEXTURE_2D` (`texelFetch`).
const PROBE_2D_SRC: &str = r#"#version 310 es
precision highp int;
layout(local_size_x = 256) in;
uniform uint u_width, u_height;
uniform highp sampler2D u_src;
layout(std430, binding = 5) writeonly buffer Out { uint o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= u_width * u_height) return;
    uint w = packUnorm4x8(texelFetch(u_src, ivec2(int(i % u_width), int(i / u_width)), 0));
    if ((w ^ (i * 2654435761u)) == 0xdeadbeefu) o[0] = i;
}
"#;

/// Shared lossless-decode compute body (v2). `read_coded`, `load_px` and
/// `store_px` come from each decoder (SSBO frame or `GL_RGBA8` image). One
/// workgroup per changed tile; a `REF`/`DPCM` tile expands one row per thread,
/// so it needs no cross-invocation scan.
const DEC_BODY: &str = r#"
void main() {
    uint k = gl_WorkGroupID.x;
    if (k >= u_count) return;
    uint tid = gl_LocalInvocationID.x;
    uint i0 = 12u + k * 3u;
    uint tile_id = read_coded(i0);
    uint off = read_coded(i0 + 1u);
    uint mode = read_coded(i0 + 2u) >> 16u;
    uint tx = tile_id % u_tiles_x;
    uint ty = tile_id / u_tiles_x;
    uint x0 = tx * TILE, y0 = ty * TILE;
    uint vw = min(TILE, u_width - x0);
    uint vh = min(TILE, u_height - y0);
    uint base = u_pbase_words + off;
    if (mode == 0u) {
        for (uint i = tid; i < vw * vh; i += 256u) {
            uint lx = i % vw, ly = i / vw;
            store_px(x0 + lx, y0 + ly, read_coded(base + i));
        }
        return;
    }
    if (tid < vh) s_cnt[tid] = read_coded(base + tid);
    barrier();
    if (tid < vh) {
        uint y = tid;
        uint roff = 0u;
        for (uint yy = 0u; yy < y; ++yy) roff += s_cnt[yy];
        uint tok = base + vh + 2u * roff;
        uint runs = s_cnt[y];
        uint x = 0u;
        uint acc = 0u;
        for (uint r = 0u; r < runs; ++r) {
            uint len = read_coded(tok);
            uint val = read_coded(tok + 1u);
            tok += 2u;
            for (uint j = 0u; j < len; ++j) {
                uint nv;
                if (mode == 1u) { nv = load_px(x0 + x, y0 + y) ^ val; }
                else { acc += val; nv = acc; }
                store_px(x0 + x, y0 + y, nv);
                x++;
            }
        }
    }
}
"#;

const DEC_PRELUDE: &str = r#"
const uint TILE = 64u;
shared uint s_cnt[256];
"#;

/// Decode into the reference/frame SSBO (the server's reference update and the
/// benchmark's GPU decode).
fn dec_src() -> String {
    format!(
        "#version 310 es\nprecision highp int;\nlayout(local_size_x = 256) in;\n\
uniform uint u_width, u_height, u_stride_px, u_tiles_x, u_count, u_pbase_words;\n\
layout(std430, binding = 2) readonly buffer In {{ uint inb[]; }};\n\
layout(std430, binding = 1) buffer Frame {{ uint frame[]; }};\n\
uint read_coded(uint i) {{ return inb[i]; }}\n\
uint load_px(uint x, uint y) {{ return frame[y * u_stride_px + x]; }}\n\
void store_px(uint x, uint y, uint v) {{ frame[y * u_stride_px + x] = v; }}\n\
{DEC_PRELUDE}\n{DEC_BODY}\n"
    )
}

/// Decode into a `GL_RGBA8` texture (the client presenter's target). The image
/// is read as well as written because `REF` reads the previous frame's texel.
fn dec_tex_src() -> String {
    format!(
        "#version 310 es\nprecision highp int;\nlayout(local_size_x = 256) in;\n\
uniform uint u_width, u_height, u_stride_px, u_tiles_x, u_count, u_pbase_words;\n\
layout(std430, binding = 4) readonly buffer In {{ uint inb[]; }};\n\
layout(rgba8, binding = 0) uniform highp image2D u_img;\n\
uint read_coded(uint i) {{ return inb[i]; }}\n\
uint load_px(uint x, uint y) {{ return packUnorm4x8(imageLoad(u_img, ivec2(int(x), int(y)))); }}\n\
void store_px(uint x, uint y, uint v) {{ imageStore(u_img, ivec2(int(x), int(y)), unpackUnorm4x8(v)); }}\n\
{DEC_PRELUDE}\n{DEC_BODY}\n"
    )
}
