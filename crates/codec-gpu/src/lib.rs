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
use rdma_display_codec::{Header, Geometry, BPP, HEADER_LEN};

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
const GL_ALL_BARRIER_BITS: u32 = 0xFFFF_FFFF;
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
const GL_WRITE_ONLY: u32 = 0x88B9;
const GL_SHADER_IMAGE_ACCESS_BARRIER_BIT: u32 = 0x0000_0020;
const GL_TEXTURE_FETCH_BARRIER_BIT: u32 = 0x0000_0008;

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
    pub fd: i32,
    pub offset: u32,
    pub stride: u32,
    pub fourcc: u32,
    pub modifier: u64,
    pub width: u32,
    pub height: u32,
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
    has_scanout: bool,
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
            let enc_tex_prog = build_program(&gl, ENC_TEX_SRC)?;

            let frame_bytes = geom.frame_bytes();
            let index_cap = geom.tile_count();
            let coded_len = geom.max_coded_len();
            let payload_base_words = ((HEADER_LEN + index_cap * 4) / 4) as u32;

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
                4,
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
                4,
                GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT,
            );
            if counter_map.is_null() {
                bail!("persistent map of the counter failed");
            }

            (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 0, cur);
            (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 1, reference);
            (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 2, coded);
            (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 3, counter);

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
                cur,
                reference,
                coded,
                counter,
                src_tex: 0,
                src_img,
                src_fbo,
                src_egl_image: std::ptr::null_mut(),
                has_scanout: false,
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

    /// Tiles changed by the most recent `encode`/`encode_tex` (0 = idle delta).
    pub fn last_count(&self) -> usize {
        self.last_count
    }

    /// Import (or re-import) a scanout dmabuf as an external texture and blit
    /// it into the RGBA8 source image, so [`encode_tex`](Self::encode_tex)
    /// encodes straight from GPU memory — no full-frame readback to host and no
    /// re-upload. Makes the codec's EGL context current itself. Errors (missing
    /// `EGL_EXT_image_dma_buf_import`, unsupported modifier, ...) are returned
    /// so the caller can fall back to the readback+upload path.
    pub fn set_scanout(&mut self, d: &DmabufImage) -> Result<()> {
        self.activate()?;
        let create = self
            .create_image
            .context("no eglCreateImageKHR (EGL_EXT_image_dma_buf_import absent)")?;
        let destroy = self.destroy_image.context("no eglDestroyImageKHR")?;
        let itt = self
            .gl
            .image_target_texture
            .context("no glEGLImageTargetTexture2DOES")?;

        // release the previous import (unbind before deleting either object)
        if !self.src_egl_image.is_null() {
            unsafe {
                (self.gl.bind_texture)(GL_TEXTURE_EXTERNAL_OES, 0);
                if self.src_tex != 0 {
                    let t = [self.src_tex];
                    (self.gl.delete_textures)(1, t.as_ptr());
                }
                (destroy)(self.dpy, self.src_egl_image);
            }
            self.src_egl_image = std::ptr::null_mut();
            self.src_tex = 0;
            self.has_scanout = false;
        }

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
            (self.gl.bind_texture)(GL_TEXTURE_EXTERNAL_OES, t[0]);
            (itt)(GL_TEXTURE_EXTERNAL_OES, img);
            (self.gl.tex_parameteri)(
                GL_TEXTURE_EXTERNAL_OES,
                GL_TEXTURE_MIN_FILTER,
                GL_NEAREST as i32,
            );
            (self.gl.tex_parameteri)(
                GL_TEXTURE_EXTERNAL_OES,
                GL_TEXTURE_MAG_FILTER,
                GL_NEAREST as i32,
            );
            (self.gl.tex_parameteri)(
                GL_TEXTURE_EXTERNAL_OES,
                GL_TEXTURE_WRAP_S,
                GL_CLAMP_TO_EDGE as i32,
            );
            (self.gl.tex_parameteri)(
                GL_TEXTURE_EXTERNAL_OES,
                GL_TEXTURE_WRAP_T,
                GL_CLAMP_TO_EDGE as i32,
            );
            let err = (self.gl.get_error)();
            if err != 0 {
                (destroy)(self.dpy, img);
                bail!("EGLImage -> external texture glError 0x{err:04x}");
            }
            self.src_tex = t[0];
        }
        self.src_egl_image = img;
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
            // no glFinish here: the encode dispatch follows on the same
            // command stream, so GPU ordering plus the barrier is enough (the
            // encode's own finish waits for both)
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
        let n = self.encode_with(prog, full)?;
        unsafe {
            let e = (self.gl.get_error)();
            if e != 0 {
                bail!("encode_tex: glError 0x{e:04x} after dispatch");
            }
        }
        Ok(n)
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
        self.encode_with(prog, full)
    }

    /// Shared encode body. `prog` is either the SSBO-input (`cur`) encoder or
    /// the image-input (`src_img`) encoder.
    fn encode_with(&mut self, prog: u32, full: bool) -> Result<usize> {
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
        };
        unsafe {
            h.write(std::slice::from_raw_parts_mut(self.coded_map, HEADER_LEN));
            *self.counter_map = 0;
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
            (self.gl.dispatch_compute)(g.tile_count() as u32, 1, 1);
            (self.gl.memory_barrier)(GL_ALL_BARRIER_BITS);
            (self.gl.finish)();
        }
        let count = unsafe { *self.counter_map } as usize;
        if count > self.index_cap {
            bail!("shader produced {count} tiles > capacity {}", self.index_cap);
        }
        self.last_count = count;
        let mut patched = h;
        patched.count = count as u32;
        unsafe {
            patched.write(std::slice::from_raw_parts_mut(self.coded_map, HEADER_LEN));
        }
        let end = HEADER_LEN + self.index_cap * 4 + count * g.tile_bytes();
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
            (self.gl.memory_barrier)(GL_ALL_BARRIER_BITS);
            (self.gl.finish)();
        }
        Ok(())
    }

    /// Wait for all submitted GPU work to finish.
    pub fn sync(&self) {
        unsafe { (self.gl.finish)() };
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
        let payload_base_words = ((HEADER_LEN + index_cap * 4) / 4) as u32;
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
            (self.gl.bind_image_texture)(0, tex, 0, 0, 0, GL_WRITE_ONLY, GL_RGBA8);
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
            (self.gl.memory_barrier)(
                GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT,
            );
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

/// Encode directly from the RGBA8 source image (the blit output). Same
/// tile-delta format as the SSBO encoder; only the word source differs.
const ENC_TEX_SRC: &str = r#"#version 310 es
precision highp int;
layout(local_size_x = 256) in;
uniform uint u_width, u_height, u_stride_px, u_tiles_x, u_full, u_pbase_words;
layout(rgba8, binding = 0) readonly uniform highp image2D u_src;
layout(std430, binding = 1) readonly buffer Ref { uint ref[]; };
layout(std430, binding = 2) writeonly buffer Out { uint outb[]; };
layout(std430, binding = 3) buffer Cnt { uint count; };
const uint TILE = 64u;
const uint TW = TILE * TILE;
shared uint s_changed;
shared uint s_k;
uint word(uint x, uint y) {
    return packUnorm4x8(imageLoad(u_src, ivec2(int(x), int(y))));
}
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
        for (uint i = tid; i < TW; i += 256u) {
            uint lx = i % TILE, ly = i / TILE;
            if (lx >= vw || ly >= vh) continue;
            uint gi = (y0 + ly) * u_stride_px + (x0 + lx);
            if (word(x0 + lx, y0 + ly) != ref[gi]) { changed = true; break; }
        }
    }
    if (tid == 0u) s_changed = 0u;
    barrier();
    if (changed) atomicOr(s_changed, 1u);
    barrier();
    if (s_changed == 0u) return;
    if (tid == 0u) { uint k = atomicAdd(count, 1u); s_k = k; outb[8u + k] = tile_id; }
    barrier();
    uint base = u_pbase_words + s_k * TW;
    for (uint i = tid; i < TW; i += 256u) {
        uint lx = i % TILE, ly = i / TILE;
        uint o = base + i;
        if (lx < vw && ly < vh) {
            outb[o] = word(x0 + lx, y0 + ly);
        } else {
            outb[o] = 0u;
        }
    }
}
"#;

const ENC_SRC: &str = r#"#version 310 es
precision highp int;
layout(local_size_x = 256) in;
uniform uint u_width, u_height, u_stride_px, u_tiles_x, u_full, u_pbase_words;
layout(std430, binding = 0) readonly buffer Cur { uint cur[]; };
layout(std430, binding = 1) readonly buffer Ref { uint ref[]; };
layout(std430, binding = 2) writeonly buffer Out { uint outb[]; };
layout(std430, binding = 3) buffer Cnt { uint count; };
const uint TILE = 64u;
const uint TW = TILE * TILE;
shared uint s_changed;
shared uint s_k;
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
        for (uint i = tid; i < TW; i += 256u) {
            uint lx = i % TILE, ly = i / TILE;
            if (lx >= vw || ly >= vh) continue;
            uint gi = (y0 + ly) * u_stride_px + (x0 + lx);
            if (cur[gi] != ref[gi]) { changed = true; break; }
        }
    }
    if (tid == 0u) s_changed = 0u;
    barrier();
    if (changed) atomicOr(s_changed, 1u);
    barrier();
    if (s_changed == 0u) return;
    if (tid == 0u) { uint k = atomicAdd(count, 1u); s_k = k; outb[8u + k] = tile_id; }
    barrier();
    uint base = u_pbase_words + s_k * TW;
    for (uint i = tid; i < TW; i += 256u) {
        uint lx = i % TILE, ly = i / TILE;
        uint o = base + i;
        if (lx < vw && ly < vh) {
            uint gi = (y0 + ly) * u_stride_px + (x0 + lx);
            outb[o] = cur[gi];
        } else {
            outb[o] = 0u;
        }
    }
}
"#;

const DEC_SRC: &str = r#"#version 310 es
precision highp int;
layout(local_size_x = 256) in;
uniform uint u_width, u_height, u_stride_px, u_tiles_x, u_count, u_pbase_words;
layout(std430, binding = 2) readonly buffer In { uint inb[]; };
layout(std430, binding = 1) buffer Frame { uint frame[]; };
const uint TILE = 64u;
const uint TW = TILE * TILE;
void main() {
    uint k = gl_WorkGroupID.x;
    if (k >= u_count) return;
    uint tile_id = inb[8u + k];
    uint tx = tile_id % u_tiles_x;
    uint ty = tile_id / u_tiles_x;
    uint x0 = tx * TILE, y0 = ty * TILE;
    uint vw = min(TILE, u_width - x0);
    uint vh = min(TILE, u_height - y0);
    uint base = u_pbase_words + k * TW;
    for (uint i = gl_LocalInvocationID.x; i < TW; i += 256u) {
        uint lx = i % TILE, ly = i / TILE;
        if (lx >= vw || ly >= vh) continue;
        uint gi = (y0 + ly) * u_stride_px + (x0 + lx);
        frame[gi] = inb[base + i];
    }
}
"#;

fn enc_src() -> String {
    ENC_SRC.to_string()
}
fn dec_src() -> String {
    DEC_SRC.to_string()
}

const DEC_TEX_SRC: &str = r#"#version 310 es
precision highp int;
layout(local_size_x = 256) in;
uniform uint u_width, u_height, u_stride_px, u_tiles_x, u_count, u_pbase_words;
layout(std430, binding = 4) readonly buffer In { uint inb[]; };
layout(rgba8, binding = 0) uniform highp writeonly image2D u_img;
const uint TILE = 64u;
const uint TW = TILE * TILE;
void main() {
    uint k = gl_WorkGroupID.x;
    if (k >= u_count) return;
    uint tile_id = inb[8u + k];
    uint tx = tile_id % u_tiles_x;
    uint ty = tile_id / u_tiles_x;
    uint x0 = tx * TILE, y0 = ty * TILE;
    uint vw = min(TILE, u_width - x0);
    uint vh = min(TILE, u_height - y0);
    uint base = u_pbase_words + k * TW;
    for (uint i = gl_LocalInvocationID.x; i < TW; i += 256u) {
        uint lx = i % TILE, ly = i / TILE;
        if (lx >= vw || ly >= vh) continue;
        uint v = inb[base + i];
        // wire word is B,G,R,A little-endian; store those four bytes in the
        // texture's R,G,B,A so the presenter's sampler sees the same texel as
        // glTexImage2D(GL_RGBA) of the wire bytes would.
        imageStore(u_img, ivec2(int(x0 + lx), int(y0 + ly)), unpackUnorm4x8(v));
    }
}
"#;

fn dec_tex_src() -> String {
    DEC_TEX_SRC.to_string()
}
