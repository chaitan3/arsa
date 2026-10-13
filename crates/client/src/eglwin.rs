//! EGL window presenter (C1, `--output egl`): the mirror of the server's
//! copy path.
//!
//! The NIC lands frames in pinned host slots (`--slot-mem host`, default);
//! the GPU *pulls* them with `glTexImage2D` (a DMA read of ordinary mapped
//! RAM — no dma-buf import anywhere, which is what KWin's `import_failed`
//! ruled out on this stack), and presents via `eglSwapBuffers` on an
//! EGL-window surface whose backings are driver-native — the only buffers
//! KWin ever sees, and always acceptable to it.
//!
//! Channel order: wire bytes are B,G,R,X; we upload them as-is (GL sees
//! r=B,g=G,b=R) and the fragment shader swizzles `.bgra` — spec-clean on
//! GLES 3.0, zero CPU work.
//!
//! Credit policy: glTexImage2D consumes the client pointer before returning
//! (GL spec), so the slot can be credited immediately after the upload call
//! — no glFinish, full pipelining with driver-side swap throttling.
//!
//! All EGL/GL/wayland-egl symbols are dlopened; nothing links at build time.

use std::ffi::c_void;

use anyhow::{bail, Context, Result};

// --- minimal EGL/GL constants -----------------------------------------------
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_WINDOW_BIT: i32 = 0x0004;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_OPENGL_ES3_BIT: i32 = 0x0040;
const EGL_ALPHA_SIZE: i32 = 0x3021;
const EGL_BUFFER_SIZE: i32 = 0x3020;
const EGL_RED_SIZE: i32 = 0x3024;
const EGL_NATIVE_VISUAL_ID: i32 = 0x302E;
const EGL_CONFIG_ID: i32 = 0x3028;
const EGL_NONE: i32 = 0x3038;
const EGL_CONTEXT_CLIENT_VERSION: i32 = 0x3098;
const EGL_CONTEXT_MINOR_VERSION: i32 = 0x30FB;
const EGL_OPENGL_ES_API: u32 = 0x30A0;

const GL_TEXTURE_2D: u32 = 0x0DE1;
/// dma-buf EGLImages bind to the external target — the only one NVIDIA's GL
/// accepts (Mesa also accepts GL_TEXTURE_2D).
const GL_TEXTURE_EXTERNAL_OES: u32 = 0x8BE7;
const GL_RGBA8: u32 = 0x8058;
const GL_RGBA: u32 = 0x1908;
const GL_UNSIGNED_BYTE: u32 = 0x1401;
const GL_TEXTURE0: u32 = 0x84C0;
const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
const GL_NEAREST: u32 = 0x2600;
const GL_LINEAR: u32 = 0x2601;
const GL_TRIANGLES: u32 = 0x0004;
const GL_BLEND: u32 = 0x0BE2;
const GL_SRC_ALPHA: u32 = 0x0302;
const GL_ONE_MINUS_SRC_ALPHA: u32 = 0x0303;
const GL_VERTEX_SHADER: u32 = 0x8B31;
const GL_FRAGMENT_SHADER: u32 = 0x8B30;
const GL_COMPILE_STATUS: u32 = 0x8B81;
const GL_LINK_STATUS: u32 = 0x8B82;

/// Effective sampler names, indexed by `effective_filter()`.
const FILTER_NAMES: [&str; 4] = ["nearest", "linear", "cubic", "area"];

// --- EGL_EXT_image_dma_buf_import (YUV planes shared from Vulkan) -----------
const EGL_WIDTH: i32 = 0x3057;
const EGL_HEIGHT: i32 = 0x3056;
const EGL_LINUX_DMA_BUF_EXT: i32 = 0x3270;
const EGL_LINUX_DRM_FOURCC_EXT: i32 = 0x3271;
const EGL_DMA_BUF_PLANE0_FD_EXT: i32 = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: i32 = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: i32 = 0x3274;
/// DRM_FORMAT_R8 ('R','8',' ',' ')
const DRM_FORMAT_R8: i32 = 0x2020_3852;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: i32 = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: i32 = 0x3444;
const EGL_EXTENSIONS: i32 = 0x3055;
const GL_EXTENSIONS: u32 = 0x1F03;
const GL_VENDOR: u32 = 0x1F00;
const GL_RENDERER: u32 = 0x1F01;
// DRM fourccs we might export from Vulkan; queried for importability below.
const DRM_FORMAT_ARGB8888: i32 = 0x3432_5241;
const DRM_FORMAT_ABGR8888: i32 = 0x3432_4241;
const DRM_FORMAT_XRGB8888: i32 = 0x3432_5258;
const DRM_FORMAT_XBGR8888: i32 = 0x3432_4258;

/// DRM fourcc -> its ASCII name (e.g. 0x20203852 -> "R8  ").
fn fourcc_name(f: i32) -> String {
    String::from_utf8_lossy(&f.to_le_bytes()).to_string()
}

/// Which GL entry point creates the texture from the EGLImage.
#[derive(Clone, Copy)]
enum BindFn {
    /// GL_OES_EGL_image
    Does(unsafe extern "system" fn(u32, *mut c_void)),
    /// GL_EXT_EGL_image_storage (recommended on modern drivers)
    Storage(unsafe extern "system" fn(u32, *mut c_void, *const i32)),
}

fn sym<T: Copy>(lib: &libloading::Library, name: &str) -> Result<T> {
    unsafe {
        let s: libloading::Symbol<T> = lib
            .get(name.as_bytes())
            .with_context(|| format!("missing symbol {name}"))?;
        Ok(*s)
    }
}

struct EglLib {
    #[allow(dead_code)]
    lib: std::sync::Arc<libloading::Library>,
    get_display: unsafe extern "system" fn(*mut c_void) -> *mut c_void,
    initialize: unsafe extern "system" fn(*mut c_void, *mut i32, *mut i32) -> u32,
    bind_api: unsafe extern "system" fn(u32) -> u32,
    choose_config:
        unsafe extern "system" fn(*mut c_void, *const i32, *mut *mut c_void, i32, *mut i32) -> u32,
    get_config_attrib:
        unsafe extern "system" fn(*mut c_void, *mut c_void, i32, *mut i32) -> u32,
    create_context: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const i32) -> *mut c_void,
    create_window_surface:
        unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const i32) -> *mut c_void,
    make_current: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> u32,
    swap_buffers: unsafe extern "system" fn(*mut c_void, *mut c_void) -> u32,
    swap_interval: unsafe extern "system" fn(*mut c_void, i32) -> u32,
    get_error: unsafe extern "system" fn() -> i32,
    get_proc_address: unsafe extern "system" fn(*const i8) -> *const c_void,
    query_string: unsafe extern "system" fn(*mut c_void, i32) -> *const i8,
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
    use_program: unsafe extern "system" fn(u32),
    gen_textures: unsafe extern "system" fn(i32, *mut u32),
    delete_textures: unsafe extern "system" fn(i32, *const u32),
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
        *const c_void,
    ),
    viewport: unsafe extern "system" fn(i32, i32, i32, i32),
    draw_arrays: unsafe extern "system" fn(u32, i32, i32),
    clear: unsafe extern "system" fn(u32),
    get_error: unsafe extern "system" fn() -> u32,
    get_string: unsafe extern "system" fn(u32) -> *const u8,
    enable: unsafe extern "system" fn(u32),
    disable: unsafe extern "system" fn(u32),
    blend_func: unsafe extern "system" fn(u32, u32),
    uniform_2f: unsafe extern "system" fn(i32, f32, f32),
    get_uniform_location: unsafe extern "system" fn(u32, *const i8) -> i32,
    uniform_1i: unsafe extern "system" fn(i32, i32),
    active_texture: unsafe extern "system" fn(u32),
}

const GL_COLOR_BUFFER_BIT: u32 = 0x4000;

fn egl_func(egl: &EglLib, name: &str) -> Result<*const c_void> {
    let cname = std::ffi::CString::new(name)?;
    let p = unsafe { (egl.get_proc_address)(cname.as_ptr()) };
    if p.is_null() {
        bail!("GL function {name} unavailable via eglGetProcAddress");
    }
    Ok(p)
}

/// `RDMA_DISPLAY_LOSSY_DEBUG=1`: verbose dma-buf import diagnostics (extension
/// strings, importable formats/modifiers, and every import attempt).
fn lossy_debug() -> bool {
    std::env::var_os("RDMA_DISPLAY_LOSSY_DEBUG").is_some()
}

const VS: &str = r#"#version 300 es
out vec2 v_uv;
void main() {
    float x = float((gl_VertexID == 1) ? 3 : -1);
    float y = float((gl_VertexID == 2) ? 3 : -1);
    v_uv = vec2(x, y) * 0.5 + 0.5;
    gl_Position = vec4(vec2(x, y), 0.0, 1.0);
}
"#;

const FS_HEAD: &str = r#"#version 300 es
precision highp float;
#define SF_SAMPLER sampler2D
uniform sampler2D tex;
uniform vec2 u_win;    // buffer (device) size, px
uniform vec2 u_frame;  // guest frame size, px
in vec2 v_uv;
out vec4 frag;
"#;

const FS_MAIN: &str = r#"
void main() {
    // Letterbox: aspect-fit the frame inside the buffer, black bars elsewhere.
    vec2 px = v_uv * u_win;              // bottom-left origin
    float scl = min(u_win.x / max(u_frame.x, 1.0), u_win.y / max(u_frame.y, 1.0));
    vec2 pic = u_frame * scl;
    vec2 org = (u_win - pic) * 0.5;
    if (all(greaterThanEqual(px, org)) && all(lessThan(px, org + pic))) {
        vec2 uv = (px - org) / pic;
        // Wire is top-first; glTexImage2D puts row 0 at v=0 -> mirror v.
        uv.y = 1.0 - uv.y;
        // Wire bytes are B,G,R,X uploaded as RGBA — swizzle to true colour.
        frag = sf(tex, uv, 1.0 / u_frame, 1.0 / max(scl, 1e-3)).bgra;
    } else {
        frag = vec4(0.0, 0.0, 0.0, 1.0);  // letterbox bar
    }
}
"#;

// Scale filter shared by both letterbox shaders (spliced between the HEAD and
// MAIN). u_filter: 0=nearest, 1=linear, 2=cubic, 3=area. The manual filters
// (cubic/area) require the texture in NEAREST mode.
const FILTER_GLSL: &str = r#"
uniform int u_filter;
float bicubic_w(float x) {
    x = abs(x);
    if (x < 1.0) return 1.5 * x * x * x - 2.5 * x * x + 1.0;
    if (x < 2.0) return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
    return 0.0;
}
vec4 sf(SF_SAMPLER t, vec2 uv, vec2 texel, float fp) {
    if (u_filter == 2) {
        // Catmull-Rom bicubic, 4x4 taps
        vec2 s = uv / texel - 0.5;
        vec2 ip = floor(s);
        vec2 f = s - ip;
        vec4 c = vec4(0.0);
        for (int j = -1; j <= 2; j++) {
            for (int i = -1; i <= 2; i++) {
                float w = bicubic_w(float(i) - f.x) * bicubic_w(float(j) - f.y);
                c += w * texture(t, (ip + vec2(float(i) + 0.5, float(j) + 0.5)) * texel);
            }
        }
        return c;
    }
    if (u_filter == 3) {
        // area/box over the source footprint (fp is in source texels)
        fp = clamp(fp, 1.0, 4.0);
        if (fp <= 1.25) return texture(t, uv);
        vec2 src = uv / texel;
        vec4 acc = vec4(0.0);
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                vec2 off = (vec2(float(i), float(j)) + 0.5) * 0.25 - 0.5;
                acc += texture(t, (src + off * fp) * texel);
            }
        }
        return acc * 0.0625;
    }
    return texture(t, uv);   // 0=nearest / 1=linear via the GL texture filter
}
"#;

// lossy YUV->RGB from the three PyroWave decode planes (linear R8 dma-bufs);
// letterbox + vertical flip + scale filter match the RGBA shader.
const FS_YUV_HEAD: &str = r#"#version 300 es
precision highp float;
#define SF_SAMPLER sampler2D
uniform sampler2D y_tex;
uniform sampler2D u_tex;
uniform sampler2D v_tex;
uniform vec2 u_win;
uniform vec2 u_frame;
in vec2 v_uv;
out vec4 frag;
"#;

// Same shader for clients whose GL only takes a dma-buf EGLImage on the
// external target (NVIDIA returns GL_INVALID_OPERATION for GL_TEXTURE_2D).
// Needs GL_OES_EGL_image_external_essl3; sampling is otherwise identical.
const FS_YUV_HEAD_EXTERNAL: &str = r#"#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
#define SF_SAMPLER samplerExternalOES
uniform samplerExternalOES y_tex;
uniform samplerExternalOES u_tex;
uniform samplerExternalOES v_tex;
uniform vec2 u_win;
uniform vec2 u_frame;
in vec2 v_uv;
out vec4 frag;
"#;

const FS_YUV_MAIN: &str = r#"
void main() {
    vec2 px = v_uv * u_win;
    float scl = min(u_win.x / max(u_frame.x, 1.0), u_win.y / max(u_frame.y, 1.0));
    vec2 pic = u_frame * scl;
    vec2 org = (u_win - pic) * 0.5;
    if (all(greaterThanEqual(px, org)) && all(lessThan(px, org + pic))) {
        vec2 uv = (px - org) / pic;
        uv.y = 1.0 - uv.y;
        vec2 texel_y = 1.0 / u_frame;
        vec2 texel_c = 2.0 / u_frame;   // chroma planes are half resolution
        float fp = 1.0 / max(scl, 1e-3);
        float y = sf(y_tex, uv, texel_y, fp).r * 255.0;
        float u = sf(u_tex, uv, texel_c, fp * 0.5).r * 255.0 - 128.0;
        float v = sf(v_tex, uv, texel_c, fp * 0.5).r * 255.0 - 128.0;
        // BT.709 full-range, identical to the old CPU conversion
        float r = clamp(y + 1.5748 * v, 0.0, 255.0);
        float g = clamp(y - 0.1873 * u - 0.4681 * v, 0.0, 255.0);
        float b = clamp(y + 1.8556 * u, 0.0, 255.0);
        frag = vec4(r, g, b, 255.0) / 255.0;
    } else {
        frag = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
"#;

// overlay: textured quad at a pixel-space rect (metrics --metrics overlay)
const VS_OVERLAY: &str = r#"#version 300 es
uniform vec2 u_origin;  // top-left of the quad, pixels
uniform vec2 u_size;    // quad size, pixels
uniform vec2 u_win;     // window size, pixels
out vec2 v_uv;
void main() {
    // two triangles from gl_VertexID 0..5 (0,0)-(1,0)-(1,1) / (1,1)-(0,1)-(0,0)
    float cx = (gl_VertexID == 1 || gl_VertexID == 2 || gl_VertexID == 3) ? 1.0 : 0.0;
    float cy = (gl_VertexID == 2 || gl_VertexID == 3 || gl_VertexID == 4) ? 1.0 : 0.0;
    vec2 px = u_origin + vec2(cx, cy) * u_size;
    v_uv = vec2(cx, cy);
    gl_Position = vec4(
        px.x / u_win.x * 2.0 - 1.0,
        1.0 - px.y / u_win.y * 2.0,
        0.0, 1.0);
}
"#;

const FS_OVERLAY: &str = r#"#version 300 es
precision highp float;
uniform sampler2D tex;
in vec2 v_uv;
out vec4 frag;
void main() {
    frag = texture(tex, v_uv);  // straight alpha; blend handles the rest
}
"#;

pub struct EglWindow {
    egl: EglLib,
    _wlegl: std::sync::Arc<libloading::Library>,
    dpy: *mut c_void,
    surf: *mut c_void,
    /// keep-alive: the context must outlive the surface usage
    #[allow(dead_code)]
    ctx: *mut c_void,
    gl: Gl,
    prog: u32,
    tex: u32,
    /// buffer (device-pixel) size — what GL renders into
    width: u32,
    height: u32,
    /// guest frame size (what we upload) — differs from the buffer when
    /// fullscreen letterboxes/scales
    frame_w: u32,
    frame_h: u32,
    /// libwayland-egl native window + resize fn (fullscreen/window resizes)
    window: *mut c_void,
    wlegl_resize: unsafe extern "C" fn(*mut c_void, i32, i32) -> i32,
    /// main program uniform locations for the letterbox shader
    loc_win: i32,
    loc_frame: i32,
    loc_filter: i32,
    // --metrics overlay (top-left text quad); all zero/disabled when off
    ov_prog: u32,
    ov_tex: u32,
    ov_loc_origin: i32,
    ov_loc_size: i32,
    ov_loc_win: i32,
    // lossy: YUV->RGB on the GPU sampling the Vulkan decode planes imported as
    // dma-bufs (no host readback, no CPU conversion)
    yuv_prog: u32,
    yuv_tex: [u32; 3],
    yuv_egl_images: [*mut c_void; 3],
    /// true when the planes are bound as GL_TEXTURE_EXTERNAL_OES (NVIDIA)
    yuv_external: bool,
    yuv_loc_win: i32,
    yuv_loc_frame: i32,
    yuv_loc_tex: [i32; 3],
    yuv_loc_filter: i32,
    /// 0=auto 1=nearest 2=linear 3=cubic 4=area (see `--scale-filter`)
    filter_mode: u32,
}

// all GL/EGL calls happen from the single client loop thread
unsafe impl Send for EglWindow {}

impl EglWindow {
    /// `wl_display_ptr`/`wl_surface_ptr`: raw libwayland C pointers from the
    /// wayland-client objects (Presenter accessors). Must be called AFTER the
    /// xdg configure (window mapped).
    pub fn new(
        wl_display_ptr: *mut c_void,
        wl_surface_ptr: *mut c_void,
        width: u32,
        height: u32,
        metrics: bool,
    ) -> Result<Self> {
        // --- libwayland-egl: wl_surface -> native window -------------------
        let wlegl = std::sync::Arc::new(unsafe {
            libloading::Library::new("libwayland-egl.so.1")
                .or_else(|_| libloading::Library::new("libwayland-egl.so"))
                .context("dlopen libwayland-egl")?
        });
        let wl_egl_window_create: unsafe extern "C" fn(
            *mut c_void,
            i32,
            i32,
        ) -> *mut c_void = sym(&wlegl, "wl_egl_window_create")?;
        let wl_egl_window_resize: unsafe extern "C" fn(
            *mut c_void,
            i32,
            i32,
        ) -> i32 = sym(&wlegl, "wl_egl_window_resize")?;
        let window =
            unsafe { wl_egl_window_create(wl_surface_ptr, width as i32, height as i32) };
        if window.is_null() {
            bail!("wl_egl_window_create failed");
        }

        // --- EGL display / context ------------------------------------------
        let lib = std::sync::Arc::new(unsafe {
            libloading::Library::new("libEGL.so.1")
                .or_else(|_| libloading::Library::new("libEGL.so"))
                .context("dlopen libEGL")?
        });
        let egl = EglLib {
            get_display: sym(&lib, "eglGetDisplay")?,
            initialize: sym(&lib, "eglInitialize")?,
            bind_api: sym(&lib, "eglBindAPI")?,
            choose_config: sym(&lib, "eglChooseConfig")?,
            get_config_attrib: sym(&lib, "eglGetConfigAttrib")?,
            create_context: sym(&lib, "eglCreateContext")?,
            create_window_surface: sym(&lib, "eglCreateWindowSurface")?,
            make_current: sym(&lib, "eglMakeCurrent")?,
            swap_buffers: sym(&lib, "eglSwapBuffers")?,
            swap_interval: sym(&lib, "eglSwapInterval")?,
            get_error: sym(&lib, "eglGetError")?,
            get_proc_address: sym(&lib, "eglGetProcAddress")?,
            query_string: sym(&lib, "eglQueryString")?,
            lib,
        };
        let err = || unsafe { (egl.get_error)() };
        let dpy = unsafe { (egl.get_display)(wl_display_ptr) };
        if dpy.is_null() || dpy as isize == -1 {
            bail!("eglGetDisplay(wl_display) failed (err {:#06x})", err());
        }
        let (mut major, mut minor) = (0, 0);
        if unsafe { (egl.initialize)(dpy, &mut major, &mut minor) } == 0 {
            bail!("eglInitialize failed (err {:#06x})", err());
        }
        eprintln!("[eglwin] EGL {major}.{minor} display on compositor");
        if unsafe { (egl.bind_api)(EGL_OPENGL_ES_API) } == 0 {
            bail!("eglBindAPI(ES) failed (err {:#06x})", err());
        }
        let cfg_attribs = [
            EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
            // no alpha in the window buffer (XRGB): the video frame is fully
            // opaque, and an XRGB buffer + the surface's opaque region lets
            // KWin direct-scanout instead of compositing
            EGL_ALPHA_SIZE, 0,
            EGL_NONE,
        ];
        let mut cfg: *mut c_void = std::ptr::null_mut();
        let mut n = 0;
        if unsafe { (egl.choose_config)(dpy, cfg_attribs.as_ptr(), &mut cfg, 1, &mut n) } == 0
            || n < 1
        {
            bail!("no ES3 window-capable EGL config (err {:#06x})", err());
        }
        unsafe {
            let attrib = |a: i32| {
                let mut v: i32 = -1;
                (egl.get_config_attrib)(dpy, cfg, a, &mut v);
                v
            };
            eprintln!(
                "[egl] window config: id={} buffer={} red={} alpha={} native_visual=0x{:x} (alpha=0 => XRGB/opaque, direct-scanout friendly)",
                attrib(EGL_CONFIG_ID),
                attrib(EGL_BUFFER_SIZE),
                attrib(EGL_RED_SIZE),
                attrib(EGL_ALPHA_SIZE),
                attrib(EGL_NATIVE_VISUAL_ID),
            );
        }
        let surf = unsafe { (egl.create_window_surface)(dpy, cfg, window, std::ptr::null()) };
        if surf.is_null() {
            bail!("eglCreateWindowSurface failed (err {:#06x})", err());
        }
        let ctx_attribs = [EGL_CONTEXT_CLIENT_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE];
        let ctx = unsafe {
            (egl.create_context)(dpy, cfg, std::ptr::null_mut(), ctx_attribs.as_ptr())
        };
        if ctx.is_null() {
            bail!("eglCreateContext(ES 3.0) failed (err {:#06x})", err());
        }
        if unsafe { (egl.make_current)(dpy, surf, surf, ctx) } == 0 {
            bail!("eglMakeCurrent failed (err {:#06x})", err());
        }
        // vsync throttle: swaps block when the compositor queue is full —
        // this is our pacing; credits flow at upload rate (see module docs)
        unsafe { (egl.swap_interval)(dpy, 1) };

        // --- GL: program + upload texture -----------------------------------
        macro_rules! glsym {
            ($n:literal, $t:ty) => {{
                let p = egl_func(&egl, $n)?;
                unsafe { std::mem::transmute::<*const c_void, $t>(p) }
            }};
        }
        let gl = Gl {
            create_shader: glsym!("glCreateShader", unsafe extern "system" fn(u32) -> u32),
            shader_source: glsym!(
                "glShaderSource",
                unsafe extern "system" fn(u32, i32, *const *const i8, *const i32)
            ),
            compile_shader: glsym!("glCompileShader", unsafe extern "system" fn(u32)),
            get_shaderiv: glsym!(
                "glGetShaderiv",
                unsafe extern "system" fn(u32, u32, *mut i32)
            ),
            get_shader_info_log: glsym!(
                "glGetShaderInfoLog",
                unsafe extern "system" fn(u32, i32, *mut i32, *mut u8)
            ),
            create_program: glsym!("glCreateProgram", unsafe extern "system" fn() -> u32),
            attach_shader: glsym!("glAttachShader", unsafe extern "system" fn(u32, u32)),
            link_program: glsym!("glLinkProgram", unsafe extern "system" fn(u32)),
            get_programiv: glsym!(
                "glGetProgramiv",
                unsafe extern "system" fn(u32, u32, *mut i32)
            ),
            use_program: glsym!("glUseProgram", unsafe extern "system" fn(u32)),
            gen_textures: glsym!("glGenTextures", unsafe extern "system" fn(i32, *mut u32)),
            delete_textures: glsym!("glDeleteTextures", unsafe extern "system" fn(i32, *const u32)),
            bind_texture: glsym!("glBindTexture", unsafe extern "system" fn(u32, u32)),
            tex_parameteri: glsym!("glTexParameteri", unsafe extern "system" fn(u32, u32, i32)),
            tex_image_2d: glsym!(
                "glTexImage2D",
                unsafe extern "system" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void)
            ),
            viewport: glsym!("glViewport", unsafe extern "system" fn(i32, i32, i32, i32)),
            draw_arrays: glsym!("glDrawArrays", unsafe extern "system" fn(u32, i32, i32)),
            clear: glsym!("glClear", unsafe extern "system" fn(u32)),
            get_error: glsym!("glGetError", unsafe extern "system" fn() -> u32),
            get_string: glsym!("glGetString", unsafe extern "system" fn(u32) -> *const u8),
            enable: glsym!("glEnable", unsafe extern "system" fn(u32)),
            disable: glsym!("glDisable", unsafe extern "system" fn(u32)),
            blend_func: glsym!("glBlendFunc", unsafe extern "system" fn(u32, u32)),
            uniform_2f: glsym!("glUniform2f", unsafe extern "system" fn(i32, f32, f32)),
            get_uniform_location: glsym!(
                "glGetUniformLocation",
                unsafe extern "system" fn(u32, *const i8) -> i32
            ),
            uniform_1i: glsym!("glUniform1i", unsafe extern "system" fn(i32, i32)),
            active_texture: glsym!("glActiveTexture", unsafe extern "system" fn(u32)),
        };

        let mut me = EglWindow {
            egl,
            _wlegl: wlegl,
            dpy,
            surf,
            ctx,
            gl,
            prog: 0,
            tex: 0,
            width,
            height,
            frame_w: width,
            frame_h: height,
            window,
            wlegl_resize: wl_egl_window_resize,
            loc_win: -1,
            loc_frame: -1,
            loc_filter: -1,
            ov_prog: 0,
            ov_tex: 0,
            ov_loc_origin: -1,
            ov_loc_size: -1,
            ov_loc_win: -1,
            yuv_prog: 0,
            yuv_tex: [0; 3],
            yuv_egl_images: [std::ptr::null_mut(); 3],
            yuv_external: false,
            yuv_loc_win: -1,
            yuv_loc_frame: -1,
            yuv_loc_tex: [-1; 3],
            yuv_loc_filter: -1,
            filter_mode: 0,
        };
        me.build_program()?;
        unsafe {
            let mut tex = 0u32;
            (me.gl.gen_textures)(1, &mut tex);
            (me.gl.bind_texture)(GL_TEXTURE_2D, tex);
            (me.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as i32);
            (me.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as i32);
            me.tex = tex;
            (me.gl.clear)(GL_COLOR_BUFFER_BIT); // black first frame
            (me.egl.swap_buffers)(me.dpy, me.surf);
        }
        if metrics {
            me.build_overlay()?;
        }
        unsafe {
            let s = |p: *const u8| {
                if p.is_null() {
                    String::new()
                } else {
                    std::ffi::CStr::from_ptr(p as *const i8).to_string_lossy().into_owned()
                }
            };
            eprintln!(
                "[eglwin] window GL: {} | {}",
                s((me.gl.get_string)(GL_VENDOR)),
                s((me.gl.get_string)(GL_RENDERER))
            );
        }
        eprintln!("[eglwin] window surface ready: {width}x{height}, ES 3.0");
        Ok(me)
    }

    /// --metrics: overlay program + font texture (content uploaded per refresh)
    fn build_overlay(&mut self) -> Result<()> {
        unsafe {
            let compile = |gl: &Gl, src: &str, ty: u32| -> Result<u32> {
                let sh = (gl.create_shader)(ty);
                let c = std::ffi::CString::new(src.as_bytes())?;
                (gl.shader_source)(
                    sh,
                    1,
                    (&[c.as_ptr()] as *const [*const i8]) as *const *const i8,
                    std::ptr::null(),
                );
                (gl.compile_shader)(sh);
                let mut ok = 0;
                (gl.get_shaderiv)(sh, GL_COMPILE_STATUS, &mut ok);
                if ok == 0 {
                    let mut log = [0u8; 512];
                    let mut n = 0;
                    (gl.get_shader_info_log)(sh, 512, &mut n, log.as_mut_ptr());
                    bail!(
                        "overlay shader compile failed: {}",
                        String::from_utf8_lossy(&log[..n as usize])
                    );
                }
                Ok(sh)
            };
            let vs = compile(&self.gl, VS_OVERLAY, GL_VERTEX_SHADER)?;
            let fs = compile(&self.gl, FS_OVERLAY, GL_FRAGMENT_SHADER)?;
            let prog = (self.gl.create_program)();
            (self.gl.attach_shader)(prog, vs);
            (self.gl.attach_shader)(prog, fs);
            (self.gl.link_program)(prog);
            let mut ok = 0;
            (self.gl.get_programiv)(prog, GL_LINK_STATUS, &mut ok);
            if ok == 0 {
                bail!("overlay program link failed");
            }
            self.ov_prog = prog;
            let name = |s: &str| -> i32 {
                let c = std::ffi::CString::new(s).unwrap();
                (self.gl.get_uniform_location)(prog, c.as_ptr())
            };
            self.ov_loc_origin = name("u_origin");
            self.ov_loc_size = name("u_size");
            self.ov_loc_win = name("u_win");
            let mut tex = 0u32;
            (self.gl.gen_textures)(1, &mut tex);
            (self.gl.bind_texture)(GL_TEXTURE_2D, tex);
            (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as i32);
            (self.gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as i32);
            (self.gl.bind_texture)(GL_TEXTURE_2D, 0);
            self.ov_tex = tex;
        }
        eprintln!("[eglwin] metrics overlay enabled");
        Ok(())
    }

    fn build_program(&mut self) -> Result<()> {
        unsafe {
            let compile = |gl: &Gl, src: &str, ty: u32| -> Result<u32> {
                let sh = (gl.create_shader)(ty);
                let c = std::ffi::CString::new(src.as_bytes())?;
                (gl.shader_source)(
                    sh,
                    1,
                    (&[c.as_ptr()] as *const [*const i8]) as *const *const i8,
                    std::ptr::null(),
                );
                (gl.compile_shader)(sh);
                let mut ok = 0;
                (gl.get_shaderiv)(sh, GL_COMPILE_STATUS, &mut ok);
                if ok == 0 {
                    let mut log = [0u8; 512];
                    let mut n = 0;
                    (gl.get_shader_info_log)(sh, 512, &mut n, log.as_mut_ptr());
                    bail!(
                        "shader compile failed: {}",
                        String::from_utf8_lossy(&log[..n as usize])
                    );
                }
                Ok(sh)
            };
            let vs = compile(&self.gl, VS, GL_VERTEX_SHADER)?;
            let fs_src = format!("{FS_HEAD}{FILTER_GLSL}{FS_MAIN}");
            let fs = compile(&self.gl, &fs_src, GL_FRAGMENT_SHADER)?;
            let prog = (self.gl.create_program)();
            (self.gl.attach_shader)(prog, vs);
            (self.gl.attach_shader)(prog, fs);
            (self.gl.link_program)(prog);
            let mut ok = 0;
            (self.gl.get_programiv)(prog, GL_LINK_STATUS, &mut ok);
            if ok == 0 {
                bail!("program link failed");
            }
            self.prog = prog;
            let name = |s: &str| -> i32 {
                let c = std::ffi::CString::new(s).unwrap();
                (self.gl.get_uniform_location)(prog, c.as_ptr())
            };
            self.loc_win = name("u_win");
            self.loc_frame = name("u_frame");
            self.loc_filter = name("u_filter");
        }
        Ok(())
    }

    /// Resize the libwayland-egl native window (fullscreen configure,
    /// fractional-scale change). The next `present` renders into the new
    /// buffer size. Also retunes the texture filter: NEAREST stays
    /// pixel-perfect for integer upscales, LINEAR avoids uneven-pixel
    /// artifacts at fractional scales.
    pub fn resize(&mut self, w: u32, h: u32) {
        if (w, h) == (self.width, self.height) {
            return;
        }
        unsafe { (self.wlegl_resize)(self.window, w as i32, h as i32) };
        self.width = w;
        self.height = h;
        let scl = self.scale();
        eprintln!(
            "[eglwin] buffer {}x{} (frame {}x{}, scale {scl:.3}, filter {})",
            w,
            h,
            self.frame_w,
            self.frame_h,
            FILTER_NAMES[self.effective_filter() as usize]
        );
    }

    /// Letterbox scale: device buffer px per frame px (scalar min of the axes).
    fn scale(&self) -> f32 {
        (self.width as f32 / self.frame_w.max(1) as f32)
            .min(self.height as f32 / self.frame_h.max(1) as f32)
    }

    /// Effective sampler: 0 nearest, 1 linear, 2 cubic, 3 area.
    fn effective_filter(&self) -> u32 {
        match self.filter_mode {
            1 => 0,
            2 => 1,
            3 => 2,
            4 => 3,
            _ => {
                let scl = self.scale();
                if (scl - scl.round()).abs() < 1e-3 {
                    0 // integer scale: pixel-perfect
                } else if scl < 1.0 {
                    3 // downscale: area/box (avoids aliasing)
                } else {
                    1 // upscale: bilinear
                }
            }
        }
    }

    /// `--scale-filter`: 0 auto, 1 nearest, 2 linear, 3 cubic, 4 area.
    pub fn set_scale_filter(&mut self, mode: u32) {
        self.filter_mode = mode;
    }

    /// Set the currently bound `target` texture's filter for `mode` (nearest
    /// for the manual filters; linear only for mode 1).
    unsafe fn set_filter(&self, target: u32, mode: u32) {
        let f = (if mode == 1 { GL_LINEAR } else { GL_NEAREST }) as i32;
        (self.gl.tex_parameteri)(target, GL_TEXTURE_MIN_FILTER, f);
        (self.gl.tex_parameteri)(target, GL_TEXTURE_MAG_FILTER, f);
    }

    /// The GL target the lossy planes were successfully imported on.
    fn yuv_target(&self) -> u32 {
        if self.yuv_external {
            GL_TEXTURE_EXTERNAL_OES
        } else {
            GL_TEXTURE_2D
        }
    }

    /// Guest frame dimensions changed (mid-session resolution change): the
    /// letterbox shader picks it up via the u_frame uniform; the filter
    /// choice is recomputed at the next `resize`. The caller must also have
    /// resized the buffer if the target window size differs.
    #[allow(dead_code)] // wired up when mid-session resolution change lands
    pub fn set_frame_size(&mut self, w: u32, h: u32) {
        self.frame_w = w;
        self.frame_h = h;
    }

    /// Upload slot bytes (the NIC-written staging buffer), draw to the
    /// surface back buffer, and swap. `overlay` = a freshly rasterized
    /// metrics text (`Some`) uploads it into the overlay texture; `None`
    /// keeps the previous one (drawn every frame so the overlay never
    /// flickers between refreshes). Returns once the GL calls are queued;
    /// the caller may credit the slot immediately (glTexImage2D consumed
    /// the pointer per spec). Returns (t4, t5): unix-us stamps taken right
    /// after the upload and after the swap (metrics sampling).
    pub fn present(
        &mut self,
        bytes: &[u8],
        overlay: Option<(&[u8], u32, u32)>,
    ) -> Result<(u64, u64)> {
        unsafe {
            (self.gl.bind_texture)(GL_TEXTURE_2D, self.tex);
            (self.gl.tex_image_2d)(
                GL_TEXTURE_2D,
                0,
                GL_RGBA8 as i32,
                self.frame_w as i32,
                self.frame_h as i32,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                bytes.as_ptr() as *const c_void,
            );
            // t4: the upload call itself. Stamp BEFORE glGetError: the error
            // query forces the driver to process queued commands, so timing
            // after it would bill the upload for the driver's flush.
            let t4 = rdma_display_proto::unix_now_us();
            let gl_err = (self.gl.get_error)();
            if gl_err != 0 {
                bail!("upload glError 0x{gl_err:04x}");
            }
            (self.gl.bind_texture)(GL_TEXTURE_2D, 0);
            self.draw_with(self.tex, overlay);
            if (self.egl.swap_buffers)(self.dpy, self.surf) == 0 {
                bail!("eglSwapBuffers failed (err {:#06x})", (self.egl.get_error)());
            }
            Ok((t4, rdma_display_proto::unix_now_us()))
        }
    }

    /// Draw an already-populated texture and swap, skipping the CPU upload.
    /// Used for the GPU lossless decode target: the decode pass wrote `tex`
    /// in this context, so the client does not touch a full host frame.
    /// `t4` is stamped at entry (the decode the caller already queued).
    pub fn present_texture(
        &mut self,
        tex: u32,
        overlay: Option<(&[u8], u32, u32)>,
    ) -> Result<(u64, u64)> {
        unsafe {
            let t4 = rdma_display_proto::unix_now_us();
            self.draw_with(tex, overlay);
            if (self.egl.swap_buffers)(self.dpy, self.surf) == 0 {
                bail!("eglSwapBuffers failed (err {:#06x})", (self.egl.get_error)());
            }
            Ok((t4, rdma_display_proto::unix_now_us()))
        }
    }

    /// Shared draw: full-screen letterboxed triangle sampling `tex`, then the
    /// metrics overlay quad if enabled. Does not swap.
    unsafe fn draw_with(&self, tex: u32, overlay: Option<(&[u8], u32, u32)>) {
        let mode = self.effective_filter();
        (self.gl.use_program)(self.prog);
        (self.gl.uniform_1i)(self.loc_filter, mode as i32);
        (self.gl.uniform_2f)(self.loc_win, self.width as f32, self.height as f32);
        (self.gl.uniform_2f)(self.loc_frame, self.frame_w as f32, self.frame_h as f32);
        (self.gl.bind_texture)(GL_TEXTURE_2D, tex);
        self.set_filter(GL_TEXTURE_2D, mode);
        (self.gl.viewport)(0, 0, self.width as i32, self.height as i32);
        (self.gl.draw_arrays)(GL_TRIANGLES, 0, 3);
        (self.gl.bind_texture)(GL_TEXTURE_2D, 0);
        self.draw_overlay(overlay);
    }

    /// Upload+draw the metrics overlay quad (no-op without --metrics).
    unsafe fn draw_overlay(&self, overlay: Option<(&[u8], u32, u32)>) {
        if self.ov_prog != 0 {
            if let Some((rgba, w, h)) = overlay {
                (self.gl.bind_texture)(GL_TEXTURE_2D, self.ov_tex);
                (self.gl.tex_image_2d)(
                    GL_TEXTURE_2D,
                    0,
                    GL_RGBA8 as i32,
                    w as i32,
                    h as i32,
                    0,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    rgba.as_ptr() as *const c_void,
                );
                (self.gl.bind_texture)(GL_TEXTURE_2D, 0);
            }
            (self.gl.use_program)(self.ov_prog);
            (self.gl.uniform_2f)(self.ov_loc_origin, 8.0, 8.0);
            (self.gl.uniform_2f)(
                self.ov_loc_size,
                crate::metrics::OVERLAY_W as f32,
                crate::metrics::OVERLAY_H as f32,
            );
            (self.gl.uniform_2f)(self.ov_loc_win, self.width as f32, self.height as f32);
            (self.gl.enable)(GL_BLEND);
            (self.gl.blend_func)(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            (self.gl.bind_texture)(GL_TEXTURE_2D, self.ov_tex);
            (self.gl.draw_arrays)(GL_TRIANGLES, 0, 6);
            (self.gl.disable)(GL_BLEND);
            (self.gl.bind_texture)(GL_TEXTURE_2D, 0);
        }
    }

    fn build_yuv_program(&mut self) -> Result<()> {
        unsafe {
            let compile = |gl: &Gl, src: &str, ty: u32| -> Result<u32> {
                let sh = (gl.create_shader)(ty);
                let c = std::ffi::CString::new(src.as_bytes())?;
                (gl.shader_source)(
                    sh,
                    1,
                    (&[c.as_ptr()] as *const [*const i8]) as *const *const i8,
                    std::ptr::null(),
                );
                (gl.compile_shader)(sh);
                let mut ok = 0;
                (gl.get_shaderiv)(sh, GL_COMPILE_STATUS, &mut ok);
                if ok == 0 {
                    let mut log = [0u8; 512];
                    let mut n = 0;
                    (gl.get_shader_info_log)(sh, 512, &mut n, log.as_mut_ptr());
                    bail!(
                        "yuv shader compile failed: {}",
                        String::from_utf8_lossy(&log[..n as usize])
                    );
                }
                Ok(sh)
            };
            let vs = compile(&self.gl, VS, GL_VERTEX_SHADER)?;
            let head = if self.yuv_external { FS_YUV_HEAD_EXTERNAL } else { FS_YUV_HEAD };
            let fs_src = format!("{head}{FILTER_GLSL}{FS_YUV_MAIN}");
            let fs = compile(&self.gl, &fs_src, GL_FRAGMENT_SHADER)?;
            let prog = (self.gl.create_program)();
            (self.gl.attach_shader)(prog, vs);
            (self.gl.attach_shader)(prog, fs);
            (self.gl.link_program)(prog);
            let mut ok = 0;
            (self.gl.get_programiv)(prog, GL_LINK_STATUS, &mut ok);
            if ok == 0 {
                bail!("yuv program link failed");
            }
            self.yuv_prog = prog;
            let name = |s: &str| -> i32 {
                let c = std::ffi::CString::new(s).unwrap();
                (self.gl.get_uniform_location)(prog, c.as_ptr())
            };
            self.yuv_loc_win = name("u_win");
            self.yuv_loc_frame = name("u_frame");
            self.yuv_loc_tex = [name("y_tex"), name("u_tex"), name("v_tex")];
            self.yuv_loc_filter = name("u_filter");
        }
        Ok(())
    }

    /// Log the DRM formats/modifiers this EGL display can import, so a
    /// rejected import is diagnosable (NVIDIA rejects some formats/tilings).
    unsafe fn log_dmabuf_support(&self) {
        let q_formats: unsafe extern "system" fn(*mut c_void, i32, *mut i32, *mut i32) -> u32 =
            match egl_func(&self.egl, "eglQueryDmaBufFormatsEXT") {
                Ok(p) => std::mem::transmute(p),
                Err(_) => return,
            };
        let mut n = 0i32;
        if q_formats(self.dpy, 0, std::ptr::null_mut(), &mut n) == 0 || n <= 0 {
            return;
        }
        let mut fmts = vec![0i32; n as usize];
        let mut got = n;
        if q_formats(self.dpy, n, fmts.as_mut_ptr(), &mut got) == 0 || got <= 0 {
            return;
        }
        let names: Vec<String> = fmts[..got as usize].iter().map(|f| fourcc_name(*f)).collect();
        eprintln!("[eglwin] EGL dma-buf formats ({}): {}", got, names.join(" "));

        let q_mods: unsafe extern "system" fn(
            *mut c_void,
            i32,
            i32,
            *mut u64,
            *mut i32,
            *mut i32,
        ) -> u32 = match egl_func(&self.egl, "eglQueryDmaBufModifiersEXT") {
            Ok(p) => std::mem::transmute(p),
            Err(_) => return,
        };
        for f in [
            DRM_FORMAT_R8,
            DRM_FORMAT_XRGB8888,
            DRM_FORMAT_ARGB8888,
            DRM_FORMAT_ABGR8888,
            DRM_FORMAT_XBGR8888,
        ] {
            let mut m = 0i32;
            if q_mods(self.dpy, f, 0, std::ptr::null_mut(), std::ptr::null_mut(), &mut m) == 0
                || m <= 0
            {
                eprintln!("[eglwin] EGL {}: not importable", fourcc_name(f).trim());
                continue;
            }
            let mut mods = vec![0u64; m as usize];
            // EGLBoolean is 32-bit: passing a byte-per-element buffer made the
            // driver write 4x the allocation and corrupt the heap.
            let mut ext = vec![0i32; m as usize];
            let mut gm = m;
            if q_mods(self.dpy, f, m, mods.as_mut_ptr(), ext.as_mut_ptr(), &mut gm) == 0 {
                continue;
            }
            let list: Vec<String> = mods[..gm as usize]
                .iter()
                .enumerate()
                .map(|(k, mm)| format!("0x{mm:x}{}", if ext[k] != 0 { "(ext)" } else { "" }))
                .collect();
            eprintln!(
                "[eglwin] EGL {} modifiers ({}): {}",
                fourcc_name(f).trim(),
                gm,
                list.join(" ")
            );
        }
    }

    /// Import the three PyroWave decode planes (linear R8 dma-bufs exported by
    /// the client's Vulkan device) as GL textures. Called once per session:
    /// no per-frame download, no CPU colour conversion.
    ///
    /// The EGLImage can, per spec, be used from `GL_TEXTURE_2D`, but drivers
    /// differ: Mesa accepts `TEXTURE_2D`, NVIDIA's GL only accepts
    /// `GL_TEXTURE_EXTERNAL_OES` (`TEXTURE_2D` gives GL_INVALID_OPERATION and a
    /// sampler that reads zeros). Try both and keep whichever binds.
    pub fn import_yuv(&mut self, planes: [(i32, u32, u32, u32); 3]) -> Result<()> {
        // extension entry points, fetched lazily so a missing
        // EGL_EXT_image_dma_buf_import never breaks raw/lossless
        let create_image: unsafe extern "system" fn(
            *mut c_void,
            *mut c_void,
            i32,
            *mut c_void,
            *const i32,
        ) -> *mut c_void =
            unsafe { std::mem::transmute(egl_func(&self.egl, "eglCreateImageKHR")?) };
        let destroy_image: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32 =
            unsafe { std::mem::transmute(egl_func(&self.egl, "eglDestroyImageKHR")?) };
        let does: unsafe extern "system" fn(u32, *mut c_void) =
            unsafe { std::mem::transmute(egl_func(&self.egl, "glEGLImageTargetTexture2DOES")?) };
        let storage: Option<unsafe extern "system" fn(u32, *mut c_void, *const i32)> =
            egl_func(&self.egl, "glEGLImageTargetTexStorageEXT")
                .ok()
                .map(|p| unsafe { std::mem::transmute(p) });

        // Diagnostic: what the driver actually offers for dma-buf -> GL.
        if lossy_debug() {
            unsafe {
                let e = (self.egl.query_string)(self.dpy, EGL_EXTENSIONS);
                if !e.is_null() {
                    eprintln!(
                        "[eglwin] EGL extensions: {}",
                        std::ffi::CStr::from_ptr(e).to_string_lossy()
                    );
                }
                let g = (self.gl.get_string)(GL_EXTENSIONS);
                if !g.is_null() {
                    let s = std::ffi::CStr::from_ptr(g as *const i8).to_string_lossy();
                    let of: Vec<&str> = s
                        .split_whitespace()
                        .filter(|x| x.contains("EGL_image") || x.contains("dma_buf"))
                        .collect();
                    eprintln!("[eglwin] GL EGL-image extensions: {}", of.join(" "));
                }
                self.log_dmabuf_support();
            }
        }

        // Drivers differ on what they accept for a dma-buf EGLImage: the
        // legacy GL_OES_EGL_image bind vs GL_EXT_EGL_image_storage, the legacy
        // vs EGL_EXT_image_dma_buf_import_modifiers attribute form, and
        // GL_TEXTURE_2D vs the external target.
        let mut attempts: Vec<(u32, bool, bool, BindFn, &str)> = vec![
            (GL_TEXTURE_2D, false, false, BindFn::Does(does), "does/legacy"),
            (GL_TEXTURE_2D, false, true, BindFn::Does(does), "does/modifier"),
        ];
        if let Some(s) = storage {
            attempts.push((GL_TEXTURE_2D, false, false, BindFn::Storage(s), "storage/legacy"));
            attempts.push((GL_TEXTURE_2D, false, true, BindFn::Storage(s), "storage/modifier"));
        }
        attempts.push((
            GL_TEXTURE_EXTERNAL_OES,
            true,
            false,
            BindFn::Does(does),
            "ext/legacy",
        ));
        attempts.push((
            GL_TEXTURE_EXTERNAL_OES,
            true,
            true,
            BindFn::Does(does),
            "ext/modifier",
        ));

        let mut last_err: Option<anyhow::Error> = None;
        for (target, external, use_modifier, bind, what) in attempts {
            match unsafe {
                self.try_import_target(
                    &planes,
                    target,
                    use_modifier,
                    &create_image,
                    &destroy_image,
                    bind,
                )
            } {
                Ok(()) => {
                    self.yuv_external = external;
                    match self.build_yuv_program() {
                        Ok(()) => {
                            eprintln!(
                                "[eglwin] lossy: YUV planes imported as dma-bufs ({} {what}, GPU YUV->RGB)",
                                if external {
                                    "GL_TEXTURE_EXTERNAL_OES"
                                } else {
                                    "GL_TEXTURE_2D"
                                },
                            );
                            return Ok(());
                        }
                        Err(e) => {
                            if lossy_debug() {
                                eprintln!("[eglwin] lossy: {what} shader failed: {e:#}");
                            }
                            unsafe { self.release_yuv_images(&destroy_image) };
                            last_err = Some(e);
                        }
                    }
                }
                Err(e) => {
                    unsafe { self.release_yuv_images(&destroy_image) };
                    if lossy_debug() {
                        eprintln!("[eglwin] lossy: dma-buf import ({what}) failed: {e:#}");
                    }
                    last_err = Some(e);
                }
            }
        }
        Err(last_err
            .unwrap_or_else(|| anyhow::anyhow!("lossy: no GL dma-buf import target worked")))
    }

    /// Create + bind the three plane EGLImages using `target`. On error the
    /// caller releases the images; nothing is left half-bound.
    unsafe fn try_import_target(
        &mut self,
        planes: &[(i32, u32, u32, u32); 3],
        target: u32,
        use_modifier: bool,
        create_image: &unsafe extern "system" fn(
            *mut c_void,
            *mut c_void,
            i32,
            *mut c_void,
            *const i32,
        ) -> *mut c_void,
        destroy_image: &unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
        bind: BindFn,
    ) -> Result<()> {
        self.release_yuv_images(destroy_image);
        self.release_yuv_textures();
        // clear any stale GL error so the binds below are attributed cleanly
        while (self.gl.get_error)() != 0 {}
        (self.gl.gen_textures)(3, self.yuv_tex.as_mut_ptr());
        for (i, (fd, w, h, pitch)) in planes.iter().enumerate() {
            if lossy_debug() {
                eprintln!(
                    "[eglwin] lossy plane {i}: fd {fd} {w}x{h} pitch {pitch} target {target:#06x} {}",
                    if use_modifier { "modifier" } else { "legacy" }
                );
            }
            let mut attribs = vec![
                EGL_WIDTH,
                *w as i32,
                EGL_HEIGHT,
                *h as i32,
                EGL_LINUX_DRM_FOURCC_EXT,
                DRM_FORMAT_R8,
                EGL_DMA_BUF_PLANE0_FD_EXT,
                *fd,
                EGL_DMA_BUF_PLANE0_OFFSET_EXT,
                0,
                EGL_DMA_BUF_PLANE0_PITCH_EXT,
                *pitch as i32,
            ];
            if use_modifier {
                // DRM_FORMAT_MOD_LINEAR == 0 (the planes are pitch-linear)
                attribs.extend_from_slice(&[
                    EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
                    0,
                    EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
                    0,
                ]);
            }
            attribs.push(EGL_NONE);
            let img = create_image(
                self.dpy,
                std::ptr::null_mut(),
                EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(),
                attribs.as_ptr(),
            );
            if img.is_null() || img as isize == -1 {
                bail!(
                    "eglCreateImageKHR(plane {i}) failed (err {:#06x})",
                    (self.egl.get_error)()
                );
            }
            self.yuv_egl_images[i] = img;
            (self.gl.bind_texture)(target, self.yuv_tex[i]);
            match bind {
                BindFn::Does(f) => f(target, img),
                BindFn::Storage(f) => f(target, img, std::ptr::null()),
            }
            let gl_err = (self.gl.get_error)();
            if gl_err != 0 {
                // A silent failure here is exactly how an unsupported import
                // shows up as an all-zero (green) frame, so make it loud.
                bail!(
                    "glEGLImageTargetTexture2DOES(plane {i}) failed: glError 0x{gl_err:04x} \
                     (fd {fd}, {w}x{h}, pitch {pitch})"
                );
            }
            (self.gl.tex_parameteri)(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR as i32);
            (self.gl.tex_parameteri)(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR as i32);
            (self.gl.bind_texture)(target, 0);
        }
        Ok(())
    }

    unsafe fn release_yuv_images(
        &mut self,
        destroy_image: &unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    ) {
        for i in 0..3 {
            if !self.yuv_egl_images[i].is_null() {
                destroy_image(self.dpy, self.yuv_egl_images[i]);
                self.yuv_egl_images[i] = std::ptr::null_mut();
            }
        }
    }

    unsafe fn release_yuv_textures(&mut self) {
        if self.yuv_tex.iter().any(|t| *t != 0) {
            (self.gl.delete_textures)(3, self.yuv_tex.as_ptr());
            self.yuv_tex = [0; 3];
        }
    }

    /// Draw the imported YUV planes (letterboxed) + overlay and swap. The
    /// Vulkan decode already signalled completion, so the planes are coherent.
    pub fn present_yuv(&mut self, overlay: Option<(&[u8], u32, u32)>) -> Result<(u64, u64)> {
        unsafe {
            let t4 = rdma_display_proto::unix_now_us();
            let mode = self.effective_filter();
            let target = self.yuv_target();
            (self.gl.use_program)(self.yuv_prog);
            (self.gl.uniform_1i)(self.yuv_loc_filter, mode as i32);
            (self.gl.uniform_2f)(self.yuv_loc_win, self.width as f32, self.height as f32);
            (self.gl.uniform_2f)(self.yuv_loc_frame, self.frame_w as f32, self.frame_h as f32);
            for i in 0..3 {
                (self.gl.active_texture)(GL_TEXTURE0 + i as u32);
                (self.gl.bind_texture)(target, self.yuv_tex[i]);
                self.set_filter(target, mode);
                (self.gl.uniform_1i)(self.yuv_loc_tex[i], i as i32);
            }
            (self.gl.viewport)(0, 0, self.width as i32, self.height as i32);
            (self.gl.draw_arrays)(GL_TRIANGLES, 0, 3);
            (self.gl.active_texture)(GL_TEXTURE0);
            self.draw_overlay(overlay);
            if (self.egl.swap_buffers)(self.dpy, self.surf) == 0 {
                bail!("eglSwapBuffers failed (err {:#06x})", (self.egl.get_error)());
            }
            Ok((t4, rdma_display_proto::unix_now_us()))
        }
    }
}
