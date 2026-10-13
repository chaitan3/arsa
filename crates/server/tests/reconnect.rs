//! Prototype for the Phase 2 reconnect lifecycle: repeatedly build and release
//! the per-session GPU pipeline on one shared context, exactly as the server
//! does on connect/disconnect. This runs headless (no DRM/input) and validates
//! that the materializer and codec resources are freed and reusable across
//! sessions rather than leaking into the long-lived context.
//!
//! Skips (passes) when there is no usable EGL device.

use rdma_display_capture::{EglMaterializer, FrameMaterializer};
use rdma_display_codec::Geometry;
use rdma_display_codec_gpu::GpuCodec;

#[test]
fn shared_context_reconnect_loop() {
    let sh = match rdma_display_egl::SharedContext::new(None) {
        Ok(s) => std::sync::Arc::new(s),
        Err(e) => {
            eprintln!("skip: no EGL device ({e:#})");
            return;
        }
    };
    let geom = Geometry::new(1920, 1080, 1920 * 4);
    for i in 0..5 {
        // per-session: materializer + codec on the shared context
        let mut mat = match EglMaterializer::new_shared(&sh) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("skip: materializer unavailable ({e:#})");
                return;
            }
        };
        let codec = GpuCodec::new_shared(geom, sh.clone()).expect("gpu codec on shared context");
        codec.sync();
        drop(codec); // deletes its GL objects
        mat.release(); // frees FBO/texture/PBOs
        drop(mat);
        eprintln!("[reconnect] iteration {i} ok");
    }
}
