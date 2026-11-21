use rdma_display_codec::{Decoder, Geometry};
use rdma_display_vkcodec::Encoder;

#[test]
#[ignore = "requires a Vulkan device; run with --ignored --nocapture"]
fn vulkan_lossless_roundtrips_full_and_delta() {
    let geom = Geometry::new(256, 256, 256 * 4);
    let n = geom.frame_bytes();
    let mut f0 = vec![0u8; n];
    for y in 0..256u32 {
        for x in 0..256u32 {
            let o = (y * 1024 + x * 4) as usize;
            f0[o] = (x as u8) ^ 0x5a;
            f0[o + 1] = y as u8;
            f0[o + 2] = (x as u8).wrapping_add(y as u8);
            f0[o + 3] = 0xff;
        }
    }

    let mut enc = Encoder::new(geom).expect("vulkan encoder");
    let mut dec = Decoder::new(geom).expect("cpu decoder");

    let n0 = enc.encode_host(&f0).expect("encode full");
    let d0 = dec.decode(&enc.coded()[..n0]).expect("decode full");
    assert_eq!(d0, &f0[..], "full frame mismatch");

    let mut f1 = f0.clone();
    for y in 64..128u32 {
        for x in 64..128u32 {
            let o = (y * 1024 + x * 4) as usize;
            f1[o..o + 4].copy_from_slice(&[0x11, 0x22, 0x33, 0xff]);
        }
    }
    let n1 = enc.encode_host(&f1).expect("encode delta");
    let d1 = dec.decode(&enc.coded()[..n1]).expect("decode delta");
    assert_eq!(d1, &f1[..], "delta frame mismatch");
}

/// The server binds the transport's host-visible staging `VkBuffer` as the
/// coded output; the bytes must be identical in effect (the client decodes
/// them) and land in that buffer, with no CPU copy.
#[test]
#[ignore = "requires a Vulkan device; run with --ignored --nocapture"]
fn vulkan_lossless_direct_output_roundtrips() {
    use std::sync::Arc;
    use rdma_display_vkcore::VkCore;

    let geom = Geometry::new(256, 256, 256 * 4);
    let n = geom.frame_bytes();
    let mut f0 = vec![0u8; n];
    for y in 0..256u32 {
        for x in 0..256u32 {
            let o = (y * 1024 + x * 4) as usize;
            f0[o] = (x as u8) ^ 0x5a;
            f0[o + 1] = y as u8;
            f0[o + 2] = (x as u8).wrapping_add(y as u8);
            f0[o + 3] = 0xff;
        }
    }
    let core = Arc::new(VkCore::new().expect("vulkan core"));
    let out = core
        .host_buffer(geom.max_coded_len() as u64 + 16)
        .expect("host output buffer");
    let mut enc = Encoder::with_core(core.clone(), geom).expect("vulkan encoder");
    enc.set_output(out.buffer, geom.max_coded_len() + 16)
        .expect("bind external output");
    let mut dec = Decoder::new(geom).expect("cpu decoder");

    let n0 = enc.encode_host(&f0).expect("encode full");
    let d0 = dec.decode(&out.as_slice()[..n0]).expect("decode full");
    assert_eq!(d0, &f0[..], "full frame mismatch");

    let mut f1 = f0.clone();
    for y in 64..128u32 {
        for x in 64..128u32 {
            let o = (y * 1024 + x * 4) as usize;
            f1[o..o + 4].copy_from_slice(&[0x11, 0x22, 0x33, 0xff]);
        }
    }
    let n1 = enc.encode_host(&f1).expect("encode delta");
    let d1 = dec.decode(&out.as_slice()[..n1]).expect("decode delta");
    assert_eq!(d1, &f1[..], "delta frame mismatch");
}

/// Exercise the production lossless zero-copy building block: the coded output
/// is a device-local exportable VkBuffer (the one the server registers as an
/// RDMA dma-buf MR). Copy it back to host memory only for the assertion.
#[test]
#[ignore = "requires a Vulkan device; run with --ignored --nocapture"]
fn vulkan_lossless_gpu_output_roundtrips() {
    use std::sync::Arc;
    use rdma_display_vkcore::{ash::vk, VkCore};

    let geom = Geometry::new(256, 256, 256 * 4);
    let n = geom.frame_bytes();
    let mut f0 = vec![0u8; n];
    for y in 0..256u32 {
        for x in 0..256u32 {
            let o = (y * 1024 + x * 4) as usize;
            f0[o] = (x as u8) ^ 0x33;
            f0[o + 1] = y as u8;
            f0[o + 2] = (x as u8).wrapping_add(y as u8);
            f0[o + 3] = 0xff;
        }
    }
    let core = Arc::new(VkCore::new().expect("vulkan core"));
    let out = core
        .create_exportable_buffer(
            geom.max_coded_len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .expect("device-local coded buffer");
    let host = core
        .host_buffer(geom.max_coded_len() as u64)
        .expect("host readback");
    let mut enc = Encoder::with_core(core.clone(), geom).expect("vulkan encoder");
    enc.set_output(out.buffer, geom.max_coded_len())
        .expect("bind GPU output");
    let mut dec = Decoder::new(geom).expect("cpu decoder");

    let n0 = enc.encode_host(&f0).expect("encode full");
    core.one_shot(|cb| unsafe {
        core.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier {
                src_access_mask: vk::AccessFlags::MEMORY_WRITE,
                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                ..Default::default()
            }],
            &[],
            &[],
        );
        core.device.cmd_copy_buffer(
            cb,
            out.buffer,
            host.buffer,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: n0 as u64,
            }],
        );
    })
    .expect("copy coded output back");
    let d0 = dec
        .decode(&host.as_slice()[..n0])
        .expect("decode full");
    assert_eq!(d0, &f0[..], "GPU-coded full frame mismatch");

    unsafe {
        core.device.destroy_buffer(out.buffer, None);
        core.device.free_memory(out.memory, None);
    }
}
