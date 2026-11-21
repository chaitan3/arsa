use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::Arc;

use anyhow::Result;
use rdma_display_codec::{Decoder, Geometry};
use rdma_display_vkcodec::Encoder;
use rdma_display_vkcore::{ash::vk, HostBuffer, VkCore};

fn frame(seed: u32) -> Vec<u8> {
    let mut f = vec![0u8; 256 * 256 * 4];
    for y in 0..256u32 {
        for x in 0..256u32 {
            let o = (y * 1024 + x * 4) as usize;
            f[o] = (x as u8) ^ (seed as u8);
            f[o + 1] = y as u8;
            f[o + 2] = (x as u8).wrapping_add(y as u8) ^ (seed as u8);
            f[o + 3] = 0xff;
        }
    }
    f
}

fn upload(core: &VkCore, image: vk::Image, host: &HostBuffer, w: u32, h: u32) -> Result<()> {
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    core.one_shot(|cb| unsafe {
        let to_dst = vk::ImageMemoryBarrier {
            old_layout: vk::ImageLayout::GENERAL,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range: range,
            src_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            ..Default::default()
        };
        core.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_dst],
        );
        let copy = vk::BufferImageCopy {
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            },
            ..Default::default()
        };
        core.device.cmd_copy_buffer_to_image(
            cb,
            host.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy],
        );
        let to_read = vk::ImageMemoryBarrier {
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range: range,
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            ..Default::default()
        };
        core.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_read],
        );
    })
}

fn scanout_roundtrip(
    core: &Arc<VkCore>,
    enc: &mut Encoder,
    dec: &mut Decoder,
    host: &mut HostBuffer,
    bytes: &[u8],
) {
    host.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
    let (img, fd, pitch) = core
        .create_exportable_image(
            vk::Format::B8G8R8A8_UNORM,
            256,
            256,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .expect("exportable image");
    upload(core, img.image, host, 256, 256).expect("upload");
    let sub = vk::ImageSubresource {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        array_layer: 0,
    };
    let layout = unsafe { core.device.get_image_subresource_layout(img.image, sub) };
    eprintln!(
        "exportable image: returned pitch={pitch} layout row_pitch={} offset={} size={}",
        layout.row_pitch, layout.offset, layout.size
    );
    let _fd = unsafe { OwnedFd::from_raw_fd(fd) };
    enc.set_scanout(
        _fd.as_raw_fd(),
        256,
        256,
        vk::Format::B8G8R8A8_UNORM,
        0,
        0,
        pitch,
    )
    .expect("set_scanout");
    let n = enc.encode_scanout().expect("encode scanout");
    let decoded = dec.decode(&enc.coded()[..n]).expect("decode");
    if decoded != bytes {
        let i = decoded
            .iter()
            .zip(bytes)
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        let px = i / 4;
        let ch = i % 4;
        let (x, y) = (px % 256, px / 256);
        panic!(
            "scanout mismatch at byte {i} (x={x} y={y} ch={ch}): \
             decoded={} expected={} | row0 decoded={:?} | row1 decoded={:?} expected={:?}",
            decoded[i],
            bytes[i],
            &decoded[..16.min(decoded.len())],
            &decoded[1024..1040.min(decoded.len())],
            &bytes[1024..1040.min(bytes.len())]
        );
    }
    core.destroy_image(img);
}

#[test]
#[ignore = "requires a Vulkan device; run with --ignored --nocapture"]
fn vulkan_lossless_scanout_full_and_delta() {
    let geom = Geometry::new(256, 256, 1024);
    let core = Arc::new(VkCore::new().expect("vkcore"));
    let mut enc = Encoder::with_core(core.clone(), geom).expect("encoder");
    let mut dec = Decoder::new(geom).expect("decoder");
    let mut host = core
        .host_buffer(geom.frame_bytes() as u64)
        .expect("host buffer");

    let f0 = frame(0x5a);
    scanout_roundtrip(&core, &mut enc, &mut dec, &mut host, &f0);

    let f1 = frame(0xa5);
    scanout_roundtrip(&core, &mut enc, &mut dec, &mut host, &f1);
}
