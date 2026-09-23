//! Smoke test for the Vulkan core: create the device, allocate a host-visible
//! buffer, fill it on the GPU, and read it back on the CPU.

use anyhow::Result;
use ash::vk;
use rdma_display_vkcore::VkCore;

fn main() -> Result<()> {
    let vk = VkCore::new()?;

    let hb = vk.host_buffer(1 << 20)?;
    vk.one_shot(|cb| unsafe {
        vk.device
            .cmd_fill_buffer(cb, hb.buffer, 0, vk::WHOLE_SIZE, 0xCAFE_D00D);
    })?;
    let first = u32::from_ne_bytes(hb.as_slice()[0..4].try_into().unwrap());
    println!("[smoke] GPU fill -> CPU sees 0x{first:08x} ({})",
        if first == 0xCAFE_D00D { "OK" } else { "MISMATCH" });

    println!("[smoke] instance_ci={:p} device_ci={:p} (for pyrowave_create_device)",
        vk.instance_ci, vk.device_ci);
    Ok(())
}
