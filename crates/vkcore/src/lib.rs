//! Minimal Vulkan core for the PyroWave lossy path.
//!
//! It owns a `VkInstance`/`VkDevice` created with exactly the features and
//! interop extensions PyroWave needs, and keeps the `VkInstanceCreateInfo` /
//! `VkDeviceCreateInfo` alive so they can be handed to
//! `pyrowave_create_device` (which shares this device). The PyroWave C API is
//! not called from here yet; this is the device/plumbing layer.

use std::ffi::{c_char, c_void, CString};

use anyhow::{bail, Context, Result};
use ash::vk;

pub use ash;

pub mod pyrowave;

/// Leaked, process-lifetime `CString` (Vulkan wants `*const c_char` that
/// outlives every use, including PyroWave's later reference to the create
/// infos).
fn leak_cstr(s: &str) -> &'static CString {
    Box::leak(Box::new(CString::new(s).unwrap()))
}

pub struct VkCore {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub command_pool: vk::CommandPool,
    pub api_version: u32,
    /// Kept alive for `pyrowave_create_device`.
    pub instance_ci: &'static vk::InstanceCreateInfo<'static>,
    pub device_ci: &'static vk::DeviceCreateInfo<'static>,
}

fn has_ext(props: &[vk::ExtensionProperties], name: &str) -> bool {
    props.iter().any(|p| {
        let n = unsafe { std::ffi::CStr::from_ptr(p.extension_name.as_ptr()) };
        n.to_bytes() == name.as_bytes()
    })
}

impl VkCore {
    pub fn new() -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load().context("load libvulkan (libvulkan.so.1)")?;

            // ---- instance ----
            let app_name = leak_cstr("arsa");
            let app_info: &'static vk::ApplicationInfo = Box::leak(Box::new(vk::ApplicationInfo {
                p_application_name: app_name.as_ptr(),
                application_version: 1,
                api_version: vk::make_api_version(0, 1, 3, 0),
                ..Default::default()
            }));
            let instance_ci: &'static vk::InstanceCreateInfo =
                Box::leak(Box::new(vk::InstanceCreateInfo {
                    p_application_info: app_info,
                    ..Default::default()
                }));
            let instance = entry
                .create_instance(instance_ci, None)
                .context("vkCreateInstance")?;

            // ---- physical device ----
            let pds = instance.enumerate_physical_devices()?;
            if pds.is_empty() {
                bail!("no Vulkan physical devices");
            }
            let required = [
                "VK_KHR_external_memory_fd",
                "VK_KHR_external_semaphore_fd",
                "VK_EXT_image_drm_format_modifier",
                "VK_KHR_dedicated_allocation",
                "VK_KHR_get_memory_requirements2",
                "VK_KHR_bind_memory2",
            ];
            let mut chosen: Option<vk::PhysicalDevice> = None;
            for pd in &pds {
                let exts = instance.enumerate_device_extension_properties(*pd)?;
                let missing: Vec<&str> = required
                    .iter()
                    .copied()
                    .filter(|n| !has_ext(&exts, n))
                    .collect();
                if missing.is_empty() {
                    chosen = Some(*pd);
                    break;
                }
                let props = instance.get_physical_device_properties(*pd);
                let name = std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
                log::info!("[vkcore] skip {name}: missing {missing:?}");
            }
            let physical_device = chosen.context("no device with the required interop extensions")?;

            let props = instance.get_physical_device_properties(physical_device);
            let api_version = props.api_version;
            let dev_name = std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
            log::info!(
                "[vkcore] device: {dev_name} (api {}.{}.{})",
                vk::api_version_major(api_version),
                vk::api_version_minor(api_version),
                vk::api_version_patch(api_version),
            );

            // ---- features PyroWave requires ----
            let feat11: &'static mut vk::PhysicalDeviceVulkan11Features =
                Box::leak(Box::new(vk::PhysicalDeviceVulkan11Features::default()));
            let feat12: &'static mut vk::PhysicalDeviceVulkan12Features =
                Box::leak(Box::new(vk::PhysicalDeviceVulkan12Features {
                    storage_buffer8_bit_access: vk::TRUE,
                    shader_float16: vk::TRUE,
                    timeline_semaphore: vk::TRUE,
                    scalar_block_layout: vk::TRUE,
                    uniform_buffer_standard_layout: vk::TRUE,
                    ..Default::default()
                }));
            let feat13: &'static mut vk::PhysicalDeviceVulkan13Features =
                Box::leak(Box::new(vk::PhysicalDeviceVulkan13Features {
                    subgroup_size_control: vk::TRUE,
                    compute_full_subgroups: vk::TRUE,
                    synchronization2: vk::TRUE,
                    ..Default::default()
                }));
            let feat2: &'static mut vk::PhysicalDeviceFeatures2 =
                Box::leak(Box::new(vk::PhysicalDeviceFeatures2 {
                    features: vk::PhysicalDeviceFeatures {
                        shader_int16: vk::TRUE,
                        ..Default::default()
                    },
                    ..Default::default()
                }));
            // chain: feat2 -> feat13 -> feat12 -> feat11
            feat13.p_next = feat12 as *mut _ as *mut c_void;
            feat12.p_next = feat11 as *mut _ as *mut c_void;
            feat2.p_next = feat13 as *mut _ as *mut c_void;

            // query what the device actually supports before asking
            let mut probe = vk::PhysicalDeviceFeatures2::default();
            let mut probe13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut probe12 = vk::PhysicalDeviceVulkan12Features::default();
            probe.p_next = &mut probe13 as *mut _ as *mut c_void;
            probe13.p_next = &mut probe12 as *mut _ as *mut c_void;
            instance.get_physical_device_features2(physical_device, &mut probe);
            log::info!(
                "[vkcore] features: shaderInt16={} subgroupSizeControl={} storageBuffer8BitAccess={} shaderFloat16={}",
                probe.features.shader_int16 == vk::TRUE,
                probe13.subgroup_size_control == vk::TRUE,
                probe12.storage_buffer8_bit_access == vk::TRUE,
                probe12.shader_float16 == vk::TRUE,
            );
            if probe.features.shader_int16 != vk::TRUE || probe13.subgroup_size_control != vk::TRUE {
                bail!("device lacks PyroWave's required features");
            }

            // ---- device ----
            let dev_exts = instance.enumerate_device_extension_properties(physical_device)?;
            let want_exts = [
                "VK_KHR_external_memory_fd",
                "VK_KHR_external_semaphore_fd",
                "VK_KHR_external_fence_fd",
                "VK_EXT_image_drm_format_modifier",
                "VK_EXT_external_memory_dma_buf",
                "VK_EXT_external_memory_host",
                "VK_KHR_dedicated_allocation",
                "VK_KHR_get_memory_requirements2",
                "VK_KHR_bind_memory2",
            ];
            let mut ext_names: Vec<&'static CString> = Vec::new();
            for n in want_exts {
                if has_ext(&dev_exts, n) {
                    ext_names.push(leak_cstr(n));
                }
            }
            let ext_ptrs: &'static [*const c_char] = Box::leak(
                ext_names
                    .iter()
                    .map(|c| c.as_ptr())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );

            let qfams = instance.get_physical_device_queue_family_properties(physical_device);
            let queue_family = qfams
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS))
                .or_else(|| qfams.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)))
                .context("no compute queue family")? as u32;

            let priorities: &'static [f32] = Box::leak(Box::new([1.0f32]));
            let queue_ci: &'static vk::DeviceQueueCreateInfo =
                Box::leak(Box::new(vk::DeviceQueueCreateInfo {
                    queue_family_index: queue_family,
                    queue_count: 1,
                    p_queue_priorities: priorities.as_ptr(),
                    ..Default::default()
                }));

            let device_ci: &'static vk::DeviceCreateInfo = Box::leak(Box::new(vk::DeviceCreateInfo {
                p_next: feat2 as *mut _ as *mut c_void,
                queue_create_info_count: 1,
                p_queue_create_infos: queue_ci,
                enabled_extension_count: ext_ptrs.len() as u32,
                pp_enabled_extension_names: ext_ptrs.as_ptr(),
                p_enabled_features: std::ptr::null(),
                ..Default::default()
            }));
            let device = instance
                .create_device(physical_device, device_ci, None)
                .context("vkCreateDevice")?;
            let queue = device.get_device_queue(queue_family, 0);

            let command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index: queue_family,
                    ..Default::default()
                },
                None,
            )?;

            log::info!(
                "[vkcore] queue family {queue_family}; {} interop extensions enabled",
                ext_ptrs.len()
            );

            Ok(VkCore {
                entry,
                instance,
                physical_device,
                device,
                queue,
                queue_family,
                command_pool,
                api_version,
                instance_ci,
                device_ci,
            })
        }
    }

    /// Wait for the device to go idle (used to await PyroWave's async GPU work).
    pub fn device_wait_idle(&self) -> Result<()> {
        unsafe { Ok(self.device.device_wait_idle()?) }
    }

    /// Run `record` on the compute queue and wait for it to complete.
    pub fn one_shot<F: FnOnce(vk::CommandBuffer)>(&self, record: F) -> Result<()> {
        unsafe {
            let cb = self.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: self.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })?[0];
            self.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;
            record(cb);
            self.device.end_command_buffer(cb)?;
            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cbs = [cb];
            let submit = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: cbs.as_ptr(),
                ..Default::default()
            };
            self.device.queue_submit(self.queue, &[submit], fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &cbs);
        }
        Ok(())
    }

    /// A host-visible, host-coherent buffer. `cpu()` is the mapped pointer the
    /// RDMA layer registers (same pattern as the existing PBO staging).
    pub fn host_buffer(&self, size: u64) -> Result<HostBuffer> {
        unsafe {
            let buffer = self.device.create_buffer(
                &vk::BufferCreateInfo {
                    size,
                    usage: vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let mt = self.find_memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let memory = self.device.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: req.size,
                    memory_type_index: mt,
                    ..Default::default()
                },
                None,
            )?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;
            let ptr = self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?;
            Ok(HostBuffer {
                buffer,
                memory,
                ptr: ptr as *mut u8,
                size,
            })
        }
    }

    pub fn find_memory_type(
        &self,
        bits: u32,
        want: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        unsafe {
            let mp = self.instance.get_physical_device_memory_properties(self.physical_device);
            for i in 0..mp.memory_type_count {
                if bits & (1 << i) != 0 && mp.memory_types[i as usize].property_flags.contains(want) {
                    return Ok(i);
                }
            }
            bail!("no memory type for bits=0x{bits:x} want={want:?}");
        }
    }
}

pub struct HostBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub ptr: *mut u8,
    pub size: u64,
}

/// A VkImage plus its backing memory (device-local or imported).
pub struct Image {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

impl VkCore {
    fn alloc_device_local(&self, req: &vk::MemoryRequirements) -> Result<vk::DeviceMemory> {
        let mt = self.find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        unsafe {
            Ok(self.device.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: req.size,
                    memory_type_index: mt,
                    ..Default::default()
                },
                None,
            )?)
        }
    }

    /// Create a device-local image (linear or optimal tiling).
    pub fn create_image(
        &self,
        format: vk::Format,
        width: u32,
        height: u32,
        usage: vk::ImageUsageFlags,
        tiling: vk::ImageTiling,
    ) -> Result<Image> {
        unsafe {
            let image = self.device.create_image(
                &vk::ImageCreateInfo {
                    image_type: vk::ImageType::TYPE_2D,
                    format,
                    extent: vk::Extent3D { width, height, depth: 1 },
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling,
                    usage,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    ..Default::default()
                },
                None,
            )?;
            let req = self.device.get_image_memory_requirements(image);
            let memory = self.alloc_device_local(&req)?;
            self.device.bind_image_memory(image, memory, 0)?;
            Ok(Image { image, memory, format, width, height })
        }
    }

    /// Import a dma-buf (scanout) with its DRM format modifier as a sampleable
    /// image. Requires `VK_EXT_external_memory_dma_buf` + `VK_EXT_image_drm_format_modifier`.
    #[allow(clippy::too_many_arguments)]
    pub fn import_dmabuf(
        &self,
        fd: i32,
        width: u32,
        height: u32,
        format: vk::Format,
        modifier: u64,
        offset: u32,
        stride: u32,
    ) -> Result<Image> {
        unsafe {
            let plane = vk::SubresourceLayout {
                offset: offset as u64,
                row_pitch: stride as u64,
                ..Default::default()
            };
            let drm = vk::ImageDrmFormatModifierExplicitCreateInfoEXT {
                drm_format_modifier: modifier,
                drm_format_modifier_plane_count: 1,
                p_plane_layouts: &plane,
                ..Default::default()
            };
            let ext = vk::ExternalMemoryImageCreateInfo {
                handle_types: vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                p_next: &drm as *const _ as *mut std::ffi::c_void,
                ..Default::default()
            };
            let ci = vk::ImageCreateInfo {
                p_next: &ext as *const _ as *mut std::ffi::c_void,
                image_type: vk::ImageType::TYPE_2D,
                format,
                extent: vk::Extent3D { width, height, depth: 1 },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };
            let image = self.device.create_image(&ci, None)?;
            let mut req2 = vk::MemoryRequirements2::default();
            self.device.get_image_memory_requirements2(
                &vk::ImageMemoryRequirementsInfo2 { image, ..Default::default() },
                &mut req2,
            );
            let req = req2.memory_requirements;
            // Vulkan takes ownership of the fd on a *successful* import and
            // must not see it used afterwards, so never hand it the caller's
            // fd: dup it and let the implementation own the duplicate. The
            // caller keeps its own fd and closes it normally (handing over the
            // caller's fd was a double-close once the caller closed it).
            let dup = libc::dup(fd);
            if dup < 0 {
                return Err(std::io::Error::last_os_error())
                    .context("dup dmabuf fd for import");
            }
            // An imported DRM-format-modifier image needs a dedicated
            // allocation tied to the image: NVIDIA reports
            // prefersDedicatedAllocation for it, and PyroWave's own external
            // import path (Granite) always chains one. Without it the driver
            // may not associate the memory with the image's modifier layout and
            // sampling reads zeros (a black frame).
            let dedicated = vk::MemoryDedicatedAllocateInfo {
                image,
                ..Default::default()
            };
            let imp = vk::ImportMemoryFdInfoKHR {
                handle_type: vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                fd: dup,
                p_next: &dedicated as *const _ as *mut std::ffi::c_void,
                ..Default::default()
            };
            let mt = self.find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let memory = match self.device.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: req.size,
                    memory_type_index: mt,
                    p_next: &imp as *const _ as *mut std::ffi::c_void,
                    ..Default::default()
                },
                None,
            ) {
                Ok(m) => m,
                Err(e) => {
                    // a failed import does not take ownership: close our dup
                    libc::close(dup);
                    return Err(anyhow::anyhow!("import dmabuf memory: {e}"));
                }
            };
            if let Err(e) = self.device.bind_image_memory(image, memory, 0) {
                // the import succeeded, so Vulkan owns `dup`; freeing the
                // memory releases it
                self.device.free_memory(memory, None);
                self.device.destroy_image(image, None);
                return Err(anyhow::anyhow!("bind imported dmabuf memory: {e}"));
            }
            Ok(Image { image, memory, format, width, height })
        }
    }

    /// Whether the chosen physical device advertises the device extension
    /// `name` (dup of `has_ext` for callers outside this module).
    pub fn has_device_ext(&self, name: &str) -> bool {
        unsafe {
            self.instance
                .enumerate_device_extension_properties(self.physical_device)
                .map(|e| has_ext(&e, name))
                .unwrap_or(false)
        }
    }

    /// A **linear** image whose memory is exportable as a dma-buf, for
    /// cross-API sharing: the client has PyroWave (Vulkan) decode into these,
    /// then imports the fd into GL and does the YUV->RGB conversion on the GPU.
    /// Returns the image, the exported fd, and the row pitch EGL needs.
    pub fn create_exportable_image(
        &self,
        format: vk::Format,
        width: u32,
        height: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<(Image, std::os::fd::RawFd, u32)> {
        // A device that does not advertise VK_EXT_external_memory_dma_buf may
        // still accept vkGetMemoryFdKHR(DMA_BUF_EXT) but hand back an opaque
        // handle whose bytes EGL then reads as garbage/zero (the classic
        // "green" frame). Refuse up front so callers can fall back cleanly.
        if !self.has_device_ext("VK_EXT_external_memory_dma_buf") {
            bail!(
                "device lacks VK_EXT_external_memory_dma_buf: \
                 cannot export decoded planes as dma-bufs"
            );
        }
        unsafe {
            let mut ext = vk::ExternalMemoryImageCreateInfo {
                handle_types: vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                ..Default::default()
            };
            let ci = vk::ImageCreateInfo {
                p_next: &mut ext as *mut _ as *mut std::ffi::c_void,
                image_type: vk::ImageType::TYPE_2D,
                format,
                extent: vk::Extent3D { width, height, depth: 1 },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::LINEAR,
                usage,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };
            let image = self.device.create_image(&ci, None)?;
            let req = self.device.get_image_memory_requirements(image);
            let mt =
                self.find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let mut exp = vk::ExportMemoryAllocateInfo {
                handle_types: vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                ..Default::default()
            };
            let memory = self.device.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: req.size,
                    memory_type_index: mt,
                    p_next: &mut exp as *mut _ as *mut std::ffi::c_void,
                    ..Default::default()
                },
                None,
            )?;
            self.device.bind_image_memory(image, memory, 0)?;
            self.transition(
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::ImageAspectFlags::COLOR,
            )?;
            let fd = ash::khr::external_memory_fd::Device::new(&self.instance, &self.device)
                .get_memory_fd(&vk::MemoryGetFdInfoKHR {
                    memory,
                    handle_type: vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                    ..Default::default()
                })?;
            let sub = vk::ImageSubresource {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                array_layer: 0,
            };
            let layout = self.device.get_image_subresource_layout(image, sub);
            Ok((
                Image { image, memory, format, width, height },
                fd,
                layout.row_pitch as u32,
            ))
        }
    }

    /// Export an image's memory as a dma-buf fd (for GL import on the client).
    pub fn export_dmabuf(&self, memory: vk::DeviceMemory) -> Result<std::os::fd::RawFd> {
        let ext = ash::khr::external_memory_fd::Device::new(&self.instance, &self.device);
        unsafe {
            Ok(ext.get_memory_fd(&vk::MemoryGetFdInfoKHR {
                memory,
                handle_type: vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                ..Default::default()
            })?)
        }
    }

    pub fn make_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        swizzle: vk::ComponentSwizzle,
        layout: vk::ImageLayout,
    ) -> Result<vk::ImageView> {
        let _ = layout;
        unsafe {
            Ok(self.device.create_image_view(
                &vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    components: vk::ComponentMapping {
                        r: swizzle,
                        g: swizzle,
                        b: swizzle,
                        a: swizzle,
                    },
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: aspect,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                },
                None,
            )?)
        }
    }

    /// Upload `host` (tightly packed, width*height*4 for RGBA8) into `image`.
    pub fn upload_image(&self, image: vk::Image, host: &HostBuffer, width: u32, height: u32) -> Result<()> {
        self.one_shot(|cb| unsafe {
            let to_dst = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );
            let copy = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_extent: vk::Extent3D { width, height, depth: 1 },
                ..Default::default()
            };
            self.device.cmd_copy_buffer_to_image(
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
                subresource_range: to_dst.subresource_range,
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier(
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

    /// Read a single-plane R8 image back into `host`. `old` must be the
    /// image's current layout (e.g. GENERAL after a PyroWave decode); using
    /// UNDEFINED would discard the contents.
    pub fn download_r8(
        &self,
        image: vk::Image,
        host: &HostBuffer,
        width: u32,
        height: u32,
        old: vk::ImageLayout,
    ) -> Result<()> {
        self.one_shot(|cb| unsafe {
            let to_src = vk::ImageMemoryBarrier {
                old_layout: old,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_src],
            );
            let copy = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_extent: vk::Extent3D { width, height, depth: 1 },
                ..Default::default()
            };
            self.device.cmd_copy_image_to_buffer(
                cb,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                host.buffer,
                &[copy],
            );
        })
    }

    /// Transition an image layout (PyroWave does not do this itself).
    pub fn transition(
        &self,
        image: vk::Image,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
        aspect: vk::ImageAspectFlags,
    ) -> Result<()> {
        self.one_shot(|cb| unsafe {
            let b = vk::ImageMemoryBarrier {
                old_layout: old,
                new_layout: new,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[b],
            );
        })
    }

    /// Destroy an image created by `create_image`/`import_dmabuf` (caller must
    /// ensure no GPU work still references it).
    pub fn destroy_image(&self, img: Image) {
        unsafe {
            self.device.destroy_image(img.image, None);
            self.device.free_memory(img.memory, None);
        }
    }
}


unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

impl HostBuffer {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size as usize) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        // leak on drop intentionally: the RDMA MR may outlive a borrowed view;
        // process-lifetime allocation keeps this simple
    }
}
