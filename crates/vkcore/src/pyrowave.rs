//! PyroWave C API bindings (loaded with `dlopen`, like the rest of the repo's
//! GPU libraries). Types mirror `pyrowave.h`; Vulkan types come from `ash`
//! (same `#[repr(C)]` layout), so no C shim or build-time dependency is
//! needed. `libpyrowave-shared.so` must be on the library path at runtime.

#![allow(non_camel_case_types)]

use std::ffi::c_void;
use std::os::fd::RawFd;

use anyhow::{bail, Context, Result};
use ash::vk;

pub const PYROWAVE_API_VERSION_MAJOR: u32 = 0;

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyrowaveResult {
    Success = 0,
    Timeout = 1,
    ErrorGeneric = -1,
    ErrorInvalidArgument = -2,
    ErrorOutOfHostMemory = -3,
    ErrorOutOfDeviceMemory = -4,
    ErrorNoVulkan = -5,
    ErrorNotImplemented = -6,
    ErrorUnsupportedExternalHandle = -7,
    ErrorFailedExternalHandle = -8,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Chroma {
    Subsampling420 = 0,
    Subsampling444 = 1,
}

/// Opaque C handles.
#[repr(C)]
pub struct pyrowave_device_opaque {
    _p: [u8; 0],
}
pub type PyrowaveDevice = *mut pyrowave_device_opaque;
#[repr(C)]
pub struct pyrowave_encoder_opaque {
    _p: [u8; 0],
}
pub type PyrowaveEncoder = *mut pyrowave_encoder_opaque;
#[repr(C)]
pub struct pyrowave_decoder_opaque {
    _p: [u8; 0],
}
pub type PyrowaveDecoder = *mut pyrowave_decoder_opaque;
#[repr(C)]
pub struct pyrowave_image_opaque {
    _p: [u8; 0],
}
pub type PyrowaveImage = *mut pyrowave_image_opaque;

#[repr(C)]
pub struct PyrowaveDeviceCreateQueueInfo {
    pub queue: vk::Queue,
    pub family_index: u32,
    pub index: u32,
}

#[repr(C)]
pub struct PyrowaveDeviceCreateInfo {
    pub get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    pub instance: vk::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: vk::Device,
    pub instance_create_info: *const vk::InstanceCreateInfo<'static>,
    pub device_create_info: *const vk::DeviceCreateInfo<'static>,
    pub queue_info: *mut PyrowaveDeviceCreateQueueInfo,
    pub queue_info_count: u32,
    pub queue_lock_callback: Option<unsafe extern "C" fn(*mut c_void)>,
    pub queue_unlock_callback: Option<unsafe extern "C" fn(*mut c_void)>,
    pub userdata: *mut c_void,
}

#[repr(C)]
pub struct PyrowaveEncoderCreateInfo {
    pub device: PyrowaveDevice,
    pub width: i32,
    pub height: i32,
    pub chroma: Chroma,
}

#[repr(C)]
pub struct PyrowaveDecoderCreateInfo {
    pub device: PyrowaveDevice,
    pub width: i32,
    pub height: i32,
    pub chroma: Chroma,
    pub fragment_path: bool,
}

#[repr(C)]
pub struct PyrowaveImageView {
    pub image: vk::Image,
    pub width: u32,
    pub height: u32,
    pub image_format: vk::Format,
    pub view_format: vk::Format,
    pub mip_level: u32,
    pub layer: u32,
    pub aspect: vk::ImageAspectFlags,
    pub swizzle: vk::ComponentSwizzle,
    pub layout: vk::ImageLayout,
}

#[repr(C)]
pub struct PyrowaveGpuBuffers {
    pub planes: [PyrowaveImageView; 3],
}

#[repr(C)]
pub struct PyrowaveSyncPoint {
    pub semaphore: vk::Semaphore,
    pub value: u64,
}

#[repr(C)]
pub struct PyrowaveGpuExternalReference {
    pub image: PyrowaveImage,
    pub queue_family_index: u32,
}

#[repr(C)]
pub struct PyrowaveGpuSyncOperation {
    pub images: *const PyrowaveGpuExternalReference,
    pub num_images: usize,
    pub sync: PyrowaveSyncPoint,
}

#[repr(C)]
pub struct PyrowaveScaledEncodeInfo {
    pub view: PyrowaveImageView,
    pub input_color_space: vk::ColorSpaceKHR,
    pub output_color_space: vk::ColorSpaceKHR,
    pub intermediate_plane_format: vk::Format,
    pub ycbcr_chroma_midpoint: f32,
    pub force_linear_filtering: bool,
    pub skip_dither: bool,
    pub crop_rect: *const vk::Rect2D,
}

#[repr(C)]
pub struct PyrowaveRateControl {
    pub maximum_bitstream_size: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PyrowavePacket {
    pub offset: usize,
    pub size: usize,
}

#[repr(C)]
pub struct PyrowaveImageCreateInfo {
    pub device: PyrowaveDevice,
    pub external_handle: usize,
    pub handle_type: vk::ExternalMemoryHandleTypeFlags,
    pub image_create_info: *const vk::ImageCreateInfo<'static>,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CpuBufferFormat {
    Nv12 = 0,
    Yuv420p = 1,
    Yuv444p = 2,
}

#[repr(C)]
pub struct PyrowaveCpuBuffer {
    pub data: [*mut c_void; 3],
    pub row_stride_in_bytes: [usize; 3],
    pub plane_size_in_bytes: [usize; 3],
    pub width: i32,
    pub height: i32,
    pub format: CpuBufferFormat,
}

#[repr(C)]
pub struct pyrowave_sync_object_opaque {
    _p: [u8; 0],
}
pub type PyrowaveSyncObject = *mut pyrowave_sync_object_opaque;

#[repr(C)]
pub struct PyrowaveSyncObjectCreateInfo {
    pub device: PyrowaveDevice,
    pub external_handle: usize,
    pub handle_type: vk::ExternalSemaphoreHandleTypeFlags,
    pub semaphore_type: vk::SemaphoreType,
    pub import_flags: vk::SemaphoreImportFlags,
}

pub struct Pyrowave {
    _lib: libloading::Library,
    pub create_device:
        unsafe extern "C" fn(*const PyrowaveDeviceCreateInfo, *mut PyrowaveDevice) -> PyrowaveResult,
    pub create_default_device: unsafe extern "C" fn(*mut PyrowaveDevice) -> PyrowaveResult,
    pub device_destroy: unsafe extern "C" fn(PyrowaveDevice),
    pub device_set_queue_type:
        unsafe extern "C" fn(PyrowaveDevice, vk::QueueFlags) -> PyrowaveResult,
    pub encoder_create: unsafe extern "C" fn(
        *const PyrowaveEncoderCreateInfo,
        *mut PyrowaveEncoder,
    ) -> PyrowaveResult,
    pub encoder_encode_gpu_scaled_synchronous: unsafe extern "C" fn(
        PyrowaveEncoder,
        *const PyrowaveGpuSyncOperation,
        *const PyrowaveGpuSyncOperation,
        *const PyrowaveScaledEncodeInfo,
        *const PyrowaveRateControl,
    ) -> PyrowaveResult,
    pub encoder_compute_num_packets:
        unsafe extern "C" fn(PyrowaveEncoder, usize, *mut usize) -> PyrowaveResult,
    pub encoder_compute_num_packets_with_padding:
        unsafe extern "C" fn(PyrowaveEncoder, usize, usize, *mut usize) -> PyrowaveResult,
    pub encoder_packetize_with_padding: unsafe extern "C" fn(
        PyrowaveEncoder,
        *mut PyrowavePacket,
        usize,
        usize,
        *mut usize,
        *mut c_void,
        usize,
    ) -> PyrowaveResult,
    pub encoder_encode_cpu_synchronous: unsafe extern "C" fn(
        PyrowaveEncoder,
        *const PyrowaveCpuBuffer,
        *const PyrowaveRateControl,
    ) -> PyrowaveResult,
    pub encoder_packetize: unsafe extern "C" fn(
        PyrowaveEncoder,
        *mut PyrowavePacket,
        usize,
        *mut usize,
        *mut c_void,
        usize,
    ) -> PyrowaveResult,
    pub encoder_destroy: unsafe extern "C" fn(PyrowaveEncoder),
    pub decoder_create: unsafe extern "C" fn(
        *const PyrowaveDecoderCreateInfo,
        *mut PyrowaveDecoder,
    ) -> PyrowaveResult,
    pub decoder_push_packet:
        unsafe extern "C" fn(PyrowaveDecoder, *const c_void, usize) -> PyrowaveResult,
    pub decoder_decode_is_ready: unsafe extern "C" fn(PyrowaveDecoder, bool) -> bool,
    pub decoder_clear: unsafe extern "C" fn(PyrowaveDecoder),
    pub decoder_decode_cpu_buffer_synchronous:
        unsafe extern "C" fn(PyrowaveDecoder, *const PyrowaveCpuBuffer) -> PyrowaveResult,
    pub decoder_decode_gpu_buffer: unsafe extern "C" fn(
        PyrowaveDecoder,
        *const PyrowaveGpuSyncOperation,
        *const PyrowaveGpuSyncOperation,
        *const PyrowaveGpuBuffers,
    ) -> PyrowaveResult,
    pub sync_object_create: unsafe extern "C" fn(
        *const PyrowaveSyncObjectCreateInfo,
        *mut PyrowaveSyncObject,
    ) -> PyrowaveResult,
    pub sync_object_get_semaphore: unsafe extern "C" fn(PyrowaveSyncObject) -> vk::Semaphore,
    pub sync_object_cpu_wait:
        unsafe extern "C" fn(PyrowaveSyncObject, u64, u64) -> PyrowaveResult,
    pub sync_object_destroy: unsafe extern "C" fn(PyrowaveSyncObject),
    pub decoder_destroy: unsafe extern "C" fn(PyrowaveDecoder),
}

fn sym<'a, T: Copy>(lib: &'a libloading::Library, name: &str) -> Result<T> {
    unsafe {
        lib.get::<T>(name.as_bytes())
            .map(|s| *s)
            .with_context(|| format!("pyrowave symbol {name}"))
    }
}

impl Pyrowave {
    pub fn load(path: Option<&str>) -> Result<Self> {
        let lib = unsafe {
            // Always keep the default candidates after an explicit path: a
            // wrong/stale PYROWAVE_LIB must not stop us finding the lib beside
            // the binary (or on the loader path).
            let mut names: Vec<std::path::PathBuf> = Vec::new();
            if let Some(p) = path {
                names.push(std::path::PathBuf::from(p));
            }
            for n in [
                "libpyrowave-shared.so.0",
                "libpyrowave-shared.so",
                "libpyrowave-shared.so.0.6.0",
            ] {
                names.push(std::path::PathBuf::from(n));
            }
            // Also try beside the running executable by absolute path:
            // $ORIGIN/RUNPATH is not always honoured (e.g. under sudo) and a
            // copied-out binary has no other way to find its sibling lib.
            if let Ok(exe) = std::env::current_exe() {
                if let Some(dir) = exe.parent() {
                    for n in [
                        "libpyrowave-shared.so.0",
                        "libpyrowave-shared.so",
                        "libpyrowave-shared.so.0.6.0",
                    ] {
                        names.push(dir.join(n));
                    }
                }
            }
            let mut l = None;
            let mut tried: Vec<String> = Vec::new();
            for n in &names {
                match libloading::Library::new(n) {
                    Ok(x) => {
                        l = Some(x);
                        break;
                    }
                    Err(e) => tried.push(format!("{} ({e})", n.display())),
                }
            }
            l.context(format!(
                "dlopen libpyrowave-shared (set PYROWAVE_LIB, put it next to the \
                 binary, or set LD_LIBRARY_PATH); tried: {}",
                tried.join(", ")
            ))?
        };
        let mut major = 0u32;
        let mut minor = 0u32;
        let mut patch = 0u32;
        let get_api_version = sym::<unsafe extern "C" fn(*mut u32, *mut u32, *mut u32)>(&lib, "pyrowave_get_api_version")?;
        unsafe { get_api_version(&mut major, &mut minor, &mut patch) };
        if major != PYROWAVE_API_VERSION_MAJOR {
            bail!("pyrowave ABI major {major} != expected {PYROWAVE_API_VERSION_MAJOR}");
        }
        eprintln!("[pyrowave] API {major}.{minor}.{patch}");
        Ok(Pyrowave {
            create_device: sym(&lib, "pyrowave_create_device")?,
            create_default_device: sym(&lib, "pyrowave_create_default_device")?,
            device_destroy: sym(&lib, "pyrowave_device_destroy")?,
            device_set_queue_type: sym(&lib, "pyrowave_device_set_queue_type")?,
            encoder_create: sym(&lib, "pyrowave_encoder_create")?,
            encoder_encode_gpu_scaled_synchronous: sym(
                &lib,
                "pyrowave_encoder_encode_gpu_scaled_synchronous",
            )?,
            encoder_compute_num_packets: sym(&lib, "pyrowave_encoder_compute_num_packets")?,
            encoder_compute_num_packets_with_padding: sym(
                &lib,
                "pyrowave_encoder_compute_num_packets_with_padding",
            )?,
            encoder_packetize: sym(&lib, "pyrowave_encoder_packetize")?,
            encoder_packetize_with_padding: sym(&lib, "pyrowave_encoder_packetize_with_padding")?,
            encoder_encode_cpu_synchronous: sym(&lib, "pyrowave_encoder_encode_cpu_synchronous")?,
            encoder_destroy: sym(&lib, "pyrowave_encoder_destroy")?,
            decoder_create: sym(&lib, "pyrowave_decoder_create")?,
            decoder_push_packet: sym(&lib, "pyrowave_decoder_push_packet")?,
            decoder_decode_is_ready: sym(&lib, "pyrowave_decoder_decode_is_ready")?,
            decoder_clear: sym(&lib, "pyrowave_decoder_clear")?,
            decoder_decode_cpu_buffer_synchronous: sym(&lib, "pyrowave_decoder_decode_cpu_buffer_synchronous")?,
            decoder_decode_gpu_buffer: sym(&lib, "pyrowave_decoder_decode_gpu_buffer")?,
            sync_object_create: sym(&lib, "pyrowave_sync_object_create")?,
            sync_object_get_semaphore: sym(&lib, "pyrowave_sync_object_get_semaphore")?,
            sync_object_cpu_wait: sym(&lib, "pyrowave_sync_object_cpu_wait")?,
            sync_object_destroy: sym(&lib, "pyrowave_sync_object_destroy")?,
            decoder_destroy: sym(&lib, "pyrowave_decoder_destroy")?,
            _lib: lib,
        })
    }

    /// Single-packet encode convenience: encode `input` (a dma-buf view),
    /// packetize into `out`, return the coded length.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_to(
        &self,
        enc: PyrowaveEncoder,
        info: &PyrowaveScaledEncodeInfo,
        rate: &PyrowaveRateControl,
        boundary: usize,
        padding: usize,
        out: &mut [u8],
    ) -> Result<EncodedFrame> {
        let r = unsafe {
            (self.encoder_encode_gpu_scaled_synchronous)(enc, std::ptr::null(), std::ptr::null(), info, rate)
        };
        if r != PyrowaveResult::Success {
            bail!("pyrowave encode: {r:?}");
        }
        let mut npackets = 0usize;
        let r = unsafe {
            (self.encoder_compute_num_packets_with_padding)(enc, boundary, padding, &mut npackets)
        };
        if r != PyrowaveResult::Success {
            bail!("pyrowave compute_num_packets: {r:?}");
        }
        let mut packets = vec![PyrowavePacket::default(); npackets.max(1)];
        let mut out_packets = 0usize;
        let r = unsafe {
            (self.encoder_packetize)(
                enc,
                packets.as_mut_ptr(),
                boundary,
                &mut out_packets,
                out.as_mut_ptr() as *mut c_void,
                out.len(),
            )
        };
        if r != PyrowaveResult::Success {
            bail!("pyrowave packetize: {r:?} (buffer {} too small?)", out.len());
        }
        // last packet end = coded length
        let mut end = 0usize;
        for p in packets.iter().take(out_packets) {
            end = end.max(p.offset + p.size);
        }
        let packets: Vec<(usize, usize)> = packets[..out_packets]
            .iter()
            .map(|p| (p.offset, p.size))
            .collect();
        Ok(EncodedFrame { len: end, packets })
    }

    /// Push one packet (RDMA-received bitstream chunk) and report readiness.
    pub fn push(&self, dec: PyrowaveDecoder, data: &[u8]) -> Result<bool> {
        let r = unsafe {
            (self.decoder_push_packet)(dec, data.as_ptr() as *const c_void, data.len())
        };
        if r != PyrowaveResult::Success && r != PyrowaveResult::Timeout {
            bail!("pyrowave push_packet: {r:?}");
        }
        Ok(unsafe { (self.decoder_decode_is_ready)(dec, false) })
    }
}

impl Pyrowave {
    /// Create a PyroWave device that *shares* our Vulkan device (so we can
    /// import the scanout dma-buf and allocate the decode planes).
    pub fn create_shared_device(&self, vk: &crate::VkCore) -> Result<PyrowaveDevice> {
        let mut qi = PyrowaveDeviceCreateQueueInfo {
            queue: vk.queue,
            family_index: vk.queue_family,
            index: 0,
        };
        let info = PyrowaveDeviceCreateInfo {
            get_instance_proc_addr: vk.entry.static_fn().get_instance_proc_addr,
            instance: vk.instance.handle(),
            physical_device: vk.physical_device,
            device: vk.device.handle(),
            instance_create_info: vk.instance_ci,
            device_create_info: vk.device_ci,
            queue_info: &mut qi,
            queue_info_count: 1,
            queue_lock_callback: None,
            queue_unlock_callback: None,
            userdata: std::ptr::null_mut(),
        };
        let mut dev: PyrowaveDevice = std::ptr::null_mut();
        let r = unsafe { (self.create_device)(&info, &mut dev) };
        if r != PyrowaveResult::Success {
            bail!("pyrowave_create_device: {r:?}");
        }
        Ok(dev)
    }

    pub fn create_default_device_checked(&self) -> Result<PyrowaveDevice> {
        let mut dev: PyrowaveDevice = std::ptr::null_mut();
        let r = unsafe { (self.create_default_device)(&mut dev) };
        if r != PyrowaveResult::Success {
            bail!("pyrowave_create_default_device: {r:?}");
        }
        Ok(dev)
    }

    pub fn encoder_new(
        &self,
        device: PyrowaveDevice,
        width: u32,
        height: u32,
        chroma: Chroma,
    ) -> Result<PyrowaveEncoder> {
        let info = PyrowaveEncoderCreateInfo {
            device,
            width: width as i32,
            height: height as i32,
            chroma,
        };
        let mut enc: PyrowaveEncoder = std::ptr::null_mut();
        let r = unsafe { (self.encoder_create)(&info, &mut enc) };
        if r != PyrowaveResult::Success {
            bail!("pyrowave_encoder_create: {r:?}");
        }
        Ok(enc)
    }

    pub fn decoder_new(
        &self,
        device: PyrowaveDevice,
        width: u32,
        height: u32,
        chroma: Chroma,
    ) -> Result<PyrowaveDecoder> {
        let info = PyrowaveDecoderCreateInfo {
            device,
            width: width as i32,
            height: height as i32,
            chroma,
            fragment_path: false,
        };
        let mut dec: PyrowaveDecoder = std::ptr::null_mut();
        let r = unsafe { (self.decoder_create)(&info, &mut dec) };
        if r != PyrowaveResult::Success {
            bail!("pyrowave_decoder_create: {r:?}");
        }
        Ok(dec)
    }

    /// GPU decode into caller-provided plane views (compute path), flushed and
    /// waited via a release timeline semaphore. `decode_gpu` alone only queues
    /// work into Granite's frame context; this is the synchronous variant.
    pub fn decode_gpu_synced(
        &self,
        device: PyrowaveDevice,
        dec: PyrowaveDecoder,
        buffers: &PyrowaveGpuBuffers,
    ) -> Result<()> {
        let cinfo = PyrowaveSyncObjectCreateInfo {
            device,
            external_handle: usize::MAX,
            handle_type: vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD,
            semaphore_type: vk::SemaphoreType::TIMELINE,
            import_flags: vk::SemaphoreImportFlags::TEMPORARY,
        };
        let mut sync: PyrowaveSyncObject = std::ptr::null_mut();
        let r = unsafe { (self.sync_object_create)(&cinfo, &mut sync) };
        if r != PyrowaveResult::Success {
            bail!("pyrowave_sync_object_create: {r:?}");
        }
        let sem = unsafe { (self.sync_object_get_semaphore)(sync) };
        let release = PyrowaveGpuSyncOperation {
            images: std::ptr::null(),
            num_images: 0,
            sync: PyrowaveSyncPoint {
                semaphore: sem,
                value: 1,
            },
        };
        let r = unsafe {
            (self.decoder_decode_gpu_buffer)(dec, std::ptr::null(), &release, buffers)
        };
        if r != PyrowaveResult::Success {
            unsafe { (self.sync_object_destroy)(sync) };
            bail!("pyrowave_decode_gpu_buffer: {r:?}");
        }
        let r = unsafe { (self.sync_object_cpu_wait)(sync, 1, 1_000_000_000) };
        unsafe { (self.sync_object_destroy)(sync) };
        if r != PyrowaveResult::Success {
            bail!("pyrowave_sync_object_cpu_wait: {r:?}");
        }
        Ok(())
    }

    /// GPU decode into caller-provided plane views (compute path).
    pub fn decode_gpu(
        &self,
        dec: PyrowaveDecoder,
        buffers: &PyrowaveGpuBuffers,
    ) -> Result<()> {
        let r = unsafe {
            (self.decoder_decode_gpu_buffer)(dec, std::ptr::null(), std::ptr::null(), buffers)
        };
        if r != PyrowaveResult::Success {
            bail!("pyrowave_decode_gpu_buffer: {r:?}");
        }
        Ok(())
    }
}

// A dma-buf fd is an `int`; keep the raw type visible for callers.
pub type DmabufFd = RawFd;

pub struct EncodedFrame {
    pub len: usize,
    pub packets: Vec<(usize, usize)>,
}

impl Pyrowave {
    /// CPU (bring-up) encode: YUV planes -> bitstream. Returns the coded
    /// length and the packet (offset, size) list for the decoder.
    pub fn encode_cpu(
        &self,
        enc: PyrowaveEncoder,
        buf: &PyrowaveCpuBuffer,
        rate: &PyrowaveRateControl,
        boundary: usize,
        padding: usize,
        out: &mut [u8],
    ) -> Result<EncodedFrame> {
        let r = unsafe { (self.encoder_encode_cpu_synchronous)(enc, buf, rate) };
        if r != PyrowaveResult::Success {
            bail!("pyrowave encode_cpu: {r:?}");
        }
        let mut npackets = 0usize;
        let r = unsafe {
            (self.encoder_compute_num_packets_with_padding)(enc, boundary, padding, &mut npackets)
        };
        if r != PyrowaveResult::Success {
            bail!("pyrowave compute_num_packets: {r:?}");
        }
        let mut packets = vec![PyrowavePacket::default(); npackets.max(1)];
        let mut out_packets = 0usize;
        let r = unsafe {
            (self.encoder_packetize_with_padding)(
                enc,
                packets.as_mut_ptr(),
                boundary,
                padding,
                &mut out_packets,
                out.as_mut_ptr() as *mut c_void,
                out.len(),
            )
        };
        if r != PyrowaveResult::Success {
            bail!("pyrowave packetize_with_padding: {r:?} (buffer {} too small?)", out.len());
        }
        let packets: Vec<(usize, usize)> = packets[..out_packets]
            .iter()
            .map(|p| (p.offset, p.size))
            .collect();
        let len = packets.iter().map(|(o, s)| o + s).max().unwrap_or(0);
        Ok(EncodedFrame { len, packets })
    }
}

