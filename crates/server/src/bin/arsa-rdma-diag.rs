//! Diagnose the Vulkan/RDMA staging paths used by the server.
//!
//! Checks, in order:
//!   * process limits: RLIMIT_MEMLOCK and CAP_IPC_LOCK/CAP_SYS_ADMIN
//!   * RDMA device/PD availability
//!   * a plain page-aligned host mapping can be registered with ibv_reg_mr
//!   * the current Vulkan host staging allocation can be registered
//!   * a Vulkan host allocation from non-DEVICE_LOCAL (system) memory can be
//!     registered (the candidate fix when the current type is BAR/VRAM)
//!   * a device-local exportable VkBuffer can be exported and registered with
//!     ibv_reg_dmabuf_mr (the GPU zero-copy path)
//!
//! It does not require KMS/root. Run it as the same user (and, if applicable,
//! with the same file capabilities) as the server.
//!
//! Usage: arsa-rdma-diag [--slots N] [--slot-len BYTES] [--device NAME]

use std::ffi::CStr;
use std::os::fd::RawFd;
use std::process::exit;

use anyhow::{bail, Context, Result};
use ibverbs::{AccessFlags, MemoryRegion, ProtectionDomain};
use rdma_display_vkcore::{ash::vk, HostBuffer, VkCore};

/// The server's 3840x2160 lossless slot from the field log.
const DEFAULT_SLOT_LEN: usize = 33_447_888;
const DEFAULT_SLOTS: usize = 5;

const CAP_IPC_LOCK: u32 = 14;
const CAP_SYS_ADMIN: u32 = 21;

struct Args {
    slots: usize,
    slot_len: usize,
    device: Option<String>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        exit(2);
    }
}

fn run() -> Result<()> {
    let args = parse_args()?;
    let staging_size = args.slot_len.saturating_add(16);

    println!("== arsa-rdma-diag ==");
    println!(
        "slots={} slot_len={} staging_size={}",
        args.slots, args.slot_len, staging_size
    );

    print_process_limits(args.slots, args.slot_len);

    let rdma = match open_rdma(args.device.as_deref()) {
        Ok(v) => v,
        Err(e) => {
            println!("[rdma] no RDMA device: {e:#}");
            None
        }
    };
    if let Some((ctx, _)) = rdma.as_ref() {
        if let Ok(Some(gid)) = ctx.routable_gid(1) {
            println!("[rdma] routable gid: {gid:?}");
        }
    }
    let pd = rdma.as_ref().map(|(_, pd)| pd);
    if pd.is_none() {
        println!("[rdma] MR checks will be skipped");
    }

    println!("\n== Vulkan ==");
    let vk = match VkCore::new() {
        Ok(v) => v,
        Err(e) => {
            println!("[vk] unavailable: {e:#}");
            println!("RESULT: cannot test Vulkan staging");
            return Ok(());
        }
    };
    let mp = unsafe {
        vk.instance
            .get_physical_device_memory_properties(vk.physical_device)
    };
    let props = unsafe { vk.instance.get_physical_device_properties(vk.physical_device) };
    let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy();
    println!("[vk] device: {name}");
    println!(
        "[vk] VK_EXT_external_memory_host: {}",
        vk.has_device_ext("VK_EXT_external_memory_host")
    );
    print_memory_types(&mp);

    let bits = buffer_memory_type_bits(&vk, staging_size)?;
    let current_mt = pick_memory_type(
        &mp,
        bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::empty(),
    );
    let system_mt = pick_memory_type(
        &mp,
        bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );
    match current_mt {
        Some(mt) => println!(
            "[vk] host staging first-match type {mt}: {} (heap {})",
            format_flags(mp.memory_types[mt as usize].property_flags),
            mp.memory_types[mt as usize].heap_index
        ),
        None => println!("[vk] no HOST_VISIBLE|HOST_COHERENT memory type for staging"),
    }
    match system_mt {
        Some(mt) => println!(
            "[vk] system host type {mt}: {} (heap {})",
            format_flags(mp.memory_types[mt as usize].property_flags),
            mp.memory_types[mt as usize].heap_index
        ),
        None => println!("[vk] no non-DEVICE_LOCAL HOST_VISIBLE|HOST_COHERENT memory type"),
    }

    println!("\n== host memory checks ==");
    if pd.is_none() {
        println!("  skipped: no RDMA ProtectionDomain");
    }
    let plain_host_ok = pd.map(|pd| test_plain_host_mr(pd, args.slot_len));

    let current_host_ok = match (pd, current_mt) {
        (Some(pd), Some(mt)) => Some(test_vulkan_host_slots(
            &vk,
            pd,
            args.slots,
            staging_size,
            args.slot_len,
            mt,
            "current",
        )),
        _ => None,
    };
    let system_host_ok = match (pd, system_mt) {
        (Some(pd), Some(mt)) => Some(test_one_vulkan_host_mr(
            &vk,
            pd,
            staging_size,
            args.slot_len,
            mt,
            "system",
        )),
        _ => None,
    };

    println!("\n== GPU dma-buf check ==");
    if pd.is_none() {
        println!("  skipped: no RDMA ProtectionDomain");
    }
    let gpu_dmabuf_ok = pd.map(|pd| test_gpu_dmabuf(&vk, pd, args.slot_len));

    println!("\n== summary ==");
    summary("plain host mapping (ibv_reg_mr)", plain_host_ok);
    summary("Vulkan host staging, current type", current_host_ok);
    summary("Vulkan host staging, system type", system_host_ok);
    summary("GPU buffer export (ibv_reg_dmabuf_mr)", gpu_dmabuf_ok);

    print_diagnosis(
        &mp,
        current_mt,
        system_mt,
        args.slots,
        args.slot_len,
        plain_host_ok,
        current_host_ok,
        system_host_ok,
        gpu_dmabuf_ok,
    );

    // The server's current setup path is the host staging path; a failure here
    // is a failure to start an RDMA session. The GPU dma-buf path is an
    // alternative that the server currently does not try first.
    if current_host_ok == Some(false) {
        exit(1);
    }
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut slots = DEFAULT_SLOTS;
    let mut slot_len = DEFAULT_SLOT_LEN;
    let mut device = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--slots" => {
                slots = args
                    .next()
                    .context("--slots needs a value")?
                    .parse()
                    .context("--slots must be a number")?;
            }
            "--slot-len" => {
                slot_len = args
                    .next()
                    .context("--slot-len needs a value")?
                    .parse()
                    .context("--slot-len must be bytes")?;
            }
            "--device" => device = Some(args.next().context("--device needs a value")?),
            "-h" | "--help" => {
                println!(
                    "usage: arsa-rdma-diag [--slots N] [--slot-len BYTES] [--device NAME]"
                );
                println!(
                    "  default: --slots {DEFAULT_SLOTS} --slot-len {DEFAULT_SLOT_LEN} (4K lossless)"
                );
                exit(0);
            }
            other => bail!("unknown argument {other:?}"),
        }
    }
    if slots == 0 {
        bail!("--slots must be >= 1");
    }
    if slot_len == 0 {
        bail!("--slot-len must be > 0");
    }
    Ok(Args {
        slots,
        slot_len,
        device,
    })
}

fn open_rdma(device: Option<&str>) -> Result<Option<(ibverbs::Context, ProtectionDomain)>> {
    println!("\n== RDMA ==");
    if let Ok(list) = ibverbs::devices() {
        print!("[rdma] devices:");
        for dev in &list {
            if let Some(name) = dev.name() {
                print!(" {}", name.to_string_lossy());
            }
        }
        println!();
    }
    match rdma_display_rdma::open_device(device) {
        Ok(ctx) => {
            let pd = ctx.alloc_pd().context("alloc_pd")?;
            println!("[rdma] opened device");
            Ok(Some((ctx, pd)))
        }
        Err(e) => {
            println!("[rdma] open failed: {e:#}");
            Ok(None)
        }
    }
}

fn print_process_limits(slots: usize, slot_len: usize) {
    let uid = unsafe { libc::geteuid() };
    let gid = unsafe { libc::getegid() };
    let mut rl = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    let got = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rl) } == 0;
    let caps = effective_caps();
    let ipc_lock = caps.is_some_and(|c| c & (1u64 << CAP_IPC_LOCK) != 0);
    let sys_admin = caps.is_some_and(|c| c & (1u64 << CAP_SYS_ADMIN) != 0);
    let need = slots.saturating_mul(slot_len);

    println!("\n== process ==");
    println!("[env] euid={uid} egid={gid}");
    if got {
        println!(
            "[env] RLIMIT_MEMLOCK soft={} hard={}",
            fmt_rlimit(rl.rlim_cur as u64),
            fmt_rlimit(rl.rlim_max as u64)
        );
    } else {
        println!("[env] RLIMIT_MEMLOCK: getrlimit failed");
    }
    match caps {
        Some(c) => println!("[env] CapEff=0x{c:016x} CAP_IPC_LOCK={ipc_lock} CAP_SYS_ADMIN={sys_admin}"),
        None => println!("[env] CapEff: unavailable"),
    }
    println!(
        "[env] estimated host staging pinning: {} B ({:.1} MiB) for {} slots x {} B",
        need,
        need as f64 / (1024.0 * 1024.0),
        slots,
        slot_len
    );
}

fn effective_caps() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some((k, v)) = line.split_once(':') {
            if k == "CapEff" {
                return u64::from_str_radix(v.trim(), 16).ok();
            }
        }
    }
    None
}

fn fmt_rlimit(v: u64) -> String {
    if v == libc::RLIM_INFINITY as u64 {
        "unlimited".to_string()
    } else {
        format!("{v} B ({:.1} MiB)", v as f64 / (1024.0 * 1024.0))
    }
}

fn buffer_create_info(size: u64) -> vk::BufferCreateInfo<'static> {
    vk::BufferCreateInfo {
        size,
        usage: vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    }
}

fn buffer_memory_type_bits(vkcore: &VkCore, size: usize) -> Result<u32> {
    unsafe {
        let buffer = vkcore
            .device
            .create_buffer(&buffer_create_info(size as u64), None)
            .context("create probe VkBuffer")?;
        let req = vkcore.device.get_buffer_memory_requirements(buffer);
        vkcore.device.destroy_buffer(buffer, None);
        Ok(req.memory_type_bits)
    }
}

fn pick_memory_type(
    mp: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    want: vk::MemoryPropertyFlags,
    avoid: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..mp.memory_type_count {
        if bits & (1u32 << i) == 0 {
            continue;
        }
        let f = mp.memory_types[i as usize].property_flags;
        if f.contains(want) && (avoid.is_empty() || !f.intersects(avoid)) {
            return Some(i);
        }
    }
    None
}

fn print_memory_types(mp: &vk::PhysicalDeviceMemoryProperties) {
    println!("[vk] memory types:");
    for i in 0..mp.memory_type_count {
        let t = &mp.memory_types[i as usize];
        let heap = &mp.memory_heaps[t.heap_index as usize];
        println!(
            "  {}: {} heap={} heap_size={} MiB",
            i,
            format_flags(t.property_flags),
            t.heap_index,
            heap.size / (1024 * 1024)
        );
    }
}

fn format_flags(f: vk::MemoryPropertyFlags) -> String {
    let mut v = Vec::new();
    if f.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
        v.push("DEVICE_LOCAL");
    }
    if f.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
        v.push("HOST_VISIBLE");
    }
    if f.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
        v.push("HOST_COHERENT");
    }
    if f.contains(vk::MemoryPropertyFlags::HOST_CACHED) {
        v.push("HOST_CACHED");
    }
    if f.contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED) {
        v.push("LAZILY_ALLOCATED");
    }
    if f.contains(vk::MemoryPropertyFlags::PROTECTED) {
        v.push("PROTECTED");
    }
    if v.is_empty() {
        format!("{f:?}")
    } else {
        v.join("|")
    }
}

fn create_host_buffer(vkcore: &VkCore, size: u64, memory_type: u32) -> Result<HostBuffer> {
    unsafe {
        let buffer = vkcore
            .device
            .create_buffer(&buffer_create_info(size), None)
            .context("create host VkBuffer")?;
        let req = vkcore.device.get_buffer_memory_requirements(buffer);
        let memory = match vkcore.device.allocate_memory(
            &vk::MemoryAllocateInfo {
                allocation_size: req.size,
                memory_type_index: memory_type,
                ..Default::default()
            },
            None,
        ) {
            Ok(m) => m,
            Err(e) => {
                vkcore.device.destroy_buffer(buffer, None);
                return Err(e).context("allocate host VkBuffer memory");
            }
        };
        if let Err(e) = vkcore.device.bind_buffer_memory(buffer, memory, 0) {
            vkcore.device.free_memory(memory, None);
            vkcore.device.destroy_buffer(buffer, None);
            return Err(e).context("bind host VkBuffer memory");
        }
        let ptr = match vkcore
            .device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        {
            Ok(p) => p,
            Err(e) => {
                vkcore.device.free_memory(memory, None);
                vkcore.device.destroy_buffer(buffer, None);
                return Err(e).context("map host VkBuffer memory");
            }
        };
        Ok(HostBuffer {
            buffer,
            memory,
            ptr: ptr as *mut u8,
            size,
        })
    }
}

fn destroy_host_buffer(vkcore: &VkCore, hb: HostBuffer) {
    unsafe {
        vkcore.device.unmap_memory(hb.memory);
        vkcore.device.destroy_buffer(hb.buffer, None);
        vkcore.device.free_memory(hb.memory, None);
    }
}

fn test_plain_host_mr(pd: &ProtectionDomain, len: usize) -> bool {
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        println!("[host] plain mmap: FAIL ({})", std::io::Error::last_os_error());
        return false;
    }
    let flags = AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_READ;
    let r = unsafe { pd.register_from_raw(ptr as *mut u8, len, flags) };
    let ok = match r {
        Ok(mr) => {
            println!("[host] plain mmap len={len}: OK (rkey={:#x})", mr.rkey());
            drop(mr);
            true
        }
        Err(e) => {
            println!("[host] plain mmap len={len}: FAIL ({e:#})");
            false
        }
    };
    unsafe { libc::munmap(ptr, len) };
    ok
}

fn test_vulkan_host_slots(
    vkcore: &VkCore,
    pd: &ProtectionDomain,
    slots: usize,
    staging_size: usize,
    slot_len: usize,
    mt: u32,
    label: &str,
) -> bool {
    println!(
        "[host] Vulkan {label} type {mt}: testing {slots} slots x {slot_len} B"
    );
    let flags = AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_READ;
    let mut mrs: Vec<MemoryRegion<()>> = Vec::new();
    let mut hosts: Vec<HostBuffer> = Vec::new();
    let mut all_ok = true;
    for slot in 0..slots {
        let hb = match create_host_buffer(vkcore, staging_size as u64, mt) {
            Ok(h) => h,
            Err(e) => {
                println!("  slot {slot}: FAIL allocate ({e:#})");
                all_ok = false;
                break;
            }
        };
        let r = unsafe { pd.register_from_raw(hb.ptr, slot_len, flags) };
        match r {
            Ok(mr) => {
                println!("  slot {slot}: OK (ptr={:p} rkey={:#x})", hb.ptr, mr.rkey());
                mrs.push(mr);
                hosts.push(hb);
            }
            Err(e) => {
                println!("  slot {slot}: FAIL ibv_reg_mr ({e:#})");
                destroy_host_buffer(vkcore, hb);
                all_ok = false;
                break;
            }
        }
    }
    drop(mrs);
    for hb in hosts {
        destroy_host_buffer(vkcore, hb);
    }
    all_ok
}

fn test_one_vulkan_host_mr(
    vkcore: &VkCore,
    pd: &ProtectionDomain,
    staging_size: usize,
    slot_len: usize,
    mt: u32,
    label: &str,
) -> bool {
    let flags = AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_READ;
    let hb = match create_host_buffer(vkcore, staging_size as u64, mt) {
        Ok(h) => h,
        Err(e) => {
            println!("[host] Vulkan {label} type {mt}: FAIL allocate ({e:#})");
            return false;
        }
    };
    let r = unsafe { pd.register_from_raw(hb.ptr, slot_len, flags) };
    let ok = match r {
        Ok(mr) => {
            println!(
                "[host] Vulkan {label} type {mt} len={slot_len}: OK (ptr={:p} rkey={:#x})",
                hb.ptr,
                mr.rkey()
            );
            drop(mr);
            true
        }
        Err(e) => {
            println!("[host] Vulkan {label} type {mt} len={slot_len}: FAIL ibv_reg_mr ({e:#})");
            false
        }
    };
    destroy_host_buffer(vkcore, hb);
    ok
}

fn test_gpu_dmabuf(vkcore: &VkCore, pd: &ProtectionDomain, slot_len: usize) -> bool {
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST;
    let buffer = match vkcore.create_exportable_buffer(slot_len as u64, usage) {
        Ok(b) => b,
        Err(e) => {
            println!("[dmabuf] create/exportable VkBuffer: FAIL ({e:#})");
            return false;
        }
    };
    let fd = match vkcore.export_dmabuf(buffer.memory) {
        Ok(fd) => fd,
        Err(e) => {
            println!("[dmabuf] export VkBuffer: FAIL ({e:#})");
            destroy_gpu_buffer(vkcore, buffer.buffer, buffer.memory);
            return false;
        }
    };
    let iova = 0x4_0000_0000u64;
    let r = pd.register_dmabuf(
        fd as i32,
        0,
        slot_len,
        iova,
        AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_READ,
    );
    unsafe { libc::close(fd as RawFd) };
    let ok = match r {
        Ok(mr) => {
            println!(
                "[dmabuf] GPU buffer fd={fd} len={slot_len}: OK (addr={:#x} rkey={:#x})",
                mr.remote().addr,
                mr.remote().rkey
            );
            drop(mr);
            true
        }
        Err(e) => {
            println!("[dmabuf] GPU buffer fd={fd} len={slot_len}: FAIL ibv_reg_dmabuf_mr ({e:#})");
            false
        }
    };
    destroy_gpu_buffer(vkcore, buffer.buffer, buffer.memory);
    ok
}

fn destroy_gpu_buffer(vkcore: &VkCore, buffer: vk::Buffer, memory: vk::DeviceMemory) {
    unsafe {
        vkcore.device.destroy_buffer(buffer, None);
        vkcore.device.free_memory(memory, None);
    }
}

fn summary(label: &str, ok: Option<bool>) {
    match ok {
        Some(true) => println!("  [ OK ] {label}"),
        Some(false) => println!("  [FAIL] {label}"),
        None => println!("  [SKIP] {label}"),
    }
}

fn print_diagnosis(
    mp: &vk::PhysicalDeviceMemoryProperties,
    current_mt: Option<u32>,
    system_mt: Option<u32>,
    slots: usize,
    slot_len: usize,
    plain_host_ok: Option<bool>,
    current_host_ok: Option<bool>,
    system_host_ok: Option<bool>,
    gpu_dmabuf_ok: Option<bool>,
) {
    println!("\n== diagnosis ==");
    let current_is_dl = current_mt.is_some_and(|mt| {
        mp.memory_types[mt as usize]
            .property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
    });

    if plain_host_ok.is_none() && current_host_ok.is_none() && gpu_dmabuf_ok.is_none() {
        println!("  No RDMA ProtectionDomain: registration checks were skipped.");
        println!("  The Vulkan memory-type listing above is still useful.");
        return;
    }

    if plain_host_ok == Some(false) {
        println!("  Plain host memory could not be registered. This points at");
        println!("  RLIMIT_MEMLOCK/CAP_IPC_LOCK before anything Vulkan-specific.");
    } else if current_host_ok == Some(false) && current_is_dl {
        println!("  Plain host memory registers, but the current Vulkan staging");
        println!("  type is DEVICE_LOCAL (BAR/VRAM), which is not pinnable by verbs.");
        if system_host_ok == Some(true) {
            println!("  The non-DEVICE_LOCAL Vulkan type registers: prefer that type");
            println!("  for VkStaging, or import system pages with");
            println!("  VK_EXT_external_memory_host.");
        }
    } else if current_host_ok == Some(false) {
        println!("  Plain host memory registers, but the current Vulkan staging");
        println!("  allocation does not. Inspect its memory type/allocator.");
    }

    if current_host_ok == Some(true) {
        println!(
            "  Current host staging path works for {} x {} B.",
            slots, slot_len
        );
    } else if gpu_dmabuf_ok == Some(true) {
        println!("  The host staging path fails, but the GPU dmabuf path works.");
        println!("  The server currently registers host staging before trying");
        println!("  GpuDirect; try GpuDirect first (or use dummy/GPU staging slots).");
    }
    if gpu_dmabuf_ok == Some(false) {
        println!("  GPU dmabuf registration also failed; host staging is the only");
        println!("  currently usable RDMA path here.");
    }
    if current_mt == system_mt {
        println!("  The current and system memory types are the same; the memory");
        println!("  type is not the differentiator.");
    }
    println!(
        "  Note: this binary has its own file capabilities and RLIMIT; run it as"
    );
    println!("  the server user, or with the same setcap/ulimit, for matching results.");
}
