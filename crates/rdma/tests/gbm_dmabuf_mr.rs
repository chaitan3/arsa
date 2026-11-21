//! Retry the client GBM → `ibv_reg_dmabuf_mr` zero-copy route, and test the
//! server's read direction.
//!
//! The C0 experiment found that a linear GBM buffer exported through NVIDIA's
//! static PRIME path was refused by `ibv_reg_dmabuf_mr` with `ENOMEM`. These
//! tests re-run the route on the current driver/hardware:
//!
//! * `gbm_linear_dmabuf_registers_and_receives_rdma_write` — the client
//!   receive direction: register, RDMA-WRITE into the GBM buffer, and verify
//!   the bytes through a CPU mapping when the exporter allows one.
//! * `gbm_linear_dmabuf_is_readable_by_the_nic` — the server send direction:
//!   register, seed the buffer with an RDMA WRITE (standing in for the GPU
//!   codec), then RDMA-READ it back into host memory. This is the operation
//!   the server's zero-copy staging slot performs.
//!
//! They need a DRM render node and an RDMA device, so they are ignored by
//! default:
//!
//! ```text
//! cargo test -p rdma-display-rdma --test gbm_dmabuf_mr -- --ignored --nocapture
//! ```
//!
//! `ARSA_DRM_NODE` (e.g. `/dev/dri/renderD128`) and `ARSA_IB_DEVICE` (e.g.
//! `mlx5_0`) select a specific node/device. Missing libgbm, DRM node, RDMA
//! device or routable GID is reported as a skip. A failed registration is the
//! result under test; a down RDMA port skips only the DMA stage.

mod common;

use std::os::fd::AsRawFd;

use common::*;
use ibverbs::AccessFlags;

#[test]
#[ignore = "requires a DRM render node and an RDMA device; run with --ignored --nocapture"]
fn gbm_linear_dmabuf_registers_and_receives_rdma_write() {
    let (gbm, drm_path) = match Gbm::open(None) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("skip: {e:#}");
            return;
        }
    };
    let bo = match gbm.alloc_linear(WIDTH, HEIGHT) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip: {e:#}");
            return;
        }
    };
    eprintln!(
        "gbm: node={drm_path} fd={} offset={} stride={} modifier={:#x} len={}",
        bo.fd.as_raw_fd(),
        bo.offset,
        bo.stride,
        bo.modifier,
        bo.size
    );
    if bo.modifier != 0 {
        eprintln!("skip: GBM returned modifier {:#x}, not linear", bo.modifier);
        return;
    }

    let (ctx, ib_name) = match open_rdma() {
        Some(x) => x,
        None => {
            eprintln!("skip: no RDMA device");
            return;
        }
    };
    let pd = ctx.alloc_pd().expect("alloc_pd");
    let cq = ctx.create_cq(64).build().expect("create_cq");

    // The client's receive slots are remote-writable; the server's staging
    // slots are local-write + remote-read. Use a superset here so the same
    // registration can be exercised in both directions.
    let flags = AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_WRITE | AccessFlags::REMOTE_READ;
    let iova = 0x1_0000_0000u64 + bo.offset;
    let dst = match pd.register_dmabuf(bo.fd.as_raw_fd(), bo.offset, bo.size, iova, flags) {
        Ok(m) => m,
        Err(e) => panic!("ibv_reg_dmabuf_mr FAILED for the GBM dma-buf: {e:#}"),
    };
    eprintln!(
        "OK: ibv_reg_dmabuf_mr accepted the GBM dma-buf \
         (rdma={ib_name} addr={:#x} rkey={:#x} lkey={:#x})",
        dst.remote().addr,
        dst.remote().rkey,
        dst.lkey()
    );

    if !port_active(&ctx) {
        eprintln!("skip: RDMA port 1 is not ACTIVE/ARMED; registration OK, DMA not attempted");
        return;
    }

    let mut qp = self_qp(&ctx, &pd, &cq, flags);
    let mut src = pd.allocate(64, AccessFlags::PERMISSIVE).expect("src MR");
    let pattern = b"arsa-gbm-dmabuf-mr";
    src.bytes_mut()[..pattern.len()].copy_from_slice(pattern);

    let remote = dst.remote().slice(..pattern.len());
    let mut batch = qp.start_send();
    batch
        .op()
        .signaled()
        .write(1, &[src.slice(..pattern.len())], remote);
    unsafe { batch.submit() }.expect("post RDMA WRITE");
    drain_one(&cq, 1);
    eprintln!("OK: RDMA WRITE completed into the GBM dma-buf");

    if let Some(p) = mmap_read(bo.fd.as_raw_fd(), bo.offset, pattern.len()) {
        let bytes = unsafe { std::slice::from_raw_parts(p, pattern.len()) };
        assert_eq!(bytes, pattern, "RDMA WRITE did not land in the GBM buffer");
        munmap_read(p, bo.offset, pattern.len());
        eprintln!("OK: content verified through a CPU mapping of the dma-buf");
    } else {
        eprintln!(
            "note: dma-buf is not CPU-mappable; registration + RDMA WRITE succeeded, \
             content not checked"
        );
    }
}

#[test]
#[ignore = "requires a DRM render node and an RDMA device; run with --ignored --nocapture"]
fn gbm_linear_dmabuf_is_readable_by_the_nic() {
    let (gbm, drm_path) = match Gbm::open(None) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("skip: {e:#}");
            return;
        }
    };
    let bo = match gbm.alloc_linear(WIDTH, HEIGHT) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip: {e:#}");
            return;
        }
    };
    eprintln!(
        "gbm: node={drm_path} fd={} offset={} stride={} modifier={:#x} len={}",
        bo.fd.as_raw_fd(),
        bo.offset,
        bo.stride,
        bo.modifier,
        bo.size
    );
    if bo.modifier != 0 {
        eprintln!("skip: GBM returned modifier {:#x}, not linear", bo.modifier);
        return;
    }

    let (ctx, ib_name) = match open_rdma() {
        Some(x) => x,
        None => {
            eprintln!("skip: no RDMA device");
            return;
        }
    };
    let pd = ctx.alloc_pd().expect("alloc_pd");
    let cq = ctx.create_cq(64).build().expect("create_cq");

    // Server staging flags are LOCAL_WRITE | REMOTE_READ; REMOTE_WRITE is only
    // used by this test to seed the buffer (the real server's GPU codec writes
    // it instead).
    let flags = AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_WRITE | AccessFlags::REMOTE_READ;
    let iova = 0x1_0000_0000u64 + bo.offset;
    let gbm_mr = match pd.register_dmabuf(bo.fd.as_raw_fd(), bo.offset, bo.size, iova, flags) {
        Ok(m) => m,
        Err(e) => panic!("ibv_reg_dmabuf_mr FAILED for the GBM dma-buf: {e:#}"),
    };
    eprintln!(
        "OK: ibv_reg_dmabuf_mr accepted the GBM dma-buf \
         (rdma={ib_name} addr={:#x} rkey={:#x} lkey={:#x})",
        gbm_mr.remote().addr,
        gbm_mr.remote().rkey,
        gbm_mr.lkey()
    );

    if !port_active(&ctx) {
        eprintln!("skip: RDMA port 1 is not ACTIVE/ARMED; registration OK, DMA not attempted");
        return;
    }

    let mut qp = self_qp(&ctx, &pd, &cq, flags);
    let mut src = pd.allocate(64, AccessFlags::PERMISSIVE).expect("src MR");
    let pattern = b"arsa-gbm-dmabuf-read";
    src.bytes_mut()[..pattern.len()].copy_from_slice(pattern);

    // Seed the GBM buffer (the server's GPU codec writes it in production).
    let remote = gbm_mr.remote().slice(..pattern.len());
    let mut batch = qp.start_send();
    batch
        .op()
        .signaled()
        .write(1, &[src.slice(..pattern.len())], remote);
    unsafe { batch.submit() }.expect("post seeding RDMA WRITE");
    drain_one(&cq, 1);

    // This is the operation the server's zero-copy staging slot performs: the
    // NIC reads the GPU buffer and writes it to the client.
    let dst = pd
        .allocate(64, AccessFlags::LOCAL_WRITE)
        .expect("dst MR");
    let remote = gbm_mr.remote().slice(..pattern.len());
    let mut batch = qp.start_send();
    batch
        .op()
        .signaled()
        .read(2, &[dst.slice(..pattern.len())], remote);
    unsafe { batch.submit() }.expect("post RDMA READ");
    drain_one(&cq, 2);

    assert_eq!(
        &dst.bytes()[..pattern.len()],
        pattern,
        "RDMA READ from the GBM buffer returned the wrong bytes"
    );
    eprintln!("OK: the NIC read the GBM dma-buf back through RDMA");
}
