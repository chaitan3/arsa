//! Stream encryption: AES-256-GCM with per-connection, per-direction keys.
//!
//! Keys are derived from the session secret (already established by the Noise
//! handshake) with HKDF, so each socket and each direction gets its own key and
//! its own nonce counter. That per-direction split is what lets the control
//! connection's reader thread and writer use the same socket without sharing
//! mutable crypto state.
//!
//! Nonce = a per-direction 64-bit counter (TCP is ordered and reliable, so a
//! strict counter also gives replay protection). A fresh session secret per
//! connection means no cross-session key reuse.
//!
//! The RDMA video data plane is *not* encrypted (it is a memory write with no
//! userspace copy to encrypt); only the TCP sockets use this.

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use anyhow::Result;
use hkdf::Hkdf;
use sha2::Sha256;

use crate::SECRET_LEN;

const INFO: &[u8] = b"rdma-display-stream";

/// Derive the key for one connection kind (`tag`) and sender direction.
/// `from_client` is true for client→server, false for server→client.
pub fn derive(secret: &[u8; SECRET_LEN], tag: u8, from_client: bool) -> [u8; 32] {
    let hk = Hkdf::<Sha256>::new(None, secret);
    let mut okm = [0u8; 32];
    hk.expand(&[INFO, &[tag], &[from_client as u8]].concat(), &mut okm)
        .expect("hkdf output is 32 bytes");
    okm
}

fn nonce(counter: u64) -> Nonce<aes_gcm::aead::consts::U12> {
    let mut n = [0u8; 12];
    n[4..].copy_from_slice(&counter.to_be_bytes());
    Nonce::from(n)
}

/// The sending half: seals messages with a monotonically increasing counter.
pub struct Seal {
    cipher: Aes256Gcm,
    counter: u64,
}

impl Seal {
    pub fn new(key: &[u8; 32]) -> Self {
        Seal { cipher: Aes256Gcm::new(key.into()), counter: 0 }
    }

    /// Seal `plain`, appending the ciphertext (with its tag) to `out`.
    pub fn seal(&mut self, plain: &[u8], out: &mut Vec<u8>) {
        let ct = self
            .cipher
            .encrypt(&nonce(self.counter), plain)
            .expect("aes-gcm encrypt");
        self.counter = self.counter.wrapping_add(1);
        out.extend_from_slice(&ct);
    }
}

/// The receiving half: opens messages, enforcing a strict counter.
pub struct Open {
    cipher: Aes256Gcm,
    counter: u64,
}

impl Open {
    pub fn new(key: &[u8; 32]) -> Self {
        Open { cipher: Aes256Gcm::new(key.into()), counter: 0 }
    }

    /// Open `ct` into `out`. A failed tag or a counter mismatch is an error.
    pub fn open(&mut self, ct: &[u8], out: &mut Vec<u8>) -> Result<()> {
        let plain = self
            .cipher
            .decrypt(&nonce(self.counter), ct)
            .map_err(|_| anyhow::anyhow!("stream decrypt failed (tamper or key mismatch)"))?;
        self.counter = self.counter.wrapping_add(1);
        out.clear();
        out.extend_from_slice(&plain);
        Ok(())
    }

    /// Open in place: `buf` is `ciphertext || tag`; on success it holds the
    /// plaintext (same length minus the tag).
    pub fn open_in_place(&mut self, buf: &mut Vec<u8>) -> Result<()> {
        let plain = self
            .cipher
            .decrypt(&nonce(self.counter), buf.as_slice())
            .map_err(|_| anyhow::anyhow!("stream decrypt failed (tamper or key mismatch)"))?;
        self.counter = self.counter.wrapping_add(1);
        *buf = plain;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_binds_direction_and_tag() {
        let secret = [9u8; SECRET_LEN];
        let mut seal = Seal::new(&derive(&secret, 0x01, true));
        let mut open = Open::new(&derive(&secret, 0x01, true));
        let mut ct = Vec::new();
        seal.seal(b"hello stream", &mut ct);
        let mut out = Vec::new();
        open.open(&ct, &mut out).unwrap();
        assert_eq!(out, b"hello stream");

        // a different tag or direction yields a different key
        assert_ne!(derive(&secret, 0x01, true), derive(&secret, 0x02, true));
        assert_ne!(derive(&secret, 0x01, true), derive(&secret, 0x01, false));
    }

    #[test]
    fn rejects_tampering() {
        let secret = [3u8; SECRET_LEN];
        let key = derive(&secret, 0x04, false);
        let mut seal = Seal::new(&key);
        let mut open = Open::new(&key);
        let mut ct = Vec::new();
        seal.seal(b"frame payload", &mut ct);
        ct[0] ^= 0xFF;
        let mut out = Vec::new();
        assert!(open.open(&ct, &mut out).is_err());
    }

    #[test]
    fn rejects_replay() {
        let secret = [5u8; SECRET_LEN];
        let key = derive(&secret, 0x02, false);
        let mut seal = Seal::new(&key);
        let mut open = Open::new(&key);
        let mut ct = Vec::new();
        seal.seal(b"one", &mut ct);
        let mut out = Vec::new();
        open.open(&ct, &mut out).unwrap();
        // replaying the same ciphertext uses a stale counter/key-stream
        assert!(open.open(&ct, &mut out).is_err());
    }

    #[test]
    fn in_place_matches() {
        let secret = [1u8; SECRET_LEN];
        let key = derive(&secret, 0x03, false);
        let mut seal = Seal::new(&key);
        let mut open = Open::new(&key);
        let mut ct = Vec::new();
        seal.seal(b"in place", &mut ct);
        open.open_in_place(&mut ct).unwrap();
        assert_eq!(ct, b"in place");
    }
}
