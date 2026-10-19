//! Secondary-socket binding: a challenge-response over the session secret.
//!
//! The video/input/audio sockets are opened after the authenticated control
//! handshake. Rather than send the session secret (a sniffable, replayable
//! bearer token), the server sends a fresh nonce and the client answers with
//! `HMAC(secret, "rdma-display-conn" ‖ tag ‖ nonce)`. The `tag` binds the proof
//! to the socket kind, so an input proof can't be replayed onto the video
//! socket.

use std::io::{Read, Write};

use anyhow::{bail, Context, Result};
use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::SECRET_LEN;

type HmacSha256 = Hmac<Sha256>;

const NONCE_LEN: usize = 16;
const PROOF_LEN: usize = 32;

fn proof(secret: &[u8; SECRET_LEN], tag: u8, nonce: &[u8]) -> Vec<u8> {
    let mut m = HmacSha256::new_from_slice(secret).expect("hmac accepts any key length");
    m.update(b"rdma-display-conn");
    m.update(&[tag]);
    m.update(nonce);
    m.finalize().into_bytes().to_vec()
}

/// Server: send a fresh nonce, then verify the client's proof.
pub fn server_conn_verify<S: Read + Write>(
    stream: &mut S,
    secret: &[u8; SECRET_LEN],
    tag: u8,
) -> Result<()> {
    let mut nonce = [0u8; NONCE_LEN];
    getrandom::getrandom(&mut nonce).map_err(|e| anyhow::anyhow!("randomness: {e}"))?;
    stream.write_all(&nonce).context("send conn nonce")?;
    stream.flush().context("flush conn nonce")?;

    let mut got = [0u8; PROOF_LEN];
    stream.read_exact(&mut got).context("read conn proof")?;
    let want = proof(secret, tag, &nonce);
    if !ct_eq(&got, &want) {
        bail!("connection proof did not verify");
    }
    Ok(())
}

/// Client: read the nonce and answer with the proof.
pub fn client_conn_proof<S: Read + Write>(
    stream: &mut S,
    secret: &[u8; SECRET_LEN],
    tag: u8,
) -> Result<()> {
    let mut nonce = [0u8; NONCE_LEN];
    stream.read_exact(&mut nonce).context("read conn nonce")?;
    let p = proof(secret, tag, &nonce);
    stream.write_all(&p).context("send conn proof")?;
    stream.flush().context("flush conn proof")?;
    Ok(())
}

/// Length-checked, branch-free comparison.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};

    fn pair() -> (TcpStream, TcpStream) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let c = TcpStream::connect(addr).unwrap();
        let (s, _) = l.accept().unwrap();
        (c, s)
    }

    #[test]
    fn correct_secret_verifies() {
        let secret = [42u8; SECRET_LEN];
        let (mut ct, mut st) = pair();
        let t = std::thread::spawn(move || {
            server_conn_verify(&mut st, &secret, 0x03).unwrap();
        });
        client_conn_proof(&mut ct, &secret, 0x03).unwrap();
        t.join().unwrap();
    }

    #[test]
    fn wrong_secret_is_rejected() {
        let (mut ct, mut st) = pair();
        let t = std::thread::spawn(move || {
            server_conn_verify(&mut st, &[1u8; SECRET_LEN], 0x03).is_err()
        });
        client_conn_proof(&mut ct, &[2u8; SECRET_LEN], 0x03).unwrap();
        assert!(t.join().unwrap(), "server must reject a wrong secret");
    }

    #[test]
    fn wrong_tag_is_rejected() {
        let secret = [7u8; SECRET_LEN];
        let (mut ct, mut st) = pair();
        let t = std::thread::spawn(move || {
            // server expects an input proof; the client answers as if video
            server_conn_verify(&mut st, &secret, 0x03).is_err()
        });
        client_conn_proof(&mut ct, &secret, 0x04).unwrap();
        assert!(t.join().unwrap(), "server must reject a proof for another socket");
    }
}
