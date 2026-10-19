//! One-time PIN pairing (a client's *first* connection to a server).
//!
//! The client asks to pair; the server prints a short PIN for the user to enter
//! on the client; both run a **SPAKE2** exchange keyed by that PIN — which
//! authenticates the PIN *and* yields a strong shared key without exposing the
//! PIN to an offline dictionary attack (the reason a plain HMAC/Noise-PSK over
//! a 4–6 digit PIN is unsafe). The two long-term static public keys are then
//! exchanged and MAC-bound to the PAKE transcript, so a wrong PIN or a
//! tampering relay is detected.
//!
//! The PIN is obtained *after* the first message, so the server can print it
//! (on receiving the request) and the client can then prompt — no pre-shared
//! PIN, no restart.
//!
//! Wire:
//!
//! ```text
//! C→S  client_name, client_static_pub(32)
//! S→C  server_static_pub(32), server_name, msg_b        (server prints the PIN)
//! C→S  msg_a                                            (client prompts for the PIN)
//!      (both) K = SPAKE2.finish(peer msg)
//! C→S  mac_c = HMAC(K, "rdma-display-pair-c" ‖ transcript)
//! S→C  mac_s = HMAC(K, "rdma-display-pair-s" ‖ transcript)
//! ```

use std::io::{Read, Write};

use anyhow::{bail, Context, Result};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use spake2::{Ed25519Group, Identity as PakeId, Password, Spake2};

use crate::identity::Identity;
use crate::stream::{read_frame, write_frame};

type HmacSha256 = Hmac<Sha256>;

/// SPAKE2 role identities (fixed strings bind the exchange to this protocol).
const PAKE_CLIENT: &[u8] = b"rdma-display-client";
const PAKE_SERVER: &[u8] = b"rdma-display-server";

/// What the client learns from a successful pairing.
#[derive(Debug)]
pub struct PairedServer {
    pub name: String,
    pub public: Vec<u8>,
}

/// Client side of pairing. `prompt_pin` is called after the server has been
/// asked (so the user can read the PIN the server printed).
pub fn pair_client<S: Read + Write>(
    mut stream: S,
    identity: &Identity,
    name: &str,
    prompt_pin: impl FnOnce() -> Result<String>,
) -> Result<(S, PairedServer)> {
    // C→S: name + our static key (no SPAKE2 message yet)
    write_frame(&mut stream, name.as_bytes()).context("send name")?;
    write_frame(&mut stream, identity.public()).context("send client pubkey")?;

    // S→C: server static key, server name, SPAKE2 msg_b
    let mut server_pub = Vec::new();
    let mut server_name = Vec::new();
    let mut msg_b = Vec::new();
    read_frame(&mut stream, &mut server_pub).context("read server pubkey")?;
    read_frame(&mut stream, &mut server_name).context("read server name")?;
    read_frame(&mut stream, &mut msg_b).context("read spake2 msg_b")?;
    if server_pub.len() != 32 {
        bail!("server static key is not 32 bytes");
    }

    let pin = prompt_pin()?;
    let (state, msg_a) = Spake2::<Ed25519Group>::start_a(
        &Password::new(pin.as_bytes()),
        &PakeId::new(PAKE_CLIENT),
        &PakeId::new(PAKE_SERVER),
    );
    write_frame(&mut stream, &msg_a).context("send spake2 msg_a")?;

    let key = state
        .finish(&msg_b)
        .map_err(|e| anyhow::anyhow!("spake2 finish (wrong PIN?): {e:?}"))?;

    let transcript = transcript(
        &msg_a,
        &msg_b,
        identity.public(),
        &server_pub,
        name.as_bytes(),
        &server_name,
    );
    let mac_c = mac(&key, b"rdma-display-pair-c", &transcript);
    write_frame(&mut stream, &mac_c).context("send confirm")?;

    let mut mac_s = Vec::new();
    read_frame(&mut stream, &mut mac_s).context("read server confirm")?;
    let want = mac(&key, b"rdma-display-pair-s", &transcript);
    if !constant_time_eq(&mac_s, &want) {
        bail!("server failed to confirm the PIN (wrong PIN?)");
    }

    Ok((
        stream,
        PairedServer {
            name: String::from_utf8_lossy(&server_name).into_owned(),
            public: server_pub,
        },
    ))
}

/// Server side of pairing. `make_pin` is called with the client's name to
/// produce (and, in the caller, print) the PIN.
pub fn pair_server<S: Read + Write>(
    mut stream: S,
    identity: &Identity,
    name: &str,
    make_pin: impl FnOnce(&str) -> Result<String>,
) -> Result<(S, String, Vec<u8>)> {
    // C→S
    let mut client_name = Vec::new();
    let mut client_pub = Vec::new();
    read_frame(&mut stream, &mut client_name).context("read client name")?;
    read_frame(&mut stream, &mut client_pub).context("read client pubkey")?;
    if client_pub.len() != 32 {
        bail!("client static key is not 32 bytes");
    }
    let client_name = String::from_utf8_lossy(&client_name).into_owned();

    let pin = make_pin(&client_name)?;
    let (state, msg_b) = Spake2::<Ed25519Group>::start_b(
        &Password::new(pin.as_bytes()),
        &PakeId::new(PAKE_CLIENT),
        &PakeId::new(PAKE_SERVER),
    );

    // S→C
    write_frame(&mut stream, identity.public()).context("send server pubkey")?;
    write_frame(&mut stream, name.as_bytes()).context("send server name")?;
    write_frame(&mut stream, &msg_b).context("send spake2 msg_b")?;

    // C→S
    let mut msg_a = Vec::new();
    read_frame(&mut stream, &mut msg_a).context("read spake2 msg_a")?;
    let key = state
        .finish(&msg_a)
        .map_err(|e| anyhow::anyhow!("spake2 finish: {e:?}"))?;

    let transcript = transcript(
        &msg_a,
        &msg_b,
        &client_pub,
        identity.public(),
        client_name.as_bytes(),
        name.as_bytes(),
    );
    let mut mac_c = Vec::new();
    read_frame(&mut stream, &mut mac_c).context("read client confirm")?;
    let want_c = mac(&key, b"rdma-display-pair-c", &transcript);
    if !constant_time_eq(&mac_c, &want_c) {
        bail!("client failed to confirm the PIN (wrong PIN?)");
    }
    let mac_s = mac(&key, b"rdma-display-pair-s", &transcript);
    write_frame(&mut stream, &mac_s).context("send server confirm")?;

    Ok((stream, client_name, client_pub))
}

/// The bytes both sides MAC: everything that must be bound together.
fn transcript(
    msg_a: &[u8],
    msg_b: &[u8],
    client_pub: &[u8],
    server_pub: &[u8],
    client_name: &[u8],
    server_name: &[u8],
) -> Vec<u8> {
    let mut t = Vec::new();
    for part in [msg_a, msg_b, client_pub, server_pub, client_name, server_name] {
        t.extend_from_slice(&(part.len() as u32).to_le_bytes());
        t.extend_from_slice(part);
    }
    t
}

fn mac(key: &[u8], label: &[u8], data: &[u8]) -> Vec<u8> {
    let mut m = HmacSha256::new_from_slice(key).expect("hmac accepts any key length");
    m.update(label);
    m.update(data);
    m.finalize().into_bytes().to_vec()
}

/// Length-checked, branch-free comparison.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

/// A short numeric PIN (default 6 digits).
pub fn random_pin(digits: u32) -> Result<String> {
    let mut buf = [0u8; 4];
    getrandom::getrandom(&mut buf)
        .map_err(|e| anyhow::anyhow!("randomness for PIN: {e}"))?;
    let modulus = 10u64.pow(digits);
    let n = (u32::from_le_bytes(buf) as u64) % modulus;
    Ok(format!("{:0width$}", n, width = digits as usize))
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
    fn pairing_exchanges_keys() {
        let server = Identity::generate().unwrap();
        let client = Identity::generate().unwrap();
        let (mut ct, mut st) = pair();
        let (spub, cpub) = (server.public().to_vec(), client.public().to_vec());

        let st2 = std::thread::spawn(move || {
            let (_s, name, pubkey) =
                pair_server(&mut st, &server, "desk", |_| Ok("483102".into())).unwrap();
            assert_eq!(name, "laptop");
            assert_eq!(pubkey, cpub);
        });

        let (_s, paired) =
            pair_client(&mut ct, &client, "laptop", || Ok("483102".into())).unwrap();
        assert_eq!(paired.name, "desk");
        assert_eq!(paired.public, spub);
        st2.join().unwrap();
    }

    #[test]
    fn wrong_pin_is_rejected() {
        let server = Identity::generate().unwrap();
        let client = Identity::generate().unwrap();
        let (mut ct, mut st) = pair();

        std::thread::spawn(move || {
            let _ = pair_server(&mut st, &server, "desk", |_| Ok("111111".into()));
        });

        let err = match pair_client(&mut ct, &client, "laptop", || Ok("999999".into())) {
            Ok(_) => panic!("pairing with the wrong PIN unexpectedly succeeded"),
            Err(e) => e,
        };
        assert!(
            format!("{err:#}").contains("confirm"),
            "expected a confirmation failure, got {err:#}"
        );
    }

    #[test]
    fn pins_are_six_digits() {
        for _ in 0..32 {
            let p = random_pin(6).unwrap();
            assert_eq!(p.len(), 6);
            assert!(p.chars().all(|c| c.is_ascii_digit()));
        }
    }
}
