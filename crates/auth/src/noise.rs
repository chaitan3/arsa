//! The Noise `IK` handshake — the *second and later* connections.
//!
//! `Noise_IK_25519_ChaChaPoly_BLAKE2s`: the client (initiator) already holds
//! the server's pinned static key; the server learns the client's static key
//! from the handshake and authorizes it against its paired list. Two messages,
//! mutual authentication, forward secrecy, and an encrypted channel.

use std::io::{Read, Write};

use anyhow::{Context, Result};
use snow::{Builder, params::NoiseParams};

use crate::NOISE_PARAMS;
use crate::identity::Identity;
use crate::stream::{EncryptedStream, read_frame, write_frame};

fn params() -> Result<NoiseParams> {
    NOISE_PARAMS.parse().context("parse noise params")
}

/// Client side: run the `IK` handshake against a server whose static key we
/// already pinned. Returns the encrypted stream.
pub fn client_handshake<S: Read + Write>(
    mut stream: S,
    identity: &Identity,
    server_public: &[u8],
) -> Result<EncryptedStream<S>> {
    let mut hs = Builder::new(params()?)
        .local_private_key(identity.private())
        .context("noise local key")?
        .remote_public_key(server_public)
        .context("noise remote key")?
        .build_initiator()
        .context("build noise initiator")?;

    let mut buf = vec![0u8; 65535];
    let n = hs.write_message(&[], &mut buf).context("noise write msg 1")?;
    write_frame(&mut stream, &buf[..n]).context("send noise msg 1")?;

    let mut msg = Vec::new();
    read_frame(&mut stream, &mut msg).context("read noise msg 2")?;
    hs.read_message(&msg, &mut buf).context("noise read msg 2")?;

    let noise = hs.into_transport_mode().context("noise transport mode")?;
    Ok(EncryptedStream::new(stream, noise))
}

/// Server side: run the `IK` handshake. Returns the encrypted stream and the
/// client's static public key (for the caller to authorize).
pub fn server_handshake<S: Read + Write>(
    mut stream: S,
    identity: &Identity,
) -> Result<(EncryptedStream<S>, Vec<u8>)> {
    let mut hs = Builder::new(params()?)
        .local_private_key(identity.private())
        .context("noise local key")?
        .build_responder()
        .context("build noise responder")?;

    let mut msg = Vec::new();
    let mut buf = vec![0u8; 65535];
    read_frame(&mut stream, &mut msg).context("read noise msg 1")?;
    hs.read_message(&msg, &mut buf).context("noise read msg 1")?;

    let n = hs.write_message(&[], &mut buf).context("noise write msg 2")?;
    write_frame(&mut stream, &buf[..n]).context("send noise msg 2")?;

    let remote = hs
        .get_remote_static()
        .map(|k| k.to_vec())
        .context("noise exposed no client static key")?;
    let noise = hs.into_transport_mode().context("noise transport mode")?;
    Ok((EncryptedStream::new(stream, noise), remote))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};

    /// A connected loopback pair.
    fn pair() -> (TcpStream, TcpStream) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let c = TcpStream::connect(addr).unwrap();
        let (s, _) = l.accept().unwrap();
        (c, s)
    }

    #[test]
    fn handshake_authenticates_and_encrypts() {
        let server = Identity::generate().unwrap();
        let client = Identity::generate().unwrap();
        let (mut ct, mut st) = pair();

        let spub = server.public().to_vec();
        let cpub = client.public().to_vec();
        let st2 = std::thread::spawn(move || {
            let (mut enc, remote) = server_handshake(&mut st, &server).unwrap();
            assert_eq!(remote, cpub, "server learned the client key");
            let mut got = [0u8; 5];
            enc.read_exact(&mut got).unwrap();
            enc.write_all(b"pong!").unwrap();
            got
        });

        let mut enc = client_handshake(&mut ct, &client, &spub).unwrap();
        enc.write_all(b"hello").unwrap();
        let mut reply = [0u8; 5];
        enc.read_exact(&mut reply).unwrap();
        assert_eq!(&reply, b"pong!");
        assert_eq!(st2.join().unwrap(), *b"hello");
    }

    #[test]
    fn wrong_server_key_is_rejected() {
        let server = Identity::generate().unwrap();
        let client = Identity::generate().unwrap();
        let imposter = Identity::generate().unwrap();
        let (mut ct, mut st) = pair();

        std::thread::spawn(move || {
            // the real server will fail to decrypt the client's msg1
            let _ = server_handshake(&mut st, &server);
        });

        // client pins the *imposter's* key, so msg1 targets the wrong key
        let err = match client_handshake(&mut ct, &client, imposter.public()) {
            Ok(_) => panic!("handshake with the wrong server key unexpectedly succeeded"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(msg.contains("noise"), "unexpected error: {msg}");
    }

    #[test]
    fn oversized_payload_round_trips_in_chunks() {
        let server = Identity::generate().unwrap();
        let client = Identity::generate().unwrap();
        let (mut ct, mut st) = pair();
        let spub = server.public().to_vec();
        let payload = vec![0xABu8; 200_000]; // > one Noise message

        let want = payload.clone();
        let st2 = std::thread::spawn(move || {
            let (mut enc, _) = server_handshake(&mut st, &server).unwrap();
            let mut got = vec![0u8; want.len()];
            enc.read_exact(&mut got).unwrap();
            assert_eq!(got, want);
        });

        let mut enc = client_handshake(&mut ct, &client, &spub).unwrap();
        enc.write_all(&payload).unwrap();
        st2.join().unwrap();
    }
}
