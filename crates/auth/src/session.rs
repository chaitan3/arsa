//! The auth preamble that runs on the control connection *before* the existing
//! `Hello` handshake: one method byte, then either a Noise `IK` handshake or a
//! PIN pairing (followed by a Noise handshake), then a server-generated
//! **session secret** carried inside the encrypted channel.
//!
//! The secret is what the secondary sockets (video/input/audio) prove
//! possession of, so they are bound to the authenticated session without a
//! handshake of their own. The preamble leaves the control connection in the
//! clear again (the secret is already delivered); full control/input/video
//! confidentiality is a separate, later step.

use std::io::{Read, Write};

use anyhow::{bail, Context, Result};

use crate::identity::Identity;
use crate::noise::{client_handshake, server_handshake};
use crate::pair::{pair_client, pair_server, PairedServer};
use crate::stream::{read_frame, write_frame};
use crate::trust::TrustFile;
use crate::{AuthMethod, fingerprint};

/// Length of the per-session secret.
pub const SECRET_LEN: usize = 32;

/// What the client got out of the preamble.
pub struct ClientAuth {
    /// the session secret (for the secondary sockets)
    pub secret: [u8; SECRET_LEN],
    /// the server's name, as pinned
    pub server_name: String,
    /// set when this connection paired (the caller should persist the pin)
    pub paired: Option<PairedServer>,
}

/// What the server got out of the preamble.
pub struct ServerAuth {
    pub secret: [u8; SECRET_LEN],
    pub client_name: String,
    pub client_public: Vec<u8>,
    /// set when this connection paired (the caller should persist the pin)
    pub newly_paired: bool,
}

/// Client: run the preamble. If the server is already pinned we authenticate
/// with Noise; otherwise we pair (prompting for the PIN the server prints).
pub fn client_auth<S: Read + Write>(
    mut stream: S,
    identity: &Identity,
    known: &TrustFile,
    server_name: &str,
    client_name: &str,
    prompt_pin: impl FnOnce() -> Result<String>,
) -> Result<(S, ClientAuth)> {
    match known.get(server_name) {
        Some(peer) => {
            let public = peer.public.clone();
            let name = peer.name.clone();
            write_method(&mut stream, AuthMethod::Noise)?;
            let mut enc = client_handshake(stream, identity, &public)?;
            let secret = read_secret(&mut enc)?;
            Ok((
                enc.into_inner(),
                ClientAuth { secret, server_name: name, paired: None },
            ))
        }
        None => {
            log::info!("[auth] {server_name} is not pinned yet; pairing");
            write_method(&mut stream, AuthMethod::Pair)?;
            let (stream, paired) = pair_client(stream, identity, client_name, prompt_pin)
                .context("PIN pairing")?;
            // we now hold the server's key: authenticate with Noise and collect
            // the session secret on the same connection
            let mut enc = client_handshake(stream, identity, &paired.public)?;
            let secret = read_secret(&mut enc)?;
            let server_name = paired.name.clone();
            Ok((
                enc.into_inner(),
                ClientAuth { secret, server_name, paired: Some(paired) },
            ))
        }
    }
}

/// Server: run the preamble. A pairing request from a new client prints a PIN
/// (via `make_pin`) and, on success, adds the client to `authorized`.
pub fn server_auth<S: Read + Write>(
    mut stream: S,
    identity: &Identity,
    authorized: &mut TrustFile,
    server_name: &str,
    make_pin: impl FnOnce(&str) -> Result<String>,
) -> Result<(S, ServerAuth)> {
    let method = read_method(&mut stream)?;
    match method {
        AuthMethod::Noise => {
            let (mut enc, client_public) = server_handshake(stream, identity)?;
            let client_name = authorized
                .iter()
                .find(|p| p.public == client_public)
                .map(|p| p.name.clone());
            let Some(client_name) = client_name else {
                bail!(
                    "unpaired client {} rejected",
                    fingerprint(&client_public)
                );
            };
            let secret = write_secret(&mut enc)?;
            Ok((
                enc.into_inner(),
                ServerAuth { secret, client_name, client_public, newly_paired: false },
            ))
        }
        AuthMethod::Pair => {
            let (stream, client_name, client_public) =
                pair_server(stream, identity, server_name, make_pin).context("PIN pairing")?;
            authorized.upsert(&client_name, &client_public);
            // the client now runs the Noise handshake; collect the secret
            let (mut enc, got) = server_handshake(stream, identity)?;
            if got != client_public {
                bail!("client key changed between pairing and handshake");
            }
            let secret = write_secret(&mut enc)?;
            Ok((
                enc.into_inner(),
                ServerAuth { secret, client_name, client_public, newly_paired: true },
            ))
        }
    }
}

fn write_method<S: Write>(w: &mut S, m: AuthMethod) -> Result<()> {
    w.write_all(&[m as u8]).context("send auth method")?;
    w.flush().context("flush auth method")?;
    Ok(())
}

fn read_method<S: Read>(r: &mut S) -> Result<AuthMethod> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).context("read auth method")?;
    AuthMethod::from_byte(b[0])
        .with_context(|| format!("unknown auth method {:#04x}", b[0]))
}

fn write_secret<S: Write>(w: &mut S) -> Result<[u8; SECRET_LEN]> {
    let mut secret = [0u8; SECRET_LEN];
    getrandom::getrandom(&mut secret).map_err(|e| anyhow::anyhow!("randomness: {e}"))?;
    write_frame(w, &secret).context("send session secret")?;
    Ok(secret)
}

fn read_secret<S: Read>(r: &mut S) -> Result<[u8; SECRET_LEN]> {
    let mut buf = Vec::new();
    read_frame(r, &mut buf).context("read session secret")?;
    buf.as_slice()
        .try_into()
        .context("session secret is not 32 bytes")
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
    fn paired_client_authenticates_and_shares_a_secret() {
        let server = Identity::generate().unwrap();
        let client = Identity::generate().unwrap();
        let mut known = TrustFile::default();
        known.upsert("desk", server.public());
        let mut authorized = TrustFile::default();
        authorized.upsert("laptop", client.public());

        let (mut ct, mut st) = pair();
        let st2 = std::thread::spawn(move || {
            // already paired: the client uses Noise, so the PIN maker is unused
            let (_s, auth) =
                server_auth(&mut st, &server, &mut authorized, "desk", |_| {
                    Ok("unused".to_string())
                })
                .unwrap();
            assert_eq!(auth.client_name, "laptop");
            auth.secret
        });

        let (_s, auth) = client_auth(&mut ct, &client, &known, "desk", "laptop", || {
            Ok("unused".to_string())
        })
        .unwrap();
        assert!(auth.paired.is_none());
        assert_eq!(auth.secret, st2.join().unwrap(), "both derive the same secret");
    }

    #[test]
    fn unpaired_client_is_rejected() {
        let server = Identity::generate().unwrap();
        let client = Identity::generate().unwrap();
        let mut known = TrustFile::default();
        known.upsert("desk", server.public());
        // server's authorized list is empty

        let (mut ct, mut st) = pair();
        std::thread::spawn(move || {
            let mut authorized = TrustFile::default();
            let _ = server_auth(&mut st, &server, &mut authorized, "desk", |_| {
                Ok("unused".to_string())
            });
        });

        let err = match client_auth(&mut ct, &client, &known, "desk", "laptop", || {
            Ok("unused".to_string())
        }) {
            Ok(_) => panic!("unpaired client unexpectedly authenticated"),
            Err(e) => e,
        };
        let _ = err; // the server drops the connection; the client sees a read error
    }

    #[test]
    fn unpinned_client_auto_pairs() {
        let server = Identity::generate().unwrap();
        let client = Identity::generate().unwrap();
        let spub = server.public().to_vec();
        let cpub = client.public().to_vec();
        let known = TrustFile::default(); // nothing pinned yet
        let mut authorized = TrustFile::default();

        let (mut ct, mut st) = pair();
        let st2 = std::thread::spawn(move || {
            let (_s, auth) =
                server_auth(&mut st, &server, &mut authorized, "desk", |_| {
                    Ok("123456".to_string())
                })
                .unwrap();
            assert!(auth.newly_paired);
            assert_eq!(auth.client_name, "laptop");
            assert!(authorized.contains_key(&cpub));
            auth.secret
        });

        let (_s, auth) = client_auth(&mut ct, &client, &known, "desk", "laptop", || {
            Ok("123456".to_string())
        })
        .unwrap();
        let paired = auth.paired.expect("paired");
        assert_eq!(paired.public, spub);
        assert_eq!(auth.secret, st2.join().unwrap());
    }
}
