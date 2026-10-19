//! Authentication for rdma-display ("method B": SSH-shaped, asymmetric).
//!
//! Trust is a pair of long-term **X25519 static keys** (one per end), plus a
//! one-time **PIN pairing** that exchanges them:
//!
//! * the **first** connection pairs: the server prints a short PIN, the user
//!   types it on the client, and a SPAKE2 exchange (offline-attack resistant)
//!   authenticates the PIN and carries the two static public keys across;
//! * **every later** connection is a Noise `IK` handshake — the client already
//!   holds the server's pinned key, the server checks the client's key against
//!   its paired list — giving mutual authentication, forward secrecy and an
//!   encrypted channel in one round trip.
//!
//! Files (under `~/.config/rdma-display/`):
//!
//! | end | file | contents |
//! |-----|------|----------|
//! | server | `host.key` | this server's X25519 static private key |
//! | server | `authorized_clients` | one line per paired client (`name key`) |
//! | client | `id.key` | this client's X25519 static private key |
//! | client | `known_servers` | pinned servers (`name key`), SSH `known_hosts`-style |
//!
//! The module is transport-agnostic: the handshakes run over any `Read + Write`
//! (a TCP stream here), and [`EncryptedStream`] wraps one with the Noise
//! transport (length-framed, chunked to the Noise message size).

pub mod identity;
pub mod noise;
pub mod pair;
pub mod paths;
pub mod stream;
pub mod trust;

pub use identity::Identity;
pub use noise::{client_handshake, server_handshake};
pub use pair::{pair_client, pair_server};
pub use stream::EncryptedStream;
pub use trust::{AuthorizedClients, KnownServers};

/// Wire tag the control connection sends first, then this, to select the auth
/// method before the Noise/SPAKE2 exchange begins.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMethod {
    /// The client has a key and will run the Noise `IK` handshake.
    Noise = 0x01,
    /// The client wants to pair (SPAKE2 with a one-time PIN).
    Pair = 0x02,
}

impl AuthMethod {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(AuthMethod::Noise),
            0x02 => Some(AuthMethod::Pair),
            _ => None,
        }
    }
}

/// The Noise pattern: mutual static-key auth, forward secrecy, one round trip
/// from the client (initiator) who already knows the server's static key.
pub const NOISE_PARAMS: &str = "Noise_IK_25519_ChaChaPoly_BLAKE2s";

/// A short fingerprint of a public key, for pin comparison and logs
/// (first 8 bytes of the raw key, hex).
pub fn fingerprint(public: &[u8]) -> String {
    hex::encode(&public[..public.len().min(8)])
}
