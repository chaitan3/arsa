//! The two trust stores: pinned servers (client side) and paired clients
//! (server side). Both are line-oriented text files — `name public-key` — so
//! they can be inspected and edited by hand, like SSH's `known_hosts`.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};

/// One trusted peer: a friendly name and its 32-byte X25519 public key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Peer {
    pub name: String,
    pub public: Vec<u8>,
}

impl Peer {
    pub fn fingerprint(&self) -> String {
        crate::fingerprint(&self.public)
    }
}

/// A `name key` list, loaded from / saved to disk.
#[derive(Default, Clone, Debug)]
pub struct TrustFile {
    peers: Vec<Peer>,
}

impl TrustFile {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text =
            fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let mut peers = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (name, key) = line.split_once(char::is_whitespace).with_context(|| {
                format!("{}:{}: expected `name public-key`", path.display(), n + 1)
            })?;
            let public =
                hex::decode(key.trim()).with_context(|| format!("{}: bad key", path.display()))?;
            if public.len() != 32 {
                bail!("{}:{}: key is not 32 bytes", path.display(), n + 1);
            }
            peers.push(Peer { name: name.trim().to_string(), public });
        }
        Ok(TrustFile { peers })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        let mut out = String::new();
        for p in &self.peers {
            out.push_str(&format!("{} {}\n", p.name, hex::encode(&p.public)));
        }
        fs::write(path, out).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Peer> {
        self.peers.iter()
    }

    /// Insert or replace the entry with this name; returns true if it changed.
    pub fn upsert(&mut self, name: &str, public: &[u8]) -> bool {
        if let Some(p) = self.peers.iter_mut().find(|p| p.name == name) {
            if p.public == public {
                return false;
            }
            p.public = public.to_vec();
        } else {
            self.peers.push(Peer { name: name.to_string(), public: public.to_vec() });
        }
        true
    }

    /// Look up by name.
    pub fn get(&self, name: &str) -> Option<&Peer> {
        self.peers.iter().find(|p| p.name == name)
    }

    /// Does any entry carry this public key?
    pub fn contains_key(&self, public: &[u8]) -> bool {
        self.peers.iter().any(|p| p.public == public)
    }
}

/// Client side: servers we have paired with and pinned (`known_hosts`).
pub type KnownServers = TrustFile;
/// Server side: clients we have paired with (`authorized_keys`).
pub type AuthorizedClients = TrustFile;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_looks_up() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("known_servers");
        let mut t = TrustFile::default();
        assert!(t.upsert("desk", &[7u8; 32]));
        assert!(!t.upsert("desk", &[7u8; 32]), "idempotent");
        assert!(t.upsert("laptop", &[9u8; 32]));
        t.save(&path).unwrap();

        let back = TrustFile::load(&path).unwrap();
        assert_eq!(back.get("desk").unwrap().public, vec![7u8; 32]);
        assert!(back.contains_key(&[9u8; 32]));
        assert!(!back.contains_key(&[1u8; 32]));
        assert_eq!(back.iter().count(), 2);
    }

    #[test]
    fn missing_file_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let t = TrustFile::load(&dir.path().join("nope")).unwrap();
        assert!(t.is_empty());
    }
}
