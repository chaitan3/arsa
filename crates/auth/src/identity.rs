//! This end's long-term X25519 static keypair.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use x25519_dalek::{PublicKey, StaticSecret};

/// An X25519 static keypair (the Noise handshake's identity).
#[derive(Clone)]
pub struct Identity {
    private: [u8; 32],
    public: [u8; 32],
}

impl Identity {
    /// Generate a fresh keypair from the OS CSPRNG.
    pub fn generate() -> Result<Self> {
        let mut private = [0u8; 32];
        getrandom::getrandom(&mut private)
            .map_err(|e| anyhow::anyhow!("gather randomness: {e}"))?;
        Self::from_private(private)
    }

    /// Load the key at `path`, generating and persisting one if absent.
    pub fn load_or_generate(path: &Path) -> Result<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            let id = Self::generate()?;
            id.save(path)?;
            log::info!(
                "[auth] generated {} (fingerprint {})",
                path.display(),
                id.fingerprint()
            );
            Ok(id)
        }
    }

    /// Load a keypair from `path` (hex-encoded private key).
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("read identity {}", path.display()))?;
        let bytes = hex::decode(text.trim())
            .with_context(|| format!("decode identity {}", path.display()))?;
        let private: [u8; 32] = bytes
            .as_slice()
            .try_into()
            .with_context(|| format!("identity {} is not 32 bytes", path.display()))?;
        Self::from_private(private)
    }

    /// Build from raw private bytes, deriving the public key.
    pub fn from_private(private: [u8; 32]) -> Result<Self> {
        let secret = StaticSecret::from(private);
        let public = PublicKey::from(&secret);
        Ok(Identity { private, public: public.to_bytes() })
    }

    /// Persist the private key to `path` (0600 on unix).
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        fs::write(path, hex::encode(self.private))
            .with_context(|| format!("write identity {}", path.display()))?;
        restrict(path)?;
        Ok(())
    }

    pub fn public(&self) -> &[u8] {
        &self.public
    }

    pub fn private(&self) -> &[u8] {
        &self.private
    }

    pub fn fingerprint(&self) -> String {
        crate::fingerprint(&self.public)
    }
}

/// Best-effort 0600 on unix.
fn restrict(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(path)?.permissions();
        perms.set_mode(0o600);
        fs::set_permissions(path, perms)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("id.key");
        let a = Identity::load_or_generate(&path).unwrap();
        let b = Identity::load(&path).unwrap();
        assert_eq!(a.public(), b.public());
        assert_eq!(a.public().len(), 32);
    }

    #[test]
    fn distinct_keys_differ() {
        let a = Identity::generate().unwrap();
        let b = Identity::generate().unwrap();
        assert_ne!(a.public(), b.public());
    }
}
