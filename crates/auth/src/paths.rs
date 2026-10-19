//! Where the key and trust files live.

use std::path::PathBuf;

/// `$XDG_CONFIG_HOME/arsa` (or `~/.config/arsa`).
pub fn config_dir() -> PathBuf {
    if let Some(x) = std::env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(x).join("arsa")
    } else if let Some(h) = std::env::var_os("HOME") {
        PathBuf::from(h).join(".config").join("arsa")
    } else {
        PathBuf::from(".arsa")
    }
}

/// This host's long-term static private key.
pub fn host_key() -> PathBuf {
    config_dir().join("host.key")
}

/// This client's long-term static private key.
pub fn id_key() -> PathBuf {
    config_dir().join("id.key")
}

/// Server: paired clients (`name public-key` per line).
pub fn authorized_clients() -> PathBuf {
    config_dir().join("authorized_clients")
}

/// Client: pinned servers (`name public-key` per line).
pub fn known_servers() -> PathBuf {
    config_dir().join("known_servers")
}
