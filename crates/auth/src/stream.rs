//! Length framing and the post-handshake encrypted byte stream.

use std::io::{self, Read, Write};

use snow::TransportState;

/// A Noise transport message is at most 65535 bytes including the 16-byte tag.
const MAX_FRAME: usize = u16::MAX as usize;
/// Largest plaintext chunk we hand to one Noise message.
const CHUNK: usize = MAX_FRAME - 16;

/// Write one length-prefixed message (`u16` big-endian length + bytes).
pub(crate) fn write_frame<W: Write>(w: &mut W, msg: &[u8]) -> io::Result<()> {
    if msg.len() > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "frame too large"));
    }
    w.write_all(&(msg.len() as u16).to_be_bytes())?;
    w.write_all(msg)?;
    w.flush()
}

/// Read one length-prefixed message into `buf` (resized to fit); returns its
/// length.
pub(crate) fn read_frame<R: Read>(r: &mut R, buf: &mut Vec<u8>) -> io::Result<usize> {
    let mut l = [0u8; 2];
    r.read_exact(&mut l)?;
    let len = u16::from_be_bytes(l) as usize;
    buf.resize(len, 0);
    r.read_exact(&mut buf[..len])?;
    Ok(len)
}

/// A `Read + Write` view over a Noise transport: every `write` becomes one
/// (or more) encrypted, framed Noise messages; every `read` decrypts and
/// buffers. This is what carries the length-prefixed control protocol.
pub struct EncryptedStream<S> {
    inner: S,
    noise: TransportState,
    /// decrypted bytes not yet handed to the caller
    plain: Vec<u8>,
    pos: usize,
    /// scratch buffers, reused per call
    cipher: Vec<u8>,
    scratch: Vec<u8>,
}

impl<S> EncryptedStream<S> {
    pub(crate) fn new(inner: S, noise: TransportState) -> Self {
        EncryptedStream {
            inner,
            noise,
            plain: Vec::new(),
            pos: 0,
            cipher: vec![0u8; MAX_FRAME],
            scratch: vec![0u8; MAX_FRAME],
        }
    }

    /// The underlying transport (for `set_nodelay`, shutdown, ...).
    pub fn get_ref(&self) -> &S {
        &self.inner
    }
}

impl<S: Read + Write> EncryptedStream<S> {
    /// Read and decrypt the next Noise message into `plain`.
    fn fill(&mut self) -> io::Result<()> {
        let n = read_frame(&mut self.inner, &mut self.cipher)?;
        if n < 16 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "short noise message"));
        }
        let out = self
            .noise
            .read_message(&self.cipher[..n], &mut self.scratch)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("noise decrypt: {e}")))?;
        self.plain.clear();
        self.plain.extend_from_slice(&self.scratch[..out]);
        self.pos = 0;
        Ok(())
    }
}

impl<S: Read + Write> Read for EncryptedStream<S> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if self.pos >= self.plain.len() {
            self.fill()?;
        }
        let n = (self.plain.len() - self.pos).min(buf.len());
        buf[..n].copy_from_slice(&self.plain[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

impl<S: Read + Write> Write for EncryptedStream<S> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = buf.len().min(CHUNK);
        if n == 0 {
            return Ok(0);
        }
        let m = self
            .noise
            .write_message(&buf[..n], &mut self.scratch)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("noise encrypt: {e}")))?;
        write_frame(&mut self.inner, &self.scratch[..m])?;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
