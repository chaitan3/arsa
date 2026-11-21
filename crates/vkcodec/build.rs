//! Compile the lossless compute shaders to SPIR-V.
//!
//! The common tile-delta body is concatenated into each source-specific
//! shader so the algorithm exists once. `glslangValidator` (the `glslang-tools`
//! package) is required at build time; the server already needs a Vulkan
//! driver, and the tool is only used to produce SPIR-V.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn glslang() -> PathBuf {
    if let Ok(p) = env::var("GLSLANG_VALIDATOR") {
        return PathBuf::from(p);
    }
    for p in ["glslangValidator", "/usr/sbin/glslangValidator", "/usr/bin/glslangValidator"] {
        if Command::new(p).arg("--version").output().is_ok() {
            return PathBuf::from(p);
        }
    }
    panic!(
        "glslangValidator not found; install glslang-tools \
         (or set GLSLANG_VALIDATOR)"
    );
}

fn compile(glslang: &Path, src: &Path, out: &Path) {
    let output = Command::new(glslang)
        .args(["-V", "--target-env", "vulkan1.1", "-S", "comp", "-o"])
        .arg(out)
        .arg(src)
        .output()
        .unwrap_or_else(|e| panic!("run {}: {e}", glslang.display()));
    if !output.status.success() {
        panic!(
            "glslangValidator {} failed:\n{}\n{}",
            src.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let common = fs::read_to_string("shaders/enc_common.glsl").expect("read enc_common.glsl");
    let glslang = glslang();

    for name in ["enc_ssbo", "enc_image"] {
        let src = fs::read_to_string(format!("shaders/{name}.glsl"))
            .unwrap_or_else(|e| panic!("read {name}: {e}"))
            .replace("//__COMMON__", &common);
        let generated = out.join(format!("{name}.comp"));
        fs::write(&generated, src).expect("write generated shader");
        compile(&glslang, &generated, &out.join(format!("{name}.spv")));
        println!("cargo:rerun-if-changed=shaders/{name}.glsl");
    }
    compile(&glslang, Path::new("shaders/enc_header.glsl"), &out.join("enc_header.spv"));
    println!("cargo:rerun-if-changed=shaders/enc_common.glsl");
    println!("cargo:rerun-if-changed=shaders/enc_header.glsl");
}
