use std::env;
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
    panic!("glslangValidator not found; install glslang-tools (or set GLSLANG_VALIDATOR)");
}

fn compile(glslang: &Path, src: &Path, out: &Path, stage: &str) {
    let output = Command::new(glslang)
        .args(["-V", "--target-env", "vulkan1.1", "-S", stage, "-o"])
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
    let glslang = glslang();
    for (name, stage) in [
        ("lossless_dec", "comp"),
        ("video", "vert"),
        ("video_frag", "frag"),
        ("yuv_frag", "frag"),
        ("overlay_frag", "frag"),
        ("raw_direct", "comp"),
        ("raw_copy", "comp"),
    ] {
        let src = Path::new("shaders").join(format!("{name}.glsl"));
        compile(&glslang, &src, &out.join(format!("{name}.spv")), stage);
        println!("cargo:rerun-if-changed=shaders/{name}.glsl");
    }
}
