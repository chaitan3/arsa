#!/usr/bin/env bash
#
# Build PyroWave's standalone C API (libpyrowave-shared.so) against Ubuntu
# 24.04's glibc (2.39) using Docker.
#
# Why: the .so is dlopen'ed at runtime by the lossy codec. The Rust binaries in
# this repo are built for/lab-hosted on Ubuntu 24.04, but if the .so is built on
# a newer distro it picks up symbol versions the lab's glibc doesn't have
# (e.g. GLIBC_2.43 from sqrtf/acosf), and dlopen fails. Building inside the
# ubuntu:24.04 image pins the C library (and libstdc++) to the lab's versions.
#
# Usage:
#   scripts/build-pyrowave-docker.sh [PYROWAVE_SRC] [OUT_DIR]
#
# Env overrides:
#   PYROWAVE_SRC   PyroWave checkout (with Granite submodules). Default below.
#   OUT_DIR        Where the libs are copied. Default: <repo>/target/release
#   UBUNTU_IMAGE   Builder image. Default: ubuntu:24.04
#   PULL=1         docker pull the image first.
#
# The library is installed next to the client/server binaries in target/release
# because .cargo/config.toml bakes in an $ORIGIN rpath and the codec dlopens
# libpyrowave-shared.so straight out of the executable's directory.
#
# The host source tree is mounted read-only; the build works on a copy so local
# modifications are never touched or reordered.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYROWAVE_SRC="${1:-${PYROWAVE_SRC:-/tmp/opencode/pw/pyrowave}}"
OUT_DIR="${2:-${OUT_DIR:-$REPO_ROOT/target/release}}"
UBUNTU_IMAGE="${UBUNTU_IMAGE:-ubuntu:24.04}"

die() { echo "error: $*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker not found in PATH"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (permissions?)"

[ -f "$PYROWAVE_SRC/CMakeLists.txt" ] \
    || die "no CMakeLists.txt in PYROWAVE_SRC=$PYROWAVE_SRC (point it at the PyroWave checkout)"

if [ ! -f "$PYROWAVE_SRC/Granite/third_party/volk/CMakeLists.txt" ] \
    || [ ! -f "$PYROWAVE_SRC/Granite/third_party/khronos/vulkan-headers/CMakeLists.txt" ]; then
    echo "note: Granite/volk/vulkan-headers missing in the source tree;"
    echo "      the container will run checkout_granite.sh (needs network)."
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "PyroWave source : $PYROWAVE_SRC"
echo "Output dir      : $OUT_DIR"
echo "Builder image   : $UBUNTU_IMAGE"
echo

[ "${PULL:-0}" = "1" ] && docker pull "$UBUNTU_IMAGE"

docker run --rm -i \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -v "$PYROWAVE_SRC:/src:ro" \
    -v "$OUT_DIR:/out" \
    "$UBUNTU_IMAGE" bash -euo pipefail -s <<'INNER'
export DEBIAN_FRONTEND=noninteractive

echo "[docker] installing build deps ..."
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    build-essential cmake ninja-build git ca-certificates python3 pkg-config

echo "[docker] $(c++ --version | head -1)"
echo "[docker] $(cmake --version | head -1)"
echo "[docker] $(ldd --version | head -1)"

# Work on a copy: keep the host tree pristine and allow checkout_granite.sh to
# write when the submodules are absent.
rm -rf /work
mkdir -p /work
cp -a /src/. /work/
cd /work

if [ ! -f Granite/third_party/volk/CMakeLists.txt ] \
    || [ ! -f Granite/third_party/khronos/vulkan-headers/CMakeLists.txt ]; then
    echo "[docker] Granite submodules missing; running checkout_granite.sh ..."
    bash checkout_granite.sh
fi

rm -rf /build /pwout
echo "[docker] configuring ..."
cmake -S /work -B /build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/pwout

echo "[docker] building (jobs=$(nproc)) ..."
cmake --build /build --parallel "$(nproc)"

echo "[docker] installing ..."
cmake --install /build >/dev/null

mkdir -p /out
cp -a /pwout/lib/libpyrowave-shared.so* /out/
chown -R "$HOST_UID:$HOST_GID" /out

so="$(ls -1 /pwout/lib/libpyrowave-shared.so.0.* | head -1)"
echo
echo "[docker] built $(basename "$so")"
echo "[docker]   NEEDED: $(objdump -p "$so" | awk '/NEEDED/{printf "%s ", $2}')"
echo "[docker]   max GLIBC   : $(objdump -T "$so" | grep -oE 'GLIBC_[0-9]+\.[0-9]+'   | sort -V -u | tail -1)"
echo "[docker]   max GLIBCXX : $(objdump -T "$so" | grep -oE 'GLIBCXX_[0-9.]+'      | sort -V -u | tail -1)"
echo "[docker]   max CXXABI  : $(objdump -T "$so" | grep -oE 'CXXABI_[0-9.]+'       | sort -V -u | tail -1)"
INNER

echo
echo "Installed into $OUT_DIR:"
ls -l "$OUT_DIR"/libpyrowave-shared.so*
