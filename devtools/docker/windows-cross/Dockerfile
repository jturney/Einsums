# Cross-compilation image for validating Windows builds from Linux/macOS.
#
# clang-cl + lld-link target x86_64-pc-windows-msvc natively; xwin fetches the
# genuine MSVC CRT and Windows SDK headers/libraries from Microsoft's servers;
# the third-party Windows libraries (OpenBLAS, HDF5, LLVM's OpenMP runtime)
# come from the SAME conda-forge win-64 packages the Windows CI environment
# uses - a .conda file is just an archive, no conda needed to extract it.
#
# What this validates: compilation of every TU against the real MSVC CRT and
# linking against real Windows import libraries (dllexport discipline, POSIX
# header leaks, /utf-8 issues, CRT API differences). What it cannot do: RUN
# tests - there is no Windows kernel here. The GitHub windows-2022 runner
# remains the source of truth for test results.
#
# Build:  docker build -t einsums-windows-cross devtools/docker/windows-cross
# Use  :  ./devtools/docker/run-ci-leg.sh windows-cross

FROM debian:trixie-slim

ARG XWIN_VERSION=0.6.5
ARG LLVM_MAJOR=19

RUN apt-get update && apt-get install -y --no-install-recommends \
        clang-${LLVM_MAJOR} clang-tools-${LLVM_MAJOR} lld-${LLVM_MAJOR} llvm-${LLVM_MAJOR} \
        cmake ninja-build git curl ca-certificates unzip zstd xz-utils bzip2 rsync python3-minimal \
    && rm -rf /var/lib/apt/lists/* \
    && for t in clang clang-cl clang++ lld-link llvm-mt llvm-rc llvm-lib llvm-dlltool llvm-readobj; do \
           ln -sf /usr/bin/${t}-${LLVM_MAJOR} /usr/local/bin/${t}; \
       done

# ── MSVC CRT + Windows SDK via xwin ─────────────────────────────────────────
# xwin ships static musl binaries for both x86_64 and aarch64 hosts, so this
# image builds natively on Apple Silicon too.
RUN arch="$(uname -m)" \
    && curl -fsSL "https://github.com/Jake-Shadle/xwin/releases/download/${XWIN_VERSION}/xwin-${XWIN_VERSION}-${arch}-unknown-linux-musl.tar.gz" \
       | tar -xz --strip-components=1 -C /usr/local/bin "xwin-${XWIN_VERSION}-${arch}-unknown-linux-musl/xwin" \
    && xwin --accept-license splat --preserve-ms-arch-notation --output /opt/xwin \
    && rm -rf /root/.xwin-cache

# ── Windows third-party libraries from conda-forge win-64 packages ──────────
# Keep these pins loosely in sync with devtools/conda-envs (the Windows CI
# env); exact versions matter less here than ABI plausibility - this sysroot
# exists so find_package() and lld-link have something real to resolve
# against.
COPY fetch-conda-win64.sh gen-implibs.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/fetch-conda-win64.sh /usr/local/bin/gen-implibs.sh \
    && fetch-conda-win64.sh /opt/winlibs \
        libopenblas \
        liblapack:openblas \
        libcblas:openblas \
        libblas:openblas \
        hdf5 \
        llvm-openmp \
        zlib \
        fmt \
        spdlog \
        catch2 \
    && gen-implibs.sh /opt/winlibs

# clang lowers C99 _Complex multiplication to __mulsc3/__muldc3; real
# clang-cl installs auto-link clang_rt.builtins but this image has no Windows
# compiler-rt. Compile the two needed builtins into a small static library.
COPY builtins_shim.c /tmp/builtins_shim.c
RUN clang-cl --target=x86_64-pc-windows-msvc /vctoolsdir /opt/xwin/crt /winsdkdir /opt/xwin/sdk \
        /c /O2 /Fo/tmp/builtins_shim.obj /tmp/builtins_shim.c \
    && llvm-lib /out:/opt/winlibs/Library/lib/einsums_wincross_builtins.lib /tmp/builtins_shim.obj \
    && rm /tmp/builtins_shim.c /tmp/builtins_shim.obj

ENV XWIN_ROOT=/opt/xwin \
    WINLIBS_ROOT=/opt/winlibs

WORKDIR /work
