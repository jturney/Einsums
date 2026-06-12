#!/usr/bin/env bash
# Run one CI leg locally inside a Linux container, avoiding GitHub Actions
# round-trip latency. The container is persistent, with a volume-mounted
# ccache and per-leg conda envs, so the first invocation pays the conda env
# creation cost of roughly 10 minutes and every subsequent rebuild/test cycle
# is incremental.
#
# Usage (run from repo root):
#
#     # First-time setup: start the persistent container.
#     ./devtools/docker/run-ci-leg.sh start
#
#     # Build + test a CI leg:
#     ./devtools/docker/run-ci-leg.sh gcc-openblas         # Linux/default/openblas RelWithDebInfo
#     ./devtools/docker/run-ci-leg.sh gcc-mkl              # Linux/default/mkl RelWithDebInfo
#     ./devtools/docker/run-ci-leg.sh clang-openblas       # Linux/clang/openblas RelWithDebInfo
#     ./devtools/docker/run-ci-leg.sh tsan                 # Sanitizers/thread (Debug, BUILD_PYTHON=ON)
#     ./devtools/docker/run-ci-leg.sh asan                 # Sanitizers/address,leak,undefined (Debug)
#
#     # Append `-arm64` to any leg name to run on native arm64 instead of
#     # x86_64-via-Rosetta. It is faster, roughly 2x for instrumented builds,
#     # and arm64's weaker memory model surfaces races more reliably. It is
#     # not a CI reproducer: CI runs x86_64 only, and our SIMD/vector code has
#     # arch-specific kernels. Use arm64 for fast sanitizer/race triage; use the
#     # default amd64 variant when chasing a specific CI failure.
#     ./devtools/docker/run-ci-leg.sh asan-arm64           # native arm64 ASan
#     ./devtools/docker/run-ci-leg.sh tsan-nopy-arm64      # native arm64 TSan, no Python
#
#     # Pass extra flags through to ctest (everything after `--`):
#     ./devtools/docker/run-ci-leg.sh gcc-openblas -- -R "CommBasic" --output-on-failure
#
#     # Tear down when done (each arch has its own container):
#     ./devtools/docker/run-ci-leg.sh stop                 # amd64 container
#     ./devtools/docker/run-ci-leg.sh stop arm64           # arm64 container
#
# Each leg gets its own:
#   - conda env  : einsums-env-${LEG}    (persisted in named volume)
#   - build dir  : /work/build-${LEG}    (persisted via the cached source mount)
#   - ccache dir : /work/ccache-${LEG}   (persisted, BLAS-independent)
#
# Source tree is bind-mounted read-only at /src; we copy it once to /work/src
# at first use (so writable for cmake's generated files) and rsync mtime on
# every re-run so ccache/ninja see edits without invalidating untouched files.

set -euo pipefail

IMAGE="condaforge/miniforge3:latest"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Per-arch container + volume namespacing. amd64 keeps the legacy unsuffixed
# names so existing volumes/containers don't need migration; arm64 gets its
# own parallel namespace so the two archs never share ccache or conda envs
# (object files are arch-specific; conda packages are linux-x86_64 vs
# linux-aarch64).
arch_container_name() {
    if [[ "$1" == "amd64" ]]; then
        echo "einsums-ci-local"
    else
        echo "einsums-ci-local-$1"
    fi
}
arch_pkg_volume() {
    if [[ "$1" == "amd64" ]]; then
        echo "einsums-ci-conda-pkgs"
    else
        echo "einsums-ci-conda-pkgs-$1"
    fi
}
arch_work_volume() {
    if [[ "$1" == "amd64" ]]; then
        echo "einsums-ci-work"
    else
        echo "einsums-ci-work-$1"
    fi
}

# ──────────────────────────────────────────────────────────────────────────
# Leg → (compiler, blas, build_type, cmake_extra) mapping
# ──────────────────────────────────────────────────────────────────────────
leg_settings() {
    # Build legs default to EINSUMS_BUILD_PYTHON=OFF so the codegen-generated
    # ComputeGraph pybind TU (very heavy, can peak >4GB) doesn't OOM Docker
    # Desktop. CI itself does build Python; use the `python` variant
    # (e.g. `gcc-openblas-py`) if you need that parity locally and your
    # Docker memory allowance is comfortable.
    case "$1" in
        gcc-openblas)
            COMPILER=default
            BLAS=openblas
            BUILD_TYPE=RelWithDebInfo
            EXTRA=("-DEINSUMS_BUILD_PYTHON=OFF")
            ;;
        gcc-openblas-py)
            COMPILER=default
            BLAS=openblas
            BUILD_TYPE=RelWithDebInfo
            EXTRA=("-DEINSUMS_BUILD_PYTHON=ON")
            ;;
        gcc-mkl)
            COMPILER=default
            BLAS=mkl
            BUILD_TYPE=RelWithDebInfo
            EXTRA=("-DEINSUMS_BUILD_PYTHON=OFF")
            ;;
        gcc-mkl-py)
            COMPILER=default
            BLAS=mkl
            BUILD_TYPE=RelWithDebInfo
            EXTRA=("-DEINSUMS_BUILD_PYTHON=ON")
            ;;
        clang-openblas)
            COMPILER=clang
            BLAS=openblas
            BUILD_TYPE=RelWithDebInfo
            EXTRA=("-DEINSUMS_BUILD_PYTHON=OFF")
            ;;
        clang-openblas-py)
            COMPILER=clang
            BLAS=openblas
            BUILD_TYPE=RelWithDebInfo
            EXTRA=("-DEINSUMS_BUILD_PYTHON=ON")
            ;;
        tsan)
            COMPILER=default
            BLAS=openblas
            BUILD_TYPE=Debug
            EXTRA=("-DEINSUMS_WITH_SANITIZERS=thread" "-DEINSUMS_BUILD_PYTHON=ON")
            ;;
        tsan-nopy)
            # TSan triage without Python. CI's tsan leg has BUILD_PYTHON=ON
            # but the codegen-generated ComputeGraph pybind TU under TSan
            # instrumentation OOMs Docker Desktop's default memory. Use this
            # variant locally to surface C++ race findings; once they're
            # addressed, run the full `tsan` leg via CI for the Python paths.
            COMPILER=default
            BLAS=openblas
            BUILD_TYPE=Debug
            EXTRA=("-DEINSUMS_WITH_SANITIZERS=thread" "-DEINSUMS_BUILD_PYTHON=OFF")
            ;;
        asan)
            COMPILER=default
            BLAS=openblas
            BUILD_TYPE=Debug
            EXTRA=("-DEINSUMS_WITH_SANITIZERS=address,leak,undefined" "-DEINSUMS_BUILD_PYTHON=OFF")
            ;;
        *)
            echo "Unknown leg: $1" >&2
            echo "Valid: gcc-openblas[-py], gcc-mkl[-py], clang-openblas[-py], tsan, tsan-nopy, asan" >&2
            echo "       (append -arm64 to any of the above for native arm64)" >&2
            exit 1
            ;;
    esac
}

# ──────────────────────────────────────────────────────────────────────────
# Container lifecycle
# ──────────────────────────────────────────────────────────────────────────
cmd_start() {
    local ARCH="${1:-amd64}"
    local CONTAINER_NAME
    CONTAINER_NAME="$(arch_container_name "${ARCH}")"
    if docker ps --filter "name=${CONTAINER_NAME}" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container ${CONTAINER_NAME} already running."
        return
    fi
    if docker ps -a --filter "name=${CONTAINER_NAME}" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Starting existing container ${CONTAINER_NAME}."
        docker start "${CONTAINER_NAME}"
        return
    fi
    echo "Pulling ${IMAGE} (linux/${ARCH})…"
    docker pull --platform "linux/${ARCH}" "${IMAGE}"
    echo "Creating container ${CONTAINER_NAME} (linux/${ARCH}, volume-mounted)…"
    docker run -d --platform "linux/${ARCH}" \
        --name "${CONTAINER_NAME}" \
        -v "${REPO_ROOT}:/src:ro" \
        -v "$(arch_pkg_volume "${ARCH}"):/opt/conda/pkgs" \
        -v "$(arch_work_volume "${ARCH}"):/work" \
        -w /work \
        "${IMAGE}" \
        sleep infinity
    # rsync isn't in the miniforge3 base image; install it now so the
    # per-leg source-sync (host edits → container) works.
    docker exec "${CONTAINER_NAME}" bash -lc \
        "apt-get update -qq >/dev/null && apt-get install -y -qq rsync >/dev/null"
    echo "Container ready. Run a leg, e.g. ./devtools/docker/run-ci-leg.sh gcc-openblas"
}

cmd_stop() {
    local ARCH="${1:-amd64}"
    local CONTAINER_NAME
    CONTAINER_NAME="$(arch_container_name "${ARCH}")"
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    echo "Container ${CONTAINER_NAME} removed."
    echo "Tip: 'docker volume rm $(arch_pkg_volume "${ARCH}") $(arch_work_volume "${ARCH}")' to also drop its cached envs and builds."
}

# ──────────────────────────────────────────────────────────────────────────
# Leg run
# ──────────────────────────────────────────────────────────────────────────
ensure_container_up() {
    local ARCH="$1"
    local CONTAINER_NAME
    CONTAINER_NAME="$(arch_container_name "${ARCH}")"
    if ! docker ps --filter "name=${CONTAINER_NAME}" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        cmd_start "${ARCH}"
    fi
}

run_leg() {
    local LEG="$1"
    shift
    # Anything after `--` is appended to the ctest invocation.
    local CTEST_EXTRA=()
    if [[ "${1:-}" == "--" ]]; then
        shift
        CTEST_EXTRA=("$@")
    fi

    # `-arm64` suffix on the leg name selects the arm64 container/volume
    # namespace. Strip it before resolving leg_settings so each leg has one
    # canonical (compiler, blas, build_type) tuple regardless of arch.
    local ARCH=amd64
    if [[ "${LEG}" == *-arm64 ]]; then
        ARCH=arm64
        LEG="${LEG%-arm64}"
    fi
    local CONTAINER_NAME
    CONTAINER_NAME="$(arch_container_name "${ARCH}")"

    leg_settings "${LEG}"
    ensure_container_up "${ARCH}"

    # Conda env is determined by the compiler and blas pair. All legs with the
    # same toolchain combo share one env so we don't pay the roughly 10-minute
    # `mamba env create` cost more than once per combo. For example, the asan
    # and tsan legs both reuse the gcc-openblas env. Build dirs and ccache stay
    # per-leg since cmake flags and sanitizer flags differ.
    local ENV_NAME="einsums-env-${COMPILER}-${BLAS}"
    local BUILD_DIR="/work/build-${LEG}"
    local CCACHE_DIR="/work/ccache-${LEG}"
    local SRC_DIR="/work/src"

    # Cache the cmake_extra so we can echo a clean configure line
    # (the `+...` form is safe when the array is empty under set -u)
    local CMAKE_EXTRA_STR=""
    for e in "${EXTRA[@]+"${EXTRA[@]}"}"; do CMAKE_EXTRA_STR+=" ${e}"; done

    # Quoted args for ctest
    local CTEST_EXTRA_STR=""
    for e in "${CTEST_EXTRA[@]+"${CTEST_EXTRA[@]}"}"; do CTEST_EXTRA_STR+=" \"${e}\""; done

    echo "▶ leg=${LEG} compiler=${COMPILER} blas=${BLAS} build_type=${BUILD_TYPE}"

    docker exec "${CONTAINER_NAME}" bash -lc "
set -e
source /opt/conda/etc/profile.d/conda.sh

# 1. Snapshot the source on first use; rsync for re-runs so untouched
#    files keep their mtime (ccache/ninja friendly).
if [[ ! -d '${SRC_DIR}' ]]; then
    echo '⤷ snapshotting /src → ${SRC_DIR}'
    cp -r /src '${SRC_DIR}'
else
    rsync -a --delete --exclude=build --exclude=.git/ /src/ '${SRC_DIR}/'
fi

# 2. Create the per-leg conda env on first use; reuse otherwise.
if ! conda env list | awk '{print \$1}' | grep -qx '${ENV_NAME}'; then
    echo '⤷ creating conda env ${ENV_NAME} (first time, slow)'
    # cp -r src dst (when dst exists) silently nests as dst/src; that makes
    # subsequent runs use a stale merge_yml.py at the top level even when
    # the source has been updated. Wipe and recopy contents to /tmp/merge.
    rm -rf /tmp/merge
    cp -r '${SRC_DIR}/devtools/conda-envs' /tmp/merge
    python /tmp/merge/merge_yml.py '${COMPILER}' '${BLAS}' --output /tmp/env-${LEG}.yml
    mamba env create -f /tmp/env-${LEG}.yml -n '${ENV_NAME}' -y >/dev/null
fi
conda activate '${ENV_NAME}'

export CCACHE_DIR='${CCACHE_DIR}'
export CCACHE_MAXSIZE=5G
mkdir -p '${CCACHE_DIR}'

# 3. Configure (idempotent; CMake re-uses cache).
mkdir -p '${BUILD_DIR}'
cmake -S '${SRC_DIR}' -B '${BUILD_DIR}' -G Ninja \\
    -DCMAKE_BUILD_TYPE='${BUILD_TYPE}' \\
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \\
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \\
    -DCMAKE_PREFIX_PATH=\"\${CONDA_PREFIX}\" \\
    -DEINSUMS_WITH_TESTS=ON \\
    -DEINSUMS_WITH_TESTS_UNIT=ON \\
    ${CMAKE_EXTRA_STR}

# 4. Build. Cap parallelism: -j2 normally, -j1 for sanitizer legs. The
#    instrumented Debug TUs, namely Transpose.cpp and the ComputeGraph pybind
#    one, can peak past Docker Desktop's default 8GB allowance when two compile
#    in parallel.
if [[ '${LEG}' == 'asan' || '${LEG}' == 'tsan' || '${LEG}' == 'tsan-nopy' ]]; then
    cmake --build '${BUILD_DIR}' -j1
else
    cmake --build '${BUILD_DIR}' -j2
fi

# 5. Run tests.
cd '${BUILD_DIR}'
ctest --output-on-failure ${CTEST_EXTRA_STR}
"
}

# ──────────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    sed -n '2,30p' "$0" >&2
    exit 1
fi

case "$1" in
    start) shift; cmd_start "${1:-amd64}" ;;
    stop)  shift; cmd_stop  "${1:-amd64}" ;;
    *)     run_leg "$@" ;;
esac
