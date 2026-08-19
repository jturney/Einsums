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
#     ./devtools/docker/run-ci-leg.sh gcc-openblas-coverage # ...as CI actually runs it: +COVERAGE (static libEinsums), +Python
#     ./devtools/docker/run-ci-leg.sh gcc-mkl              # Linux/default/mkl RelWithDebInfo
#     ./devtools/docker/run-ci-leg.sh clang-openblas       # Linux/clang/openblas RelWithDebInfo
#     ./devtools/docker/run-ci-leg.sh no-profiler          # EINSUMS_WITH_PROFILER=OFF (BUILD_PYTHON=ON)
#     ./devtools/docker/run-ci-leg.sh tsan                 # Sanitizers/thread (Debug, BUILD_PYTHON=ON)
#     ./devtools/docker/run-ci-leg.sh asan                 # Sanitizers/address,leak,undefined (Debug, BUILD_PYTHON=ON)
#     ./devtools/docker/run-ci-leg.sh asan-nopy            # same without Python, if the pybind TUs exhaust memory
#     ./devtools/docker/run-ci-leg.sh free-threaded        # free-threaded CPython (cp314t), BUILD_PYTHON=ON
#     ./devtools/docker/run-ci-leg.sh valgrind             # memcheck under valgrind (Debug, BUILD_PYTHON=OFF)
#
#     # Valgrind runs each test 20-50x slower, so the whole suite is an
#     # overnight proposition. Narrow it:
#     ./devtools/docker/run-ci-leg.sh valgrind -- -L UNIT_ONLY
#     ./devtools/docker/run-ci-leg.sh valgrind -- -R "Modules.Tensor"
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
# IF `start` HANGS: the `docker pull` below consults Docker Desktop's
# credential helper, and `docker-credential-desktop get` can block indefinitely
# even for a public image that is already cached locally. It looks like a slow
# network and is not. Run with a config that has no credential store:
#
#     mkdir -p /tmp/dockercfg && echo '{}' > /tmp/dockercfg/config.json
#     DOCKER_CONFIG=/tmp/dockercfg ./devtools/docker/run-ci-leg.sh start arm64
#
# Nothing here needs authentication, so dropping the helper costs nothing.
# Prefer this over editing ~/.docker/config.json, which is shared with every
# other Docker use on the machine.
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
    #
    # Extra args for merge_yml.py. Legs that need a different conda env than the
    # plain (compiler, blas) pair set this; it also namespaces the env, see
    # ENV_NAME in run_leg. Reset here so it never leaks between calls.
    MERGE_EXTRA=""
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
        gcc-openblas-coverage)
            # The REAL shape of CI's Linux/default/openblas leg, which carries
            # -DEINSUMS_WITH_COVERAGE=ON from an `include:` entry in
            # linux-build-and-test.yml that is easy to miss when reading the
            # matrix. Neither gcc-openblas nor gcc-openblas-py reproduced it, so
            # "run the CI leg locally" quietly tested a different build than the
            # one that was failing.
            #
            # Coverage is not just extra flags: libs/CMakeLists.txt selects a
            # STATIC libEinsums when it is on, which folds a private copy of the
            # library's state into every Python extension module. That is a
            # different program, and it is the one CI runs.
            COMPILER=default
            BLAS=openblas
            BUILD_TYPE=RelWithDebInfo
            EXTRA=("-DEINSUMS_BUILD_PYTHON=ON" "-DEINSUMS_WITH_COVERAGE=ON")
            ;;
        gcc-mkl)
            COMPILER=default
            BLAS=mkl
            BUILD_TYPE=RelWithDebInfo
            EXTRA=("-DEINSUMS_BUILD_PYTHON=OFF")
            ;;
        intel)
            # Intel oneAPI icx/icpx (dpcpp_linux-64) + MKL. merge_yml.py pulls
            # the pinned dpcpp_linux-64 (currently 2026.x, conda-forge
            # intel-compiler-repack) and forces BLAS=mkl; the conda activation
            # sets CC=icx / CXX=icpx, which CMake picks up. Intel is x86_64-only,
            # so this leg has no -arm64 variant (and on Apple silicon it runs
            # under Rosetta -- fine to compile, but AVX-512 codegen won't execute
            # under emulation, so use a real x86_64 host to run the tests).
            COMPILER=intel
            BLAS=mkl
            BUILD_TYPE=Debug
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
        no-profiler)
            # EINSUMS_WITH_PROFILER=OFF replaces the instrumentation API with
            # no-op shims. Python stays ON: the einsums.profile bindings are
            # generated from the shims, so leaving it OFF would skip the half
            # of the configuration most likely to break.
            COMPILER=clang
            BLAS=openblas
            BUILD_TYPE=RelWithDebInfo
            EXTRA=("-DEINSUMS_WITH_PROFILER=OFF" "-DEINSUMS_BUILD_PYTHON=ON")
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
            # Python ON: the bindings are the one part of the tree with no
            # memory-error coverage otherwise, since the tsan leg builds them
            # for races and nothing else builds them under a sanitizer at all.
            # einsums_add_python_unit_test already preloads the sanitizer
            # runtime for the test process (EINSUMS_PYTEST_SANITIZER_PRELOAD)
            # and disables LSan there, CPython leaking at shutdown by design.
            COMPILER=default
            BLAS=openblas
            BUILD_TYPE=Debug
            EXTRA=("-DEINSUMS_WITH_SANITIZERS=address,leak,undefined" "-DEINSUMS_BUILD_PYTHON=ON")
            ;;
        asan-nopy)
            # The variant to reach for if the instrumented pybind TUs exhaust
            # the container's memory; mirrors tsan-nopy.
            COMPILER=default
            BLAS=openblas
            BUILD_TYPE=Debug
            EXTRA=("-DEINSUMS_WITH_SANITIZERS=address,leak,undefined" "-DEINSUMS_BUILD_PYTHON=OFF")
            ;;
        free-threaded)
            # Free-threaded CPython (cp314t), mirroring the CI leg of the same
            # name. Python must be ON -- the whole point is the bindings.
            #
            # The Python_FIND_ABI 4th field is the free-threaded flag (CMake
            # 3.30+); without it FindPython refuses the t-build and silently
            # falls back to another interpreter, which looks like a pass while
            # testing nothing. MERGE_EXTRA gets this leg its own conda env, so
            # it does not inherit the regular-ABI interpreter from the shared
            # gcc-openblas env.
            #
            # After the normal ctest, the runner re-runs the PYTHON label with
            # PYTHON_GIL=0; see step 5 below for why that second pass matters.
            COMPILER=default
            BLAS=openblas
            BUILD_TYPE=RelWithDebInfo
            MERGE_EXTRA="--free-threaded"
            # The single quotes around the ABI list are load-bearing: EXTRA is
            # flattened into a plain string and interpolated into the `docker
            # exec bash -lc "..."` command, so bare semicolons would be read as
            # command separators by the container's shell. \${CONDA_PREFIX} is
            # escaped for the same reason -- it must reach the container
            # unexpanded and be resolved there, after conda activate.
            EXTRA=("-DEINSUMS_BUILD_PYTHON=ON"
                   "-DPython_FIND_ABI='OFF;ANY;ANY;ON'"
                   "-DPython_ROOT_DIR=\${CONDA_PREFIX}"
                   "-DPython_FIND_STRATEGY=LOCATION")
            ;;
        valgrind)
            # Memcheck. Debug rather than RelWithDebInfo on purpose: valgrind
            # reports uninitialised values in terms of what the optimizer
            # emitted, so -O2 yields both confusing line numbers and reports
            # with no source-level cause. -O0 costs build time and buys signal.
            #
            # Python is off. The python tests are registered with add_test
            # directly (einsums_add_python_unit_test), not through
            # einsums_add_test, so they never get the valgrind wrapper - they
            # would only add runtime. Running CPython itself under memcheck
            # additionally needs Python's own suppression file to be readable.
            #
            # The options are one CMake list: --error-exitcode makes a finding
            # fail the test rather than merely printing, and leak checking is
            # limited to definite losses because OpenMP and BLAS runtimes leave
            # still-reachable pools behind by design. The suppression path is
            # the in-container source tree, i.e. SRC_DIR below.
            COMPILER=default
            BLAS=openblas
            BUILD_TYPE=Debug
            # Single quotes around the semicolon-separated list are load-bearing;
            # see the free-threaded leg for why.
            EXTRA=("-DEINSUMS_WITH_TESTS_VALGRIND=ON"
                   "-DEINSUMS_BUILD_PYTHON=OFF"
                   "-DEINSUMS_WITH_TESTS_VALGRIND_OPTIONS='--error-exitcode=1;--leak-check=full;--errors-for-leak-kinds=definite;--suppressions=/work/src/devtools/sanitizers/valgrind.supp'")
            ;;
        *)
            echo "Unknown leg: $1" >&2
            echo "Valid: gcc-openblas[-py], gcc-mkl[-py], clang-openblas[-py], intel, no-profiler, tsan, tsan-nopy, asan, asan-nopy, free-threaded, valgrind, windows-cross" >&2
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
    # Legs that pass extra merge_yml.py args resolve a DIFFERENT dependency set
    # (the free-threaded leg swaps the CPython ABI), so they must not share the
    # plain (compiler, blas) env - reusing it would silently build against the
    # regular interpreter and pass while testing nothing.
    local ENV_NAME="einsums-env-${COMPILER}-${BLAS}"
    if [[ -n "${MERGE_EXTRA}" ]]; then
        ENV_NAME="${ENV_NAME}-$(echo "${MERGE_EXTRA}" | tr -d ' -')"
    fi
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
    python /tmp/merge/merge_yml.py ${MERGE_EXTRA} '${COMPILER}' '${BLAS}' --output /tmp/env-${LEG}.yml
    mamba env create -f /tmp/env-${LEG}.yml -n '${ENV_NAME}' -y >/dev/null
fi
conda activate '${ENV_NAME}'

export CCACHE_DIR='${CCACHE_DIR}'
export CCACHE_MAXSIZE=5G
mkdir -p '${CCACHE_DIR}'

# 2a. OpenBLAS picks a tuned aarch64 kernel whose sdot is wrong in this
#     container, so LinearAlgebra.dot, TensorAlgebra.Dot and GPU.Runtime fail
#     against a library that is perfectly fine. Asking for the baseline ARMv8
#     kernel avoids it. x86_64 needs none of this, which is also why amd64 is
#     the faithful target and arm64 is for triage.
#
#     This was first seen under valgrind and blamed on its aarch64 emulation,
#     so the guard used to be valgrind-only. It is not: a plain
#     gcc-openblas-arm64 run fails the same three tests, and setting this makes
#     all three pass. Under valgrind the same wrongness additionally reads as
#     an uninitialised value, which is what made it look emulation-specific.
if [[ \$(uname -m) == 'aarch64' ]]; then
    export OPENBLAS_CORETYPE=ARMV8
fi

# 2b. valgrind is not in the miniforge image, and Einsums_AddTest does
#     find_program(valgrind REQUIRED) at configure time when the option is on,
#     so install it before cmake runs rather than before ctest. Done here and
#     not at container creation so existing containers pick it up without
#     being recreated, and so the other legs do not carry it.
if [[ '${LEG}' == 'valgrind' ]] && ! command -v valgrind >/dev/null; then
    echo '⤷ installing valgrind (first time for this container)'
    apt-get update -qq >/dev/null && apt-get install -y -qq valgrind >/dev/null
fi

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
#
#    The coverage leg belongs in the same category: --coverage inflates the
#    ComputeGraph pybind TU the same way the sanitizers do, and at -j2 it was
#    OOM-killed outright -- cc1plus terminated by signal -- even against a
#    20GB allowance. Note this whole block is inside a double-quoted heredoc,
#    so a comment here must not contain a double quote.
#
#    The ComputeGraph half of that story is now smaller: its binding is
#    generated as 8 shard TUs (PYBIND_NUM_TU in its CMakeLists), which took
#    the worst instrumented shard from 7.5GB to 4.7GB measured with clang.
#    Two of those still overrun an 8GB allowance in the worst case, so the
#    cap stays -- but the cliff it was guarding against is gone, and raising
#    the allowance is now enough to lift it.
if [[ '${LEG}' == 'asan' || '${LEG}' == 'tsan' || '${LEG}' == 'tsan-nopy' || '${LEG}' == 'gcc-openblas-coverage' ]]; then
    cmake --build '${BUILD_DIR}' -j1
else
    cmake --build '${BUILD_DIR}' -j2
fi

# 5. Run tests.
cd '${BUILD_DIR}'
ctest --output-on-failure ${CTEST_EXTRA_STR}

# 5b. Free-threaded leg only: the run above imports einsums with the GIL
#     auto-re-enabled, because _core does not declare Py_MOD_GIL_NOT_USED and
#     CPython falls back for any such module (it warns when it does). That
#     proves the tree works against the 't' ABI but exercises no free-threading
#     at all. PYTHON_GIL=0 overrides the fallback. The assert is after the
#     import on purpose: if loading _core flipped the GIL back on, this pass
#     would silently duplicate the one above instead of failing.
if [[ '${LEG}' == 'free-threaded' ]]; then
    echo '⤷ re-running the PYTHON label with the GIL disabled'
    # The interpreter CMake resolved, not PATH's: FindPython picks the
    # t-build via Python_FIND_ABI, but the env's default python can be the
    # regular-ABI one, where PYTHON_GIL=0 aborts rather than being ignored.
    FT_PY=\$(grep '^Python_EXECUTABLE:' CMakeCache.txt | cut -d= -f2-)
    PYTHONPATH='${BUILD_DIR}/lib' PYTHON_GIL=0 \"\${FT_PY}\" -c \"
import sys, einsums
assert not sys._is_gil_enabled(), 'importing einsums re-enabled the GIL'
print('GIL still disabled after importing einsums')
\"
    PYTHON_GIL=0 ctest -L PYTHON --output-on-failure ${CTEST_EXTRA_STR}
fi
"
}

# ──────────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    sed -n '2,30p' "$0" >&2
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────────
# windows-cross: compile+link validation for x86_64-pc-windows-msvc.
# Self-contained (own image with clang-cl/lld-link + xwin CRT/SDK + conda
# win-64 libraries); does NOT use the conda-leg container. Cannot RUN tests -
# there is no Windows kernel - the GitHub windows runner stays the source of
# truth for test results. Everything after `--` is passed to ninja.
run_windows_cross() {
    shift # drop leg name
    local image=einsums-windows-cross
    local ctx="$REPO_ROOT/devtools/docker/windows-cross"

    if ! docker image inspect "$image" >/dev/null 2>&1; then
        echo ">> building $image (first run: downloads the MSVC CRT/SDK, ~10 min)"
        docker build -t "$image" "$ctx"
    fi

    local ninja_args=()
    if [[ "${1:-}" == "--" ]]; then
        shift
        ninja_args=("$@")
    fi

    docker run --rm \
        -v "$REPO_ROOT":/src:ro \
        -v einsums-wincross-build:/wb \
        "$image" sh -c "
            cmake -S /src -B /wb/build -GNinja \
                -DCMAKE_TOOLCHAIN_FILE=/src/devtools/docker/windows-cross/toolchain-clang-cl.cmake \
                -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                -DEINSUMS_BUILD_PYTHON=OFF \
                -DEINSUMS_WITH_TESTS=ON \
                -DEINSUMS_WITH_BACKTRACES=OFF \
                > /wb/configure.log 2>&1 || { tail -40 /wb/configure.log; exit 1; }
            ninja -C /wb/build ${ninja_args[*]:-}
        "
}

case "$1" in
    start) shift; cmd_start "${1:-amd64}" ;;
    stop)  shift; cmd_stop  "${1:-amd64}" ;;
    windows-cross) run_windows_cross "$@" ;;
    *)     run_leg "$@" ;;
esac
