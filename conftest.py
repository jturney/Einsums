# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Root pytest configuration, loaded for every test in the tree.

Holds only what has to happen before any test runs, in any suite.
"""

import os
import secrets
import sys

import pytest

# Re-arm the sanitizer preload for subprocesses.
#
# On macOS dyld honors DYLD_INSERT_LIBRARIES and then REMOVES it from the
# environment the process can read, so a pytest run that ctest correctly
# preloaded cannot pass the preload on: a subprocess it spawns inherits an
# environment with no DYLD_INSERT_LIBRARIES, dlopens the instrumented
# extension with no runtime beneath it, and dies with
#
#     ERROR: Interceptors are not working. This may be because
#     AddressSanitizer is loaded too late (e.g. via dlopen).
#
# on its first intercepted call - which for this library is any OpenMP region,
# so it reaches tests that merely ask a subprocess what it measured. The build
# hands us the same path a second time under a name dyld does not scrub; put
# the real one back so children inherit it.
#
# Inert everywhere else: without a sanitizer build the carrier is unset, and on
# Linux LD_PRELOAD survives exec on its own, so the setdefault does nothing.
_preload = os.environ.get("EINSUMS_SANITIZER_PRELOAD")
if _preload:
    os.environ.setdefault(
        "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD",
        _preload,
    )

# Turn a hung test into a stack trace instead of a stopwatch reading.
#
# The DLPNO triples suites hang in CI perhaps one run in three, on Linux/mkl
# and Windows, while every other test on the same runner goes FASTER than on
# runs where they pass - so it is a hang, not contention. ctest reports it as
# "Timeout 1500 sec" and discards the test's output when it kills it, which
# says nothing about where the process stopped. 68 local runs across four
# thread and load configurations did not reproduce it, so the information has
# to come from the machine that does.
#
# faulthandler dumps every thread's Python stack, which is what identifies a
# deadlock: the C++ frames are absent but the thread that is waiting, and the
# call it is waiting in, are not. repeat=True because ONE dump cannot tell a
# deadlock from slow progress and two, a few minutes apart, can - identical
# stacks mean stuck.
#
# To a file rather than stderr: the whole problem is that the output of a
# killed test does not survive. The workflows already collect crash dumps, so
# this lands somewhere collectable. Off by setting the timeout to 0.
_stackdump_seconds = float(os.environ.get("EINSUMS_TEST_STACKDUMP_SECONDS", "600"))
if _stackdump_seconds > 0:
    import faulthandler
    import tempfile

    _stackdump_dir = os.environ.get("EINSUMS_TEST_STACKDUMP_DIR") or tempfile.gettempdir()
    try:
        os.makedirs(_stackdump_dir, exist_ok=True)
        # Kept open for the life of the process: faulthandler writes through
        # this object from its timer thread, and a closed file would take the
        # dump with it. Line buffered so a dump survives the kill that follows.
        _stackdump_file = open(  # noqa: SIM115
            os.path.join(_stackdump_dir, f"einsums-stackdump-{os.getpid()}.txt"),
            "w",
            buffering=1,
        )
    except OSError:
        # A read-only or missing dump directory is not a reason to fail a test
        # run; the dump is a diagnostic, not a subject.
        _stackdump_file = None
    if _stackdump_file is not None:
        # Say whose file this is. A process that dies without hanging never
        # reaches the cleanup below, so its file is left behind, and without a
        # header it would be an empty file that names no test. With one, a file
        # holding only the header says "this run died abruptly": look for its
        # report in that test's ctest output.
        _stackdump_file.write(f"# pid {os.getpid()}: {' '.join(sys.argv)}\n")
        _stackdump_header_size = _stackdump_file.tell()
        faulthandler.dump_traceback_later(
            _stackdump_seconds, repeat=True, file=_stackdump_file, exit=False
        )

        import atexit

        @atexit.register
        def _drop_empty_stackdump() -> None:
            """Leave a file behind only when there was something to say.

            Every python test process arms this, and nearly all of them finish
            long before the timer fires. Keeping their empty files would bury
            the one dump that matters under hundreds that say nothing.
            """
            faulthandler.cancel_dump_traceback_later()
            try:
                empty = _stackdump_file.tell() == _stackdump_header_size
                _stackdump_file.close()
                if empty:
                    os.unlink(_stackdump_file.name)
            except OSError:
                pass


# Deterministic random tensors.
#
# The C++ engine behind create_random_tensor, create_random_definite and the
# randomized LAPACK-style algorithms is seeded from the clock when einsums is
# imported, so a test that fails at a tolerance edge fails by chance and cannot
# be rerun into the same state. Every test instead starts the engine from a
# seed derived from the session seed and the test's node id, the same scheme
# the C++ test main uses with Catch2's run seed and the test case name. The
# session seed is random unless given, so runs still vary, and it is printed
# in the header and beside every failure; passing it back with --einsums-seed
# (or EINSUMS_TEST_SEED under ctest) reproduces the draws of any single test,
# however the run was filtered.
#
# Tests that seed the engine themselves keep working: they reseed after this.
_SEED_ENV = "EINSUMS_TEST_SEED"


def _test_case_seed(run_seed: int, name: str) -> int:
    """32-bit FNV-1a over the seed's little-endian bytes, then the name.

    Mirrors ``einsums::testing::test_case_seed`` in the C++ test header, so a
    seed means the same thing in both suites.
    """
    value = 2166136261
    for byte in run_seed.to_bytes(4, "little") + name.encode():
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--einsums-seed",
        type=int,
        default=None,
        help=f"Session seed for the einsums random engine (default: ${_SEED_ENV}, else random).",
    )


def pytest_configure(config: pytest.Config) -> None:
    seed = config.getoption("--einsums-seed")
    if seed is None:
        from_env = os.environ.get(_SEED_ENV, "").strip()
        seed = int(from_env) if from_env else secrets.randbits(32)
    config.einsums_seed = seed & 0xFFFFFFFF


def _rerun_hint(config: pytest.Config) -> str:
    return f"einsums random seed: {config.einsums_seed} (reproduce with --einsums-seed={config.einsums_seed} or {_SEED_ENV}={config.einsums_seed})"


def pytest_report_header(config: pytest.Config) -> str:
    return _rerun_hint(config)


@pytest.fixture(autouse=True)
def _seed_einsums_random_engine(request: pytest.FixtureRequest) -> None:
    # Only seed an einsums the test has already imported: importing it here
    # would load the extension into tests that never touch it.
    einsums = sys.modules.get("einsums")
    seed_random = getattr(einsums, "seed_random", None)
    if seed_random is not None:
        seed_random(_test_case_seed(request.config.einsums_seed, request.node.nodeid))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> pytest.TestReport:
    report = yield
    if report.failed:
        report.sections.append(("einsums random seed", _rerun_hint(item.config)))
    return report


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    # The header is hidden under -q, which is how ctest runs every suite, so
    # repeat the seed where a failing run's log will show it.
    if terminalreporter.stats.get("failed") or terminalreporter.stats.get("error"):
        terminalreporter.write_line(_rerun_hint(config))


# Dtype coverage by opting in.
#
# A test that takes a ``dtype`` argument runs once per dtype in
# einsums.testing.ALL_DTYPES. A test that parametrizes ``dtype`` itself, with
# a narrower list, keeps its own list.
@pytest.fixture
def dtype(request: pytest.FixtureRequest) -> str:
    """One of einsums.testing.ALL_DTYPES; a test opts in by taking this argument."""
    return request.param


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "dtype" not in metafunc.fixturenames:
        return
    for marker in metafunc.definition.iter_markers("parametrize"):
        argnames = marker.args[0] if marker.args else marker.kwargs.get("argnames", ())
        if isinstance(argnames, str):
            argnames = [name.strip() for name in argnames.split(",")]
        if "dtype" in argnames:
            return
    # Imported here rather than at the top so the tree's pure tooling tests
    # never load the extension.
    from einsums.testing import ALL_DTYPES

    metafunc.parametrize("dtype", ALL_DTYPES, indirect=True)
