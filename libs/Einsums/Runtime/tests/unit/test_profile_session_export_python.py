# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Where the profiler session is exported, checked from outside the process.

``--einsums:profile:save`` used to be handled where the runtime hands control back, which is
after user code for a caller using ``einsums::start`` and directly after start-up for one
driving ``initialize`` and ``finalize`` itself. The second wrote a file holding nothing but the
runtime's own start-up zones, and said nothing about it.

This has to be a subprocess: the question is when in a process lifetime the export runs, and a
test binary that has already initialized the runtime cannot ask it.

``HelloWorld2`` is the fixture because it uses the entry point that was broken. The assertion is
that the session carries a zone that only exists once teardown has begun, which is the thing an
export placed after start-up cannot contain however long the program runs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

# Recorded when finalize() runs the shutdown hooks, so it exists only during teardown.
TEARDOWN_ZONE = "shutdown"

# ...and recorded through LabeledSectionInternal, like every zone the runtime places around
# start-up and teardown. With EINSUMS_WITH_PROFILER_INTERNAL off the preprocessor removes all
# of them, so HelloWorld2 records no zones at all and there is nothing for the export to be
# checked against. The build says which one this is; see the test CMakeLists.
INTERNAL_ZONES = os.environ.get("EINSUMS_TEST_PROFILER_INTERNAL") == "1"


def _binary() -> str:
    """The HelloWorld2 example, found through the build directory the harness points at.

    einsums_add_python_unit_test runs these with ``PYTHONPATH`` set to ``<build>/lib``, so the
    build tree is one directory up from there. Deriving it beats walking parents from this file,
    which finds the source tree and not the build, and skipping on a wrong guess would leave a
    test that reports green while asserting nothing.
    """
    candidates = []
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry:
            build = os.path.dirname(os.path.abspath(entry.rstrip(os.sep)))
            candidates.append(os.path.join(build, "bin", "HelloWorld2"))
    found = shutil.which("HelloWorld2")
    if found:
        candidates.append(found)
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    pytest.skip(f"HelloWorld2 was not found; looked in {candidates}")


def _zone_names(node: dict) -> list[str]:
    out = [node.get("name", "")]
    for child in node.get("children") or []:
        out += _zone_names(child)
    return out


def _run(tmp_path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_binary(), "--einsums:debug:no-attach-debugger", *extra],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.skipif(
    not INTERNAL_ZONES,
    reason="the runtime's teardown zones are LabeledSectionInternal; "
    "configure with -DEINSUMS_WITH_PROFILER_INTERNAL=ON to check where the export runs",
)
def test_session_export_covers_teardown(tmp_path):
    """The export runs during teardown, so the session reaches the shutdown zones."""
    out = tmp_path / "session.json"
    proc = _run(
        tmp_path,
        "--einsums:profile:server",
        "--einsums:profile:port=19281",
        f"--einsums:profile:save={out}",
    )
    assert proc.returncode == 0, proc.stderr

    assert out.exists(), "no session file was written"
    payload = json.loads(out.read_text())

    names = []
    for thread in payload.get("threads", {}).values():
        for child in thread.get("children") or []:
            names += _zone_names(child)
    assert names, "the session carries no zones at all"

    joined = " ".join(names).lower()
    assert TEARDOWN_ZONE in joined, (
        "the session holds no teardown zone, so it was exported before the program finished: "
        f"{sorted(set(names))}"
    )


def test_session_export_without_a_server_is_reported(tmp_path):
    """Asking for a session without a server cannot be served, and must not be silent."""
    out = tmp_path / "session.json"
    proc = _run(tmp_path, f"--einsums:profile:save={out}")

    assert proc.returncode == 0, proc.stderr
    assert not out.exists(), "a session was written without a server"

    combined = (proc.stdout + proc.stderr).lower()
    assert "profile:server" in combined, (
        "no diagnostic named the missing option; the request was refused silently. "
        f"stdout/stderr was: {combined[:400]!r}"
    )
