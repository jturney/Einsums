# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""``einsums info``, ``einsums options`` and ``einsums completion``."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from einsums import _core, rc
from einsums.cli import main
from einsums.cli.completion import bash, command_tree, fish


def test_info_reports_the_build_and_machine(capsys):
    assert main(["info", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["einsums"]["version"] == _core._version.full_version_as_string()
    assert "EINSUMS_WITH_PROFILER" in info["einsums"]["options"]
    assert info["cpu"]["logical_cores"] >= 1
    assert set(info["python"]["optional"]) >= {"numpy", "textual"}
    assert main(["info"]) == 0
    assert "Einsums" in capsys.readouterr().out


def test_options_lists_the_registry_with_its_spellings():
    # In a fresh interpreter, as `einsums options` runs: the runtime is never started,
    # so the registry must know each environment variable before initialize().
    # (Regression: the EINSUMS_ prefix was set only at start-up, and every name came back empty.)
    out = subprocess.run(
        [sys.executable, "-m", "einsums", "options", "--json"], capture_output=True, text=True, check=True
    ).stdout
    listed = {o["name"]: o for o in json.loads(out)}
    assert set(listed) == {o["name"] for o in _core._registered_options()}
    level = listed["einsums:log:level"]
    assert level["flag"] == "--einsums:log:level" and level["env"] == "EINSUMS_LOG_LEVEL"
    assert level["rc"] == "einsums.rc.log_level" and hasattr(rc, "log_level")
    assert all(o["env"].startswith("EINSUMS_") for o in listed.values())


def test_options_filter_and_no_match(capsys):
    assert main(["options", "profile:port"]) == 0
    out = capsys.readouterr().out
    assert "--einsums:profile:port" in out and "--einsums:log:level" not in out
    assert main(["options", "no-such-option-anywhere"]) == 1


def test_completion_covers_forwarded_commands():
    tree = command_tree()
    assert {"profiler", "bench", "stages", "info", "options", "completion"} <= set(tree[()][0])
    assert "report" in tree[("profiler",)][0] and "--fail-above" in tree[("profiler", "diff")][1]
    assert "promote" in tree[("stages",)][0] and "--jobs" in tree[("bench", "run")][1]
    assert "complete -c einsums" in fish(tree)


def find_bash() -> str | None:
    """A bash that runs scripts. On Windows, System32's bash.exe launches WSL, which fails where no
    Linux distribution is installed (as on CI's runners), so Git's bash, also on PATH, is taken."""
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for directory in os.get_exec_path():
        found = shutil.which("bash", path=directory)
        if found is not None and not (sys.platform == "win32" and Path(found).parent.resolve() == system32.resolve()):
            return found
    return None


@pytest.mark.skipif(find_bash() is None, reason="needs bash")
def test_bash_completion_completes():
    script = bash(command_tree())
    shell = find_bash()

    # The script goes to bash on stdin, as bytes: a Windows temp path pasted into the command
    # loses its backslashes to bash's quoting, and a text-mode write turns every LF into CRLF,
    # which bash reads as part of each command.
    def complete(*words):
        line = " ".join(["einsums", *words])
        probe = (
            f"{script}\nCOMP_WORDS=({line}); COMP_CWORD={len(words)}; "
            '_einsums_complete; printf "%s\\n" "${COMPREPLY[@]}"\n'
        )
        result = subprocess.run([shell, "-s"], input=probe.encode(), capture_output=True)
        assert result.returncode == 0, f"{shell} failed: {result.stderr.decode(errors='replace')}"
        return result.stdout.decode().split()

    assert "bench" in complete("b") and "profiler" in complete("p")
    assert complete("bench", "tr") == ["trend"]
    assert "--fail-above" in complete("profiler", "diff", "--fa")
    assert complete("stages", "pro") == ["promote"]
