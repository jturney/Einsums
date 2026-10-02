# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""``einsums info``, ``einsums options`` and ``einsums completion``."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

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


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_bash_completion_completes(tmp_path):
    script = tmp_path / "einsums.bash"
    script.write_text(bash(command_tree()))

    def complete(*words):
        line = " ".join(["einsums", *words])
        probe = (
            f"source {script}; COMP_WORDS=({line}); COMP_CWORD={len(words)}; "
            '_einsums_complete; printf "%s\\n" "${COMPREPLY[@]}"'
        )
        return subprocess.run(["bash", "-c", probe], capture_output=True, text=True, check=True).stdout.split()

    assert "bench" in complete("b") and "profiler" in complete("p")
    assert complete("bench", "tr") == ["trend"]
    assert "--fail-above" in complete("profiler", "diff", "--fa")
    assert complete("stages", "pro") == ["promote"]
