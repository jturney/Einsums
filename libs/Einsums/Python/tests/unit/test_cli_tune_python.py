# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""``einsums tune`` and ``python -m einsums.tune``: measuring a machine and managing its profile.

Each case runs the command in a fresh process with its own cache and site directories, as a user
would, so neither the developer's own profile nor one the case writes leaks into another.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from einsums import hardware as hw

#: Sizes small enough that a full measuring run takes a second or two. The numbers are not looked
#: at; what is checked is that a run writes a profile this machine then uses.
QUICK = ["--min-size", "16", "--max-size", "32", "--num-points", "2", "--warmup", "0", "--repeats", "1", "--no-thread-sweep"]


def _env(tmp_path: Path) -> dict[str, str]:
    for name in ("cache", "site"):
        (tmp_path / name).mkdir(exist_ok=True)
    return {**os.environ, "EINSUMS_CACHE_DIR": str(tmp_path / "cache"), "EINSUMS_HARDWARE_SITE_PROFILES": str(tmp_path / "site"),
            "EINSUMS_HARDWARE_PROFILE": "", "OMP_NUM_THREADS": "2"}


def _tune(tmp_path: Path, *args: str, module: str = "einsums.tune") -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", module, *(["tune"] if module == "einsums" else []), *args]
    return subprocess.run(command, capture_output=True, text=True, env=_env(tmp_path), timeout=300)


def _cached(tmp_path: Path, fingerprint: str | None = None) -> Path:
    return tmp_path / "cache" / "profiles" / f"{fingerprint or hw.machine_fingerprint()}.json"


def _fingerprint(machine: dict) -> str:
    """The library's fingerprint: FNV-1a of the machine fields' canonical JSON."""
    fields = {key: value for key, value in machine.items() if key != "fingerprint"}
    value = 0xcbf29ce484222325
    for byte in json.dumps(fields, separators=(",", ":"), ensure_ascii=False).encode():
        value = ((value ^ byte) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _profile(**changes) -> dict:
    document = json.loads(hw.profile_header_json())
    document["omp_region_cost_ns"] = {"2": 4242.0}
    document.update(changes)
    return document


def test_without_a_profile_the_reports_say_so(tmp_path):
    shown = _tune(tmp_path, "--show")
    assert shown.returncode == 0
    assert hw.machine_fingerprint() in shown.stdout and "none" in shown.stdout
    assert str(_cached(tmp_path)) in shown.stdout, "the search path tells a user where a profile would go"
    assert _tune(tmp_path, "--check").returncode == 1
    assert _tune(tmp_path, "--export").returncode == 1


def test_a_measured_profile_is_found_by_the_next_process(tmp_path):
    measured = _tune(tmp_path, *QUICK)
    assert measured.returncode == 0, measured.stderr
    assert _cached(tmp_path).is_file()
    document = json.loads(_cached(tmp_path).read_text())
    assert {"machine", "software", "measured", "omp_region_cost_ns", "cost_model"} <= set(document)

    checked = _tune(tmp_path, "--check")
    assert checked.returncode == 0 and checked.stdout.strip() == str(_cached(tmp_path))
    assert _tune(tmp_path, "--export").stdout.strip() == str(_cached(tmp_path))
    assert "loaded" in _tune(tmp_path, "--show").stdout


def test_einsums_tune_is_the_same_command(tmp_path):
    """`einsums tune` and `python -m einsums.tune` run the one implementation."""
    assert _tune(tmp_path, *QUICK, module="einsums").returncode == 0
    assert _tune(tmp_path, "--check", module="einsums").returncode == 0


def test_a_profile_for_this_machine_is_imported_for_it(tmp_path):
    source = tmp_path / "from-a-colleague.json"
    source.write_text(json.dumps(_profile()))
    imported = _tune(tmp_path, "--import", str(source))
    assert imported.returncode == 0 and "for this machine" in imported.stdout
    assert _cached(tmp_path).is_file()
    assert _tune(tmp_path, "--check").returncode == 0


def test_a_profile_for_another_machine_is_filed_for_that_machine(tmp_path):
    """On a cluster's shared home directory, a login node can import the compute nodes' profile."""
    machine = json.loads(hw.profile_header_json())["machine"]
    machine["logical_cores"] += 64
    machine["fingerprint"] = _fingerprint(machine)
    source = tmp_path / "compute-node.json"
    source.write_text(json.dumps(_profile(machine=machine)))

    imported = _tune(tmp_path, "--import", str(source))
    assert imported.returncode == 0 and "another kind" in imported.stdout
    assert _cached(tmp_path, machine["fingerprint"]).is_file()
    assert not _cached(tmp_path).exists()
    assert _tune(tmp_path, "--check").returncode == 1


def test_a_stale_profile_is_not_imported(tmp_path):
    software = json.loads(hw.profile_header_json())["software"]
    source = tmp_path / "old.json"
    source.write_text(json.dumps(_profile(software={**software, "blas": "SomeOtherBLAS 9.9"})))
    imported = _tune(tmp_path, "--import", str(source))
    assert imported.returncode == 1 and "not importing" in imported.stdout
    assert not (tmp_path / "cache" / "profiles").exists()


@pytest.mark.parametrize("args", [["--max-szie", "64"], ["--repeats", "many"], ["--import"], ["--dtype", "half"]])
def test_bad_arguments_are_refused_rather_than_ignored(tmp_path, args):
    """A misspelt size used to run silently with the default."""
    refused = _tune(tmp_path, *args)
    assert refused.returncode == 2 and refused.stderr.strip()
    assert not (tmp_path / "cache" / "profiles").exists()


def test_help_lists_the_actions(tmp_path):
    helped = _tune(tmp_path, "--help")
    assert helped.returncode == 0
    assert all(action in helped.stdout for action in ("--show", "--check", "--import", "--export", "--output"))
