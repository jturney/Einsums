#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The hardware cost facts, as Python sees them.

These are bindings over measurements, so the assertions are about shape and
internal consistency rather than values: what a cost model reading them is
entitled to assume.
"""

import json
import os
import subprocess
import sys

import pytest

from einsums import hardware as hw


def test_region_cost_is_a_nonnegative_number():
    cost = hw.omp_region_cost_ns()
    assert isinstance(cost, float)
    assert cost >= 0.0
    # An empty parallel region is microseconds, not milliseconds. A value this
    # large means the measurement ran against a loaded machine and every
    # threshold derived from it is nonsense.
    assert cost < 1e7


def test_thresholds_are_nonnegative():
    assert hw.omp_min_parallel_elements() >= 0
    assert hw.omp_min_parallel_flops() >= 0


def test_thresholds_follow_the_region_cost():
    """Both thresholds are derived from the region cost, so they vanish together.

    A serial build or a single-threaded run measures no region cost, and then
    there is no work size below which parallelizing loses - because nothing is
    parallelized either way. Any nonzero threshold with a zero region cost would
    mean a hardcoded constant had crept back in.
    """
    cost = hw.omp_region_cost_ns()
    elements = hw.omp_min_parallel_elements()
    flops = hw.omp_min_parallel_flops()

    if cost == 0.0:
        assert elements == 0
        assert flops == 0
    else:
        assert elements > 0
        assert flops > 0
        # Same derivation, converted at different rates: flops is the more
        # conservative of the two, so it never sits below the element count.
        assert flops >= elements


def test_values_are_stable_within_a_process():
    """Detected once and cached, so a cost model can call them in a loop."""
    assert hw.omp_region_cost_ns() == hw.omp_region_cost_ns()
    assert hw.omp_min_parallel_elements() == hw.omp_min_parallel_elements()
    assert hw.omp_min_parallel_flops() == hw.omp_min_parallel_flops()


def _in_subprocess(expression, env):
    """``expression`` evaluated by a fresh process, which reads the profile anew."""
    out = subprocess.run(
        [sys.executable, "-c",
         f"import json; from einsums import hardware as hw; print(json.dumps({expression}))"],
        capture_output=True, text=True, check=True,
        env={**os.environ, **env},
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def _isolated(tmp_path, **extra):
    """An environment that sees only the profiles a test writes.

    The developer's own cache directory and the installation's site directory
    would otherwise decide what a test sees.
    """
    cache = tmp_path / "cache"
    site = tmp_path / "site"
    cache.mkdir(exist_ok=True)
    site.mkdir(exist_ok=True)
    env = {"EINSUMS_CACHE_DIR": str(cache), "EINSUMS_HARDWARE_SITE_PROFILES": str(site),
           "EINSUMS_HARDWARE_PROFILE": "", "OMP_NUM_THREADS": "2"}
    env.update(extra)
    return env


def _profile(**changes):
    """A profile for this machine and build, with the region cost at 2 and 4 threads.

    ``changes`` replaces top-level sections, or adds them.
    """
    document = json.loads(hw.profile_header_json())
    document["measured"] = {"date": "2026-10-12"}
    document["omp_region_cost_ns"] = {"1": 0.0, "2": 4242.0, "4": 8484.0}
    document.update(changes)
    return document


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document) if not isinstance(document, str) else document)
    return path


def _cached(tmp_path):
    """Where the cache directory of :func:`_isolated` files this machine's profile."""
    return tmp_path / "cache" / "profiles" / f"{hw.machine_fingerprint()}.json"


def _software(**changes):
    software = json.loads(hw.profile_header_json())["software"]
    software.update(changes)
    return software


def test_a_profiled_value_is_used_and_is_identical_across_processes(tmp_path):
    """The point of a profile: same machine, same team, same number.

    Measured per process the region cost drifts by tens of percent with whatever
    else the machine is doing, which is harmless for a threshold and not harmless
    for a chooser ranking discrete options against the rate.
    """
    _write(_cached(tmp_path), _profile())
    env = _isolated(tmp_path)
    assert [_in_subprocess("hw.omp_region_cost_ns()", env) for _ in range(3)] == [4242.0] * 3
    assert _in_subprocess("hw.profile_state()", env) == "loaded"
    assert _in_subprocess("hw.profile_path()", env) == str(_cached(tmp_path))


def test_the_entry_matching_this_team_size_is_the_one_used(tmp_path):
    """A run at four threads must not be handed the two-thread number."""
    _write(_cached(tmp_path), _profile())
    assert _in_subprocess("hw.omp_region_cost_ns()", _isolated(tmp_path, OMP_NUM_THREADS="4")) == 8484.0


def test_a_team_size_with_no_entry_falls_back_to_measuring(tmp_path):
    """A profile is still useful for the team sizes it does cover."""
    _write(_cached(tmp_path), _profile())
    env = _isolated(tmp_path, OMP_NUM_THREADS="3")
    assert _in_subprocess("hw.omp_region_cost_ns()", env) not in (4242.0, 8484.0)
    assert not _in_subprocess("hw.region_cost_is_calibrated()", env)


def test_an_explicit_pin_wins(tmp_path):
    """So a benchmark can hold the rate fixed without touching any file."""
    _write(_cached(tmp_path), _profile())
    env = _isolated(tmp_path, EINSUMS_HARDWARE_OMP_REGION_COST_NS="12345")
    assert _in_subprocess("hw.omp_region_cost_ns()", env) == 12345.0


def test_the_site_directory_is_searched_after_the_cache(tmp_path):
    """An administrator's profile serves every user who has not measured their own."""
    site = tmp_path / "site" / f"{hw.machine_fingerprint()}.json"
    _write(site, _profile(omp_region_cost_ns={"2": 1111.0}))
    env = _isolated(tmp_path)
    assert _in_subprocess("hw.omp_region_cost_ns()", env) == 1111.0
    assert _in_subprocess("hw.profile_path()", env) == str(site)

    # The user's own measurement takes precedence over the site's.
    _write(_cached(tmp_path), _profile())
    assert _in_subprocess("hw.omp_region_cost_ns()", env) == 4242.0


def test_a_named_profile_is_used_in_place_of_the_search(tmp_path):
    named = _write(tmp_path / "shared.json", _profile(omp_region_cost_ns={"2": 2222.0}))
    _write(_cached(tmp_path), _profile())
    env = _isolated(tmp_path, EINSUMS_HARDWARE_PROFILE=str(named))
    assert _in_subprocess("hw.omp_region_cost_ns()", env) == 2222.0
    assert _in_subprocess("hw.profile_path()", env) == str(named)


def _fingerprint(machine):
    """The library's fingerprint, recomputed: FNV-1a of the fields' canonical JSON."""
    fields = {key: value for key, value in machine.items() if key != "fingerprint"}
    text = json.dumps(fields, separators=(",", ":"), ensure_ascii=False).encode()
    value = 0xcbf29ce484222325
    for byte in text:
        value = ((value ^ byte) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _foreign_machine():
    """This machine with more cores: another kind of machine, consistently fingerprinted."""
    machine = json.loads(hw.profile_header_json())["machine"]
    machine["logical_cores"] += 64
    machine["fingerprint"] = _fingerprint(machine)
    return machine


def test_the_fingerprint_is_the_hash_of_the_machine_fields():
    """Pins the recomputation the foreign-machine cases rely on."""
    machine = json.loads(hw.profile_header_json())["machine"]
    assert _fingerprint(machine) == machine["fingerprint"] == hw.machine_fingerprint()


def test_a_profile_from_another_machine_is_skipped(tmp_path):
    """Two kinds of machine never share numbers by accident."""
    _write(_cached(tmp_path), _profile(machine=_foreign_machine()))
    env = _isolated(tmp_path)
    assert _in_subprocess("hw.profile_state()", env) == "foreign"
    assert "logical cores" in _in_subprocess("hw.profile_reason()", env)
    assert not _in_subprocess("hw.region_cost_is_calibrated()", env)


def test_a_named_profile_from_another_machine_is_used_with_a_warning(tmp_path):
    """A user who names a file means it."""
    named = _write(tmp_path / "theirs.json", _profile(machine=_foreign_machine()))
    env = _isolated(tmp_path, EINSUMS_HARDWARE_PROFILE=str(named))
    assert _in_subprocess("hw.profile_state()", env) == "loaded-foreign"
    assert _in_subprocess("hw.omp_region_cost_ns()", env) == 4242.0


def test_an_edited_machine_section_makes_a_profile_invalid(tmp_path):
    """A fingerprint that disagrees with the fields means the file was edited by hand."""
    machine = json.loads(hw.profile_header_json())["machine"]
    machine["memory_gib"] += 1
    _write(_cached(tmp_path), _profile(machine=machine))
    env = _isolated(tmp_path)
    assert _in_subprocess("hw.profile_state()", env) == "invalid"
    assert "fingerprint" in _in_subprocess("hw.profile_reason()", env)


def test_another_einsums_minor_version_makes_a_profile_stale(tmp_path):
    """Kernels change between minor versions, so their measurements do too."""
    major, minor, _ = (int(part) for part in _software()["einsums"].split("."))
    _write(_cached(tmp_path), _profile(software=_software(einsums=f"{major}.{minor + 1}.0")))
    env = _isolated(tmp_path)
    assert _in_subprocess("hw.profile_state()", env) == "stale"
    assert "Einsums" in _in_subprocess("hw.profile_reason()", env)
    assert not _in_subprocess("hw.region_cost_is_calibrated()", env)


def test_another_patch_version_still_applies(tmp_path):
    """A patch release keeps its kernels, so a profile survives it."""
    major, minor, patch = (int(part) for part in _software()["einsums"].split("."))
    _write(_cached(tmp_path), _profile(software=_software(einsums=f"{major}.{minor}.{patch + 7}")))
    assert _in_subprocess("hw.profile_state()", _isolated(tmp_path)) == "loaded"


def test_another_blas_makes_a_profile_stale(tmp_path):
    """Swapping the BLAS behind an environment changes every GEMM number."""
    _write(_cached(tmp_path), _profile(software=_software(blas="SomeOtherBLAS 9.9")))
    env = _isolated(tmp_path)
    assert _in_subprocess("hw.profile_state()", env) == "stale"
    assert "SomeOtherBLAS" in _in_subprocess("hw.profile_reason()", env)


def test_a_stale_cache_profile_does_not_hide_a_current_site_profile(tmp_path):
    """A profile that does not apply is skipped, not a dead end."""
    _write(_cached(tmp_path), _profile(software=_software(blas="SomeOtherBLAS 9.9")))
    _write(tmp_path / "site" / f"{hw.machine_fingerprint()}.json", _profile(omp_region_cost_ns={"2": 3333.0}))
    env = _isolated(tmp_path)
    assert _in_subprocess("hw.profile_state()", env) == "loaded"
    assert _in_subprocess("hw.omp_region_cost_ns()", env) == 3333.0


def test_an_unknown_section_makes_a_profile_invalid(tmp_path):
    """Strict like the graph IR: a section this build does not read is not skipped silently."""
    _write(_cached(tmp_path), _profile(options={"einsums:graph:tiling-budget-share": 0.2}))
    env = _isolated(tmp_path)
    assert _in_subprocess("hw.profile_state()", env) == "invalid"
    assert "'options'" in _in_subprocess("hw.profile_reason()", env)


def test_another_format_makes_a_profile_invalid(tmp_path):
    _write(_cached(tmp_path), _profile(einsums_profile=99))
    assert _in_subprocess("hw.profile_state()", _isolated(tmp_path)) == "invalid"


def test_a_corrupt_profile_is_survived(tmp_path):
    """Anything unreadable means measure: the file is never load-bearing."""
    _write(_cached(tmp_path), "{ this is not json")
    env = _isolated(tmp_path)
    assert _in_subprocess("hw.profile_state()", env) == "invalid"
    assert _in_subprocess("hw.omp_region_cost_ns()", env) >= 0.0


def test_no_profile_is_survived_and_none_is_written(tmp_path):
    """The library reads profiles. Only einsums-tune writes them."""
    env = _isolated(tmp_path)
    assert _in_subprocess("hw.profile_state()", env) == "none"
    assert _in_subprocess("hw.omp_region_cost_ns()", env) >= 0.0
    assert not any((tmp_path / "cache").rglob("*")), "the library must never write a profile"


def test_a_missing_named_profile_says_so(tmp_path):
    env = _isolated(tmp_path, EINSUMS_HARDWARE_PROFILE=str(tmp_path / "absent.json"))
    assert _in_subprocess("hw.profile_state()", env) == "invalid"
    assert "does not exist" in _in_subprocess("hw.profile_reason()", env)


def test_a_single_thread_counts_as_calibrated(tmp_path):
    """No region to enter means the cost is exactly zero, not an estimate of it."""
    assert _in_subprocess("hw.region_cost_is_calibrated()", _isolated(tmp_path, OMP_NUM_THREADS="1"))


def test_the_profile_paths_are_reported(tmp_path):
    """So a user can find, inspect or delete a profile without guessing."""
    env = _isolated(tmp_path)
    assert _in_subprocess("hw.default_profile_path()", env) == str(_cached(tmp_path))
    assert _in_subprocess("hw.profile_search_path()", env) == [
        str(_cached(tmp_path)), str(tmp_path / "site" / f"{hw.machine_fingerprint()}.json")]


def test_the_fingerprint_is_stable_and_short():
    """The name a profile is filed under, the same in every process."""
    fingerprint = hw.machine_fingerprint()
    assert len(fingerprint) == 16
    int(fingerprint, 16)
    assert _in_subprocess("hw.machine_fingerprint()", {}) == fingerprint


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
