#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Tests for ``einsums.sealed``: world identity and the stage-module handshake.

The C++ primitives are covered by ``test_world_python.py`` and the C++ ABI
suite. What is tested here is the policy layered on top: which mismatches are
refused, and whether the refusal says enough to act on.
"""

import types

import pytest

import einsums  # noqa: F401  (loads the runtime)
from einsums import sealed
from einsums.sealed import WorldMismatch


def _good_module(name="fake_stage_module"):
    """A module that looks exactly like a correctly built stage module."""
    m = types.ModuleType(name)
    m.__einsums_world__ = sealed.world_identity()
    m.__einsums_world_info__ = dict(sealed.world())
    # Registration by side effect is what the SDK macro does at import.
    einsums._core._register_stage_module(name)
    return m


# ----------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------
def test_world_reports_the_fields_a_mismatch_report_needs():
    w = sealed.world()
    for key in ("identity", "config_fingerprint", "layout_fingerprint", "version", "library_path"):
        assert key in w, f"world() must carry {key} for the error message to be actionable"
    assert w["identity"] == sealed.world_identity()


def test_mapped_libraries_includes_our_own():
    libs = sealed.mapped_einsums_libraries()
    if not libs:
        pytest.skip("this platform cannot enumerate loaded images")
    assert sealed.world()["library_path"] in libs


# ----------------------------------------------------------------------
# The handshake
# ----------------------------------------------------------------------
def test_a_correctly_built_module_is_accepted():
    m = _good_module("accepted_module")
    sealed.verify_stage_module(m)  # must not raise


def test_a_module_that_never_registered_is_refused_as_cross_world():
    """The registration side effect is the cross-world check: no entry, no load."""
    m = types.ModuleType("never_registered")
    m.__einsums_world__ = sealed.world_identity()
    m.__einsums_world_info__ = dict(sealed.world())

    with pytest.raises(WorldMismatch, match="bound to a different copy"):
        sealed.verify_stage_module(m)


def test_a_plain_module_is_refused_with_a_diagnosis():
    m = types.ModuleType("not_a_stage_module_at_all")
    with pytest.raises(WorldMismatch, match="does not look like an Einsums stage module"):
        sealed.verify_stage_module(m)


def test_a_stale_header_build_is_refused_even_in_one_world():
    """Guarantee 3: same library, module compiled against different headers."""
    m = _good_module("stale_headers")
    m.__einsums_world_info__ = dict(m.__einsums_world_info__)
    m.__einsums_world_info__["config_fingerprint"] ^= 0xDEADBEEF

    with pytest.raises(WorldMismatch, match="build configuration"):
        sealed.verify_stage_module(m)


def test_a_layout_mismatch_is_refused_and_named_separately():
    m = _good_module("stale_layouts")
    m.__einsums_world_info__ = dict(m.__einsums_world_info__)
    m.__einsums_world_info__["layout_fingerprint"] ^= 0xDEADBEEF

    with pytest.raises(WorldMismatch, match="type layout"):
        sealed.verify_stage_module(m)


def test_the_refusal_names_both_library_paths():
    """A mismatch report with two hex numbers and no paths is not actionable."""
    m = types.ModuleType("path_report")
    m.__einsums_world_info__ = {"library_path": "/somewhere/else/libEinsums.so"}

    with pytest.raises(WorldMismatch) as exc:
        sealed.verify_stage_module(m)
    text = str(exc.value)
    assert "/somewhere/else/libEinsums.so" in text
    assert sealed.world()["library_path"] in text


# ----------------------------------------------------------------------
# Host probes
# ----------------------------------------------------------------------
def test_a_host_probe_is_only_called_when_the_host_is_imported():
    calls = []

    sealed.register_host_probe("a_host_that_is_not_imported", lambda: calls.append(1) or "9.9")
    assert sealed.host_versions() == {}
    assert calls == [], "the bridge must never import a host to ask it a question"


def test_a_throwing_host_probe_does_not_break_the_caller():
    def boom():
        raise RuntimeError("host is unhappy")

    sealed.register_host_probe("sys", boom)  # sys is definitely imported
    assert "sys" not in sealed.host_versions()


# ----------------------------------------------------------------------
# load_stage_module
# ----------------------------------------------------------------------
def test_load_stage_module_refuses_a_module_from_another_world(monkeypatch):
    """M1's pending policy: the handshake now actually acts on a mismatch."""
    import sys

    from einsums import stages

    bad = types.ModuleType("bad_world_stages")
    bad.stage_something = lambda: None
    monkeypatch.setitem(sys.modules, "bad_world_stages", bad)

    with pytest.raises(WorldMismatch):
        stages.load_stage_module("bad_world_stages")


def test_load_stage_module_does_not_fall_back_to_python(monkeypatch):
    """Decision 4: a refused module is a hard error, never a silent demotion."""
    import sys

    from einsums import stages
    from einsums.stages import TensorD, stage

    @stage
    def has_only_python(a: TensorD) -> None: ...

    bad = types.ModuleType("refused_stages")
    bad.stage_has_only_python = lambda a: None
    monkeypatch.setitem(sys.modules, "refused_stages", bad)

    with pytest.raises(WorldMismatch):
        stages.load_stage_module("refused_stages")

    assert "cpp" not in stages.get_stage("has_only_python").backends
    assert stages.selected_backend("has_only_python") == "python"


def test_a_cpp_stage_with_no_python_counterpart_is_an_error(monkeypatch):
    import sys

    from einsums import stages

    m = _good_module("orphan_stages")
    m.stage_no_python_side = lambda: None
    monkeypatch.setitem(sys.modules, "orphan_stages", m)

    with pytest.raises(Exception, match="no Python counterpart"):
        stages.load_stage_module("orphan_stages")


def test_a_verified_module_registers_its_cpp_backend(monkeypatch):
    import sys

    from einsums import stages
    from einsums.stages import TensorD, stage

    calls = []

    @stage
    def dual_backend(a: TensorD) -> None:
        calls.append("python")

    m = _good_module("dual_stages")
    m.stage_dual_backend = lambda a: calls.append("cpp")
    monkeypatch.setitem(sys.modules, "dual_stages", m)

    stages.load_stage_module("dual_stages")
    assert "cpp" in stages.get_stage("dual_backend").backends

    dual_backend(None)
    stages.select(dual_backend="cpp")
    dual_backend(None)
    assert calls == ["python", "cpp"]
