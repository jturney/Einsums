#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""``EINSUMS_STAGE_MODULE`` against a real compiled stage module.

``test_sealed_python.py`` covers the policy using hand-built stand-ins. This
covers the other end: that the macro a stage module actually writes publishes
the right things.

The load-bearing case is the refusal. The macro is supposed to publish the
fingerprints the *headers* computed in the module's own translation unit. An
implementation that read them back from the runtime ``world()`` record would
agree with the library every single time, and would sail through a test that
only ever compiles a matching build. So the same probe source is compiled at
two language levels: one matches the library and must be accepted, the other
does not and must be refused.
"""

import importlib
import os

import pytest

import einsums  # noqa: F401  (loads the runtime)
from einsums import sealed, stages
from einsums.sealed import WorldMismatch

MATCHING = f"einsums_stage_probe_cxx{os.environ.get('EINSUMS_STAGE_PROBE_STD', '20')}"
SKEWED = f"einsums_stage_probe_cxx{os.environ.get('EINSUMS_STAGE_PROBE_SKEW', '23')}"


def _import(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        pytest.skip(f"{name} was not built (no C++23 toolchain?)")


@pytest.fixture(scope="module")
def matching():
    return _import(MATCHING)


@pytest.fixture(scope="module")
def skewed():
    return _import(SKEWED)


def test_the_two_probes_really_were_compiled_differently(matching, skewed):
    """Assert the discriminator varied before asserting anything about the fold.

    Without this, "the macro publishes runtime values" and "the build system
    compiled both probes the same way" have the same symptom, and the refusal
    test below would pass or fail for reasons that have nothing to do with the
    macro.
    """
    assert matching.__probe_cplusplus__ != skewed.__probe_cplusplus__, (
        "both probes compiled at the same language level, so this file cannot "
        "test what it claims to"
    )


# ----------------------------------------------------------------------
# What the macro publishes
# ----------------------------------------------------------------------
def test_the_macro_publishes_all_three_things_the_handshake_looks_for(matching):
    assert matching.__einsums_world__ == sealed.world_identity()
    info = matching.__einsums_world_info__
    for key in ("config_fingerprint", "layout_fingerprint", "version", "library_path", "module"):
        assert key in info, f"__einsums_world_info__ is missing {key}"
    assert MATCHING in sealed.registered_stage_modules()


def test_a_matching_build_is_accepted(matching):
    sealed.verify_stage_module(matching)  # must not raise


def test_a_skewed_build_is_refused(skewed):
    """The whole reason this file compiles the probe twice."""
    with pytest.raises(WorldMismatch, match="build configuration"):
        sealed.verify_stage_module(skewed)


def test_the_skewed_build_still_registered(skewed):
    """It is one world; only the headers differ.

    Worth pinning because it says which check did the refusing. Registration
    succeeded, so the module reached this libEinsums, and the refusal came from
    the fingerprint rather than from the cross-world path. If this ever starts
    failing alongside the refusal, the test above has stopped testing
    guarantee 3 and started testing guarantee 1 by accident.
    """
    assert SKEWED in sealed.registered_stage_modules()


# ----------------------------------------------------------------------
# Through the loader
# ----------------------------------------------------------------------
def test_load_stage_module_accepts_the_matching_probe(matching):
    from einsums.stages import stage

    @stage(contract=False)
    def probe_add(a, b):
        return -1

    stages.load_stage_module(MATCHING, prefix="stage_")
    assert "cpp" in stages.get_stage("probe_add").backends

    stages.select(probe_add="cpp")
    assert probe_add(2, 3) == 5


def test_load_stage_module_refuses_the_skewed_probe(skewed):
    with pytest.raises(WorldMismatch):
        stages.load_stage_module(SKEWED, prefix="stage_")
