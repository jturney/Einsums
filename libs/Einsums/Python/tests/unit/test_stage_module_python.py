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
CAPTURE = "einsums_shared_capture_probe"


def _import(name, optional=False):
    """Import a probe, skipping only when it genuinely was not built.

    The `optional` narrowing matters. A blanket `except ImportError: skip` hid a
    real defect here once: the C++23 probe was building fine and failing to
    dlopen on a missing symbol, and the skip turned "the skew test is broken"
    into "the skew test did not run", which looks identical in a green suite.
    A module whose file exists must import or fail loudly.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        if optional and "No module named" in str(exc):
            pytest.skip(f"{name} was not built (no C++23 toolchain?)")
        raise


@pytest.fixture(scope="module")
def matching():
    return _import(MATCHING)


@pytest.fixture(scope="module")
def skewed():
    return _import(SKEWED, optional=True)


@pytest.fixture(scope="module")
def capture_probe():
    return _import(CAPTURE)


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


# ----------------------------------------------------------------------
# Shared-graph capture: the definition of decision 2
# ----------------------------------------------------------------------
def test_a_cpp_stage_emits_into_pythons_graph(capture_probe):
    """A C++ stage called under a Python capture joins THAT graph.

    This is the contract the whole framework rests on. If a C++ stage captured
    into a private graph instead, every stage boundary would be a graph
    boundary and no optimization pass would ever see a whole method - which is
    the design that was considered and rejected.

    The node count is asserted as hard as the numbers are, and that is
    deliberate: three separate graphs produce exactly the same answers, so a
    correctness check alone cannot tell the two designs apart.
    """
    import numpy as np

    import einsums.graph as cg

    n = 4
    A = einsums.create_random_tensor("A", [n, n])
    B = einsums.create_random_tensor("B", [n, n])
    C = einsums.create_zero_tensor("C", [n, n])
    D = einsums.create_zero_tensor("D", [n, n])
    E = einsums.create_zero_tensor("E", [n, n])

    a, b = np.array(A), np.array(B)

    g = cg.Graph("shared")
    with cg.capture(g):
        einsums.linalg.gemm(1.0, A, B, 0.0, C)          # python
        capture_probe.stage_shared_capture_einsum(A, B, D)   # C++, same graph
        einsums.linalg.gemm(1.0, A, B, 0.0, E)          # python again

    assert g.num_nodes() == 3, (
        f"expected one graph holding all three contributions, got {g.num_nodes()} nodes; "
        f"a C++ stage that opened its own graph would leave 2 here"
    )

    # Stronger than the count, and far more legible in a failure: the C++ node
    # is not merely present, it is sandwiched between the two Python ones in
    # capture order. Nothing but a genuinely shared graph produces this.
    import json

    labels = [node["label"] for node in json.loads(g.to_json())["nodes"]]
    assert labels[0].startswith("gemm") and labels[2].startswith("gemm"), labels
    assert labels[1].startswith("einsum"), (
        f"the middle node should be the C++ stage's einsum, got {labels}"
    )

    # Nothing has run yet: capture defers, so the C++ stage's output is still
    # zero. Worth asserting, because a stage that executed eagerly instead of
    # capturing would pass every other check in this test.
    assert np.max(np.abs(np.array(D))) == 0.0, "the C++ stage ran eagerly instead of capturing"

    g.execute()

    expected = a @ b
    for name, T in (("C", C), ("D", D), ("E", E)):
        np.testing.assert_allclose(np.array(T), expected, rtol=1e-12, err_msg=f"{name} is wrong")


def test_the_cpp_emitted_node_took_the_fast_path(capture_probe):
    """A generic-loop fallback gives the right answer, so only the route shows it."""
    import einsums.graph as cg

    n = 8
    A = einsums.create_random_tensor("A", [n, n])
    B = einsums.create_random_tensor("B", [n, n])
    C = einsums.create_zero_tensor("C", [n, n])

    g = cg.Graph("route")
    with cg.capture(g):
        capture_probe.stage_shared_capture_einsum(A, B, C)
    g.execute()

    route = capture_probe.last_dispatch_route()
    assert route not in ("none", "generic_loop"), (
        f"the C++-emitted einsum took route {route!r}; a matrix-multiply spec falling back "
        f"to the generic loop still computes the right answer, which is why this is checked"
    )


def test_a_cpp_stage_outside_capture_runs_eagerly(capture_probe):
    """Stage code must be capture-transparent, or it cannot be debugged.

    The same call with no capture open has to compute and return, not silently
    record into nothing.
    """
    import numpy as np

    n = 4
    A = einsums.create_random_tensor("A", [n, n])
    B = einsums.create_random_tensor("B", [n, n])
    C = einsums.create_zero_tensor("C", [n, n])

    capture_probe.stage_shared_capture_einsum(A, B, C)
    np.testing.assert_allclose(np.array(C), np.array(A) @ np.array(B), rtol=1e-12)
