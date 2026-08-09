#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Scaffolded once by 'python -m einsums.stages promote'; this file is yours.

"""Differential test for the ``transform_pnos`` stage: python against cpp.

Inputs come from the water-ccpvdz fixture through the real planner: the
truncated preset gives 13 distinct upper pairs over one shared domain, which
exercises the deduplicated ``fits``/``dom_*`` crossing without being big.

The comparison uses the contract's own ``cmp`` rules, so a field declared
``up_to_sign`` is compared that way here without this file knowing which field
that is. That is the point of putting the rules on the contract.

A passing differential proves nothing until you know both sides ran (learned
in M3, nearly the hard way): the provenance assertion pins that the second
result really is the C++ module's bound class rather than the Python
dataclass a silent fallback would have produced.
"""

import os

import pytest

from einsums import stages

import dlpno.stages  # noqa: F401  (registers the python backend)

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, os.pardir, "fixtures", "water-ccpvdz.npz")


def make_inputs():
    """The stage's arguments, built by the real planner from a real fixture."""
    from dlpno.mp2 import DLPNOMP2
    from dlpno.reference_io import load_reference
    from dlpno.thresholds import Thresholds

    reference, extras = load_reference(FIXTURE)
    t_cut = float(extras["metadata"].get("t_cut_pno", 1e-8))
    mp2 = DLPNOMP2(reference, Thresholds.preset("NORMAL", t_cut_pno=t_cut, n_buckets=4),
                   verbose=False)
    for phase in ("setup_orbitals", "compute_doi", "prep_sparsity", "compute_metric",
                  "compute_qia", "precompute_fits"):
        getattr(mp2, phase)()
    return mp2.plan_pno_transform()


@pytest.fixture(scope="module")
def stage_module():
    # Idempotent: another test file in the same pytest process may have
    # loaded the module already, and the registry (rightly) refuses a second
    # cpp registration for a stage.
    if "cpp" in stages.get_stage("transform_pnos").backends:
        return None
    return stages.load_stage_module("dlpno_stages")


def test_transform_pnos_backends_agree(stage_module):
    import numpy as np

    inputs = make_inputs()

    st = stages.get_stage("transform_pnos")

    st.select("python")
    expected = st.call(**inputs)
    st.select("cpp")
    actual = st.call(**inputs)

    # Both sides actually ran: the C++ result is the stage module's bound
    # class, the Python one is the @contract dataclass. A fallback that
    # quietly re-ran Python would agree to 0.000e+00 and fail here.
    assert type(expected).__module__.startswith("dlpno"), type(expected)
    assert type(actual).__module__ == "dlpno_stages", type(actual)

    # The contract's cmp rules, applied by hand until the M5 comparator
    # (einsums.stages.compare) exists: n_pno exact, everything tensor-shaped
    # close. In practice the two backends emit the same einsums ops on the
    # same values and agree bit for bit; `close` is the contract's promise,
    # not the observed slack.
    assert list(actual.n_pno) == list(expected.n_pno)
    for field in ("K_pno", "T_pno", "X_pno", "e_pno",
                  "e_initial", "e_os_initial", "e_trunc", "e_os_trunc"):
        exp, act = getattr(expected, field), getattr(actual, field)
        assert len(exp) == len(act), field
        for u, (a, b) in enumerate(zip(exp, act)):
            np.testing.assert_allclose(
                np.asarray(b), np.asarray(a), rtol=1e-12, atol=1e-13,
                err_msg=f"{field}[{u}]")
