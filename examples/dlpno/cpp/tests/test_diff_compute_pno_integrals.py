#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Scaffolded once by 'python -m einsums.stages promote'; this file is yours.

"""Differential test for the ``compute_pno_integrals`` stage: python against cpp.

Inputs come from the water-ccpvdz fixture through the real planner and the real
coupled-cluster prescreening cascade, because the shapes this stage is easiest to
get wrong are the ones a synthetic case would not have: a diagonal pair, an
off-diagonal pair, a weak pair with no density-fitted factors at all, and a
neighbour slot whose pair carries no PNOs.

Every field of the contract is compared, not only the energies a full run would
compare. Fifteen families reach three different consumers, and two of them - the
non-projected pair - flatten their merged axis in OPPOSITE orders, so a
transposed slab reproduces the right integral and only goes wrong several
equations later. An energy comparison finds that eventually; this finds it here.

A passing differential proves nothing until you know both sides ran (learned in
M3, nearly the hard way): the provenance assertion pins that the second result
really is the C++ module's bound class rather than the Python dataclass a silent
fallback would have produced.
"""

import os

import numpy as np
import pytest

from einsums import stages

import dlpno.stages  # noqa: F401  (registers the python backend)

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, os.pardir, "fixtures", "water-ccpvdz.npz")

# Every list-valued field of PnoIntegralBlocks. Named rather than read off the
# contract so that a family added later fails here loudly instead of quietly
# going uncompared.
FIELDS = ("K_mibj", "J_ijmb", "K_ivvv", "K_mjai", "K_jvvv",
          "i_Qk", "i_Qa", "j_Qk", "j_Qa", "Qma", "Qab",
          "J_ikac", "K_iakc", "J_jkac", "K_jakc")


def make_inputs():
    """The stage's arguments, built by the real planner from a real fixture."""
    from dlpno.cc_integrals import _chunks, _plan_chunk
    from dlpno.ccsd import DLPNOCCSD
    from dlpno.reference_io import load_reference
    from dlpno.thresholds import Thresholds

    reference, _ = load_reference(FIXTURE)
    cc = DLPNOCCSD(reference, Thresholds.preset("NORMAL", method="cc", n_buckets=4),
                   verbose=False)
    # The cascade, stopping short of compute_pno_integrals itself: it is what
    # decides which pairs are strong and what each one's PNO basis is, and those
    # decisions are most of this stage's input.
    cc.compute_energy(method="mp2")
    upper = [ij for ij, (i, j) in enumerate(cc.ij_to_i_j)
             if i <= j and cc.n_pno[ij]]
    chunks = _chunks(cc, upper)
    assert chunks, "the fixture produced no pairs to build integrals for"
    return cc, _plan_chunk(cc, chunks[0])


@pytest.fixture(scope="module")
def stage_module():
    # Idempotent: another test file in the same pytest process may have
    # loaded the module already, and the registry (rightly) refuses a second
    # cpp registration for a stage.
    if "cpp" in stages.get_stage("compute_pno_integrals").backends:
        return None
    return stages.load_stage_module("dlpno_stages")


def test_compute_pno_integrals_backends_agree(stage_module):
    cc, inputs = make_inputs()

    st = stages.get_stage("compute_pno_integrals")

    st.select("python")
    expected = st.call(**inputs)
    st.select("cpp")
    actual = st.call(**inputs)

    # Both sides actually ran: the C++ result is the stage module's bound
    # class, the Python one is the @contract dataclass. A fallback that
    # quietly re-ran Python would agree to 0.000e+00 and fail here.
    assert type(expected).__module__.startswith("dlpno"), type(expected)
    assert type(actual).__module__ == "dlpno_stages", type(actual)

    # The cases this fixture has to exercise for the comparison to mean
    # anything. Asserted, because a differential over a chunk that happened to
    # be all diagonal strong pairs would pass while leaving three of the
    # fifteen families empty on both sides.
    strong, nb_ij = inputs["strong"], inputs["nb_ij"]
    off_diagonal = [p for p in range(len(strong))
                    if inputs["i_lmo"][p] != inputs["j_lmo"][p]]
    assert any(strong), "no strong pair: eleven of the fifteen families are absent"
    assert off_diagonal, "no off-diagonal pair: the ji families are absent"
    assert any(len(nb_ij[p]) for p in range(len(strong))), "no neighbour slots"

    for field in FIELDS:
        exp, act = getattr(expected, field), getattr(actual, field)
        assert len(exp) == len(act), field
        for u, (a, b) in enumerate(zip(exp, act)):
            a, b = np.asarray(a), np.asarray(b)
            assert a.shape == b.shape, f"{field}[{u}]: {a.shape} against {b.shape}"
            # The contract promises `close`; both backends emit the same
            # operations on the same values and in practice agree bit for bit,
            # which is what check_backends.py asserts on the energies.
            np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-13,
                                       err_msg=f"{field}[{u}]")
