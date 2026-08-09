#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Scaffolded once by 'python -m einsums.stages promote'; this file is yours.

"""Differential test for the ``mp2_energy`` stage: python against cpp.

Inputs are synthetic and host-free: a random symmetric ``(ia|jb)`` and
plausible orbital energies. No psi4 is needed to run this, which is the
point of the plugin's host-free method half - the physics validation against
psi4 lives in ``input.dat``, and this test only needs the two backends to be
the same function.

A passing differential proves nothing until you know both sides ran: the
provenance assertion pins that the second result really is the C++ module's
bound class rather than the Python dataclass a silent fallback would have
produced.
"""

import numpy as np
import pytest

from einsums import stages

import hybrid_mp2.stages  # noqa: F401  (registers the python backend)


def make_inputs():
    """Synthetic MP2 inputs: 4 occupied, 7 virtual, seeded."""
    import einsums

    rng = np.random.default_rng(2026)
    nocc, nvir = 4, 7
    iajb = rng.standard_normal((nocc, nvir, nocc, nvir))
    iajb = 0.5 * (iajb + iajb.transpose(2, 3, 0, 1))  # (ia|jb) = (jb|ia)
    eps_occ = -np.sort(rng.uniform(0.5, 2.0, nocc))
    eps_vir = np.sort(rng.uniform(0.2, 3.0, nvir))

    def tensor(name, arr):
        T = einsums.create_zero_tensor(name, list(arr.shape), dtype="float64")
        np.asarray(T)[...] = arr
        return T

    return dict(
        iajb=tensor("(ia|jb)", iajb),
        eps_occ=tensor("eps occ", eps_occ),
        eps_vir=tensor("eps vir", eps_vir),
    )


@pytest.fixture(scope="module")
def stage_module():
    # Idempotent: another test file in the same pytest process may have
    # loaded the module already, and the registry (rightly) refuses a second
    # cpp registration for a stage.
    if "cpp" in stages.get_stage("mp2_energy").backends:
        return None
    return stages.load_stage_module("hybrid_mp2_stages")


def test_mp2_energy_backends_agree(stage_module):
    inputs = make_inputs()

    st = stages.get_stage("mp2_energy")

    st.select("python")
    expected = st.call(**inputs)
    st.select("cpp")
    actual = st.call(**inputs)

    # Both sides actually ran: the C++ result is the stage module's bound
    # class, the Python one is the @contract dataclass.
    assert type(expected).__module__.startswith("hybrid_mp2"), type(expected)
    assert type(actual).__module__ == "hybrid_mp2_stages", type(actual)

    # The contract's cmp rules, applied by hand until the M5 comparator
    # (einsums.stages.compare) exists. In practice the two backends emit the
    # same einsums ops on the same values and agree bit for bit; `close` is
    # the contract's promise, not the observed slack.
    for field in ("e_corr", "e_os"):
        np.testing.assert_allclose(
            np.asarray(getattr(actual, field)),
            np.asarray(getattr(expected, field)),
            rtol=1e-12, atol=1e-14, err_msg=field,
        )
