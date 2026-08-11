#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The integral-source seam: the contract, and that the dense source honours it.

No psi4. These run against a saved fixture, which is the point - the dense
source is the one implementation that needs nothing but the frozen buffers, and
that is what qualifies it as the oracle a screened source gets checked against.

    PYTHONPATH=/path/to/Einsums/build/lib python -m pytest examples/dlpno/test_integrals.py
"""

import glob
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno import integrals
from dlpno.mp2 import DLPNOMP2
from dlpno.reference_io import load_reference
from dlpno.thresholds import Thresholds

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
SMALLEST = os.path.join(FIXTURES, "water-ccpvdz.npz")


def _solver(**kwargs):
    reference, _ = load_reference(SMALLEST)
    mp2 = DLPNOMP2(reference, Thresholds.untruncated(), verbose=False, **kwargs)
    mp2.setup_orbitals(); mp2.compute_doi(); mp2.prep_sparsity()
    mp2.compute_metric()
    return mp2


def test_dense_source_satisfies_the_protocol():
    reference, _ = load_reference(SMALLEST)
    assert isinstance(integrals.DenseSource(reference.eri_3index), integrals.ThreeIndexSource)


def test_the_dense_source_declares_itself_exact():
    """The whole validation rests on this.

    Untruncated DLPNO-MP2 has to reproduce canonical DF-MP2 to 1e-13, and it can
    only do that if the integrals are exact. A source that screened silently
    would break it in a way that reads as a domain bug, so the exactness is a
    declared property rather than something inferred from a passing energy.
    """
    reference, _ = load_reference(SMALLEST)
    assert integrals.DenseSource(reference.eri_3index).screening_threshold == 0.0


def test_the_untruncated_path_uses_an_exact_source():
    """Guards the seam rather than the implementation behind it.

    Whatever source is installed, an untruncated run is only meaningful if it is
    exact. This is the assertion that should fail the day someone defaults the
    solver to a screened producer.
    """
    assert _solver().integrals.screening_threshold == 0.0


def test_declare_is_given_every_distinct_domain():
    """A producer is entitled to the whole demand before it builds anything."""
    captured = {}

    class Recording(integrals.DenseSource):
        def declare(self, spaces, demand):
            captured["spaces"] = spaces
            captured["demand"] = demand
            super().declare(spaces, demand)

    reference, _ = load_reference(SMALLEST)
    mp2 = _solver(integral_source=Recording(reference.eri_3index))
    mp2.compute_qia()

    demand = captured["demand"]
    assert not demand.is_empty()
    # Distinct domains, not one per pair: with screening off every pair shares
    # the same two, and a producer wants the set.
    assert len(demand.aux_domains) <= len(mp2.lmopair_to_ribfs)
    assert len(demand.pao_domains) <= len(mp2.lmopair_to_paos)
    # And the spaces are the ones the transform will actually use.
    assert captured["spaces"].C_lmo is mp2.C_lmo
    assert captured["spaces"].C_pao is mp2.C_pao


def test_a_substituted_source_is_the_one_that_gets_used():
    """The seam is real: replacing the source replaces the numbers."""

    class Doubled(integrals.DenseSource):
        def build(self):
            super().build()
            # Through the accessor, not the storage. The dense source keeps its
            # blocks in a dict now that it serves three integral classes, and a
            # test that names the field would break every time that changes
            # while proving nothing extra.
            np.asarray(self.q_ia(), copy=False)[...] *= 2.0

    reference, _ = load_reference(SMALLEST)
    plain = _solver(); plain.compute_qia()
    doubled = _solver(integral_source=Doubled(reference.eri_3index)); doubled.compute_qia()

    assert np.allclose(np.asarray(doubled.q_ia, copy=False),
                       2.0 * np.asarray(plain.q_ia, copy=False))


def test_build_before_declare_is_an_error():
    """A producer that has not been told the spaces cannot invent them."""
    reference, _ = load_reference(SMALLEST)
    with pytest.raises(RuntimeError):
        integrals.DenseSource(reference.eri_3index).build()


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(FIXTURES, "*.npz"))))
def test_the_seam_changes_no_number(path):
    """The refactor's whole claim, on every fixture.

    ``compute_qia`` used to transform inline; it now asks a source. The two
    contractions here are the two the source runs, in the source's layout, so
    the tensor must be bit-identical - not close, identical.

    The layout has to be mirrored, not merely the arithmetic. The source reads
    the AO integrals reversed and carries a reversed half-transform, and a
    contraction written over a different layout blocks differently inside BLAS,
    which reassociates the sums. The result then agrees to roughly 1e-16 but not
    to the bit, and whether the last bits happen to match is decided by the BLAS
    vendor: writing this the old way passes under MKL and Accelerate and fails
    under OpenBLAS. Bit-exactness is only a property of running the same
    operations in the same order, so this must track
    :meth:`dlpno.integrals.DenseSource.build`.
    """
    reference, _ = load_reference(path)
    mp2 = DLPNOMP2(reference, Thresholds.untruncated(), verbose=False)
    mp2.setup_orbitals(); mp2.compute_doi(); mp2.prep_sparsity(); mp2.compute_metric()
    mp2.compute_qia()

    import einsums
    from dlpno import tensors as ten
    Qmn = ten.from_numpy_reversed("(n m Q)", reference.eri_3index)
    naocc, npao = ten.shape(mp2.C_lmo)[1], ten.shape(mp2.C_pao)[1]
    half = ten.empty("(n i Q)", [reference.nbf, naocc, reference.naux])
    einsums.einsum("niQ <- nmQ ; mi", half, Qmn, mp2.C_lmo)
    expected = ten.empty("(Q|i u)", [reference.naux, naocc, npao])
    einsums.einsum("Qiu <- niQ ; nu", expected, half, mp2.C_pao)

    assert np.array_equal(np.asarray(mp2.q_ia, copy=False),
                          np.asarray(expected, copy=False))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
