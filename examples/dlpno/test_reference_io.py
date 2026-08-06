#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Round-trip tests for :mod:`dlpno.reference_io`.

Numpy only: no psi4, and no einsums either. The point of the fixture format is
that loading it cannot fail for reasons belonging to the library under test, and
a test that imported einsums would give that property up.

    python -m pytest examples/dlpno/test_reference_io.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno.reference import Reference
from dlpno.reference_io import FORMAT_VERSION, load_reference, save_reference


def make_reference(nbf=6, naux=9, naocc=2, nocc=3, with_grid=True, with_dipole=True):
    """A shape-valid Reference of arbitrary numbers.

    The values are not physical and do not need to be: these tests are about
    the container preserving what it was handed, exactly.
    """
    rng = np.random.default_rng(7)
    grid = None
    if with_grid:
        # Deliberately ragged in both directions, which is the case the flat
        # layout exists to handle: block 1 has fewer points AND fewer basis
        # functions than block 0.
        blocks = [
            (rng.standard_normal((5, 4)), rng.standard_normal(5), [0, 2, 3, 5]),
            (rng.standard_normal((3, 2)), rng.standard_normal(3), [1, 4]),
        ]

        def grid():
            yield from blocks

    return Reference(
        S=rng.standard_normal((nbf, nbf)),
        F=rng.standard_normal((nbf, nbf)),
        C_occ=rng.standard_normal((nbf, nocc)),
        C_lmo=rng.standard_normal((nbf, naocc)),
        eri_3index=rng.standard_normal((naux, nbf, nbf)),
        metric=rng.standard_normal((naux, naux)),
        atom_to_bf=[[0, 1, 2], [3, 4], [5]],
        atom_to_ribf=[[0, 1, 2, 3], [4, 5, 6], [7, 8]],
        dipole_ao=rng.standard_normal((3, nbf, nbf)) if with_dipole else None,
        grid_blocks=grid,
        n_core=1,
        e_scf=-76.02663273485651,
    )


def test_round_trip_is_exact(tmp_path):
    ref = make_reference()
    path = save_reference(ref, str(tmp_path / "ref.npz"))

    back, extras = load_reference(path)

    for name in ("S", "F", "C_occ", "C_lmo", "eri_3index", "metric", "dipole_ao"):
        np.testing.assert_array_equal(getattr(back, name), getattr(ref, name),
                                      err_msg=f"{name} changed across the round trip")
    assert back.atom_to_bf == ref.atom_to_bf
    assert back.atom_to_ribf == ref.atom_to_ribf
    assert back.n_core == ref.n_core
    # Exact, not approximate: the SCF energy is what a replay reports as a total.
    assert back.e_scf == ref.e_scf
    assert extras == {"energies": {}, "metadata": {}}


def test_grid_blocks_replay_identically(tmp_path):
    ref = make_reference()
    original = [(phi.copy(), w.copy(), list(bf)) for phi, w, bf in ref.grid_blocks()]

    back, _ = load_reference(save_reference(ref, str(tmp_path / "ref.npz")))
    replayed = list(back.grid_blocks())

    assert len(replayed) == len(original)
    for (phi_a, w_a, map_a), (phi_b, w_b, map_b) in zip(original, replayed):
        np.testing.assert_array_equal(phi_b, phi_a)
        np.testing.assert_array_equal(w_b, w_a)
        assert map_b == map_a


def test_grid_blocks_that_alias_one_buffer_are_captured(tmp_path):
    """A provider may hand out views into a buffer it reuses for the next block.

    That is what psi4's does: ``PHI`` is one allocation sized to the largest
    block, and the trimmed slice of it is sometimes already contiguous, so the
    yielded array can be a view onto memory the next iteration overwrites. The
    streaming contract allows it - a block is consumed before the next is asked
    for - and ``prep_sparsity`` honours it, but ``save_reference`` accumulates.
    Without an eager copy every block sharing the buffer stores the LAST
    block's values, and the only symptom downstream is that differential-overlap
    screening picks the wrong domains.
    """
    scratch = np.zeros((4, 3))
    wanted = [np.full((4, 3), float(i)) for i in range(3)]

    def aliasing_grid():
        for i, block in enumerate(wanted):
            scratch[:] = block          # reuse one buffer, as psi4 does
            yield scratch, np.full(4, float(i)), [0, 1, 2]

    ref = make_reference(with_grid=False)
    ref.grid_blocks = aliasing_grid

    back, _ = load_reference(save_reference(ref, str(tmp_path / "ref.npz")))

    replayed = [phi.copy() for phi, _, _ in back.grid_blocks()]
    assert len(replayed) == len(wanted)
    for i, (got, want) in enumerate(zip(replayed, wanted)):
        np.testing.assert_array_equal(got, want, err_msg=f"block {i} was not captured")


def test_grid_is_replayable_more_than_once(tmp_path):
    """The loaded provider is a generator *factory*, like the psi4 one.

    ``prep_sparsity`` calls it once, but nothing in the contract says it may
    only be called once, and a bare generator would silently yield nothing the
    second time.
    """
    back, _ = load_reference(save_reference(make_reference(), str(tmp_path / "ref.npz")))
    first = [phi.copy() for phi, _, _ in back.grid_blocks()]
    second = [phi.copy() for phi, _, _ in back.grid_blocks()]

    assert len(first) == len(second) > 0
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("with_grid, with_dipole", [(False, True), (True, False), (False, False)])
def test_optional_fields_survive_being_absent(tmp_path, with_grid, with_dipole):
    """``grid_blocks=None`` is the untruncated reference calculation, and
    ``dipole_ao=None`` disables the dipole prescreening. Both are legal."""
    ref = make_reference(with_grid=with_grid, with_dipole=with_dipole)
    back, _ = load_reference(save_reference(ref, str(tmp_path / "ref.npz")))

    assert (back.grid_blocks is None) == (not with_grid)
    assert (back.dipole_ao is None) == (not with_dipole)


def test_energies_and_metadata_round_trip(tmp_path):
    ref = make_reference()
    path = save_reference(ref, str(tmp_path / "ref.npz"),
                          energies={"psi4_df_mp2": -0.204124711603},
                          metadata={"molecule": "water", "basis": "cc-pvdz"})

    _, extras = load_reference(path)

    # Exact: a replay compares its own energy against this to 1e-9.
    assert extras["energies"]["psi4_df_mp2"] == -0.204124711603
    assert extras["metadata"] == {"molecule": "water", "basis": "cc-pvdz"}


def test_loads_without_pickle(tmp_path):
    """Nothing in the file may need ``allow_pickle``, which load_reference does
    not pass. Asserted directly so a future field cannot quietly reintroduce an
    object array."""
    path = save_reference(make_reference(), str(tmp_path / "ref.npz"))
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            assert data[key].dtype != object, f"{key} is an object array"


def test_rejects_a_future_format_version(tmp_path):
    path = str(tmp_path / "ref.npz")
    save_reference(make_reference(), path)

    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["format_version"] = np.array(FORMAT_VERSION + 1, dtype=np.int64)
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="format version"):
        load_reference(path)
