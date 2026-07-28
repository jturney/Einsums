#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Assemble einsums tensors from psi4's neutral integral exports.

psi4's MintsHelper hands out plain ``psi4.core.Matrix`` objects (per-irrep
numpy blocks reachable through ``.nph``) plus a little index metadata; this
module reshapes those buffers into einsums tensors.
Nothing here imports psi4: every function is duck-typed against the data it
needs (``nirrep()``/``symmetry()``/``.nph`` for symmetry-blocked matrices,
or any 2D array-like for single-block data), so einsums and psi4 never need
to be compiled against each other and either side can be rebuilt freely.

The psi4 side of the contract (all chemists' notation):

- ``mints.so_overlap()`` etc. -> symmetry-blocked Matrix -> :func:`from_matrix`
- ``mints.so_eri_blocked()`` -> list of ``([hp, hq, hr, hs], Matrix)`` pairs,
  each Matrix a row-major ``(d1*d2, d3*d4)`` block -> :func:`so_eri`
- ``mints.mo_bra_half_transform(C1, C2)`` -> ``(n1*n2, nbf*nbf)`` Matrix
  -> :func:`mo_bra_half_transform`
- ``DFTensor.Qso()``/``Qov()``/... -> ``(naux, d2*d3)`` Matrix
  -> :func:`df_tensor`
"""

import math as _math

import numpy as _np

# Resolve tensor classes through the package (not _core directly) so first use
# goes through einsums.__getattr__ and brings the runtime up (honoring rc).
import einsums as _ein

__all__ = [
    "from_matrix",
    "so_eri",
    "mo_bra_half_transform",
    "df_tensor",
    "dense",
]


def _block_array(obj, require_2d=True):
    """A float64 ndarray from a single-block (C1) Matrix or any array-like.

    psi4 Matrices normally view as their 2D block, but some accessors (e.g.
    ``ao_eri()``) expose an already-shaped higher-rank array; callers that only
    need the raw buffer pass ``require_2d=False``.
    """
    if hasattr(obj, "nph"):
        blocks = obj.nph
        if len(blocks) != 1:
            raise ValueError(
                f"expected a single-block (C1) matrix, got {len(blocks)} irrep blocks; "
                "use from_matrix for symmetry-blocked matrices"
            )
        obj = blocks[0]
    arr = _np.asarray(obj, dtype=_np.float64)
    if require_2d and arr.ndim != 2:
        raise ValueError(f"expected a 2D block, got shape {arr.shape}")
    return arr


def _matrix_name(mat, name, default):
    if name is not None:
        return name
    attr = getattr(mat, "name", None)
    if callable(attr):
        attr = attr()
    return attr if attr else default


def from_matrix(mat, name=None):
    """Copy a symmetry-blocked matrix into a rank-2 ``TiledRuntimeTensorD``.

    ``mat`` needs ``nirrep()``, ``symmetry()``, and ``.nph`` (the per-irrep
    2D blocks, block ``h`` of shape ``(rowspi[h], colspi[h ^ symmetry])``) --
    psi4's ``Matrix`` and any look-alike qualify. Block ``h`` lands in tile
    ``(h, h ^ symmetry)``; empty blocks stay unstored (structural zeros).
    """
    nirrep = mat.nirrep()
    sym = mat.symmetry()
    blocks = [_np.asarray(b, dtype=_np.float64) for b in mat.nph]
    if len(blocks) != nirrep:
        raise ValueError(f"nph has {len(blocks)} blocks but nirrep() is {nirrep}")

    rows = [b.shape[0] for b in blocks]
    cols = [0] * nirrep
    for h in range(nirrep):
        cols[h ^ sym] = blocks[h].shape[1]

    T = _ein.TiledRuntimeTensorD(_matrix_name(mat, name, "psi4 matrix"), [rows, cols])
    for h in range(nirrep):
        hc = h ^ sym
        if rows[h] == 0 or cols[hc] == 0:
            continue
        _np.asarray(T.tile_view([h, hc]))[...] = blocks[h]
    return T


def so_eri(blocked, sopi, name="SO ERI"):
    """Assemble ``mints.so_eri_blocked()`` into a rank-4 ``TiledRuntimeTensorD``.

    ``blocked`` is an iterable of ``(quad, block)`` pairs where ``quad`` is the
    irrep quadruple ``[hp, hq, hr, hs]`` and ``block`` a row-major
    ``(sopi[hp]*sopi[hq], sopi[hr]*sopi[hs])`` matrix over the within-irrep
    compound indices. ``sopi`` is the SO-per-irrep dimension (e.g.
    ``wfn.nsopi().to_tuple()``). Only the supplied (symmetry-allowed) blocks
    are stored; every axis of the result is tiled by ``sopi``.
    """
    sopi = [int(d) for d in sopi]
    T = _ein.TiledRuntimeTensorD(name, [sopi, sopi, sopi, sopi])
    for quad, block in blocked:
        hp, hq, hr, hs = (int(h) for h in quad)
        dims = (sopi[hp], sopi[hq], sopi[hr], sopi[hs])
        arr = _block_array(block)
        expected = (dims[0] * dims[1], dims[2] * dims[3])
        if arr.shape != expected:
            raise ValueError(
                f"block {quad}: expected shape {expected} from sopi {sopi}, got {arr.shape}"
            )
        _np.asarray(T.tile_view([hp, hq, hr, hs]))[...] = arr.reshape(dims)
    return T


def dense(mat, shape, name="psi4 block"):
    """Reshape a single-block (C1) matrix into a dense ``RuntimeTensorD``.

    ``mat`` is any row-major array-like holding the requested ``shape``'s
    elements (e.g. ``mints.ao_eri()``'s chemists'-pair matrix with
    ``shape=(nbf, nbf, nbf, nbf)``). The element count must match exactly.
    """
    arr = _block_array(mat, require_2d=False)
    shape = [int(s) for s in shape]
    count = _math.prod(shape)
    if arr.size != count:
        raise ValueError(f"matrix has {arr.size} elements, expected {count} for shape {tuple(shape)}")
    T = _ein.RuntimeTensorD(name, shape)
    _np.asarray(T)[...] = arr.reshape(shape)
    return T


def mo_bra_half_transform(mat, n1, n2, name="MO bra half-transform"):
    """Reshape ``mints.mo_bra_half_transform(C1, C2)`` into a dense rank-4 tensor.

    ``mat`` is the ``(n1*n2, nbf*nbf)`` row-major flat matrix; the result is
    the ``(n1, n2, nbf, nbf)`` ``RuntimeTensorD`` it flattens, chemists'
    notation ``(pq|ls)`` with the bra transformed and the ket still AO.
    """
    # psi4 Matrices may view flat (n1*n2, nbf*nbf) or already shaped via their
    # numpy-shape hint; validate by element count and reshape either way.
    arr = _block_array(mat, require_2d=False)
    n1, n2 = int(n1), int(n2)
    if n1 * n2 == 0 or arr.size % (n1 * n2) != 0:
        raise ValueError(f"matrix has {arr.size} elements, not divisible into n1*n2 = {n1 * n2} rows")
    nbf2 = arr.size // (n1 * n2)
    nbf = _math.isqrt(nbf2)
    if nbf * nbf != nbf2:
        raise ValueError(f"matrix has {nbf2} elements per MO pair, expected a square nbf*nbf")
    H = _ein.RuntimeTensorD(name, [n1, n2, nbf, nbf])
    _np.asarray(H)[...] = arr.reshape(n1, n2, nbf, nbf)
    return H


def df_tensor(mat, d2, d3, name="DF 3-index"):
    """Reshape a DF 3-index block (``DFTensor.Qso()``/``Qov()``/...) into a
    dense rank-3 ``RuntimeTensorD``.

    ``mat`` is the ``(naux, d2*d3)`` row-major matrix psi4 returns; the result
    has shape ``(naux, d2, d3)``.
    """
    # Flat (naux, d2*d3) or already shaped (naux, d2, d3) both cross fine.
    arr = _block_array(mat, require_2d=False)
    d2, d3 = int(d2), int(d3)
    if d2 * d3 == 0 or arr.size % (d2 * d3) != 0:
        raise ValueError(f"matrix has {arr.size} elements, not divisible by d2*d3 = {d2 * d3}")
    naux = arr.size // (d2 * d3)
    T = _ein.RuntimeTensorD(name, [naux, d2, d3])
    _np.asarray(T)[...] = arr.reshape(naux, d2, d3)
    return T
