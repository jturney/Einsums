#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Save and load a :class:`~dlpno.reference.Reference` as a plain ``.npz`` file.

This is what lets DLPNO run with no psi4 in the process. ``psi4_source.py``
builds a Reference from a converged wavefunction; this module writes that
Reference to disk and reads it back, so a test fixture can be produced once on a
machine that has psi4 and replayed anywhere.

A FCIDUMP would not do: it carries MO-basis ``h_pq``, ``(pq|rs)`` and the
nuclear repulsion, while a Reference is AO-basis throughout and additionally
carries the auxiliary metric, the *raw* three-index integrals, the atom-to-basis
maps the domain construction keys off, and the DFT grid the differential overlap
integrals are quadratured on. None of those have a FCIDUMP field.

Only numpy is imported here, deliberately. A fixture that needed einsums to load
could not distinguish "the fixture is broken" from "the library under test is
broken", which is most of the value of having one.

Nothing is pickled, so the files load under ``allow_pickle=False``. The two
ragged structures - the per-atom basis-function lists and the DFT grid blocks -
are stored flat with explicit counts and rebuilt on load.
"""

import numpy as np

from .reference import Reference

__all__ = ["save_reference", "load_reference", "FORMAT_VERSION"]

#: Bumped whenever the on-disk layout changes incompatibly. ``load_reference``
#: refuses anything it does not recognise rather than misreading it.
FORMAT_VERSION = 1


def _flatten_ragged(lists):
    """``[[0, 1], [2]]`` -> ``(flat=[0, 1, 2], counts=[2, 1])``."""
    counts = np.array([len(x) for x in lists], dtype=np.int64)
    flat = np.array([i for x in lists for i in x], dtype=np.int64)
    return flat, counts


def _unflatten_ragged(flat, counts):
    """Inverse of :func:`_flatten_ragged`, as a list of lists of ``int``."""
    out, pos = [], 0
    for n in counts:
        out.append([int(v) for v in flat[pos:pos + int(n)]])
        pos += int(n)
    return out


def _require_dense_integrals(ref):
    """A fixture is buffers, and a live integral source is not one.

    ``from_psi4(integrals="dfhelper")`` leaves ``eri_3index`` unset because
    nothing in a running calculation reads it. Freezing that reference would
    produce a fixture that cannot be replayed, so say so here rather than at
    load time.
    """
    if ref.eri_3index is None:
        raise ValueError(
            "save_reference: this Reference carries a live integral source rather than "
            "eri_3index, and a source cannot be frozen. Rebuild it with "
            "from_psi4(..., integrals='dense').")


def save_reference(ref, path, energies=None, metadata=None):
    """Write @p ref to ``path`` as a compressed ``.npz``.

    The DFT grid is consumed here: ``ref.grid_blocks`` is a one-shot stream, so
    it is drained into arrays. Collocating the whole grid is what the streaming
    interface exists to avoid, which bounds this to fixture-sized molecules -
    the point of a smoke test, not a substitute for the psi4 path.

    :param energies: optional dict of reference energies (psi4's DF-MP2 and
        DLPNO-MP2, say) recorded alongside the data, so a replay can check
        itself without psi4 present.
    :param metadata: optional dict of short strings (molecule, basis, ...)
        describing where the fixture came from.
    """
    _require_dense_integrals(ref)
    ref.validate()

    arrays = {
        "format_version": np.array(FORMAT_VERSION, dtype=np.int64),
        "S": np.ascontiguousarray(ref.S, dtype=np.float64),
        "F": np.ascontiguousarray(ref.F, dtype=np.float64),
        "C_occ": np.ascontiguousarray(ref.C_occ, dtype=np.float64),
        "C_lmo": np.ascontiguousarray(ref.C_lmo, dtype=np.float64),
        "eri_3index": np.ascontiguousarray(ref.eri_3index, dtype=np.float64),
        "metric": np.ascontiguousarray(ref.metric, dtype=np.float64),
        "n_core": np.array(ref.n_core, dtype=np.int64),
        "e_scf": np.array(ref.e_scf, dtype=np.float64),
    }

    bf_flat, bf_counts = _flatten_ragged(ref.atom_to_bf)
    ri_flat, ri_counts = _flatten_ragged(ref.atom_to_ribf)
    arrays["atom_to_bf_flat"] = bf_flat
    arrays["atom_to_bf_counts"] = bf_counts
    arrays["atom_to_ribf_flat"] = ri_flat
    arrays["atom_to_ribf_counts"] = ri_counts

    arrays["has_dipole"] = np.array(ref.dipole_ao is not None)
    if ref.dipole_ao is not None:
        arrays["dipole_ao"] = np.ascontiguousarray(ref.dipole_ao, dtype=np.float64)

    # The grid: one concatenated buffer per component, plus the per-block shapes
    # needed to cut it back up. phi is (npoints, nbf_block) and both dimensions
    # vary block to block, so neither can be inferred from the other.
    arrays["has_grid"] = np.array(ref.grid_blocks is not None)
    if ref.grid_blocks is not None:
        phis, ws, maps, npts, nbfs = [], [], [], [], []
        for phi, w, bf_map in ref.grid_blocks():
            # copy=True, not ascontiguousarray. A provider is entitled to yield
            # views into a buffer it reuses for the next block - the contract is
            # that a block is consumed before the next is asked for - and this
            # loop accumulates instead of consuming. Without the copy, every
            # block sharing a buffer ends up holding the last one's values, and
            # the only symptom is that differential-overlap screening quietly
            # chooses the wrong domains.
            phi = np.array(phi, dtype=np.float64, copy=True, order="C")
            npts.append(phi.shape[0])
            nbfs.append(phi.shape[1])
            phis.append(phi.ravel())
            ws.append(np.array(w, dtype=np.float64, copy=True))
            maps.append(np.asarray(bf_map, dtype=np.int64))
        arrays["grid_phi_flat"] = np.concatenate(phis) if phis else np.zeros(0)
        arrays["grid_w_flat"] = np.concatenate(ws) if ws else np.zeros(0)
        arrays["grid_bfmap_flat"] = np.concatenate(maps) if maps else np.zeros(0, dtype=np.int64)
        arrays["grid_npoints"] = np.array(npts, dtype=np.int64)
        arrays["grid_nbf"] = np.array(nbfs, dtype=np.int64)

    for key, value in (energies or {}).items():
        arrays[f"energy_{key}"] = np.array(value, dtype=np.float64)
    for key, value in (metadata or {}).items():
        arrays[f"meta_{key}"] = np.array(str(value))

    # Compressed: the grid collocation is mostly near-zero (basis functions
    # decay away from their centre) and packs to about a fifth of its size,
    # which is the difference between a fixture worth committing and one that
    # is not. Decompression is a few milliseconds against a run of seconds.
    np.savez_compressed(path, **arrays)
    return path


def _grid_provider(phi_flat, w_flat, bfmap_flat, npoints, nbf):
    """Rebuild the streaming ``grid_blocks`` callable from the flat buffers.

    Yields views into the loaded arrays rather than copies; the consumer copies
    into einsums tensors anyway.
    """
    def blocks():
        p = w = m = 0
        for npt, nb in zip(npoints, nbf):
            npt, nb = int(npt), int(nb)
            phi = phi_flat[p:p + npt * nb].reshape(npt, nb)
            yield phi, w_flat[w:w + npt], [int(v) for v in bfmap_flat[m:m + nb]]
            p += npt * nb
            w += npt
            m += nb

    return blocks


def load_reference(path):
    """Read a Reference written by :func:`save_reference`.

    Returns ``(reference, extras)``, where ``extras`` holds whatever energies
    and metadata were recorded at dump time, so a replay can check itself.
    """
    with np.load(path, allow_pickle=False) as data:
        version = int(data["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path}: reference format version {version}, this build reads "
                f"{FORMAT_VERSION}; regenerate the fixture with dump_reference.py"
            )

        grid = None
        if bool(data["has_grid"]):
            grid = _grid_provider(data["grid_phi_flat"], data["grid_w_flat"],
                                  data["grid_bfmap_flat"], data["grid_npoints"],
                                  data["grid_nbf"])

        ref = Reference(
            S=data["S"],
            F=data["F"],
            C_occ=data["C_occ"],
            C_lmo=data["C_lmo"],
            eri_3index=data["eri_3index"],
            metric=data["metric"],
            atom_to_bf=_unflatten_ragged(data["atom_to_bf_flat"], data["atom_to_bf_counts"]),
            atom_to_ribf=_unflatten_ragged(data["atom_to_ribf_flat"], data["atom_to_ribf_counts"]),
            dipole_ao=data["dipole_ao"] if bool(data["has_dipole"]) else None,
            grid_blocks=grid,
            n_core=int(data["n_core"]),
            e_scf=float(data["e_scf"]),
        )

        extras = {
            "energies": {k[len("energy_"):]: float(data[k])
                         for k in data.files if k.startswith("energy_")},
            "metadata": {k[len("meta_"):]: str(data[k])
                         for k in data.files if k.startswith("meta_")},
        }

    return ref.validate(), extras
