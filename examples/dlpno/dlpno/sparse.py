#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Sparse-map bookkeeping, ported from psi4's ``dlpno/sparse.cc``.

A ``SparseMap`` is a list of sorted, unique index lists: entry ``x`` holds the
indices ``y`` that ``x`` couples to. DLPNO is built almost entirely out of these
maps (which PAOs span an LMO's virtual space, which aux functions fit an LMO
pair, and so on), so the helpers below are the exact analogues of the psi4
functions and keep the same names.

The submatrix helpers are the seam where domain restriction meets tensor
algebra. Einsums has no gather-by-index-list primitive yet, so they go through
the tensor's numpy view; a real implementation wants that gather in the library
so a domain extraction can be captured into a ComputeGraph instead of running
eagerly on the host.
"""

import numpy as np

from . import tensors as ten

__all__ = [
    "merge_lists",
    "index_list",
    "contract_lists",
    "block_list",
    "invert_map",
    "chain_maps",
    "extend_maps",
    "submatrix_rows",
    "submatrix_cols",
    "submatrix_rows_and_cols",
]


def merge_lists(l1, l2):
    """Sorted union of two sorted lists."""
    return sorted(set(l1) | set(l2))


def index_list(l1, l2):
    """Positions in ``l1`` of each element of ``l2``.

    Both lists must be sorted with unique elements and ``l2`` must be a subset
    of ``l1``, matching psi4's contract.
    """
    where = {v: i for i, v in enumerate(l1)}
    try:
        return [where[v] for v in l2]
    except KeyError as exc:
        raise ValueError(
            "index_list: l2 is not a subset of l1 (missing %r)" % (exc.args[0],)
        ) from None


def contract_lists(y, A_to_y):
    """Union of the lists in ``A_to_y`` that share at least one element with ``y``.

    Used to widen a raw index selection out to whole atoms: pass the surviving
    basis functions and the atom-to-bf map, and get back every basis function on
    every atom that contributed.
    """
    y_set = set(y)
    out = []
    for y_vals in A_to_y:
        if y_set.intersection(y_vals):
            out.extend(y_vals)
    return out


def block_list(x_list, x_to_y_map):
    """Map a list of indices through a dense per-index map, collapsing runs.

    ``x_to_y_map`` is dense (one ``y`` per ``x``, e.g. basis function to atom);
    consecutive duplicates collapse, exactly as in psi4.
    """
    out = []
    for x in x_list:
        y = x_to_y_map[x]
        if not out or out[-1] != y:
            out.append(y)
    return out


def invert_map(x_to_y, ny):
    """Invert a SparseMap: which ``x`` reach each ``y``."""
    y_to_x = [[] for _ in range(ny)]
    for x, ys in enumerate(x_to_y):
        for y in ys:
            y_to_x[y].append(x)
    return y_to_x


def chain_maps(x_to_y, y_to_z):
    """Compose two SparseMaps into a map from ``x`` to ``z``."""
    out = []
    for ys in x_to_y:
        zs = set()
        for y in ys:
            zs.update(y_to_z[y])
        out.append(sorted(zs))
    return out


def extend_maps(x_to_y, xpairs):
    """Union each ``x``'s list with those of every partner it pairs with."""
    xext_to_y = [[] for _ in x_to_y]
    for x1, x2 in xpairs:
        xext_to_y[x1] = merge_lists(xext_to_y[x1], x_to_y[x2])
    return xext_to_y


# => domain restriction of tensors <= #


def submatrix_rows(mat, rows, name=None):
    """Gather rows of a rank-2 tensor into a fresh tensor."""
    a = ten.view(mat)
    return ten.from_numpy(name or "submatrix rows", a[np.asarray(rows, dtype=int), :])


def submatrix_cols(mat, cols, name=None):
    """Gather columns of a rank-2 tensor into a fresh tensor."""
    a = ten.view(mat)
    return ten.from_numpy(name or "submatrix cols", a[:, np.asarray(cols, dtype=int)])


def submatrix_rows_and_cols(mat, rows, cols, name=None):
    """Gather a rectangular index-list submatrix into a fresh tensor."""
    a = ten.view(mat)
    r = np.asarray(rows, dtype=int)
    c = np.asarray(cols, dtype=int)
    return ten.from_numpy(name or "submatrix", a[np.ix_(r, c)])
