#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Canonical closed-shell (T), in plain numpy, over a :class:`~dlpno.canonical_ccsd.OrbitalBasis`.

The independent value the local triples port is measured against, and the
companion to :mod:`dlpno.canonical_ccsd`. Nothing here imports einsums or psi4,
for the same reason: a shared mistake cannot cancel if there is no shared code.

**Two different quantities live here, and confusing them is the first thing that
will go wrong.**

:func:`t0_energy` is the SEMICANONICAL correction, ``(T0)``. It is what psi4's
``DLPNOCCSD_T::compute_lccsd_t0`` computes and what the port computes, and its
energy denominator carries the DIAGONAL of the occupied Fock matrix,
``F_ii + F_jj + F_kk``. In a localized occupied basis that is an approximation:
the off-diagonal ``F_il`` couplings between triplets are dropped, which is
exactly the coupling the iterative ``(T)`` of milestone M6 puts back.

:func:`spin_orbital_t_energy` is the textbook ``(T)`` of Raghavachari et al., in
the spin-orbital form of Crawford's programming project 6. It is only correct
where the Fock matrix is diagonal.

**The two coincide when the occupied orbitals are canonical, and that is the
whole validation strategy.** Rotating the occupied space to diagonalize
``F_oo`` makes the semicanonical denominator exact, so ``t0_energy`` in the
canonical basis must equal ``spin_orbital_t_energy`` to machine precision - two
implementations sharing nothing but the integrals. Once that pins
:func:`t0_energy` down, it can be trusted in the LOCALIZED basis, where nothing
else can check it and where the port actually works.

Note that ``(T0)`` is NOT invariant to a rotation of the occupied space, unlike
CCSD. That is a property of the approximation, not a defect, and
``test_canonical_triples.py`` asserts the difference is there rather than
asserting it away.

The equations, in psi4's spelling (Jiang et al., JCP 161, 082502 (2024),
Eq. 109, 110, 53), over unique triplets ``i <= j <= k``::

    W_ijk^abc = P_ijk^abc [ (i a | b d) t_kj^cd  -  t_il^ab (j l | k c) ]
    V_ijk^abc = W_ijk^abc + t_i^a (jb|kc) + t_j^b (ia|kc) + t_k^c (ia|jb)
    T_ijk^abc = -W_ijk^abc / (e_a + e_b + e_c - F_ii - F_jj - F_kk)
    e_ijk     = p T_ijk . (8 V^abc - 4 V^cba - 4 V^acb - 4 V^bac
                                  + 2 V^cab + 2 V^bca)

with ``p = 1/2`` when exactly two of ``i, j, k`` coincide and ``1`` otherwise,
and ``P_ijk^abc X = X^abc(ijk) + X^acb(ikj) + X^bac(jik) + X^bca(jki)
+ X^cab(kij) + X^cba(kji)``.

Triplets with ``i == j == k`` are skipped, as psi4 skips them, and that is exact
rather than an approximation: every one of ``W``, ``V`` and ``T`` is fully
symmetric in ``abc`` there, so the six coefficients ``8 - 4 - 4 - 4 + 2 + 2``
cancel and the contribution is identically zero.
"""

import numpy as np

__all__ = ["t0_energy", "triplet_blocks", "unique_triplets",
           "spin_orbital_t_energy"]


def unique_triplets(nocc):
    """``i <= j <= k`` with not all three equal, in psi4's enumeration order.

    psi4 walks pairs ``ij`` with ``i <= j`` and then ``k`` in that pair's
    neighbour list with ``k >= i, j``; untruncated the neighbour list is every
    LMO, so the two enumerations agree set for set. The order matters only for
    reading a per-triplet dump next to psi4's.
    """
    return [(i, j, k)
            for i in range(nocc)
            for j in range(i, nocc)
            for k in range(j, nocc)
            if not (i == j and j == k)]


def _chemist(basis):
    """``g[p, q, r, s] = (pq|rs)``, from the basis's physicist-ordered ``eri``."""
    return basis.eri.transpose(0, 2, 1, 3)


def triplet_blocks(basis, t1, t2, i, j, k, g=None):
    """``(W, V)`` for one triplet, over the whole virtual space.

    Rank 3 and indexed ``[a, b, c]``, which is psi4's ``(a, b * nvir + c)``
    laid out so the indices are named rather than counted. Exposed separately
    from :func:`t0_energy` because a block comparison localizes a defect that a
    summed energy only reports: the port's ``W`` for one triplet, rotated into
    this basis, is comparable element for element.
    """
    g = _chemist(basis) if g is None else g
    no, nv = basis.nocc, basis.nvir
    o, v = slice(0, no), slice(no, no + nv)
    g_ovvv = g[o, v, v, v]            # (x a | b d)
    g_ooov = g[o, o, o, v]            # (x l | y c)
    g_ovov = g[o, v, o, v]            # (x a | y b)

    perms = [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]
    #: How each permutation's block enters ``P_ijk^abc``; see the module
    #: docstring. Index ``p`` of the tuple says which of ``a, b, c`` sits in
    #: axis ``p`` of that permutation's own block.
    scatter = ["abc", "acb", "bac", "bca", "cab", "cba"]

    W = np.zeros((nv, nv, nv))
    for (x, y, z), order in zip(perms, scatter):
        # Eq. 109a: + (x a | b d) t_zy^cd, and Eq. 109b: - t_xl^ab (y l | z c).
        block = np.einsum("abd,cd->abc", g_ovvv[x], t2[z, y], optimize=True)
        block -= np.einsum("lab,lc->abc", t2[x, :], g_ooov[y, :, z],
                           optimize=True)
        W += np.einsum(f"{order}->abc", block)

    V = W.copy()
    V += np.einsum("a,bc->abc", t1[i], g_ovov[j, :, k], optimize=True)
    V += np.einsum("b,ac->abc", t1[j], g_ovov[i, :, k], optimize=True)
    V += np.einsum("c,ab->abc", t1[k], g_ovov[i, :, j], optimize=True)
    return W, V


def t0_energy(basis, t1, t2, per_triplet=False):
    """The semicanonical ``(T0)`` correction over ``basis``.

    Args:
        basis: an :class:`~dlpno.canonical_ccsd.OrbitalBasis`.
        t1: ``(nocc, nvir)`` singles amplitudes.
        t2: ``(nocc, nocc, nvir, nvir)`` doubles, ``t2[i, j, a, b]``.
        per_triplet: also return ``{(i, j, k): e_ijk}``, which is what the
            triplet screening compares and what localizes a disagreement to a
            triplet rather than to a total.

    The denominator uses only the DIAGONAL of the occupied Fock matrix, which is
    what makes this ``(T0)`` rather than ``(T)``; see the module docstring.
    """
    g = _chemist(basis)
    f_o = np.diag(basis.F_oo)
    f_v = np.diag(basis.F_vv)
    e_abc = (f_v[:, None, None] + f_v[None, :, None] + f_v[None, None, :])

    total = 0.0
    per = {}
    for i, j, k in unique_triplets(basis.nocc):
        W, V = triplet_blocks(basis, t1, t2, i, j, k, g=g)
        D = e_abc - (f_o[i] + f_o[j] + f_o[k])
        T = -W / D

        prefactor = 0.5 if (i == j or j == k or i == k) else 1.0
        bracket = (8.0 * V
                   - 4.0 * V.transpose(2, 1, 0)      # V^cba
                   - 4.0 * V.transpose(0, 2, 1)      # V^acb
                   - 4.0 * V.transpose(1, 0, 2)      # V^bac
                   + 2.0 * V.transpose(1, 2, 0)      # V^cab
                   + 2.0 * V.transpose(2, 0, 1))     # V^bca
        e_ijk = prefactor * float(np.sum(bracket * T))
        per[i, j, k] = e_ijk
        total += e_ijk
    return (total, per) if per_triplet else total


# -- the spin-orbital cross-check ------------------------------------------


def spin_orbital_t_energy(basis, t1, t2, f_tol=1e-9):
    """The textbook ``(T)`` correction, spin-orbital, blocked over ``i < j < k``.

    Raghavachari's correction in the form of Crawford's programming project 6::

        Xc[abc] = sum_e t2[jk,ae] <ei||bc> - sum_m t2[im,bc] <ma||jk>
        Xd[abc] = t1[i,a] <jk||bc>
        W = P(i/jk) P(a/bc) Xc,  V = P(i/jk) P(a/bc) Xd
        E = sum_{i<j<k} (1/6) sum_abc W (W + V) / D

    The ``1/6`` rather than the ``1/36`` of the whole-tensor form: the summand
    is invariant under a permutation of ``ijk`` and, separately, of ``abc``
    (both ``W`` and ``V`` are antisymmetric in each, and the square of a sign is
    one), so restricting the occupied sum to ``i < j < k`` multiplies by six and
    leaves the virtual sum whole.

    Shares nothing with :func:`t0_energy` except the integrals, which is the
    point. It is only valid where the Fock matrix is diagonal, so it refuses
    rather than returning a plausible wrong number when handed a localized
    basis.
    """
    from .canonical_ccsd import _spin_block, spin_block_amplitudes

    off = basis.F - np.diag(np.diag(basis.F))
    if np.abs(off).max() > f_tol:
        raise ValueError(
            "spin_orbital_t_energy needs a diagonal Fock matrix (largest "
            f"off-diagonal element {np.abs(off).max():.3e}); build the basis "
            "with occupied='canonical'")

    anti, f = _spin_block(basis)
    T1, T2 = spin_block_amplitudes(t1, t2)
    no, nv = 2 * basis.nocc, 2 * basis.nvir
    o, v = slice(0, no), slice(no, no + nv)

    vovv = anti[v, o, v, v]           # <e i || b c>
    ovoo = anti[o, v, o, o]           # <m a || j k>
    oovv = anti[o, o, v, v]           # <j k || b c>
    f_o, f_v = np.diag(f)[o], np.diag(f)[v]
    d_abc = -(f_v[:, None, None] + f_v[None, :, None] + f_v[None, None, :])

    def antisymmetrize(x):
        """``P(i/jk) P(a/bc)`` applied to the three ``i``-permuted blocks."""
        base, swap_ij, swap_ik = x
        out = np.zeros_like(base)
        for block, sign in ((base, 1.0), (swap_ij, -1.0), (swap_ik, -1.0)):
            out += sign * (block
                           - block.transpose(1, 0, 2)
                           - block.transpose(2, 1, 0))
        return out

    total = 0.0
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                cs, ds = [], []
                for p, q, r in ((i, j, k), (j, i, k), (k, j, i)):
                    c = np.einsum("ae,ebc->abc", T2[q, r], vovv[:, p],
                                  optimize=True)
                    c -= np.einsum("mbc,ma->abc", T2[p, :], ovoo[:, :, q, r],
                                   optimize=True)
                    cs.append(c)
                    ds.append(np.einsum("a,bc->abc", T1[p], oovv[q, r],
                                        optimize=True))
                W = antisymmetrize(cs)
                V = antisymmetrize(ds)
                D = d_abc + (f_o[i] + f_o[j] + f_o[k])
                total += float(np.sum(W * (W + V) / D)) / 6.0
    return total
