#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Canonical closed-shell DF-CCSD, in plain numpy, from a :class:`~dlpno.reference.Reference`.

The independent value the local coupled-cluster port is measured against. With
every truncation switched off, local CCSD is an exact rotation of canonical
CCSD, so the two must agree to the convergence threshold and any residue is a
defect in the equations rather than in the method. Nothing here imports einsums
or psi4: the whole point is that it shares no code with the thing it is judging,
so a shared mistake cannot cancel.

**The occupied block is deliberately allowed to be non-diagonal.** The
interesting basis is the localized one - it is the basis the port works in, and
the basis in which its amplitudes can be compared term by term - and in that
basis ``F_ij`` has large off-diagonal elements. So the residuals below carry the
full ``F_oo`` and ``F_vv`` matrices and only the DIAGONAL enters the Jacobi
preconditioner. That is what makes :func:`CanonicalCCSD.solve` invariant to a
unitary rotation of the occupied space, which is the property the whole
comparison rests on and which
``test_canonical_ccsd.py::test_the_energy_is_invariant_to_an_occupied_rotation``
asserts.

Two implementations of the same equations live here:

* :meth:`CanonicalCCSD.residuals` is the spin-adapted closed-shell form, the one
  that gets used. It is written against spatial orbitals and reads the port's
  amplitudes directly.
* :func:`spin_orbital_residuals` is the Stanton, Gauss, Watts and Bartlett
  (JCP 94, 4334 (1991)) spin-orbital form, which is the textbook statement of
  the same equations and is written here with no reference to the closed-shell
  one.

The second exists to check the first at ARBITRARY amplitudes, not just at the
solution. A residual that is right at convergence is not automatically right
away from it, and this module's other job is to evaluate residuals at somebody
else's amplitudes, where being right away from the solution is the whole
measurement. The spin-orbital form scales as ``(2 nmo)^4``, so it is a checking
tool for fixture-sized molecules and nothing more.

Integrals are density-fitted with the same convention as everything else in this
example, ``(pq|rs) = (Q|pq) [J^-1]_QP (P|rs)``, which is exact against psi4's own
DF-CCSD because psi4 fits the same way.
"""

import numpy as np

__all__ = ["CanonicalCCSD", "OrbitalBasis", "build_orbital_basis",
           "spin_orbital_residuals", "spin_block_amplitudes"]


class OrbitalBasis:
    """The molecular orbitals, the Fock matrix and the MO integrals over them.

    ``eri`` is in PHYSICIST notation, ``eri[p, q, r, s] = <pq|rs> = (pr|qs)``,
    over the correlated space with the occupied orbitals first. Everything below
    slices it, so the notation is fixed in exactly one place.
    """

    def __init__(self, C_occ, C_vir, F, eri):
        self.C_occ = C_occ
        self.C_vir = C_vir
        self.F = F
        self.eri = eri
        self.nocc = C_occ.shape[1]
        self.nvir = C_vir.shape[1]
        self.o = slice(0, self.nocc)
        self.v = slice(self.nocc, self.nocc + self.nvir)

    def block(self, spaces):
        """``basis.block("oovv")`` is ``<ij|ab>``, and so on."""
        s = {"o": self.o, "v": self.v}
        return self.eri[s[spaces[0]], s[spaces[1]], s[spaces[2]], s[spaces[3]]]

    @property
    def F_oo(self):
        return self.F[self.o, self.o]

    @property
    def F_ov(self):
        return self.F[self.o, self.v]

    @property
    def F_vv(self):
        return self.F[self.v, self.v]


def _symmetric_inverse_sqrt(M, cut=1e-12):
    """``M^-1/2`` for a symmetric positive definite ``M``."""
    w, V = np.linalg.eigh(M)
    keep = w > cut * w.max()
    return (V[:, keep] / np.sqrt(w[keep])) @ V[:, keep].T


def build_orbital_basis(reference, occupied="canonical", occupied_rotation=None):
    """Orbitals and integrals for the canonical calculation.

    The VIRTUAL space is built the way the port builds its PAOs and for the same
    reason: project the occupied space out of the AO basis, drop the resulting
    linear dependence, and diagonalize the Fock matrix in what is left. That
    gives exactly the ``nbf - nocc`` virtual orbitals of the reference
    determinant, with no dependence on how psi4 happened to order or phase them.

    The OCCUPIED space is the active one the reference carries, ``C_lmo``, and
    the argument chooses the basis inside it:

    ``"localized"``
        ``C_lmo`` itself, which is what the port works in and in which ``F_oo``
        is not diagonal.
    ``"canonical"``
        ``C_lmo`` rotated by the eigenvectors of ``F_lmo``, so ``F_oo`` is
        diagonal and the calculation is the textbook one.
    an array
        an explicit ``naocc x naocc`` unitary applied to ``C_lmo``, which is how
        the rotation-invariance check feeds in a random rotation.

    Frozen core needs no special handling: the inactive occupied orbitals are in
    ``C_occ`` but not in ``C_lmo``, so they define the virtual space and the
    Fock matrix and are simply not correlated.
    """
    S = np.asarray(reference.S)
    F_ao = np.asarray(reference.F)
    C_occ_all = np.asarray(reference.C_occ)
    C_lmo = np.asarray(reference.C_lmo)
    nbf = S.shape[0]

    # -- virtual space: the complement of ALL occupied orbitals ------------
    P = np.eye(nbf) - C_occ_all @ (C_occ_all.T @ S)
    S_pao = P.T @ S @ P
    w, V = np.linalg.eigh(S_pao)
    keep = w > 1e-8 * w.max()
    X = P @ (V[:, keep] / np.sqrt(w[keep]))
    F_v = X.T @ F_ao @ X
    e_vir, U = np.linalg.eigh(F_v)
    C_vir = X @ U

    nvir_expected = nbf - C_occ_all.shape[1]
    if C_vir.shape[1] != nvir_expected:
        raise RuntimeError(
            f"build_orbital_basis: got {C_vir.shape[1]} virtual orbitals, expected "
            f"{nvir_expected}; the occupied projection is not clean")

    # -- occupied space ----------------------------------------------------
    if occupied_rotation is not None:
        C_o = C_lmo @ np.asarray(occupied_rotation)
    elif occupied == "localized":
        C_o = C_lmo
    elif occupied == "canonical":
        _, U_o = np.linalg.eigh(C_lmo.T @ F_ao @ C_lmo)
        C_o = C_lmo @ U_o
    else:
        raise ValueError(f"build_orbital_basis: unknown occupied basis {occupied!r}")

    C = np.hstack([C_o, C_vir])
    F = C.T @ F_ao @ C

    # -- density-fitted MO integrals ---------------------------------------
    # B^Q_pq = [J^-1/2]_QP (P|pq), so (pq|rs) = B^Q_pq B^Q_rs exactly. The
    # symmetric power rather than the full inverse on one side only: both give
    # the same four-index integral, and this one keeps the factorization.
    eri_3index = np.asarray(reference.eri_3index)
    Jhalf = _symmetric_inverse_sqrt(np.asarray(reference.metric))
    B = np.einsum("QP,Pmn,mp,nq->Qpq", Jhalf, eri_3index, C, C, optimize=True)
    chem = np.einsum("Qpq,Qrs->pqrs", B, B, optimize=True)
    eri = chem.transpose(0, 2, 1, 3)          # (pq|rs) -> <pr|qs>

    return OrbitalBasis(C_o, C_vir, F, eri)


class CanonicalCCSD:
    """Closed-shell CCSD over a :class:`OrbitalBasis`, spin-adapted.

    The residual convention is the one the port uses, so the two are directly
    comparable: :meth:`residuals` returns ``R`` such that ``R = 0`` at the
    solution and the Jacobi step is ``t += R / D`` with
    ``D_ia = F_ii - F_aa`` and ``D_ijab = F_ii + F_jj - F_aa - F_bb`` built from
    the DIAGONAL of the Fock matrix only. Everything off-diagonal, occupied
    block included, stays inside ``R``.
    """

    def __init__(self, reference=None, occupied="canonical", occupied_rotation=None,
                 basis=None):
        self.basis = basis if basis is not None else build_orbital_basis(
            reference, occupied=occupied, occupied_rotation=occupied_rotation)
        b = self.basis
        self.nocc, self.nvir = b.nocc, b.nvir

        f_o = np.diag(b.F_oo)
        f_v = np.diag(b.F_vv)
        self.D_ia = f_o[:, None] - f_v[None, :]
        self.D_ijab = (f_o[:, None, None, None] + f_o[None, :, None, None]
                       - f_v[None, None, :, None] - f_v[None, None, None, :])

        self.t1 = np.zeros((self.nocc, self.nvir))
        self.t2 = b.block("oovv") / self.D_ijab
        self.e_corr = 0.0
        self.n_iterations = 0

    # -- energy ------------------------------------------------------------

    def energy(self, t1=None, t2=None):
        """``E = sum (2<ij|ab> - <ij|ba>) (t_ij^ab + t_i^a t_j^b) + 2 f_ia t_i^a``.

        The singles-Fock term is zero for a converged SCF reference and is
        carried anyway, because nothing else in this module assumes Brillouin.
        """
        t1 = self.t1 if t1 is None else t1
        t2 = self.t2 if t2 is None else t2
        oovv = self.basis.block("oovv")
        L = 2.0 * oovv - oovv.transpose(0, 1, 3, 2)
        tau = t2 + np.einsum("ia,jb->ijab", t1, t1, optimize=True)
        return (float(np.sum(L * tau))
                + 2.0 * float(np.sum(self.basis.F_ov * t1)))

    def mp2_energy(self):
        """The first-order guess energy, which is canonical DF-MP2.

        Only in the canonical occupied basis: the guess amplitudes are built
        from the diagonal denominators, so in a localized basis this is the
        first iteration of local MP2 rather than MP2 itself.
        """
        return self.energy(np.zeros_like(self.t1), self.basis.block("oovv") / self.D_ijab)

    # -- the spin-adapted closed-shell residuals ---------------------------

    def residuals(self, t1=None, t2=None):
        """``(R_ia, R_ijab)``, the closed-shell CCSD residuals.

        The Hirata/Crawford spin-adapted form, with the intermediates named as
        they usually are. The Fock matrices go in COMPLETE, diagonal included,
        which is the one departure from how these are normally written: the
        usual presentation zeroes the diagonals and divides by them at the end,
        and that only works when the occupied block is diagonal to begin with.
        """
        b = self.basis
        t1 = self.t1 if t1 is None else t1
        t2 = self.t2 if t2 is None else t2

        oooo, ooov, oovo = b.block("oooo"), b.block("ooov"), b.block("oovo")
        oovv, ovov, ovvo = b.block("oovv"), b.block("ovov"), b.block("ovvo")
        ovvv, vvvv = b.block("ovvv"), b.block("vvvv")
        vvvo, ovoo = b.block("vvvo"), b.block("ovoo")

        tau = t2 + np.einsum("ia,jb->ijab", t1, t1, optimize=True)
        ttau = t2 + 0.5 * np.einsum("ia,jb->ijab", t1, t1, optimize=True)

        # -- one-particle intermediates -----------------------------------
        Fae = b.F_vv.copy()
        Fae -= 0.5 * np.einsum("me,ma->ae", b.F_ov, t1, optimize=True)
        Fae += 2.0 * np.einsum("mf,mafe->ae", t1, ovvv, optimize=True)
        Fae -= np.einsum("mf,maef->ae", t1, ovvv, optimize=True)
        Fae -= 2.0 * np.einsum("mnaf,mnef->ae", ttau, oovv, optimize=True)
        Fae += np.einsum("mnaf,mnfe->ae", ttau, oovv, optimize=True)

        Fmi = b.F_oo.copy()
        Fmi += 0.5 * np.einsum("ie,me->mi", t1, b.F_ov, optimize=True)
        Fmi += 2.0 * np.einsum("ne,mnie->mi", t1, ooov, optimize=True)
        Fmi -= np.einsum("ne,mnei->mi", t1, oovo, optimize=True)
        Fmi += 2.0 * np.einsum("inef,mnef->mi", ttau, oovv, optimize=True)
        Fmi -= np.einsum("inef,mnfe->mi", ttau, oovv, optimize=True)

        Fme = b.F_ov.copy()
        Fme += 2.0 * np.einsum("nf,mnef->me", t1, oovv, optimize=True)
        Fme -= np.einsum("nf,mnfe->me", t1, oovv, optimize=True)

        # -- singles residual, Eq. T1 -------------------------------------
        r1 = b.F_ov.copy()
        r1 += np.einsum("ie,ae->ia", t1, Fae, optimize=True)
        r1 -= np.einsum("ma,mi->ia", t1, Fmi, optimize=True)
        r1 += 2.0 * np.einsum("imae,me->ia", t2, Fme, optimize=True)
        r1 -= np.einsum("imea,me->ia", t2, Fme, optimize=True)
        r1 += 2.0 * np.einsum("nf,nafi->ia", t1, ovvo, optimize=True)
        r1 -= np.einsum("nf,naif->ia", t1, ovov, optimize=True)
        r1 += 2.0 * np.einsum("mief,maef->ia", t2, ovvv, optimize=True)
        r1 -= np.einsum("mife,maef->ia", t2, ovvv, optimize=True)
        r1 -= 2.0 * np.einsum("mnae,nmei->ia", t2, oovo, optimize=True)
        r1 += np.einsum("mnae,mnei->ia", t2, oovo, optimize=True)

        # -- two-particle intermediates -----------------------------------
        Wmnij = oooo.copy()
        Wmnij += np.einsum("je,mnie->mnij", t1, ooov, optimize=True)
        Wmnij += np.einsum("ie,mnej->mnij", t1, oovo, optimize=True)
        Wmnij += np.einsum("ijef,mnef->mnij", tau, oovv, optimize=True)

        half = 0.5 * t2 + np.einsum("jf,nb->jnfb", t1, t1, optimize=True)

        Wmbej = ovvo.copy()
        Wmbej += np.einsum("jf,mbef->mbej", t1, ovvv, optimize=True)
        Wmbej -= np.einsum("nb,mnej->mbej", t1, oovo, optimize=True)
        Wmbej -= np.einsum("jnfb,mnef->mbej", half, oovv, optimize=True)
        Wmbej += np.einsum("njfb,mnef->mbej", t2, oovv, optimize=True)
        Wmbej -= 0.5 * np.einsum("njfb,mnfe->mbej", t2, oovv, optimize=True)

        Wmbje = -ovov.copy()
        Wmbje -= np.einsum("jf,mbfe->mbje", t1, ovvv, optimize=True)
        Wmbje += np.einsum("nb,mnje->mbje", t1, ooov, optimize=True)
        Wmbje += np.einsum("jnfb,mnfe->mbje", half, oovv, optimize=True)

        Zmbij = np.einsum("mbef,ijef->mbij", ovvv, tau, optimize=True)

        # -- doubles residual ---------------------------------------------
        # Every term below is written for one of (ij, ab) and symmetrized by
        # the (ij) <-> (ji), (ab) <-> (ba) transpose, which is the closed-shell
        # form of the spin-orbital P(ij) P(ab) permutation operator.
        def P(x):
            return x + x.transpose(1, 0, 3, 2)

        r2 = oovv.copy()

        tmp = Fae - 0.5 * np.einsum("mb,me->be", t1, Fme, optimize=True)
        r2 += P(np.einsum("ijae,be->ijab", t2, tmp, optimize=True))

        tmp = Fmi + 0.5 * np.einsum("je,me->mj", t1, Fme, optimize=True)
        r2 -= P(np.einsum("imab,mj->ijab", t2, tmp, optimize=True))

        r2 += np.einsum("mnab,mnij->ijab", tau, Wmnij, optimize=True)
        r2 += np.einsum("ijef,abef->ijab", tau, vvvv, optimize=True)
        r2 -= P(np.einsum("ma,mbij->ijab", t1, Zmbij, optimize=True))

        # The ring terms. Two of them contract the antisymmetrized amplitude
        # combination against Wmbej and one contracts the plain amplitude
        # against Wmbje in the other index order; getting the split wrong shows
        # up as a wrong energy at the third decimal, not as anything subtle.
        r2 += P(np.einsum("imae,mbej->ijab", t2 - t2.transpose(0, 1, 3, 2),
                          Wmbej, optimize=True))
        r2 += P(np.einsum("imae,mbej->ijab", t2, Wmbej, optimize=True)
                + np.einsum("imae,mbje->ijab", t2, Wmbje, optimize=True))
        r2 += P(np.einsum("mjae,mbie->ijab", t2, Wmbje, optimize=True))

        tmp = np.einsum("ie,ma->imea", t1, t1, optimize=True)
        r2 -= P(np.einsum("imea,mbej->ijab", tmp, ovvo, optimize=True))
        tmp = np.einsum("ie,mb->imeb", t1, t1, optimize=True)
        r2 -= P(np.einsum("imeb,maje->ijab", tmp, ovov, optimize=True))

        r2 += P(np.einsum("ie,abej->ijab", t1, vvvo, optimize=True))
        r2 -= P(np.einsum("ma,mbij->ijab", t1, ovoo, optimize=True))
        return r1, r2

    # -- the solve ---------------------------------------------------------

    def solve(self, max_iter=100, e_convergence=1e-12, r_convergence=1e-10,
              diis_max_vecs=8, singles=True, verbose=False, t1=None, t2=None):
        """Iterate to convergence and return the correlation energy.

        ``singles=False`` clamps ``T1`` at zero throughout, which is the
        doubles-only calculation the bisection in ``check_ccsd_defect.py``
        needs on both sides of the comparison.
        """
        if t1 is not None:
            self.t1 = np.array(t1, dtype=float)
        if t2 is not None:
            self.t2 = np.array(t2, dtype=float)
        if not singles:
            self.t1 = np.zeros_like(self.t1)

        diis = _DIIS(diis_max_vecs)
        e_prev = self.energy()
        for iteration in range(max_iter):
            r1, r2 = self.residuals()
            if not singles:
                r1 = np.zeros_like(r1)
            self.t1 = self.t1 + r1 / self.D_ia
            self.t2 = self.t2 + r2 / self.D_ijab

            flat = diis.extrapolate(
                np.concatenate([self.t1.ravel(), self.t2.ravel()]),
                np.concatenate([r1.ravel(), r2.ravel()]))
            self.t1 = flat[:self.t1.size].reshape(self.t1.shape)
            self.t2 = flat[self.t1.size:].reshape(self.t2.shape)
            if not singles:
                self.t1 = np.zeros_like(self.t1)

            e_curr = self.energy()
            rms = max(np.sqrt(np.mean(r1 ** 2)), np.sqrt(np.mean(r2 ** 2)))
            if verbose:
                print(f"    {iteration:>4}  {e_curr:>18.12f} "
                      f"{e_curr - e_prev:>12.3e} {rms:>10.3e}", flush=True)
            converged = (iteration > 0 and abs(e_curr - e_prev) < e_convergence
                         and rms < r_convergence)
            e_prev = e_curr
            if converged:
                self.n_iterations = iteration + 1
                self.e_corr = e_curr
                return e_curr
        raise RuntimeError("canonical CCSD did not converge")


class _DIIS:
    """Pulay extrapolation over the concatenated ``(T1, T2)`` vector.

    The same shape as the port's, deliberately: the comparison is between two
    sets of equations, and matching the accelerator keeps the convergence
    behaviour of the two runs comparable when something goes wrong.
    """

    def __init__(self, max_vecs=8):
        self.max_vecs = max_vecs
        self.amplitudes = []
        self.errors = []

    def extrapolate(self, vector, error):
        self.amplitudes.append(np.array(vector, dtype=float))
        self.errors.append(np.array(error, dtype=float))
        if len(self.amplitudes) > self.max_vecs:
            self.amplitudes.pop(0)
            self.errors.pop(0)

        n = len(self.errors)
        if n < 2:
            return self.amplitudes[-1]
        B = np.empty((n + 1, n + 1))
        B[:n, :n] = np.array([[float(a @ b) for b in self.errors] for a in self.errors])
        B[:n, n] = B[n, :n] = -1.0
        B[n, n] = 0.0
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0
        try:
            coeffs = np.linalg.solve(B, rhs)[:n]
        except np.linalg.LinAlgError:
            return self.amplitudes[-1]
        return sum(c * a for c, a in zip(coeffs, self.amplitudes))


# -- the spin-orbital cross-check ------------------------------------------


def _spin_block(basis):
    """``(<pq||rs>, f)`` over spin orbitals, alpha and beta interleaved.

    Spin orbital ``2p`` is the alpha part of spatial orbital ``p`` and ``2p+1``
    the beta part, so the occupied spin orbitals are the first ``2 nocc`` and
    the slicing below stays as simple as the spatial version's.
    """
    n = basis.nocc + basis.nvir
    idx = np.arange(2 * n) // 2
    spin = np.arange(2 * n) % 2

    phys = basis.eri[np.ix_(idx, idx, idx, idx)]
    same13 = (spin[:, None, None, None] == spin[None, None, :, None])
    same24 = (spin[None, :, None, None] == spin[None, None, None, :])
    phys = phys * same13 * same24
    anti = phys - phys.transpose(0, 1, 3, 2)

    f = basis.F[np.ix_(idx, idx)] * (spin[:, None] == spin[None, :])
    return anti, f


def spin_block_amplitudes(t1, t2):
    """Spatial closed-shell amplitudes as spin-orbital ones.

    ``t(i sigma, a sigma) = t_i^a`` and
    ``t(i s, j s', a s, b s') = t_ij^ab``, antisymmetrized, which is the
    standard closed-shell relation and the inverse of reading the alpha-beta
    block back out.
    """
    no, nv = t1.shape
    io, iv = np.arange(2 * no) // 2, np.arange(2 * nv) // 2
    so, sv = np.arange(2 * no) % 2, np.arange(2 * nv) % 2

    T1 = t1[np.ix_(io, iv)] * (so[:, None] == sv[None, :])

    direct = t2[np.ix_(io, io, iv, iv)]
    d_ia = (so[:, None, None, None] == sv[None, None, :, None])
    d_jb = (so[None, :, None, None] == sv[None, None, None, :])
    d_ib = (so[:, None, None, None] == sv[None, None, None, :])
    d_ja = (so[None, :, None, None] == sv[None, None, :, None])
    T2 = direct * d_ia * d_jb - direct.transpose(0, 1, 3, 2) * d_ib * d_ja
    return T1, T2


def spin_orbital_residuals(basis, t1, t2):
    """``(R_ia, R_ijab)`` from the spin-orbital CCSD equations, spin-summed back.

    Stanton, Gauss, Watts and Bartlett, JCP 94, 4334 (1991), with the full Fock
    matrix left in place rather than its diagonal removed, so what comes back is
    the residual in the same convention :meth:`CanonicalCCSD.residuals` uses.
    The spatial residual is read out of the alpha-beta block, which is the
    inverse of :func:`spin_block_amplitudes`.

    This is the check on the closed-shell equations, so it shares nothing with
    them beyond the integrals.
    """
    anti, f = _spin_block(basis)
    T1, T2 = spin_block_amplitudes(t1, t2)

    no, nv = 2 * basis.nocc, 2 * basis.nvir
    o, v = slice(0, no), slice(no, no + nv)
    ein = lambda *a: np.einsum(*a, optimize=True)

    tt = T2 + 0.5 * (ein("ia,jb->ijab", T1, T1) - ein("ib,ja->ijab", T1, T1))
    tau = T2 + ein("ia,jb->ijab", T1, T1) - ein("ib,ja->ijab", T1, T1)

    Fae = f[v, v].copy()
    Fae -= 0.5 * ein("me,ma->ae", f[o, v], T1)
    Fae += ein("mf,mafe->ae", T1, anti[o, v, v, v])
    Fae -= 0.5 * ein("mnaf,mnef->ae", tt, anti[o, o, v, v])

    Fmi = f[o, o].copy()
    Fmi += 0.5 * ein("ie,me->mi", T1, f[o, v])
    Fmi += ein("ne,mnie->mi", T1, anti[o, o, o, v])
    Fmi += 0.5 * ein("inef,mnef->mi", tt, anti[o, o, v, v])

    Fme = f[o, v] + ein("nf,mnef->me", T1, anti[o, o, v, v])

    Wmnij = anti[o, o, o, o].copy()
    tmp = ein("je,mnie->mnij", T1, anti[o, o, o, v])
    Wmnij += tmp - tmp.transpose(0, 1, 3, 2)
    Wmnij += 0.25 * ein("ijef,mnef->mnij", tau, anti[o, o, v, v])

    Wabef = anti[v, v, v, v].copy()
    tmp = ein("mb,amef->abef", T1, anti[v, o, v, v])
    Wabef -= tmp - tmp.transpose(1, 0, 2, 3)
    Wabef += 0.25 * ein("mnab,mnef->abef", tau, anti[o, o, v, v])

    Wmbej = anti[o, v, v, o].copy()
    Wmbej += ein("jf,mbef->mbej", T1, anti[o, v, v, v])
    Wmbej -= ein("nb,mnej->mbej", T1, anti[o, o, v, o])
    Wmbej -= ein("jnfb,mnef->mbej", 0.5 * T2 + ein("jf,nb->jnfb", T1, T1),
                 anti[o, o, v, v])

    r1 = f[o, v].copy()
    r1 += ein("ie,ae->ia", T1, Fae)
    r1 -= ein("ma,mi->ia", T1, Fmi)
    r1 += ein("imae,me->ia", T2, Fme)
    r1 -= ein("nf,naif->ia", T1, anti[o, v, o, v])
    r1 -= 0.5 * ein("imef,maef->ia", T2, anti[o, v, v, v])
    r1 -= 0.5 * ein("mnae,nmei->ia", T2, anti[o, o, v, o])

    r2 = anti[o, o, v, v].copy()
    tmp = ein("ijae,be->ijab", T2, Fae - 0.5 * ein("mb,me->be", T1, Fme))
    r2 += tmp - tmp.transpose(0, 1, 3, 2)
    tmp = ein("imab,mj->ijab", T2, Fmi + 0.5 * ein("je,me->mj", T1, Fme))
    r2 -= tmp - tmp.transpose(1, 0, 2, 3)
    r2 += 0.5 * ein("mnab,mnij->ijab", tau, Wmnij)
    r2 += 0.5 * ein("ijef,abef->ijab", tau, Wabef)
    tmp = ein("imae,mbej->ijab", T2, Wmbej)
    tmp -= ein("ie,ma,mbej->ijab", T1, T1, anti[o, v, v, o])
    r2 += tmp - tmp.transpose(1, 0, 2, 3) - tmp.transpose(0, 1, 3, 2) \
        + tmp.transpose(1, 0, 3, 2)
    tmp = ein("ie,abej->ijab", T1, anti[v, v, v, o])
    r2 += tmp - tmp.transpose(1, 0, 2, 3)
    tmp = ein("ma,mbij->ijab", T1, anti[o, v, o, o])
    r2 -= tmp - tmp.transpose(0, 1, 3, 2)

    # Read the closed-shell residual back out of the alpha-beta block.
    io = 2 * np.arange(basis.nocc)
    iv = 2 * np.arange(basis.nvir)
    return (r1[np.ix_(io, iv)],
            r2[np.ix_(io, io + 1, iv, iv + 1)])
