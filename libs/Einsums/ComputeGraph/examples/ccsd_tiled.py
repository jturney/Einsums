#!/usr/bin/env python
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Spin-orbital CCSD on TiledRuntimeTensor, blocked by point-group irrep.

Every orbital axis is partitioned by irrep, so a tile IS a symmetry block and a
block exists only when the direct product of its irreps is totally symmetric.
The C2v irrep group is Z2 x Z2, so that test is an XOR of the tile coordinates.
Structural sparsity then does real work: `t2` stores 64 of 256 blocks, and the
contraction engine skips absent operand pairs without being told to.

Permutation operators are the interesting part. P(ij) and P(ab) are normally a
transpose, and `permute` is not bound for tiled operands -- but a contraction
does not need one, because the OUTPUT index labels can be written in any order.
`"ijba <- ijae ; be"` deposits the ab-swapped contribution directly, so each
P(...) is a second accumulating einsum rather than a sort. This is the same
trade psi4's DPD makes with buf4_sort, resolved in favour of relabelling.

The model Hamiltonian is synthetic: two-electron integrals are built from a
Cholesky-like factor whose auxiliary index carries an irrep, which yields exact
8-fold permutational symmetry and exact C2v block sparsity. Canonical HF is
assumed (diagonal Fock), as the spin-orbital equations do. It is not a physical
molecule -- the point is to exercise the tiled machinery against a dense oracle
running the identical equations on the identical integrals. Physical validation
belongs with the psi4 bridge examples.

Run:
    PYTHONPATH=build/lib python libs/Einsums/ComputeGraph/examples/ccsd_tiled.py
"""

from __future__ import annotations

import itertools

import numpy as np

import einsums

TRT = einsums.TiledRuntimeTensorD
NIRREP = 4  # C2v: A1 A2 B1 B2, closed under XOR


# ── model system ────────────────────────────────────────────────────────────


def build_model(n_occ, n_vir, n_aux, seed=11):
    """Spin-orbital integrals with exact C2v block sparsity.

    Spatial orbitals are ordered occupied-then-virtual, each group sorted by
    irrep, so every orbital axis is contiguous per irrep and a tile boundary is
    an irrep boundary. Spin orbitals interleave the two spins of one spatial
    orbital, which keeps that grouping intact and doubles each block.
    """
    rng = np.random.default_rng(seed)

    occ_irrep = np.concatenate([np.full(n_occ[h], h) for h in range(NIRREP)])
    vir_irrep = np.concatenate([np.full(n_vir[h], h) for h in range(NIRREP)])
    irrep = np.concatenate([occ_irrep, vir_irrep])
    n_mo = irrep.size
    n_occ_tot = occ_irrep.size

    # Auxiliary index carries an irrep; B[p,q,x] may be nonzero only when
    # irrep(p) ^ irrep(q) == irrep(x). Then (pq|rs) = sum_x B[pq,x] B[rs,x] is
    # nonzero only when irrep(p)^irrep(q)^irrep(r)^irrep(s) == 0, which is
    # exactly the totally-symmetric condition, and 8-fold symmetry is automatic.
    aux_irrep = np.concatenate([np.full(n_aux[h], h) for h in range(NIRREP)])
    B = rng.standard_normal((n_mo, n_mo, aux_irrep.size)) * 0.25
    B = 0.5 * (B + B.transpose(1, 0, 2))  # symmetric in (p,q)
    mask = (irrep[:, None, None] ^ irrep[None, :, None]) == aux_irrep[None, None, :]
    B *= mask

    g_chem = np.einsum("pqx,rsx->pqrs", B, B, optimize=True)  # (pq|rs)

    # Spin orbitals: <pq|rs> = (PR|QS) delta_{sp,sr} delta_{sq,ss}
    n_so = 2 * n_mo
    spin = np.arange(n_so) % 2
    spatial = np.arange(n_so) // 2
    phys = g_chem.transpose(0, 2, 1, 3)  # (PR|QS) -> <PQ|RS> ordering
    g = phys[np.ix_(spatial, spatial, spatial, spatial)]
    g = g * (spin[:, None, None, None] == spin[None, None, :, None])
    g = g * (spin[None, :, None, None] == spin[None, None, None, :])
    g = g - g.transpose(0, 1, 3, 2)  # <pq||rs>

    # Canonical HF is assumed; the orbital energies only enter the denominators.
    eps = np.concatenate(
        [
            -2.0 - 0.35 * np.arange(n_occ_tot),
            0.6 + 0.30 * np.arange(n_mo - n_occ_tot),
        ]
    )
    eps_so = np.repeat(eps, 2)

    n_o = 2 * n_occ_tot
    occ_part = [2 * int(n) for n in n_occ]
    vir_part = [2 * int(n) for n in n_vir]
    return g, eps_so, n_o, occ_part, vir_part


# ── tiled helpers ───────────────────────────────────────────────────────────


def symmetry_allowed(coord):
    """A block survives only if the direct product of its irreps is totally
    symmetric. For Z2 x Z2 that is an XOR of the tile coordinates."""
    acc = 0
    for c in coord:
        acc ^= c
    return acc == 0


class Blocking:
    """Per-axis irrep partitions, and construction of tiled tensors over them."""

    def __init__(self, occ_part, vir_part):
        self.part = {"o": occ_part, "v": vir_part}

    def grid(self, spec):
        return [self.part[c] for c in spec]

    def make(self, name, spec, dense=None):
        t = TRT(name, self.grid(spec))
        for coord in itertools.product(range(NIRREP), repeat=len(spec)):
            if symmetry_allowed(coord):
                t.add_tile(list(coord))
        t.materialize()
        if dense is not None:
            self.load(t, spec, dense)
        return t

    def _slices(self, t, spec, coord):
        sizes, offs = t.tile_sizes(), t.tile_offsets()
        return tuple(
            slice(offs[ax][coord[ax]], offs[ax][coord[ax]] + sizes[ax][coord[ax]])
            for ax in range(len(spec))
        )

    def load(self, t, spec, dense):
        for coord in itertools.product(range(NIRREP), repeat=len(spec)):
            c = list(coord)
            if t.has_tile(c):
                np.asarray(t.tile_view(c))[...] = dense[self._slices(t, spec, coord)]

    def gather(self, t, spec, shape):
        out = np.zeros(shape)
        for coord in itertools.product(range(NIRREP), repeat=len(spec)):
            c = list(coord)
            if t.has_tile(c):
                out[self._slices(t, spec, coord)] = np.asarray(t.tile_view(c))
        return out

    def divide_by(self, t, spec, denom_dense):
        """Amplitude update. There is no tiled elementwise divide bound (no
        `direct_division` over tiled operands), so this drops to numpy per tile.
        The contractions all stay in einsums; only this step does not."""
        for coord in itertools.product(range(NIRREP), repeat=len(spec)):
            c = list(coord)
            if t.has_tile(c):
                np.asarray(t.tile_view(c))[...] /= denom_dense[self._slices(t, spec, coord)]


def zero(t):
    einsums.linalg.scale(0.0, t)


def dot(a, b):
    r = einsums.RuntimeTensorD("r", [1])
    einsums.linalg.dot(r, a, b)
    return float(np.asarray(r)[0])


# ── CCSD ────────────────────────────────────────────────────────────────────


class TiledCCSD:
    """Spin-orbital CCSD over irrep-blocked tiled tensors.

    Every contraction is a tiled einsum, and the amplitude update is a tiled
    `direct_division`, so a whole iteration is expressible as graph operations
    with nothing dropping to numpy. `iterate()` therefore runs identically eagerly
    or inside a CaptureGuard, which is what lets the same code be replayed through
    the optimizer.

    P(ij) and P(ab) are expressed by permuting the OUTPUT index labels rather than
    transposing, since `permute` is not bound for tiled operands and a contraction
    does not need one.
    """

    def __init__(self, blk, blocks, d_ia, d_ijab):
        self.blk = blk
        self.g = blocks
        self.T1 = blk.make("t1", "ov")
        self.T2 = blk.make("t2", "oovv")
        self.D1 = blk.make("D1", "ov", d_ia)
        self.D2 = blk.make("D2", "oovv", d_ijab)
        for nm, spec in (
            ("TAU", "oovv"), ("TAUT", "oovv"), ("FAE", "vv"), ("FMI", "oo"),
            ("FME", "ov"), ("WMNIJ", "oooo"), ("WABEF", "vvvv"), ("WMBEJ", "ovvo"),
            ("XJNFB", "oovv"), ("TMPVV", "vv"), ("TMPOO", "oo"), ("SIBJM", "ovoo"),
            ("T1N", "ov"), ("T2N", "oovv"), ("U", "oovv"),
        ):
            setattr(self, nm, blk.make(nm.lower(), spec))
        # MP2 start: t2 = <ij||ab> / D
        einsums.linalg.direct_division(1.0, self.g["oovv"], self.D2, 0.0, self.T2)

    def energy(self):
        zero(self.U)
        einsums.einsum("ijab <- ia ; jb", self.U, self.T1, self.T1, c_pf=0.0, ab_pf=1.0)
        return 0.25 * dot(self.g["oovv"], self.T2) + 0.5 * dot(self.g["oovv"], self.U)

    def iterate(self):
        """One CCSD iteration. Pure graph ops: safe to call under capture."""
        ein = einsums.einsum
        axpy = einsums.linalg.axpy
        g = self.g
        T1, T2, T1N, T2N = self.T1, self.T2, self.T1N, self.T2N

        # tau_t = t2 + 0.5 (t1[ia] t1[jb] - t1[ib] t1[ja]);  tau uses 1.0
        for dst, half in ((self.TAUT, 0.5), (self.TAU, 1.0)):
            zero(dst)
            axpy(1.0, T2, dst)
            ein("ijab <- ia ; jb", dst, T1, T1, c_pf=1.0, ab_pf=half)
            ein("ijab <- ib ; ja", dst, T1, T1, c_pf=1.0, ab_pf=-half)

        # One-particle intermediates
        ein("ae <- mf ; amef", self.FAE, T1, g["vovv"], c_pf=0.0, ab_pf=1.0)
        ein("ae <- mnaf ; mnef", self.FAE, self.TAUT, g["oovv"], c_pf=1.0, ab_pf=-0.5)
        ein("mi <- ne ; mnie", self.FMI, T1, g["ooov"], c_pf=0.0, ab_pf=1.0)
        ein("mi <- inef ; mnef", self.FMI, self.TAUT, g["oovv"], c_pf=1.0, ab_pf=0.5)
        ein("me <- nf ; mnef", self.FME, T1, g["oovv"], c_pf=0.0, ab_pf=1.0)

        # Two-particle intermediates. P(ij) on Wmnij acts on axes (2,3) -> "mnji";
        # P(ab) on Wabef acts on (0,1) -> "baef".
        zero(self.WMNIJ)
        axpy(1.0, g["oooo"], self.WMNIJ)
        ein("mnij <- je ; mnie", self.WMNIJ, T1, g["ooov"], c_pf=1.0, ab_pf=1.0)
        ein("mnji <- je ; mnie", self.WMNIJ, T1, g["ooov"], c_pf=1.0, ab_pf=-1.0)
        ein("mnij <- ijef ; mnef", self.WMNIJ, self.TAU, g["oovv"], c_pf=1.0, ab_pf=0.25)

        zero(self.WABEF)
        axpy(1.0, g["vvvv"], self.WABEF)
        ein("abef <- mb ; amef", self.WABEF, T1, g["vovv"], c_pf=1.0, ab_pf=-1.0)
        ein("baef <- mb ; amef", self.WABEF, T1, g["vovv"], c_pf=1.0, ab_pf=1.0)
        ein("abef <- mnab ; mnef", self.WABEF, self.TAU, g["oovv"], c_pf=1.0, ab_pf=0.25)

        zero(self.XJNFB)
        axpy(0.5, T2, self.XJNFB)
        ein("jnfb <- jf ; nb", self.XJNFB, T1, T1, c_pf=1.0, ab_pf=1.0)

        zero(self.WMBEJ)
        axpy(1.0, g["ovvo"], self.WMBEJ)
        ein("mbej <- jf ; mbef", self.WMBEJ, T1, g["ovvv"], c_pf=1.0, ab_pf=1.0)
        ein("mbej <- nb ; mnej", self.WMBEJ, T1, g["oovo"], c_pf=1.0, ab_pf=-1.0)
        ein("mbej <- jnfb ; mnef", self.WMBEJ, self.XJNFB, g["oovv"], c_pf=1.0, ab_pf=-1.0)

        # ── T1 ──
        ein("ia <- ie ; ae", T1N, T1, self.FAE, c_pf=0.0, ab_pf=1.0)
        ein("ia <- ma ; mi", T1N, T1, self.FMI, c_pf=1.0, ab_pf=-1.0)
        ein("ia <- imae ; me", T1N, T2, self.FME, c_pf=1.0, ab_pf=1.0)
        ein("ia <- nf ; naif", T1N, T1, g["ovov"], c_pf=1.0, ab_pf=-1.0)
        ein("ia <- imef ; maef", T1N, T2, g["ovvv"], c_pf=1.0, ab_pf=-0.5)
        ein("ia <- mnae ; nmei", T1N, T2, g["oovo"], c_pf=1.0, ab_pf=-0.5)

        # ── T2 ──
        zero(T2N)
        axpy(1.0, g["oovv"], T2N)

        zero(self.TMPVV)
        axpy(1.0, self.FAE, self.TMPVV)
        ein("be <- mb ; me", self.TMPVV, T1, self.FME, c_pf=1.0, ab_pf=-0.5)
        ein("ijab <- ijae ; be", T2N, T2, self.TMPVV, c_pf=1.0, ab_pf=1.0)
        ein("ijba <- ijae ; be", T2N, T2, self.TMPVV, c_pf=1.0, ab_pf=-1.0)

        zero(self.TMPOO)
        axpy(1.0, self.FMI, self.TMPOO)
        ein("mj <- je ; me", self.TMPOO, T1, self.FME, c_pf=1.0, ab_pf=0.5)
        ein("ijab <- imab ; mj", T2N, T2, self.TMPOO, c_pf=1.0, ab_pf=-1.0)
        ein("jiab <- imab ; mj", T2N, T2, self.TMPOO, c_pf=1.0, ab_pf=1.0)

        ein("ijab <- mnab ; mnij", T2N, self.TAU, self.WMNIJ, c_pf=1.0, ab_pf=0.5)
        ein("ijab <- ijef ; abef", T2N, self.TAU, self.WABEF, c_pf=1.0, ab_pf=0.5)

        # P(ij)P(ab)[ t2[imae] Wmbej[mbej] - t1[ie] t1[ma] g_ovvo[mbej] ]
        for out, sign in (("ijab", 1.0), ("jiab", -1.0), ("ijba", -1.0), ("jiba", 1.0)):
            ein(f"{out} <- imae ; mbej", T2N, T2, self.WMBEJ, c_pf=1.0, ab_pf=sign)
        ein("ibjm <- ie ; mbej", self.SIBJM, T1, g["ovvo"], c_pf=0.0, ab_pf=1.0)
        for out, sign in (("ijab", -1.0), ("jiab", 1.0), ("ijba", 1.0), ("jiba", -1.0)):
            ein(f"{out} <- ibjm ; ma", T2N, self.SIBJM, T1, c_pf=1.0, ab_pf=sign)

        ein("ijab <- ie ; abej", T2N, T1, g["vvvo"], c_pf=1.0, ab_pf=1.0)
        ein("jiab <- ie ; abej", T2N, T1, g["vvvo"], c_pf=1.0, ab_pf=-1.0)
        ein("ijab <- ma ; mbij", T2N, T1, g["ovoo"], c_pf=1.0, ab_pf=-1.0)
        ein("ijba <- ma ; mbij", T2N, T1, g["ovoo"], c_pf=1.0, ab_pf=1.0)

        # Amplitude update: divide by the denominators straight into t1/t2. This
        # is the step that used to drop to numpy for want of a tiled divide.
        einsums.linalg.direct_division(1.0, T1N, self.D1, 0.0, T1)
        einsums.linalg.direct_division(1.0, T2N, self.D2, 0.0, T2)

    def run(self, n_iter=60, tol=1e-11):
        e_old = self.energy()
        print(f"  MP2 (tiled)  = {e_old:.10f}")
        for it in range(n_iter):
            self.iterate()
            e_new = self.energy()
            if abs(e_new - e_old) < tol:
                print(f"  converged in {it + 1} iterations")
                return e_new
            e_old = e_new
        print("  NOT converged")
        return e_old


def ccsd_dense(g, o, v, d_ia, d_ijab, n_iter=60, tol=1e-11):
    """The identical equations in dense numpy: the oracle."""
    ee = lambda *a, **k: np.einsum(*a, optimize=True, **k)  # noqa: E731
    oovv = g[o, o, v, v]
    t1 = np.zeros((oovv.shape[0], oovv.shape[2]))
    t2 = oovv / d_ijab

    def P_ij(x):
        return x - x.transpose(1, 0, 2, 3)

    def P_ab(x):
        return x - x.transpose(0, 1, 3, 2)

    def energy(t1, t2):
        return 0.25 * ee("ijab,ijab->", oovv, t2) + 0.5 * ee("ijab,ia,jb->", oovv, t1, t1)

    e_old = energy(t1, t2)
    print(f"  MP2 (dense)  = {e_old:.10f}")
    for it in range(n_iter):
        tau_t = t2 + 0.5 * (ee("ia,jb->ijab", t1, t1) - ee("ib,ja->ijab", t1, t1))
        tau = t2 + ee("ia,jb->ijab", t1, t1) - ee("ib,ja->ijab", t1, t1)

        Fae = ee("mf,amef->ae", t1, g[v, o, v, v]) - 0.5 * ee("mnaf,mnef->ae", tau_t, oovv)
        Fmi = ee("ne,mnie->mi", t1, g[o, o, o, v]) + 0.5 * ee("inef,mnef->mi", tau_t, oovv)
        Fme = ee("nf,mnef->me", t1, oovv)

        wt = ee("je,mnie->mnij", t1, g[o, o, o, v])
        Wmnij = g[o, o, o, o] + (wt - wt.transpose(0, 1, 3, 2)) + 0.25 * ee("ijef,mnef->mnij", tau, oovv)
        wt = ee("mb,amef->abef", t1, g[v, o, v, v])
        Wabef = g[v, v, v, v] - (wt - wt.transpose(1, 0, 2, 3)) + 0.25 * ee("mnab,mnef->abef", tau, oovv)
        Wmbej = (
            g[o, v, v, o]
            + ee("jf,mbef->mbej", t1, g[o, v, v, v])
            - ee("nb,mnej->mbej", t1, g[o, o, v, o])
            - ee("jnfb,mnef->mbej", (0.5 * t2 + ee("jf,nb->jnfb", t1, t1)), oovv)
        )

        t1n = (
            ee("ie,ae->ia", t1, Fae)
            - ee("ma,mi->ia", t1, Fmi)
            + ee("imae,me->ia", t2, Fme)
            - ee("nf,naif->ia", t1, g[o, v, o, v])
            - 0.5 * ee("imef,maef->ia", t2, g[o, v, v, v])
            - 0.5 * ee("mnae,nmei->ia", t2, g[o, o, v, o])
        )
        t1n /= d_ia

        t2n = oovv.copy()
        tmp = Fae - 0.5 * ee("mb,me->be", t1, Fme)
        t2n += P_ab(ee("ijae,be->ijab", t2, tmp))
        tmp = Fmi + 0.5 * ee("je,me->mj", t1, Fme)
        t2n -= P_ij(ee("imab,mj->ijab", t2, tmp))
        t2n += 0.5 * ee("mnab,mnij->ijab", tau, Wmnij)
        t2n += 0.5 * ee("ijef,abef->ijab", tau, Wabef)
        tmp = ee("imae,mbej->ijab", t2, Wmbej) - ee("ie,ma,mbej->ijab", t1, t1, g[o, v, v, o])
        t2n += P_ij(P_ab(tmp))
        t2n += P_ij(ee("ie,abej->ijab", t1, g[v, v, v, o]))
        t2n -= P_ab(ee("ma,mbij->ijab", t1, g[o, v, o, o]))
        t2n /= d_ijab

        t1, t2 = t1n, t2n
        e_new = energy(t1, t2)
        if abs(e_new - e_old) < tol:
            print(f"  converged in {it + 1} iterations")
            return e_new, t1, t2
        e_old = e_new
    print("  NOT converged")
    return e_old, t1, t2


def main():
    n_occ = [2, 1, 1, 1]
    n_vir = [2, 2, 2, 2]
    n_aux = [3, 2, 2, 2]
    g, eps, n_o, occ_part, vir_part = build_model(n_occ, n_vir, n_aux)
    n_so = g.shape[0]
    o, v = slice(0, n_o), slice(n_o, n_so)
    eo, ev = eps[o], eps[v]
    d_ia = eo[:, None] - ev[None, :]
    d_ijab = (
        eo[:, None, None, None]
        + eo[None, :, None, None]
        - ev[None, None, :, None]
        - ev[None, None, None, :]
    )

    print(f"model: {n_so} spin orbitals ({n_o} occupied, {n_so - n_o} virtual), C2v")
    print(f"       occ blocks {occ_part}, vir blocks {vir_part}")

    print("\ndense reference:")
    e_dense, _, t2_dense = ccsd_dense(g, o, v, d_ia, d_ijab)

    blk = Blocking(occ_part, vir_part)
    specs = {
        "oovv": (o, o, v, v),
        "vovv": (v, o, v, v),
        "ooov": (o, o, o, v),
        "oooo": (o, o, o, o),
        "vvvv": (v, v, v, v),
        "ovvo": (o, v, v, o),
        "ovvv": (o, v, v, v),
        "oovo": (o, o, v, o),
        "ovov": (o, v, o, v),
        "vvvo": (v, v, v, o),
        "ovoo": (o, v, o, o),
    }
    blocks = {name: blk.make(name, name, g[sl]) for name, sl in specs.items()}

    # The integrals must already be block sparse, or the model is wrong.
    stored = blocks["oovv"].num_filled_tiles()
    print(f"\ntiled: <ij||ab> stores {stored} of {NIRREP ** 4} blocks "
          f"({100.0 * stored / NIRREP ** 4:.0f}%)")

    print("\ntiled CCSD (eager):")
    cc = TiledCCSD(blk, blocks, d_ia, d_ijab)
    e_tiled = cc.run()
    T2 = cc.T2

    t2_got = blk.gather(T2, "oovv", t2_dense.shape)
    amp_err = float(np.max(np.abs(t2_got - t2_dense)))

    print("\n" + "=" * 58)
    print(f"dense CCSD correlation = {e_dense:.12f}")
    print(f"tiled CCSD correlation = {e_tiled:.12f}")
    print(f"|difference|           = {abs(e_tiled - e_dense):.3e}")
    print(f"max |t2 tiled - dense| = {amp_err:.3e}")

    # Symmetry is preserved, not merely started with: nothing may have created a
    # block whose direct product is not totally symmetric.
    bad = [
        list(c)
        for c in itertools.product(range(NIRREP), repeat=4)
        if T2.has_tile(list(c)) and not symmetry_allowed(c)
    ]
    print(f"symmetry-forbidden t2 blocks created: {len(bad)}")

    # NOTE: capturing iterate() into a graph and running it through
    # default_pass_manager() works and is bit-identical to eager -- except that
    # GEMMBatching intermittently corrupts the result (~3% of runs) on the
    # expanded tiled graph. That is an open bug, not a limitation of this
    # example, so the pipeline arm is left out here rather than shipped flaky.
    ok = abs(e_tiled - e_dense) < 1e-10 and amp_err < 1e-9 and not bad
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
