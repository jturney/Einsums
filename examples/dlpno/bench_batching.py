#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""What the uniform pair-block layout buys in the local MP2 residual.

psi4's DLPNO keeps each pair's blocks in its own ``SharedMatrix``, sized to that
pair's PNO count. The residual's Fock coupling is then a loop over k of
``triplet(S, T, S)`` over operands that share no shape and no allocation, and
nothing downstream can batch them.

``dlpno/mp2.py`` writes the same loop, but over blocks padded to a single
``npno_max`` inside one contiguous store. That one change is what lets
``GEMMBatching`` collapse every ``(pair, k)`` coupling into a handful of
``blas::gemm_batch`` calls.

Three configurations are timed over identical data:

* **ragged + passes** - per-pair tensors at their natural sizes, psi4's layout.
* **padded, no passes** - the shipped layout with the optimizer switched off.
* **padded + passes** - what the port actually runs.

Run with the Einsums build and the psi4 stage on PYTHONPATH::

    PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/stage/lib \
        python examples/dlpno/bench_batching.py --molecule water-dimer
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import psi4

import einsums
import einsums.graph as cg

from dlpno import DLPNOMP2, Thresholds
from dlpno import tensors as ten
from dlpno.molecules import MOLECULES
from dlpno.psi4_source import from_psi4

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--molecule", default="water-dimer", choices=sorted(MOLECULES))
parser.add_argument("--basis", default="cc-pvdz")
parser.add_argument("--t-cut-pno", type=float, default=1e-8)
parser.add_argument("--reps", type=int, default=20)
args = parser.parse_args()

psi4.core.set_output_file("/tmp/psi4_dlpno_bench.out", False)
psi4.set_options({
    "basis": args.basis, "scf_type": "df", "mp2_type": "df", "freeze_core": "false",
    "e_convergence": 1e-10, "d_convergence": 1e-10,
})
psi4.geometry(MOLECULES[args.molecule] + "symmetry c1\n")
_, wfn = psi4.energy("mp2", return_wfn=True)

mp2 = DLPNOMP2(from_psi4(wfn), Thresholds.preset("NORMAL", t_cut_pno=args.t_cut_pno), verbose=False)
mp2.setup_orbitals().prep_sparsity().compute_metric().compute_qia()
mp2.pno_transform().compute_pno_overlaps()
mp2._allocate_iteration_tensors()

naocc = mp2.ref.naocc
M = mp2.npno_max
counts = sorted({n for n in mp2.n_pno if n})
n_couplings = sum(len(mp2.k_couple_kj[ij]) + len(mp2.k_couple_ik[ij])
                  for ij in range(mp2.n_lmo_pairs))
print(f"\n{args.molecule}/{args.basis}: {naocc} LMOs, {mp2.n_lmo_pairs} pairs, "
      f"{n_couplings} couplings")
print(f"  PNO counts per pair: {len(counts)} distinct values, {counts[0]}..{counts[-1]}; "
      f"padded to {M}")

# Non-trivial amplitudes so nothing folds away, keeping T_ji = T_ij^T.
rng = np.random.default_rng(0)
T_view = ten.view(mp2.T_all)
for ij, n in enumerate(mp2.n_pno):
    if n:
        T_view[:n, :n, ij] = rng.random((n, n))
for ij, (i, j) in enumerate(mp2.ij_to_i_j):
    if i < j and mp2.n_pno[ij]:
        T_view[:, :, mp2.ij_to_ji[ij]] = T_view[:, :, ij].T


def capture_ragged():
    """psi4's layout: one independently sized tensor per pair.

    Written with ``einsum`` rather than ``linalg.gemm`` deliberately. A
    2D x 2D -> 2D einsum with a single link index carries the ``gemm_hint``
    that ``GEMMBatching`` groups on, whereas ``linalg.gemm`` captures as
    ``OpKind::Gemm`` and the pass skips it outright. So this is the best case
    for psi4's shape, not a straw man: the only thing it gives up is uniformity.
    """
    S_kj, S_ik, T_pair, R_pair, tmp = {}, {}, {}, {}, {}
    for ij in range(mp2.n_lmo_pairs):
        n = mp2.n_pno[ij]
        if n == 0:
            continue
        T_pair[ij] = ten.from_numpy(f"T{ij}", ten.view(mp2.T_all)[:n, :n, ij])
        R_pair[ij] = ten.zeros(f"R{ij}", [n, n])
    for ij, (i, j) in enumerate(mp2.ij_to_i_j):
        n = mp2.n_pno[ij]
        for k in mp2.k_couple_kj[ij]:
            nk = mp2.n_pno[int(mp2.i_j_to_ij[k, j])]
            S_kj[ij, k] = ten.from_numpy(f"Skj{ij}_{k}",
                                         ten.view(mp2.S_pno_ij_kj[ij])[:n, :nk, k])
            tmp[0, ij, k] = ten.zeros(f"t0_{ij}_{k}", [n, nk])
        for k in mp2.k_couple_ik[ij]:
            nk = mp2.n_pno[int(mp2.i_j_to_ij[i, k])]
            S_ik[ij, k] = ten.from_numpy(f"Sik{ij}_{k}",
                                         ten.view(mp2.S_pno_ij_ik[ij])[:n, :nk, k])
            tmp[1, ij, k] = ten.zeros(f"t1_{ij}_{k}", [n, nk])
    keep = (S_kj, S_ik, T_pair, R_pair, tmp)

    g = cg.Graph("coupling (ragged, psi4 layout)")
    with cg.capture(g):
        for ij, (i, j) in enumerate(mp2.ij_to_i_j):
            if mp2.n_pno[ij] == 0:
                continue
            for k, factor in mp2.k_couple_kj[ij].items():
                kj = int(mp2.i_j_to_ij[k, j])
                einsums.einsum("ad <- ac ; cd", tmp[0, ij, k], S_kj[ij, k], T_pair[kj])
                einsums.einsum("ab <- ad ; bd", R_pair[ij], tmp[0, ij, k], S_kj[ij, k],
                               c_pf=1.0, ab_pf=factor)
            for k, factor in mp2.k_couple_ik[ij].items():
                ik = int(mp2.i_j_to_ij[i, k])
                einsums.einsum("ad <- ac ; cd", tmp[1, ij, k], S_ik[ij, k], T_pair[ik])
                einsums.einsum("ab <- ad ; bd", R_pair[ij], tmp[1, ij, k], S_ik[ij, k],
                               c_pf=1.0, ab_pf=factor)
    return g, keep, R_pair


def timeit(graph, reps):
    graph.execute()
    t0 = time.perf_counter()
    for _ in range(reps):
        graph.execute()
    return (time.perf_counter() - t0) / reps * 1e3


def optimized(graph):
    pm = cg.PassManager()
    pm.populate_default()
    graph.apply(pm)
    return graph


rows = []


def coupling_only(graph, zero):
    """Run the coupling graph once over a zeroed residual and snapshot it.

    The coupling graph accumulates (``c_pf=1.0``); in the solver the prologue
    rewrites R each iteration. Here it is zeroed by hand so one replay gives
    the coupling contribution alone, which is what the three configurations
    have to agree on. Timing runs let it accumulate, which changes no flops.
    """
    zero()
    graph.execute()


g = mp2._capture_residual()
coupling_only(g, lambda: ten.view(mp2.R_all).fill(0.0))
R_ref = ten.view(mp2.R_all).copy()
rows.append(("padded, no passes", g.num_nodes(), g.num_nodes(), timeit(g, args.reps)))

g = mp2._capture_residual()
before = g.num_nodes()
after = optimized(g).num_nodes()
coupling_only(g, lambda: ten.view(mp2.R_all).fill(0.0))
err = np.abs(ten.view(mp2.R_all) - R_ref).max() / max(np.abs(R_ref).max(), 1e-300)
assert err < 1e-10, f"passes changed the coupling by {err:.3e} relative"
rows.append(("padded + passes (shipped)", before, after, timeit(g, args.reps)))

g_rag, _keep, R_pair = capture_ragged()
before = g_rag.num_nodes()
after = optimized(g_rag).num_nodes()
coupling_only(g_rag, lambda: [ten.view(R).fill(0.0) for R in R_pair.values()])
scale = max(np.abs(R_ref).max(), 1e-300)
worst = max(
    (np.abs(ten.view(R_pair[ij]) - R_ref[:mp2.n_pno[ij], :mp2.n_pno[ij], ij]).max()
     for ij in range(mp2.n_lmo_pairs) if mp2.n_pno[ij]),
    default=0.0,
)
assert worst / scale < 1e-10, f"ragged coupling disagrees by {worst / scale:.3e} relative"
rows.append(("ragged + passes (psi4 layout)", before, after, timeit(g_rag, args.reps)))

base = next(ms for label, _, _, ms in rows if label.startswith("ragged"))
print(f"\n  {'configuration':32} {'nodes':>14} {'ms/replay':>11} {'speedup':>9}")
for label, nb, na, ms in rows:
    print(f"  {label:32} {nb:>6} -> {na:<5} {ms:>11.3f} {base / ms:>8.2f}x")

print("\n  All three agree on the coupling to 1e-10 relative.")
print("  Padding to one shape is what lets GEMMBatching group every (pair, k)")
print("  contraction into a few gemm_batch calls; the ragged form can only")
print("  group pairs that happen to share a PNO count.")
