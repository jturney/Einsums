#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""DF-MP2 correlation energy built as an einsums ComputeGraph, then optimized.

Companion to df_mp2_energy.py: same memory-optimal, pair-driven algorithm, which
never forms the O(o^2 v^2) four-index (ia|jb), the whole point of density
fitting. Here it is recorded into a cg.Graph and run through the deferred path
capture -> optimize -> execute instead of eagerly.

For each occupied pair i<=j the per-pair work is captured:

    I_ab   = sum_Q (Q|ia)(Q|jb)             einsum   (nvir x nvir GEMM)
    K_ab   = 2 I_ab - I_ba                  permute, axpby, axpby
    W_ab   = 1 / (e_i + e_j - e_a - e_b)     axpby, axpby, element_transform
    T_ab   = I_ab * W_ab                     direct_product
    e_pair = sum_ab K_ab T_ab                dot           -> 1-element tensor
    E     += (2 - d_ij) e_pair               axpby         -> accumulate into E[0]

The slabs B[:, i, :] are sliced eagerly, since integer-index slicing is not
capture-aware yet, then the captured ops reference those views. The default
optimization passes run over the resulting graph at roughly 10 nodes per pair,
then it executes. Checked against psi4's own DF-MP2.

Pass ``--show-passes`` to apply each optimization pass on its own and report
which ones modify the graph, plus the node execution order before vs after
optimization. For this graph the only mover is Reorder, which reschedules the
nodes; there is no redundancy for CSE/DNE/fusion to exploit.

Run with the Einsums build and psi4 stage on PYTHONPATH, using the conda-env
Python::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        /Users/jturney/Code/Einsums/Einsums/examples/psi4-bridge/df_mp2_graph.py
"""
import argparse
import json

import numpy as np
import einsums
from einsums import linalg as la
import einsums.graph as cg   # the graph.py shell (capture/default_pass_manager); NOT
                             # `from einsums import graph`, which resolves to the bare
                             # _core.graph submodule and lacks the `capture` helper.
import psi4

_argp = argparse.ArgumentParser(description="DF-MP2 in einsums via ComputeGraph.")
_argp.add_argument(
    "--show-passes", action="store_true",
    help="apply each optimization pass on its own and report which modify the graph, "
         "plus the node execution order before vs after optimization",
)
_args = _argp.parse_args()


def _exec_order(graph):
    """Node 'kind's in execution order (the to_json 'nodes' array order)."""
    return [n["kind"] for n in json.loads(graph.to_json())["nodes"]]


def _optimize_verbose(graph):
    """Apply each Python-exposed pass on its own, in pipeline order, reporting
    which ones modify the graph and how the execution order changes."""
    before = _exec_order(graph)
    ordered = [
        cg.ConstantFolding, cg.ScaleAbsorption, cg.CSE, cg.DeadNodeElimination,
        cg.ElementWiseFusion, cg.LoopInvariantHoisting, cg.Reorder,
        cg.InplaceOptimization, cg.MemoryPlanning, cg.SymmetryPropagation,
        cg.ChainParenthesization,
    ]
    print(f"\n  {'pass':24} modified  nodes")
    for P in ordered:
        pm = cg.PassManager()
        pm.add(P())
        modified = graph.apply(pm)
        print(f"  {P.__name__:24} {str(modified):8} {graph.num_nodes()}")
    after = _exec_order(graph)

    # Compact one-char-per-node view of the execution order (10 nodes/group).
    abbr = {"Einsum": "E", "Permute": "P", "Axpby": "X", "ElementTransform": "T",
            "DirectProduct": "D", "Dot": "o"}
    spare = iter("123456789")
    for k in dict.fromkeys(before + after):
        abbr.setdefault(k, next(spare))

    def fmt(kinds):
        return " ".join("".join(abbr[k] for k in kinds[p:p + 10]) for p in range(0, len(kinds), 10))

    print("\n  legend: " + ", ".join(f"{c}={k}" for k, c in abbr.items() if k in set(before)))
    print(f"\n  execution order BEFORE optimization ({len(before)} nodes):\n    {fmt(before)}")
    print(f"\n  execution order AFTER  optimization ({len(after)} nodes):\n    {fmt(after)}")
    moved = sum(1 for a, b in zip(before, after) if a != b)
    print(f"\n  {moved} of {len(before)} positions changed node kind after reordering")

psi4.core.set_output_file("/tmp/psi4_df_mp2_graph.out", False)
psi4.set_options({
    "basis": "cc-pvdz", "scf_type": "df", "mp2_type": "df",
    "freeze_core": "false", "e_convergence": 1e-10, "d_convergence": 1e-10,
})

mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n")

_, wfn = psi4.energy("mp2", return_wfn=True)
ref_corr = psi4.variable("MP2 CORRELATION ENERGY")
print(f"psi4 DF-MP2 corr = {ref_corr:.10f}")

# ---- DF integrals + constants (eager prep) ---------------------------------
primary = wfn.basisset()
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "", "RIFIT", primary.name())
C = wfn.Ca()
nocc = wfn.nalpha()
nvir = wfn.nmo() - nocc
eps = np.asarray(wfn.epsilon_a())
eo = eps[:nocc]                                          # occupied energies (scalars)

dft = psi4.core.DFTensor(primary, aux, C, nocc, nvir)
B = dft.Qov_einsums()                                    # (naux, nocc, nvir)
# Per-i (naux, nvir) slabs, sliced eagerly (integer-index slicing isn't recorded
# by the capture-aware __getitem__); the captured einsums read these views.
Bslab = [B[:, i, :] for i in range(nocc)]

ev = einsums.create_zero_tensor("ev", [nvir], dtype="float64")
np.asarray(ev)[:] = eps[nocc:]
Dbase = einsums.create_zero_tensor("-ea-eb", [nvir, nvir], dtype="float64")
la.outer_sum(Dbase, [ev, ev], [-1.0, -1.0])              # -e_a - e_b
ones = einsums.create_zero_tensor("ones", [nvir, nvir], dtype="float64")
la.element_transform(ones, lambda _: 1.0)
recip = lambda x: 1.0 / x

# Scratch reused across pairs; the graph tracks the dependencies (the passes can
# break the reuse for scheduling). Memory stays O(v^2) + the 3-index B.
I = einsums.create_zero_tensor("I(ab)", [nvir, nvir], dtype="float64")
IT = einsums.create_zero_tensor("I(ba)", [nvir, nvir], dtype="float64")
K = einsums.create_zero_tensor("K(ab)", [nvir, nvir], dtype="float64")
W = einsums.create_zero_tensor("w(ab)", [nvir, nvir], dtype="float64")
T = einsums.create_zero_tensor("t(ab)", [nvir, nvir], dtype="float64")
e_pair = einsums.create_zero_tensor("e_pair", [1], dtype="float64")
E = einsums.create_zero_tensor("E_corr", [1], dtype="float64")

# ---- capture the pair-driven energy into a ComputeGraph --------------------
g = cg.Graph("df-mp2 (pair-driven)")
with cg.capture(g):
    for i in range(nocc):
        for j in range(i, nocc):
            einsums.einsum("ab <- Qa ; Qb", I, Bslab[i], Bslab[j])  # I_ab = (ia|jb)
            einsums.permute("ab <- ba", IT, I)                       # IT = I^T
            la.axpby(2.0, I, 0.0, K)                                 # K = 2 I
            la.axpby(-1.0, IT, 1.0, K)                               # K = 2 I - I^T
            la.axpby(1.0, Dbase, 0.0, W)                             # W = -e_a - e_b
            la.axpby(eo[i] + eo[j], ones, 1.0, W)                    # W += e_i + e_j
            la.element_transform(W, recip)                           # W = 1 / denom
            la.direct_product(1.0, I, W, 0.0, T)                     # T = I * W
            la.dot(e_pair, K, T)                                     # e_pair = sum K*T
            la.axpby(1.0 if i == j else 2.0, e_pair, 1.0, E)         # E += (2-d_ij) e_pair
print(f"captured graph '{g.name}': {g.num_nodes()} nodes, {g.num_tensors()} tensors")

# ---- run the optimization passes -------------------------------------------
if _args.show_passes:
    _optimize_verbose(g)
else:
    pm = cg.PassManager()
    pm.populate_default()
    before = g.num_nodes()
    modified = g.apply(pm)
    print(f"optimization: {pm.size} passes, modified={modified}, nodes {before} -> {g.num_nodes()}")

# ---- execute the (optimized) graph -----------------------------------------
g.execute()
e_corr = float(np.asarray(E)[0])

print(f"einsums (graph) DF-MP2 corr = {e_corr:.10f}")
print(f"difference vs psi4           = {abs(e_corr - ref_corr):.2e}")
assert abs(e_corr - ref_corr) < 1e-6, "graph DF-MP2 disagrees with psi4"
print("DF-MP2 (einsums ComputeGraph: capture -> optimize -> execute) MATCHES psi4")
