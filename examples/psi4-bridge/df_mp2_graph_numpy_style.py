#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""DF-MP2 as an einsums ComputeGraph, written in numpy-style einsums.

The graph companion to df_mp2_numpy_style.py, and the numpy-style twin of
df_mp2_graph.py. It is the same memory-optimal, pair-driven DF-MP2, which never
forms the O(o^2 v^2) four-index (ia|jb), recorded into a cg.Graph and run through
the deferred path capture -> optimize -> execute, but with the per-pair tensor
math expressed via einsums' numpy-like operators wherever they are capture-safe.

Per occupied pair i<=j, captured into one graph:

    I      = Bi.T @ Bj                        .T view + @ (gemm)
    K      = 2.0 * I - I.T                     scalar*, -, .T view
    W      = Dbase (reused); W += e_i+e_j      in-place scalar shift (W = denom)
    T      = I / W                            /                         (Hadamard divide operator)
    e_pair = sum_ab K_ab T_ab                 einsums.linalg.dot        (reduction -> [1])
    E     += (2 - d_ij) * e_pair              scalar*, +=               (operators -> E[0])

Every operator, including ``Bi.T @ Bj`` and ``2.0 * I - I.T``, is recorded
into the graph. ``.T`` is capture-safe: inside ``cg.capture`` it routes through
``cg.permute_view``, which records a graph-registered, parent-aliasing transpose
view. The raw ``transpose_view`` shares the parent's data pointer and is
invisible to the graph, which is why a naive ``.T`` in capture used to fault.
Operator outputs are allocated graph-owned, so they survive optimize/execute.

The denominator W is reused scratch rather than a fresh per-pair tensor:
graph-owned operator results live for the whole execution, so allocating one
per pair would pin nocc^2 denominators in memory. Overwriting W from Dbase and
shifting it in place with ``W += (e_i + e_j)``, the in-place scalar form that
allocates nothing, keeps the denominator footprint at O(v^2).

Only one op has no operator spelling and stays explicit, the same as the eager
numpy-style file: the full reduction dot, written into a 1-element tensor for
in-graph accumulation. The amplitude denominator is now the '/' operator backed
by direct_division, which records a graph DirectDivision node with no per-element
reciprocal callback, so the whole per-pair body is operators plus one dot.

numpy itself appears only to ingest psi4 data and read scalar orbital energies.
Checked against psi4's own DF-MP2.

Pass ``--show-passes`` to apply each optimization pass on its own and report
which ones modify the graph, plus the node execution order before vs after.

Run with the Einsums build and psi4 stage on PYTHONPATH, using the conda-env
Python::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        /Users/jturney/Code/Einsums/Einsums/examples/psi4-bridge/df_mp2_graph_numpy_style.py
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

_argp = argparse.ArgumentParser(description="DF-MP2 in numpy-style einsums via ComputeGraph.")
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

# ---- DF integrals + constants (eager prep, numpy-style ingest) -------------
primary = wfn.basisset()
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "", "RIFIT", primary.name())
C = wfn.Ca()
nocc = wfn.nalpha()
nvir = wfn.nmo() - nocc
eps = np.asarray(wfn.epsilon_a())
eo = eps[:nocc]                                          # occupied energies (scalars)
ev_np = eps[nocc:]                                       # virtual energies (numpy, scalars)

dft = psi4.core.DFTensor(primary, aux, C, nocc, nvir)
B = dft.Qov_einsums()                                    # (naux, nocc, nvir)
print(f"B (Q|ov) shape = {B.shape}")

# D_ab = -e_a - e_b, ingested with einsums.asarray (one-time setup, not loop math).
Dbase = einsums.asarray(-ev_np[:, None] - ev_np[None, :], name="-ea-eb")

# e_pair / E accumulate across pairs. W is a reused denominator scratch:
# unlike eager mode (where operator results are GC'd), every operator output in
# a graph is a graph-owned tensor that lives for the whole execution, so a
# per-pair ``Dbase + scalar`` would leave nocc^2 denominator tensors resident.
# Reusing one W (overwrite it from Dbase, then shift in place with ``+=``) keeps
# the denominator at O(v^2). I/K/T are still fresh per pair to stay readable.
e_pair = einsums.zeros((1,), name="e_pair")
E = einsums.zeros((1,), name="E_corr")
W = einsums.zeros((nvir, nvir), name="w(ab)")

# ---- capture the pair-driven energy into a ComputeGraph --------------------
g = cg.Graph("df-mp2 numpy-style (pair-driven)")
with cg.capture(g):
    for i in range(nocc):
        Bi = B[:, i, :]                                             # (naux, nvir) slab: rank-reducing index, recorded in capture
        for j in range(i, nocc):
            Bj = B[:, j, :]
            I = Bi.T @ Bj                           # I_ab = (ia|jb)
            K = 2.0 * I - I.T                       # K = 2 I - I^T
            la.axpby(1.0, Dbase, 0.0, W)            # W = Dbase  (reuse, no alloc)
            W += eo[i] + eo[j]                      # W = denom = (e_i+e_j) - e_a - e_b  (in-place scalar shift)
            T = I / W                               # T = I / denom  (Hadamard divide)
            la.dot(e_pair, K, T)                    # e_pair = sum K*T
            E += (1.0 if i == j else 2.0) * e_pair  # E += (2-d_ij) e_pair
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

print(f"einsums (numpy-style graph) DF-MP2 corr = {e_corr:.10f}")
print(f"difference vs psi4                       = {abs(e_corr - ref_corr):.2e}")
assert abs(e_corr - ref_corr) < 1e-6, "numpy-style graph DF-MP2 disagrees with psi4"
print("DF-MP2 (numpy-style einsums ComputeGraph: capture -> optimize -> execute) MATCHES psi4")
