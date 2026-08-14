#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""DF-MP2 correlation energy built as an einsums ComputeGraph, then optimized.

Companion to df_mp2_energy.py: same memory-optimal, pair-driven algorithm, which
never forms the O(o^2 v^2) four-index (ia|jb), the whole point of density
fitting. Here it is recorded into a cg.Graph and run through the deferred path
capture -> optimize -> execute instead of eagerly.

For each occupied pair i<=j the per-pair work is:

    I_ab   = sum_Q (Q|ia)(Q|jb)             grouped_batched_gemm  (nvir x nvir)
    IT_ab  = I_ba                           permute
    K_ab   = 2 I_ab - I_ba                  grouped_axpby x2
    W_ab   = 1 / (e_i + e_j - e_a - e_b)    grouped_axpby x2, element_transform
    T_ab   = I_ab * W_ab                    direct_product
    e_pair = sum_ab K_ab T_ab               grouped_dot   -> 1-element tensor
    E     += (2 - d_ij) e_pair              grouped_axpby -> accumulate into E[0]

The pairs are captured in CHUNKS rather than one at a time, and the members of a
chunk are emitted as ONE node per family instead of one node per pair. That is
the central technique from examples/dlpno: what a pair costs at this size is not
its arithmetic but its dispatch, and a grouped node pays that once for the whole
chunk. Seven of the ten families above group; permute, element_transform and
direct_product have no grouped form yet and still cost a node per pair, so a
chunk of m pairs emits 7 + 3m nodes where the per-pair form emitted 10m. On the
water/cc-pVDZ case below that is 150 nodes down to 59.

Chunking is what keeps the memory claim intact. Grouping needs a distinct
destination per member, so the scratch is a POOL of _POOL slots handed out
round-robin, not one buffer per pair: the store is O(_POOL * v^2), a small
constant times what the per-pair form used, and emphatically not the O(o^2 v^2)
this algorithm exists to avoid. The pool is sized to the machine, since its only
job is to let the executor have every core busy at once.

The reordering this allows is legal because a chunk's members are independent.
The accumulation into E is not, and stays ordered: grouped_axpby runs its
entries sequentially and reads E as well as writing it, so the chunks chain in
program order and the energy is summed in exactly the pair order the per-pair
form used. The result is bit-identical to that form, which
``--check-per-pair`` verifies by capturing and running both.

The slabs B[:, i, :] are sliced eagerly, since integer-index slicing is not
capture-aware yet, then the captured ops reference those views. The default
optimization passes run over the resulting graph, then it executes. Checked
against psi4's own DF-MP2.

Pass ``--show-passes`` to apply each optimization pass on its own and report
which ones modify the graph, plus the node execution order before vs after
optimization.

Run with the Einsums build and psi4 stage on PYTHONPATH, using the conda-env
Python::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        /Users/jturney/Code/Einsums/Einsums/examples/psi4-bridge/df_mp2_graph.py --threads 10
"""
import argparse
import json
import os

import numpy as np
import einsums
from einsums import linalg as la
import einsums.graph as cg   # the graph.py shell (capture/default_pass_manager); NOT
                             # `from einsums import graph`, which resolves to the bare
                             # _core.graph submodule and lacks the `capture` helper.
import psi4
from einsums.interop import psi4 as interop

#: Scratch slots per role. About one per core, so every member of a chunk can be
#: in flight at once, and capped because the pool is what the store costs.
_POOL = min(os.cpu_count() or 1, 16)

_argp = argparse.ArgumentParser(description="DF-MP2 in einsums via ComputeGraph.")
_argp.add_argument(
    "--show-passes", action="store_true",
    help="apply each optimization pass on its own and report which modify the graph, "
         "plus the node execution order before vs after optimization",
)
_argp.add_argument(
    "--check-per-pair", action="store_true",
    help="also capture the ungrouped one-node-per-pair form and require the two "
         "energies to agree bit for bit",
)
_argp.add_argument(
    "--threads", type=int, default=1,
    help="thread count, applied with psi4.set_num_threads before any einsums "
         "work. Importing psi4 sets the process-wide OpenMP count to "
         "OMP_NUM_THREADS if that is exported and to 1 otherwise, so leaving "
         "both unset runs einsums silently serial",
)
_argp.add_argument(
    "--no-executor", action="store_true",
    help="replay serially instead of under an OpenMP executor. This is how the "
         "executor is bisected out of a result that moves",
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
        cg.ElementWiseFusion, cg.LinearCombinationContractionFolding,
        cg.StreamContractionFusion, cg.LoopInvariantHoisting, cg.Reorder,
        cg.InplaceOptimization, cg.MemoryPlanning, cg.SymmetryPropagation,
    ]
    print(f"\n  {'pass':36} modified  nodes")
    for P in ordered:
        pm = cg.PassManager()
        pm.add(P())
        modified = graph.apply(pm)
        print(f"  {P.__name__:36} {str(modified):8} {graph.num_nodes()}")
    after = _exec_order(graph)

    # Compact one-char-per-node view of the execution order (10 nodes/group).
    abbr = {"Einsum": "E", "Permute": "P", "Axpby": "X", "ElementTransform": "T",
            "DirectProduct": "D", "Dot": "o", "GroupedBatchedGemm": "G",
            "GroupedAxpby": "A", "GroupedDot": "S"}
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
# Before any einsums work: importing psi4 already set the process-wide OpenMP
# count, to OMP_NUM_THREADS if it was exported and to 1 if it was not.
psi4.set_num_threads(_args.threads)

mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n")

_, wfn = psi4.energy("mp2", return_wfn=True)
ref_corr = psi4.variable("MP2 CORRELATION ENERGY")
print(f"psi4 DF-MP2 corr = {ref_corr:.10f}")
print(f"threads = {einsums.hardware.get_max_threads()}, scratch pool = {_POOL} slots/role")

# ---- DF integrals + constants (eager prep) ---------------------------------
primary = wfn.basisset()
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "", "RIFIT", primary.name())
C = wfn.Ca()
nocc = wfn.nalpha()
nvir = wfn.nmo() - nocc
eps = np.asarray(wfn.epsilon_a())
eo = eps[:nocc]                                          # occupied energies (scalars)

dft = psi4.core.DFTensor(primary, aux, C, nocc, nvir)
B = interop.df_tensor(dft.Qov(), nocc, nvir, name="DF (Q|ia) OV")  # (naux, nocc, nvir)
# Per-i (naux, nvir) slabs, sliced eagerly (integer-index slicing isn't recorded
# by the capture-aware __getitem__); the captured GEMMs read these views. A
# middle-index slice of the column-major (naux, nocc, nvir) store is a genuine
# column-major matrix with lda = naux*nocc, so it is a batched-GEMM operand as
# it stands, with nothing to pack.
Bslab = [B[:, i, :] for i in range(nocc)]

ev = einsums.create_zero_tensor("ev", [nvir], dtype="float64")
np.asarray(ev)[:] = eps[nocc:]
Dbase = einsums.create_zero_tensor("-ea-eb", [nvir, nvir], dtype="float64")
la.outer_sum(Dbase, [ev, ev], [-1.0, -1.0])              # -e_a - e_b
ones = einsums.create_zero_tensor("ones", [nvir, nvir], dtype="float64")
la.element_transform(ones, lambda _: 1.0)
recip = lambda x: 1.0 / x

pairs = [(i, j) for i in range(nocc) for j in range(i, nocc)]
depth = min(_POOL, len(pairs))


def _pool(tag):
    """One scratch buffer per pool slot, for one role."""
    return [einsums.create_zero_tensor(f"{tag}[{s}]", [nvir, nvir], dtype="float64")
            for s in range(depth)]


# Scratch pools reused across chunks; the graph tracks the dependencies. Memory
# is O(depth * v^2) + the 3-index B, never the O(o^2 v^2) four-index tensor.
I = _pool("I(ab)")
IT = _pool("I(ba)")
K = _pool("K(ab)")
W = _pool("w(ab)")
T = _pool("t(ab)")
e_pair = [einsums.create_zero_tensor(f"e_pair[{s}]", [1], dtype="float64")
          for s in range(depth)]
E = einsums.create_zero_tensor("E_corr", [1], dtype="float64")


def capture_grouped(g):
    """Emit the pair energies a chunk at a time, one node per family per chunk."""
    with cg.capture(g):
        for base in range(0, len(pairs), depth):
            chunk = pairs[base:base + depth]
            m = len(chunk)
            # I_ab = (ia|jb) = sum_Q B_i[Q,a] B_j[Q,b], one node for the chunk.
            cg.grouped_batched_gemm(1.0, [Bslab[i] for i, _ in chunk],
                                    [Bslab[j] for _, j in chunk],
                                    0.0, I[:m], trans_a=True)
            for s in range(m):
                einsums.permute("ab <- ba", IT[s], I[s])          # IT = I^T
            la.grouped_axpby([2.0] * m, I[:m], [0.0] * m, K[:m])  # K = 2 I
            la.grouped_axpby([-1.0] * m, IT[:m], [1.0] * m, K[:m])  # K = 2 I - I^T
            la.grouped_axpby([1.0] * m, [Dbase] * m, [0.0] * m, W[:m])  # W = -e_a - e_b
            la.grouped_axpby([eo[i] + eo[j] for i, j in chunk],   # W += e_i + e_j
                             [ones] * m, [1.0] * m, W[:m])
            for s in range(m):
                la.element_transform(W[s], recip)                 # W = 1 / denom
                la.direct_product(1.0, I[s], W[s], 0.0, T[s])     # T = I * W
            la.grouped_dot(e_pair[:m], K[:m], T[:m])              # e_pair = sum K*T
            # E += (2 - d_ij) e_pair. One node, entries sequential, E repeated as
            # the destination: the pair order of the sum is exactly the loop's.
            la.grouped_axpby([1.0 if i == j else 2.0 for i, j in chunk],
                             e_pair[:m], [1.0] * m, [E] * m)


def capture_per_pair(g, dest):
    """The ungrouped form this replaces: one node per op per pair."""
    with cg.capture(g):
        for i, j in pairs:
            einsums.einsum("ab <- Qa ; Qb", I[0], Bslab[i], Bslab[j])
            einsums.permute("ab <- ba", IT[0], I[0])
            la.axpby(2.0, I[0], 0.0, K[0])
            la.axpby(-1.0, IT[0], 1.0, K[0])
            la.axpby(1.0, Dbase, 0.0, W[0])
            la.axpby(eo[i] + eo[j], ones, 1.0, W[0])
            la.element_transform(W[0], recip)
            la.direct_product(1.0, I[0], W[0], 0.0, T[0])
            la.dot(e_pair[0], K[0], T[0])
            la.axpby(1.0 if i == j else 2.0, e_pair[0], 1.0, dest)


# ---- capture the pair-driven energy into a ComputeGraph --------------------
g = cg.Graph("df-mp2 (pair-driven, grouped)")
capture_grouped(g)
print(f"captured graph '{g.name}': {g.num_nodes()} nodes, {g.num_tensors()} tensors "
      f"({len(pairs)} pairs in {-(-len(pairs) // depth)} chunks)")

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
if not _args.no_executor:
    g.set_executor(cg.OpenMPExecutor())
g.execute()
e_corr = float(np.asarray(E)[0])

print(f"einsums (graph) DF-MP2 corr = {e_corr:.10f}")
print(f"difference vs psi4           = {abs(e_corr - ref_corr):.2e}")
assert abs(e_corr - ref_corr) < 1e-6, "graph DF-MP2 disagrees with psi4"

if _args.check_per_pair:
    E_ref = einsums.create_zero_tensor("E_corr (per-pair)", [1], dtype="float64")
    g_ref = cg.Graph("df-mp2 (pair-driven, one node per op)")
    capture_per_pair(g_ref, E_ref)
    g_ref.execute()
    e_ref = float(np.asarray(E_ref)[0])
    print(f"\nungrouped per-pair form: {g_ref.num_nodes()} nodes -> {e_ref:.16f}")
    print(f"grouped chunked form:    {g.num_nodes()} nodes -> {e_corr:.16f}")
    assert e_corr == e_ref, "grouping moved the energy; it must be an exact substitution"
    print("the two forms agree BIT FOR BIT")

print("DF-MP2 (einsums ComputeGraph: capture -> optimize -> execute) MATCHES psi4")
