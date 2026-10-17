# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""A tensor that lives in a file: one graph writes it a slab at a time, another reads it back.

The virtual-virtual block of the density-fitted integrals, Bvv[Q,a,b], is the
largest tensor a DF-CCSD program holds, and the particle ladder reads it
twice per iteration. Annotated with a backing, it never has to be held whole.

The first graph produces it. Annotated with ``einsums.io.etn(path, entry,
write=True)``, Bvv is stored to that entry of an ``.etn`` file. The
contraction that produces it runs in a loop, and each iteration writes the
block of Bvv it computed: AxisTiling's loop over the half-transformed tensor
when that one is too large as well, as here, or a loop of SlabStreaming's own
over Bvv's last axis.

The second graph reads it. Annotated with ``einsums.io.etn(path, entry)``,
Bvv is filled from the file at every replay. Each contraction that reads it
reads one block at a time: the box AxisTiling's loop names, or a block of
Bvv's last axis in a loop of SlabStreaming's own.

Neither graph asks for any of this beyond the annotation. The default pipeline
streams a backed tensor when it is larger than its share of
``--einsums:max-memory``, set small below so that a cc-pVTZ water is enough to
show it. A backed tensor that fits is loaded or stored whole, once per replay.

Run with::

    PYTHONPATH=<einsums-build>/lib python out_of_core.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

import einsums.rc

# A budget of 16 MB gives each tensor a share of 2.6 MB, which the 3.2 MB Bvv of
# cc-pVTZ water exceeds. The runtime reads it when it starts, so it is set first.
einsums.rc.max_memory = "16MB"

import einsums  # noqa: E402
import einsums.graph as cg  # noqa: E402

_FIXTURE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "examples", "dlpno", "fixtures", "water-ccpvtz.npz"))


def _tensor(name, array):
    tensor = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(tensor)[...] = np.ascontiguousarray(array)
    return tensor


def load():
    """Canonical virtual orbitals, the fitted AO three-index tensor, and stand-in amplitudes."""
    z = np.load(_FIXTURE, allow_pickle=False)
    overlap, fock = z["S"], z["F"]
    values, vectors = np.linalg.eigh(overlap)
    orthogonalizer = vectors @ np.diag(values ** -0.5) @ vectors.T
    energies, rotation = np.linalg.eigh(orthogonalizer.T @ fock @ orthogonalizer)
    coefficients = orthogonalizer @ rotation
    nocc = int(z["C_occ"].shape[1])

    mvals, mvecs = np.linalg.eigh(z["metric"])
    inv_sqrt = mvecs @ np.diag(np.where(mvals > 1e-10, mvals ** -0.5, 0.0)) @ mvecs.T
    fitted = np.einsum("PQ,Qmn->Pmn", inv_sqrt, z["eri_3index"])

    # MP2 amplitudes from the same integrals stand in for an iteration's tau.
    cv = coefficients[:, nocc:]
    co = coefficients[:, :nocc]
    bov = np.einsum("Qmn,mi,na->Qia", fitted, co, cv)
    eo, ev = energies[:nocc], energies[nocc:]
    denominators = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
    tau = np.einsum("Qia,Qjb->ijab", bov, bov) / denominators
    return {"fitted": fitted, "cv": np.ascontiguousarray(cv), "tau": tau}


def produce(path, problem):
    """Bvv[Q,a,b] = sum_mn (Q|mn) C[m,a] C[n,b], stored to the file and never held whole."""
    fitted, cv = problem["fitted"], problem["cv"]
    naux, nbf, nvir = fitted.shape[0], fitted.shape[1], cv.shape[1]

    graph = cg.Graph("produce_bvv")
    bvv = graph.declare_tensor("Bvv", [naux, nvir, nvir])
    cg.annotate(bvv, backing=einsums.io.etn(path, "Bvv", write=True), graph=graph)
    # Like H in the ladder, only this graph reads the half-transformed tensor, so it is an
    # intermediate and the graph may build it a block at a time.
    half = graph.declare_tensor("half", [naux, nvir, nbf], intermediate=True)
    P = _tensor("P", fitted)
    C = _tensor("Cv", cv)
    with cg.capture(graph):
        einsums.einsum("Q,a,n <- Q,m,n ; m,a", half, P, C)
        einsums.einsum("Q,a,b <- Q,a,n ; n,b", bvv, half, C)
    return graph


def ladder(path, problem):
    """The DF particle ladder, r2[i,j,a,b] = sum_ef (ae|bf) tau[i,j,e,f], reading Bvv from the file."""
    tau = problem["tau"]
    nocc, nvir = tau.shape[0], tau.shape[2]
    naux = problem["fitted"].shape[0]

    graph = cg.Graph("ladder")
    bvv = graph.declare_tensor("Bvv", [naux, nvir, nvir])
    cg.annotate(bvv, backing=einsums.io.etn(path, "Bvv"), graph=graph)
    # H is the largest tensor here, 25 times Bvv, and only this graph reads it. Declared as an
    # intermediate, the graph may build it a block at a time; declared without, it is a result the
    # caller could read after execute(), so it would be held whole, over the budget, and the
    # AxisTiling line of the report would say so.
    half = graph.declare_tensor("H", [naux, nvir, nvir, nocc, nocc], intermediate=True)
    T = _tensor("tau", tau)
    r2 = einsums.create_zero_tensor("r2", [nocc, nocc, nvir, nvir])
    with cg.capture(graph):
        einsums.einsum("Q,a,f,i,j <- Q,a,e ; i,j,e,f", half, bvv, T)
        einsums.einsum("i,j,a,b <- Q,a,f,i,j ; Q,b,f", r2, half, bvv)
    return graph, r2


def print_explanation(manager):
    """What SlabStreaming and AxisTiling did, and the peak MemoryPlanning found, from the pass report."""
    # Each line reads "  - [category] Pass: what it did".
    for line in manager.explain().splitlines():
        _, _, entry = line.partition("] ")
        if entry.startswith(("SlabStreaming:", "AxisTiling:", "Materialization:", "MemoryPlanning:")):
            print(f"    | {entry}")


def main():
    problem = load()
    fd, path = tempfile.mkstemp(suffix=".etn", prefix="einsums_out_of_core_")
    os.close(fd)
    os.remove(path)
    try:
        print("Producing Bvv into", os.path.basename(path))
        producer = produce(path, problem)
        manager = cg.default_pass_manager()
        producer.apply(manager)
        print_explanation(manager)
        producer.execute()

        bvv = np.einsum("Qmn,ma,nb->Qab", problem["fitted"], problem["cv"], problem["cv"])
        stored = einsums.create_zero_tensor("Bvv", list(bvv.shape))
        einsums.io.read(path, "Bvv", stored)
        error = float(np.max(np.abs(np.asarray(stored) - bvv)))
        print(f"    stored Bvv against numpy, largest difference {error:.2e}")
        assert error < 1e-12

        print("\nReading Bvv back for the particle ladder")
        reader, r2 = ladder(path, problem)
        manager = cg.default_pass_manager()
        reader.apply(manager)
        print_explanation(manager)
        reader.execute()

        expected = np.einsum("Qae,Qbf,ijef->ijab", bvv, bvv, problem["tau"], optimize=True)
        error = float(np.max(np.abs(np.asarray(r2) - expected)))
        print(f"    ladder against numpy, largest difference {error:.2e}")
        assert error < 1e-10
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    main()
