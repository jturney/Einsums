# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""A tensor that lives in a file: written once, read a block at a time, never held whole.

The virtual-virtual block of the density-fitted integrals, Bvv[Q,a,b], is the
largest tensor a DF-CCSD program holds, and the particle ladder reads it
twice per iteration. Annotated with a backing, it never has to be held whole.

The first part is one program, built as a CC code is: a setup body produces
Bvv once, into the entry of an ``.etn`` file ``einsums.io.etn(path, entry,
write=True)`` names, and every iteration of the loop runs the ladder on it.
The passes see the writer and every reader together. AxisTiling loops both
over blocks of the auxiliary index Q, the second contraction of the ladder
summing its blocks into the residual; SlabStreaming writes and reads exactly
the block of Bvv each iteration names; and since this program creates the
entry, it stores it with Q last, so each of those blocks is one contiguous run
of the file.

The second part is a later run that reuses the file. Annotated with
``einsums.io.etn(path, entry)``, Bvv is read in the order the entry records,
and the ladder reads it a block at a time as before. A file another program
wrote keeps the order it was written in; the report says so when another
order would at least halve the runs its blocks cost.

Neither program asks for any of this beyond the annotation. The default
pipeline streams a backed tensor when it is larger than its share of
``--einsums:max-memory``, set small below so that a cc-pVTZ water is enough to
show it. A backed tensor that fits is loaded or stored whole, once per replay.

Each transformation is one einsum over all of its factors, written as the
equation reads. Einsums contracts the factors two at a time, in the order
that costs least at these extents: for the ladder of water, Bvv with the
amplitudes first, which forms a four-index partial product named
``Bvv*tau``, 25 times the size of Bvv. A partial product is an intermediate
the graph owns, so AxisTiling builds it a block at a time and it is never
held whole. Written instead as two einsums through a tensor of your own,
that tensor would have to be declared ``intermediate=True`` for the same: a
tensor declared on the graph is otherwise a result the caller may read after
``execute()``, held whole.

What to read in the report the example prints:

* ``SlabStreaming: stores 'Bvv' with its axes in the order [1,2,0]``: the
  entry keeps Q last, and the blocks the program moves are 144 runs of the
  file per replay rather than 404496 in the declared order.
* ``writes 'Bvv' a box at a time`` and ``reads 'Bvv' a box at a time``: each
  iteration of a loop moves the block its view names, one run of the file,
  in place of the whole tensor.
* ``AxisTiling: slicing Bvv*tau[0]``: the ladder's partial product is built
  three values of Q at a time, and the residual collects each block's share.
* ``MemoryPlanning: peak``: 5.75 MB for the program and 2.74 MB for the later
  run, under a budget of 16 MB; Bvv alone is 3.2 MB, and ``Bvv*tau`` 79 MB.

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


def program(path, problem, iterations):
    """Bvv produced once into the file by a setup body, and the DF particle ladder run on it per iteration.

    Bvv[Q,a,b] = sum_mn (Q|mn) C[m,a] C[n,b]; r2[i,j,a,b] = sum_Qef Bvv[Q,a,e] Bvv[Q,b,f] tau[i,j,e,f].
    """
    fitted, cv, tau = problem["fitted"], problem["cv"], problem["tau"]
    naux, nvir, nocc = fitted.shape[0], cv.shape[1], tau.shape[0]

    graph = cg.Graph("ladder_program")
    bvv = graph.declare_tensor("Bvv", [naux, nvir, nvir], backing=einsums.io.etn(path, "Bvv", write=True))
    P = _tensor("P", fitted)
    C = _tensor("Cv", cv)
    T = _tensor("tau", tau)
    r2 = einsums.create_zero_tensor("r2", [nocc, nocc, nvir, nvir])

    setup = graph.add_setup("integrals")
    with cg.capture(setup):
        einsums.einsum("Q,a,b <- Q,m,n ; m,a ; n,b", bvv, P, C, C)
    body = graph.add_loop("iterations", iterations, lambda it: it + 1 < iterations)
    with cg.capture(body):
        einsums.einsum("i,j,a,b <- Q,a,e ; Q,b,f ; i,j,e,f", r2, bvv, bvv, T)
    return graph, r2


def later_run(path, problem):
    """The ladder alone, in a run that reuses the file the program wrote."""
    tau = problem["tau"]
    nocc, nvir = tau.shape[0], tau.shape[2]

    graph = cg.Graph("ladder")
    # The entry says how large Bvv is and what it holds, so the declaration names only the entry.
    bvv = graph.declare_tensor("Bvv", backing=einsums.io.etn(path, "Bvv"))
    T = _tensor("tau", tau)
    r2 = einsums.create_zero_tensor("r2", [nocc, nocc, nvir, nvir])
    with cg.capture(graph):
        einsums.einsum("i,j,a,b <- Q,a,e ; Q,b,f ; i,j,e,f", r2, bvv, bvv, T)
    return graph, r2


def print_explanation(manager):
    """What SlabStreaming and AxisTiling did, and the peak MemoryPlanning planned, from the passes' decisions."""
    for d in manager.decisions:
        if d.kind == cg.DecisionKind.Applied and d.pass_name in ("SlabStreaming", "AxisTiling"):
            text = d.text if d.text.startswith(d.pass_name) else f"{d.pass_name}: {d.text}"
            print(f"    | {text}")
        elif d.code == "memory-planning.peak":
            # MiB, as the pass's own report prints it.
            print(f"    | MemoryPlanning: peak {d.fact('peak') / 2**20:.2f} MB of {d.fact('total') / 2**20:.2f} MB total")


def check(label, got, want, tolerance):
    error = float(np.max(np.abs(np.asarray(got) - want)))
    print(f"    {label} against numpy, largest difference {error:.2e}")
    assert error < tolerance


def main():
    problem = load()
    bvv = np.einsum("Qmn,ma,nb->Qab", problem["fitted"], problem["cv"], problem["cv"])
    expected = np.einsum("Qae,Qbf,ijef->ijab", bvv, bvv, problem["tau"], optimize=True)

    fd, path = tempfile.mkstemp(suffix=".etn", prefix="einsums_out_of_core_")
    os.close(fd)
    os.remove(path)
    try:
        print("One program: Bvv produced into", os.path.basename(path), "by a setup body, read by every iteration")
        graph, r2 = program(path, problem, iterations=3)
        manager = cg.default_pass_manager()
        graph.apply(manager)
        print_explanation(manager)
        graph.execute()

        stored = einsums.create_zero_tensor("Bvv", list(bvv.shape))
        einsums.io.read(path, "Bvv", stored)
        check("stored Bvv", stored, bvv, 1e-12)
        check("ladder", r2, expected, 1e-10)

        print("\nA later run reusing the file")
        graph, r2 = later_run(path, problem)
        manager = cg.default_pass_manager()
        graph.apply(manager)
        print_explanation(manager)
        graph.execute()
        check("ladder", r2, expected, 1e-10)
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    main()
