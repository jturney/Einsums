# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Closed-shell CCSD with the integrals and the doubles amplitudes in files.

The two-electron integrals and T2 are the tensors a CCSD program cannot keep
in memory as the system grows: the integrals reach v^4 and T2 is o^2 v^2.
Here each lives in an entry of an ``.etn`` file, and the graphs that use them
read them from there. T1, o v, stays in memory.

The program has three parts.

* A transform graph produces the integral blocks <pq|rs> = (pr|qs) from the
  density-fitted three-index tensors, each into its entry of ``ints.etn``
  (a write backing), together with the MP2 guess for T2 in ``amps.etn``.
  Nothing reads the blocks in that graph, so each is computed a slab at a
  time and stored, and the v^4 block is never held whole.
* An iteration graph runs one CCSD iteration. It reads the integrals from
  their entries (read backings) and T2 through an update backing,
  ``amps.entry("t2", update=True)``: one tensor that every replay starts
  holding what the entry holds, and whose final value the entry holds once
  the replay ends. The residual r2 is an ordinary tensor of the graph, and
  the update t2 = r2 / D, with the diagonal of the Fock matrix moved into the
  denominator D, is the one statement that writes T2. Every term before it
  reads the amplitudes the iteration started from, and the energy after it
  reads the new ones.
* The new T2 goes into the scratch entry ``t2.next``, and at the end of the
  replay that entry replaces ``t2`` in one commit of the file. A crash before
  then leaves ``t2`` holding the previous iteration's amplitudes. The graph is
  captured once and replayed until the energy settles.

T1 lives in the same file through an update backing of its own. The two
update backings of one file are published together, in one commit, so the
file holds one iteration's amplitudes in ``t1`` and ``t2`` whichever
iteration the program stopped after, or was killed in, for a later run to
start from.

The default pipeline decides what to hold whole. A backed tensor larger than
its share of ``--einsums:max-memory`` is read a block at a time by the
contractions that use it; a smaller one is loaded once per replay. The budget
below is small enough that T2 (0.56 MB for water in cc-pVTZ) exceeds its
share, as v^4 (63 MB) and o v^3 (6 MB) do.

What to read in the report the example prints:

* ``SlabStreaming: ... reads 'vvvv' in blocks``: the particle ladder reads the
  v^4 block a slab at a time.
* ``SlabStreaming: ... reads 't2' in blocks``: contractions with the current
  amplitudes read them from the file.
* ``SlabStreaming: ... writes 'vvvv' a box at a time``: the transform stores
  the v^4 block one (a, b) pair at a time, each box one run of the file.
* ``LinearCombinationContractionFolding: does not fold 1 pair(s) over 'ovvv'``: the pair
  2 t1 ovvv(mafe) - t1 ovvv(maef) is usually folded into one contraction with
  L = 2 ovvv(mafe) - ovvv(maef), but building L holds o v^3 whole, and an L as
  large. The two terms are left as they are, and each reads o v^3 a slab at
  a time.
* ``MemoryPlanning: peak``: the plan's estimate of what one iteration holds
  at once, 5.7 MB, with every tensor freed after its last use. The new T2
  is summed in memory and stored once per iteration, the statements that
  write it being more than one loop over its axes can hold; so are the
  o^2 v^2 intermediates. A contraction of T2 with o v^3 reads o v^3 a slab
  at a time and holds T2, the smaller, whole.
* ``last iteration``, printed after the energy, is what the last iteration
  measured as it ran (``graph.last_run()``): the bytes read and written, and
  the most the graph held at once, 7.7 MB, within about 2 MB of the plan.

Run with::

    PYTHONPATH=<einsums-build>/lib python ccsd_out_of_core.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

import einsums.rc

# A budget of 3 MB gives each tensor a share of 0.5 MB, which T2 exceeds. The
# runtime reads it when it starts, so it is set first.
einsums.rc.max_memory = "3MB"

import einsums  # noqa: E402
import einsums.graph as cg  # noqa: E402
from einsums import linalg as la  # noqa: E402

_FIXTURE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "..", "examples", "dlpno", "fixtures", "water-ccpvtz.npz"))

# The integral blocks the CCSD equations read, physicist's notation <pq|rs>.
BLOCKS = ["oooo", "ooov", "oovo", "oovv", "ovoo", "ovov", "ovvo", "ovvv", "vvvo", "vvvv"]


def _tensor(name, array):
    tensor = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(tensor)[...] = np.ascontiguousarray(array)
    return tensor


def load():
    """Orbital energies and the fitted MO three-index tensors B[Q,p,q] over each pair of spaces."""
    z = np.load(_FIXTURE, allow_pickle=False)
    values, vectors = np.linalg.eigh(z["S"])
    orthogonalizer = vectors @ np.diag(values ** -0.5) @ vectors.T
    energies, rotation = np.linalg.eigh(orthogonalizer.T @ z["F"] @ orthogonalizer)
    coefficients = orthogonalizer @ rotation
    nocc = int(z["C_occ"].shape[1])

    mvals, mvecs = np.linalg.eigh(z["metric"])
    inv_sqrt = mvecs @ np.diag(np.where(mvals > 1e-10, mvals ** -0.5, 0.0)) @ mvecs.T
    fitted = np.einsum("PQ,Qmn->Pmn", inv_sqrt, z["eri_3index"])
    orbitals = {"o": coefficients[:, :nocc], "v": coefficients[:, nocc:]}
    three_index = {p + q: np.einsum("Qmn,mp,nq->Qpq", fitted, orbitals[p], orbitals[q], optimize=True)
                   for p in "ov" for q in "ov"}
    return energies[:nocc], energies[nocc:], three_index


def denominators(eo, ev):
    d1 = eo[:, None] - ev[None, :]
    d2 = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
    return d1, d2


def transform(ints_path, amps_path, three_index, d2):
    """Each integral block into its entry, <pq|rs> = sum_Q B[Q,p,r] B[Q,q,s], and the MP2 guess for T2."""
    graph = cg.Graph("transform")
    B = {pair: _tensor("B" + pair, array) for pair, array in three_index.items()}
    D = _tensor("D", d2)
    blocks = {}
    for name in BLOCKS:
        extents = [three_index[name[i] + name[i]].shape[1] for i in range(4)]
        # Left to itself, the transform stores each entry in the order that suits its own writes.
        # The iterations read every block many times over, in slabs of its last axis, so the
        # entries are stored in the declared order, in which such a slab is one run of the file.
        backing = einsums.io.etn(ints_path, name, write=True, order=[0, 1, 2, 3])
        blocks[name] = graph.declare_tensor(name, extents, backing=backing)
    t2 = graph.declare_tensor("t2", list(d2.shape), backing=einsums.io.etn(amps_path, "t2", write=True))
    with cg.capture(graph):
        for name, block in blocks.items():
            p, q, r, s = name
            einsums.einsum("p,q,r,s <- Q,p,r ; Q,q,s", block, B[p + r], B[q + s])
        la.direct_division(1.0, blocks["oovv"], D, 0.0, t2)
    return graph


def iteration(ints_path, amps_path, energy, d1, d2):
    """One CCSD iteration: T1 and T2 read from their entries and their new values left there, together.

    The equations are the closed-shell ones of Crawford's ccenergy, with the
    diagonal Fock moved into the denominators: Fae and Fmi hold only the
    terms in the amplitudes, and the new amplitudes are the residuals over D.
    """
    nocc, nvir = d1.shape
    graph = cg.Graph("ccsd_iteration")
    ints, amps = einsums.io.EtnFile(ints_path), einsums.io.EtnFile(amps_path)
    # Each entry says how large it is and what it holds, so the inputs are declared from the file alone.
    G = {name: graph.declare_tensor(name, backing=ints.entry(name)) for name in BLOCKS}
    # Read at the start of each replay, written once by the update, and published together when the replay ends.
    t1 = graph.declare_tensor("t1", backing=amps.entry("t1", update=True))
    t2 = graph.declare_tensor("t2", backing=amps.entry("t2", update=True))
    r2 = graph.declare_tensor("r2", list(d2.shape), intermediate=True)
    D1, D2 = _tensor("D1", d1), _tensor("D2", d2)

    shapes = {"Tau": "oovv", "Taut": "oovv", "Fae": "vv", "Fmi": "oo", "Fme": "ov", "Wmnij": "oooo", "Wmbej": "ovvo",
              "Wmbje": "ovov", "Zmbij": "ovoo", "jnfb": "oovv", "r1": "ov", "be": "vv", "jm": "oo", "imea": "oovv",
              "imeb": "oovv", "tmp": "oovv", "tmpP": "oovv", "e1": "", "e2": ""}
    S = {name: graph.declare_zero_tensor(name, [nocc if c == "o" else nvir for c in spaces] or [1], dtype="float64",
                                         intermediate=True)
         for name, spaces in shapes.items()}

    def ein(spec, out, A, B, pf=1.0, accumulate=False):
        einsums.einsum(spec, out, A, B, c_pf=1.0 if accumulate else 0.0, ab_pf=pf)

    def symmetrized(spec, A, B, pf, sign=1.0):
        """r2 += sign * (X[i,j,a,b] + X[j,i,b,a]) with X = pf * A B."""
        ein(spec, S["tmp"], A, B, pf)
        la.axpby(sign, S["tmp"], 1.0, r2)
        einsums.permute("j,i,b,a <- i,j,a,b", S["tmpP"], S["tmp"])
        la.axpby(sign, S["tmpP"], 1.0, r2)

    with cg.capture(graph):
        Tau, Taut = S["Tau"], S["Taut"]
        ein("i,j,a,b <- i,a ; j,b", Tau, t1, t1)
        la.axpby(1.0, t2, 1.0, Tau)
        ein("i,j,a,b <- i,a ; j,b", Taut, t1, t1, 0.5)
        la.axpby(1.0, t2, 1.0, Taut)

        Fae, Fmi, Fme = S["Fae"], S["Fmi"], S["Fme"]
        ein("ae <- mf ; mafe", Fae, t1, G["ovvv"], 2.0)
        ein("ae <- mf ; maef", Fae, t1, G["ovvv"], -1.0, True)
        ein("ae <- mnaf ; mnef", Fae, Taut, G["oovv"], -2.0, True)
        ein("ae <- mnaf ; mnfe", Fae, Taut, G["oovv"], 1.0, True)
        ein("mi <- ne ; mnie", Fmi, t1, G["ooov"], 2.0)
        ein("mi <- ne ; mnei", Fmi, t1, G["oovo"], -1.0, True)
        ein("mi <- inef ; mnef", Fmi, Taut, G["oovv"], 2.0, True)
        ein("mi <- inef ; mnfe", Fmi, Taut, G["oovv"], -1.0, True)
        ein("me <- nf ; mnef", Fme, t1, G["oovv"], 2.0)
        ein("me <- nf ; mnfe", Fme, t1, G["oovv"], -1.0, True)

        Wmnij, Wmbej, Wmbje, jnfb = S["Wmnij"], S["Wmbej"], S["Wmbje"], S["jnfb"]
        la.axpby(1.0, G["oooo"], 0.0, Wmnij)
        ein("mnij <- je ; mnie", Wmnij, t1, G["ooov"], 1.0, True)
        ein("mnij <- ie ; mnej", Wmnij, t1, G["oovo"], 1.0, True)
        ein("mnij <- ijef ; mnef", Wmnij, Tau, G["oovv"], 1.0, True)
        ein("j,n,f,b <- j,f ; n,b", jnfb, t1, t1)
        la.axpby(0.5, t2, 1.0, jnfb)
        la.axpby(1.0, G["ovvo"], 0.0, Wmbej)
        ein("mbej <- jf ; mbef", Wmbej, t1, G["ovvv"], 1.0, True)
        ein("mbej <- nb ; mnej", Wmbej, t1, G["oovo"], -1.0, True)
        ein("mbej <- jnfb ; mnef", Wmbej, jnfb, G["oovv"], -1.0, True)
        ein("mbej <- njfb ; mnef", Wmbej, t2, G["oovv"], 1.0, True)
        ein("mbej <- njfb ; mnfe", Wmbej, t2, G["oovv"], -0.5, True)
        la.axpby(-1.0, G["ovov"], 0.0, Wmbje)
        ein("mbje <- jf ; mbfe", Wmbje, t1, G["ovvv"], -1.0, True)
        ein("mbje <- nb ; mnje", Wmbje, t1, G["ooov"], 1.0, True)
        ein("mbje <- jnfb ; mnfe", Wmbje, jnfb, G["oovv"], 1.0, True)
        ein("mbij <- mbef ; ijef", S["Zmbij"], G["ovvv"], Tau)

        r1 = S["r1"]
        ein("ia <- ie ; ae", r1, t1, Fae)
        ein("ia <- ma ; mi", r1, t1, Fmi, -1.0, True)
        ein("ia <- imae ; me", r1, t2, Fme, 2.0, True)
        ein("ia <- imea ; me", r1, t2, Fme, -1.0, True)
        ein("ia <- nf ; nafi", r1, t1, G["ovvo"], 2.0, True)
        ein("ia <- nf ; naif", r1, t1, G["ovov"], -1.0, True)
        ein("ia <- mief ; maef", r1, t2, G["ovvv"], 2.0, True)
        ein("ia <- mife ; maef", r1, t2, G["ovvv"], -1.0, True)
        ein("ia <- mnae ; nmei", r1, t2, G["oovo"], -2.0, True)
        ein("ia <- mnae ; nmie", r1, t2, G["ooov"], 1.0, True)

        # The doubles residual. The first statement overwrites it.
        la.axpby(1.0, G["oovv"], 0.0, r2)
        symmetrized("ijab <- ijae ; be", t2, Fae, 1.0)
        ein("be <- mb ; me", S["be"], t1, Fme)
        symmetrized("ijab <- ijae ; be", t2, S["be"], 0.5, -1.0)
        symmetrized("ijab <- imab ; mj", t2, Fmi, 1.0, -1.0)
        ein("jm <- je ; me", S["jm"], t1, Fme)
        symmetrized("ijab <- imab ; jm", t2, S["jm"], 0.5, -1.0)
        ein("ijab <- mnab ; mnij", r2, Tau, Wmnij, 1.0, True)
        ein("ijab <- ijef ; abef", r2, Tau, G["vvvv"], 1.0, True)
        symmetrized("ijab <- ma ; mbij", t1, S["Zmbij"], 1.0, -1.0)
        ein("ijab <- imae ; mbej", S["tmp"], t2, Wmbej)
        ein("ijab <- imea ; mbej", S["tmp"], t2, Wmbej, -1.0, True)
        la.axpby(1.0, S["tmp"], 1.0, r2)
        einsums.permute("j,i,b,a <- i,j,a,b", S["tmpP"], S["tmp"])
        la.axpby(1.0, S["tmpP"], 1.0, r2)
        ein("ijab <- imae ; mbej", S["tmp"], t2, Wmbej)
        ein("ijab <- imae ; mbje", S["tmp"], t2, Wmbje, 1.0, True)
        la.axpby(1.0, S["tmp"], 1.0, r2)
        einsums.permute("j,i,b,a <- i,j,a,b", S["tmpP"], S["tmp"])
        la.axpby(1.0, S["tmpP"], 1.0, r2)
        symmetrized("ijab <- mjae ; mbie", t2, Wmbje, 1.0)
        ein("i,m,e,a <- i,e ; m,a", S["imea"], t1, t1)
        symmetrized("ijab <- imea ; mbej", S["imea"], G["ovvo"], 1.0, -1.0)
        ein("i,m,e,b <- i,e ; m,b", S["imeb"], t1, t1)
        symmetrized("ijab <- imeb ; maje", S["imeb"], G["ovov"], 1.0, -1.0)
        symmetrized("ijab <- ie ; abej", t1, G["vvvo"], 1.0)
        symmetrized("ijab <- ma ; mbij", t1, G["ovoo"], 1.0, -1.0)

        # The new amplitudes.
        la.direct_division(1.0, r2, D2, 0.0, t2)
        la.direct_division(1.0, r1, D1, 0.0, t1)

        # E = sum_ijab tau_ijab (2 <ij|ab> - <ij|ba>), from the new amplitudes.
        ein("i,j,a,b <- i,a ; j,b", Tau, t1, t1)
        la.axpby(1.0, t2, 1.0, Tau)
        ein("<- i,j,a,b ; i,j,a,b", S["e1"], Tau, G["oovv"])
        ein("<- i,j,a,b ; i,j,b,a", S["e2"], Tau, G["oovv"])
        la.axpby(2.0, S["e1"], 0.0, energy)
        la.axpby(-1.0, S["e2"], 1.0, energy)
    return graph


def print_explanation(manager):
    """What SlabStreaming did, the folds left to it, and the peak MemoryPlanning planned, from the
    passes' decisions."""
    unfolded = {}
    for d in manager.decisions:
        if d.code == "lccf.backed-over-share":
            unfolded[d.tensors[0]] = unfolded.get(d.tensors[0], 0) + 1
    for name, pairs in unfolded.items():
        print(f"    | LinearCombinationContractionFolding: does not fold {pairs} pair(s) over '{name}', "
              "which is read a slab at a time")
    for d in manager.decisions:
        if d.kind == cg.DecisionKind.Applied and d.pass_name == "SlabStreaming":
            text = d.text if d.text.startswith(d.pass_name) else f"{d.pass_name}: {d.text}"
            print(f"    | {text}")
        elif d.code == "memory-planning.peak":
            # In MB, as every other figure the example prints.
            print(f"    | MemoryPlanning: peak {d.fact('peak') / 1e6:.2f} MB of {d.fact('total') / 1e6:.2f} MB total")


def print_measured(run):
    """What the last iteration moved and held, as it ran, beside the plan the report gave."""
    read = sum(b.bytes_read for b in run.backings)
    written = sum(b.bytes_written for b in run.backings)
    print(f"    last iteration: {run.wall_ms:.0f} ms, {read / 1e6:.1f} MB read and {written / 1e6:.2f} MB written, "
          f"{run.peak_bytes / 1e6:.2f} MB held at most")
    for b in run.backings:
        if b.tensor in ("t1", "t2"):
            print(f"    | {b.tensor}: {b.reads} read(s), {b.writes} write(s), {b.publishes} publish")


def reference(eo, ev, three_index, iterations):
    """The same equations in numpy, on the same integrals, for the same number of iterations."""
    def eri(name):
        p, q, r, s = name
        return np.einsum("Qpr,Qqs->pqrs", three_index[p + r], three_index[q + s], optimize=True)

    I = {name: eri(name) for name in BLOCKS}
    E = lambda *args: np.einsum(*args, optimize=True)  # noqa: E731
    sym = lambda x: x + x.transpose(1, 0, 3, 2)  # noqa: E731
    d1, d2 = denominators(eo, ev)
    t1, t2 = np.zeros(d1.shape), I["oovv"] / d2
    for _ in range(iterations):
        tau = t2 + E("ia,jb->ijab", t1, t1)
        taut = t2 + 0.5 * E("ia,jb->ijab", t1, t1)
        Fae = 2 * E("mf,mafe->ae", t1, I["ovvv"]) - E("mf,maef->ae", t1, I["ovvv"]) \
            - 2 * E("mnaf,mnef->ae", taut, I["oovv"]) + E("mnaf,mnfe->ae", taut, I["oovv"])
        Fmi = 2 * E("ne,mnie->mi", t1, I["ooov"]) - E("ne,mnei->mi", t1, I["oovo"]) \
            + 2 * E("inef,mnef->mi", taut, I["oovv"]) - E("inef,mnfe->mi", taut, I["oovv"])
        Fme = 2 * E("nf,mnef->me", t1, I["oovv"]) - E("nf,mnfe->me", t1, I["oovv"])
        Wmnij = I["oooo"] + E("je,mnie->mnij", t1, I["ooov"]) + E("ie,mnej->mnij", t1, I["oovo"]) \
            + E("ijef,mnef->mnij", tau, I["oovv"])
        jnfb = 0.5 * t2 + E("jf,nb->jnfb", t1, t1)
        Wmbej = I["ovvo"] + E("jf,mbef->mbej", t1, I["ovvv"]) - E("nb,mnej->mbej", t1, I["oovo"]) \
            - E("jnfb,mnef->mbej", jnfb, I["oovv"]) + E("njfb,mnef->mbej", t2, I["oovv"]) \
            - 0.5 * E("njfb,mnfe->mbej", t2, I["oovv"])
        Wmbje = -I["ovov"] - E("jf,mbfe->mbje", t1, I["ovvv"]) + E("nb,mnje->mbje", t1, I["ooov"]) \
            + E("jnfb,mnfe->mbje", jnfb, I["oovv"])
        r1 = E("ie,ae->ia", t1, Fae) - E("ma,mi->ia", t1, Fmi) + 2 * E("imae,me->ia", t2, Fme) \
            - E("imea,me->ia", t2, Fme) + 2 * E("nf,nafi->ia", t1, I["ovvo"]) - E("nf,naif->ia", t1, I["ovov"]) \
            + 2 * E("mief,maef->ia", t2, I["ovvv"]) - E("mife,maef->ia", t2, I["ovvv"]) \
            - 2 * E("mnae,nmei->ia", t2, I["oovo"]) + E("mnae,nmie->ia", t2, I["ooov"])
        r2 = I["oovv"] + sym(E("ijae,be->ijab", t2, Fae - 0.5 * E("mb,me->be", t1, Fme)))
        r2 -= sym(E("imab,mj->ijab", t2, Fmi + 0.5 * E("je,me->mj", t1, Fme)))
        r2 += E("mnab,mnij->ijab", tau, Wmnij) + E("ijef,abef->ijab", tau, I["vvvv"])
        r2 -= sym(E("ma,mbij->ijab", t1, E("mbef,ijef->mbij", I["ovvv"], tau)))
        r2 += sym(E("imae,mbej->ijab", t2 - t2.transpose(0, 1, 3, 2), Wmbej))
        r2 += sym(E("imae,mbej->ijab", t2, Wmbej + Wmbje.transpose(0, 1, 3, 2)))
        r2 += sym(E("mjae,mbie->ijab", t2, Wmbje))
        r2 -= sym(E("ie,ma,mbej->ijab", t1, t1, I["ovvo"]) + E("ie,mb,maje->ijab", t1, t1, I["ovov"]))
        r2 += sym(E("ie,abej->ijab", t1, I["vvvo"]) - E("ma,mbij->ijab", t1, I["ovoo"]))
        t1, t2 = r1 / d1, r2 / d2
    tau = t2 + E("ia,jb->ijab", t1, t1)
    return 2 * E("ijab,ijab->", tau, I["oovv"]) - E("ijab,ijba->", tau, I["oovv"]), t1, t2


def main():
    eo, ev, three_index = load()
    d1, d2 = denominators(eo, ev)
    workdir = tempfile.mkdtemp(prefix="einsums_ccsd_out_of_core_")
    ints_path, amps_path = os.path.join(workdir, "ints.etn"), os.path.join(workdir, "amps.etn")
    try:
        print(f"o {len(eo)}, v {len(ev)}: T2 {d2.nbytes / 1e6:.2f} MB, <ab|cd> {len(ev) ** 4 * 8 / 1e6:.1f} MB")
        print("The integrals and the MP2 guess, into", os.path.basename(ints_path), "and", os.path.basename(amps_path))
        graph = transform(ints_path, amps_path, three_index, d2)
        manager = cg.default_pass_manager()
        graph.apply(manager)
        print_explanation(manager)
        graph.execute()
        # The guess for T1 is zero.
        einsums.io.write(amps_path, "t1", einsums.create_zero_tensor("t1", list(d1.shape)))

        energy = einsums.create_zero_tensor("energy", [1])
        graph = iteration(ints_path, amps_path, energy, d1, d2)
        print("\nOne iteration, T1 and T2 read from 't1' and 't2' and their new values published there together")
        manager = cg.default_pass_manager()
        graph.apply(manager)
        print_explanation(manager)

        previous, count = 0.0, 0
        while True:
            graph.execute()
            count += 1
            current = float(np.asarray(energy)[0])
            if abs(current - previous) < 1e-10 or count == 50:
                break
            previous = current
        print(f"\nCCSD correlation energy {current:.10f} after {count} iterations")
        print_measured(graph.last_run())

        want, t1_want, t2_want = reference(eo, ev, three_index, count)
        t1_stored = einsums.create_zero_tensor("t1", list(d1.shape))
        t2_stored = einsums.create_zero_tensor("t2", list(d2.shape))
        einsums.io.read(amps_path, "t1", t1_stored)
        einsums.io.read(amps_path, "t2", t2_stored)
        errors = {"energy": abs(current - want),
                  "stored T1": float(np.max(np.abs(np.asarray(t1_stored) - t1_want))),
                  "stored T2": float(np.max(np.abs(np.asarray(t2_stored) - t2_want)))}
        for label, error in errors.items():
            print(f"    {label} against numpy, largest difference {error:.2e}")
            assert error < 1e-10
    finally:
        for path in (ints_path, amps_path):
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(workdir)


if __name__ == "__main__":
    main()
