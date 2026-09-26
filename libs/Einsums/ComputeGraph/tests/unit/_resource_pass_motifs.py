# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Motif programs and a shuffled-pipeline driver for the resource and backend passes.

The pool corpus in ``_fuzz_diff_common`` is built from small matrices, vectors
and rank-3 tensors a few elements on a side. None of the passes this module
targets can act on that: TiledExpansion needs a tiled operand, LayoutAssignment
a rank-3 graph-owned deferred intermediate, StreamContractionFusion a stream of
at least 4096 elements eight times larger than its weights, FreeInsertion an
intermediate of at least 1 MiB, IOPrefetch a captured disk read, the space
passes annotated operands, and the GPU passes a float32 GEMM large enough for
the cost model to offload. So the programs here are built from MOTIFS, each one
shaped to make one of those passes fire, each carrying its own numpy oracle.

A program is two to four motifs captured into one graph in sequence, some of
them inside a loop body. The motifs share no tensor, so what they exercise
together is scheduling: every pass sees the other motifs' nodes on either side
of the ones it rewrites.

The driver applies one pass per ``PassManager``, in a shuffled order, and always
closes with Materialization (see ``check_program_rich_pipeline`` for why). For
the passes Python exposes as classes the pass's own counters decide whether it
fired; the others run as the default pipeline with every other pass disabled,
and their ``modified`` verdict is the signal.
"""

from __future__ import annotations

import itertools
import math
import os
import tempfile

import numpy as np

import einsums
import einsums.graph as cg
import einsums._core.graph as _G

from _fuzz_diff_common import _DEFAULT_PASS_NAMES, _DTYPE_TOL, _apply_one_pass

# ──────────────────────────────────────────────────────────────────────────
# The passes under test
# ──────────────────────────────────────────────────────────────────────────

#: Resource and backend passes of the default pipeline that no shuffled arm ran before.
TARGET_PASSES = (
    "TiledExpansion", "ScratchPrivatization", "LayoutAssignment", "StreamContractionFusion",
    "IOPrefetch", "FreeInsertion", "SpacePropagation", "CrossSpaceValidation", "ScalingAnalysis",
    "DistributionPlanning", "InputSlicing", "SUMMAExpansion", "CommunicationInsertion",
    "CommunicationElimination", "CommunicationScheduling", "GPUPlacement", "TransferInsertion",
    "TransferElimination", "GPUDiagnostics", "StreamAssignment",
)

#: Default-pipeline neighbours drawn into some orders, so the targets also meet
#: the passes they sit beside in the real pipeline. ContractionPlanning meets
#: chains of mixed dtype here, since the GPU motif is float32 whatever the
#: program's dtype.
NEIGHBOUR_PASSES = ("Reorder", "InplaceOptimization", "MemoryPlanning", "CSE", "DeadNodeElimination",
                    "SymmetryPropagation", "LoopInvariantHoisting", "ContractionPlanning")

#: Passes that can only run on more than one MPI rank. The mock backend's
#: ``comm::world_size()`` is hard-wired to 1, so on a serial build they are
#: guaranteed no-ops (CommunicationElimination has no such guard, but its only
#: input is an Allreduce node, which only CommunicationInsertion creates).
DISTRIBUTED_PASSES = ("DistributionPlanning", "InputSlicing", "SUMMAExpansion", "CommunicationInsertion",
                      "CommunicationElimination", "CommunicationScheduling")

#: Passes that exist in this build only when a GPU backend or the GPU mock is compiled in.
GPU_PASSES = ("GPUPlacement", "TransferInsertion", "TransferElimination", "GPUDiagnostics", "StreamAssignment")

#: Per pass: trials that ran it and trials on which it fired.
STATS = {name: {"ran": 0, "fired": 0} for name in TARGET_PASSES + NEIGHBOUR_PASSES + ("Materialization",)}

#: Per motif kind: how many trials carried one.
MOTIF_STATS = {}


def default_pipeline_pass_names():
    """The names ``populate_default`` builds in THIS binary, by elimination.

    ``PassManager`` has no name listing, but a run records which disabled
    names matched a pass, so disabling every candidate and running the manager
    over an empty graph lists what the build has. Every candidate is disabled,
    so the run itself does nothing.
    """
    pm = cg.PassManager()
    pm.populate_default()
    for name in _DEFAULT_PASS_NAMES:
        pm.disable(name)
    pm.run(cg.Graph("pass_census"))
    return list(pm.disabled_passes())


# Counters that decide whether a Python-exposed pass fired. A read-only pass
# never reports a modification, so its own findings are the only signal.
def _fired_scratch(p):
    return p.num_tensors_privatized > 0


def _fired_layout(p):
    return p.num_relaid_out > 0 or p.num_copies_removed > 0


def _fired_spaces(p):
    return p.num_inferred > 0


def _fired_cross(p):
    return len(list(p.findings)) > 0


def _fired_scaling(p):
    return p.num_analyzed > 0


_COUNTERS = {
    "ScratchPrivatization": _fired_scratch,
    "LayoutAssignment": _fired_layout,
    "SpacePropagation": _fired_spaces,
    "CrossSpaceValidation": _fired_cross,
    "ScalingAnalysis": _fired_scaling,
}


def apply_pass(g, name):
    """Run pass @p name alone on @p g and return whether it fired.

    A Python class is used when one exists and the pass has a counter, since
    the counter also sees read-only passes act. ScratchPrivatization is built
    with its default gate (an installed executor), as the default pipeline
    builds it. Everything else goes through ``_apply_one_pass``, which isolates
    the default pipeline's own instance, with the cost model it was built with.
    """
    if name in _COUNTERS:
        p = getattr(_G, name)()
        pm = cg.PassManager()
        pm.add(p)
        modified = g.apply(pm)
        return bool(modified) or _COUNTERS[name](p)
    return bool(_apply_one_pass(g, name))


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

_uid = itertools.count()
_IS_COMPLEX = {"complex64", "complex128"}


def _rand(rng, shape, dtype):
    a = rng.standard_normal(shape)
    if dtype in _IS_COMPLEX:
        a = a + 1j * rng.standard_normal(shape)
    return a.astype(np.dtype(dtype))


def _tensor(name, array):
    t = einsums.create_zero_tensor(name, list(array.shape), dtype=str(array.dtype))
    np.asarray(t)[...] = array
    return t


def _scalar(rng):
    return float(rng.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 2.0]))


class Motif:
    """One self-contained piece of a program.

    ``build`` creates the motif's tensors and captures its operations into
    ``target`` (the graph, or a loop body of it); ``step`` applies one
    execution of the same operations to numpy arrays; ``observe`` reads back
    every caller-visible tensor. Graph-owned scratch is never observed.
    """

    kind = "motif"
    dtype = "float64"
    loop = 0  # 0 = captured at top level, else the loop's iteration count

    def __init__(self, rng, dtype, tag):
        self.tag = tag
        self.dtype = dtype
        self.np = {}

    def build(self, g, target):  # pragma: no cover (abstract)
        raise NotImplementedError

    def step(self, a):  # pragma: no cover (abstract)
        raise NotImplementedError

    def observe(self):
        return {name: np.asarray(t).copy() for name, t in self.t.items()}

    def expected(self, runs):
        a = {k: v.copy() for k, v in self.np.items()}
        for _ in range(runs * max(1, self.loop)):
            self.step(a)
        return a

    def cleanup(self):
        pass

    def describe(self):
        return f"{self.kind}[{self.dtype}]{'(loop %d)' % self.loop if self.loop else ''}"


# ──────────────────────────────────────────────────────────────────────────
# Motifs
# ──────────────────────────────────────────────────────────────────────────

#: GEMV-shaped members over one streamed rank-4 tensor: every C and W letter is one of S's.
_STREAM_SPECS = ("mn <- mnls ; ls", "mn <- mlns ; ls", "ls <- mnls ; mn", "ml <- mnls ; ns", "ns <- mnls ; ml")


class StreamJK(Motif):
    """Two or three contractions streaming one TEI against a small density (J / K build).

    n is 8 to 10, so the stream holds 4096 to 10000 elements, above the pass's
    4096 floor and 64 or more times the size of each weight and output.
    ``shared`` sends every member into one output, where members after the
    first must accumulate (``c_pf == 1``) for the group to fuse.
    """

    kind = "stream"

    def __init__(self, rng, dtype, tag):
        super().__init__(rng, dtype, tag)
        n = int(rng.integers(8, 11))
        self.n = n
        k = int(rng.integers(2, 4))
        self.specs = [str(s) for s in rng.choice(_STREAM_SPECS, size=k, replace=False)]
        self.shared = bool(rng.random() < 0.3)
        self.pf = [(float(rng.choice([0.0, 1.0, 0.5])), _scalar(rng)) for _ in self.specs]
        if self.shared:
            self.pf = [self.pf[0]] + [(1.0, ab) for _, ab in self.pf[1:]]
        self.np["S"] = _rand(rng, (n, n, n, n), dtype)
        self.np["D"] = _rand(rng, (n, n), dtype)
        outs = 1 if self.shared else len(self.specs)
        for j in range(outs):
            self.np[f"O{j}"] = _rand(rng, (n, n), dtype)

    def _out(self, j):
        return "O0" if self.shared else f"O{j}"

    def build(self, g, target):
        self.t = {k: _tensor(f"{self.tag}_{k}", v) for k, v in self.np.items()}
        with cg.capture(target):
            for j, (spec, (cpf, ab)) in enumerate(zip(self.specs, self.pf)):
                einsums.einsum(spec, self.t[self._out(j)], self.t["S"], self.t["D"], c_pf=cpf, ab_pf=ab)

    def step(self, a):
        for j, (spec, (cpf, ab)) in enumerate(zip(self.specs, self.pf)):
            lhs, rhs = spec.split("<-")
            sa, sb = (x.strip() for x in rhs.split(";"))
            o = self._out(j)
            a[o] = cpf * a[o] + ab * np.einsum(f"{sa},{sb}->{lhs.strip()}", a["S"], a["D"])

    def describe(self):
        return super().describe() + f" n={self.n} specs={self.specs} pf={self.pf} shared={self.shared}"


class LayoutChain(Motif):
    """A rank-3 graph-owned deferred intermediate read by contractions that want another order.

    ``W[i,j,x] = A[i,k] B[k,x,j]`` then ``R[i,x,y] (+)= W D[j,y]``, with W
    captured in a random axis order. The ``permute`` variant routes the
    consumer through an explicit permute of W into a second deferred tensor,
    which LayoutAssignment can delete by storing the copy in its source's order.
    """

    kind = "layout"

    def __init__(self, rng, dtype, tag):
        super().__init__(rng, dtype, tag)
        self.dims = {c: int(rng.integers(2, 6)) for c in "ijxky"}
        self.w_order = "".join(rng.permutation(list("ijx")))
        self.via_permute = bool(rng.random() < 0.4)
        self.p_order = "".join(rng.permutation(list("ijx")))
        self.r_cpf = float(rng.choice([0.0, 1.0]))
        self.ab = _scalar(rng)
        d = self.dims
        self.np["A"] = _rand(rng, (d["i"], d["k"]), dtype)
        self.np["B"] = _rand(rng, (d["k"], d["x"], d["j"]), dtype)
        self.np["D"] = _rand(rng, (d["j"], d["y"]), dtype)
        self.np["R"] = _rand(rng, (d["i"], d["x"], d["y"]), dtype)

    def _shape(self, letters):
        return [self.dims[c] for c in letters]

    def build(self, g, target):
        self.t = {k: _tensor(f"{self.tag}_{k}", v) for k, v in self.np.items()}
        W = g.declare_zero_tensor(f"{self.tag}_W", self._shape(self.w_order), intermediate=True, dtype=self.dtype)
        self._keep = [W]
        w = ",".join(self.w_order)
        with cg.capture(target):
            einsums.einsum(f"{w} <- i,k ; k,x,j", W, self.t["A"], self.t["B"], c_pf=0.0, ab_pf=1.0)
            src, s = W, w
            if self.via_permute:
                P = g.declare_zero_tensor(f"{self.tag}_P", self._shape(self.p_order), intermediate=True,
                                          dtype=self.dtype)
                self._keep.append(P)
                p = ",".join(self.p_order)
                einsums.permute(f"{p} <- {w}", P, W, c_pf=0.0, a_pf=1.0)
                src, s = P, p
            einsums.einsum(f"i,x,y <- {s} ; j,y", self.t["R"], src, self.t["D"], c_pf=self.r_cpf, ab_pf=self.ab)

    def step(self, a):
        w = np.einsum("ik,kxj->ijx", a["A"], a["B"])
        a["R"] = self.r_cpf * a["R"] + self.ab * np.einsum("ijx,jy->ixy", w, a["D"])

    def describe(self):
        return super().describe() + (f" dims={self.dims} W={self.w_order}"
                                     f" permute={self.p_order if self.via_permute else None} r_cpf={self.r_cpf}")


class ScratchReuse(Motif):
    """One scratch buffer recycled through several write-then-read episodes (the CCSD idiom).

    Every episode opens with a pure overwrite (an einsum or a permute with a
    zero C prefactor), which is what makes it a generation ScratchPrivatization
    can rename onto a clone.
    """

    kind = "scratch"

    def __init__(self, rng, dtype, tag):
        super().__init__(rng, dtype, tag)
        n = int(rng.integers(3, 7))
        self.n = n
        self.gens = []
        for k in range(int(rng.integers(2, 5))):
            op = "permute" if rng.random() < 0.3 else "einsum"
            self.gens.append((op, _scalar(rng), float(rng.choice([1.0, 0.5]))))
            self.np[f"A{k}"] = _rand(rng, (n, n), dtype)
            self.np[f"B{k}"] = _rand(rng, (n, n), dtype)
        self.np["acc"] = _rand(rng, (n, n), dtype)
        # An eager scratch carries an Alloc node, and a tensor any lifecycle
        # node touches is one the pass leaves alone, so most draws are deferred.
        self.deferred = bool(rng.random() < 0.8)

    def build(self, g, target):
        self.t = {k: _tensor(f"{self.tag}_{k}", v) for k, v in self.np.items()}
        mk = g.declare_zero_tensor if self.deferred else g.create_zero_tensor
        tmp = mk(f"{self.tag}_tmp", [self.n, self.n], intermediate=True, dtype=self.dtype)
        self._keep = [tmp]
        with cg.capture(target):
            for k, (op, s, beta) in enumerate(self.gens):
                if op == "einsum":
                    einsums.einsum("ij <- ik ; kj", tmp, self.t[f"A{k}"], self.t[f"B{k}"], c_pf=0.0, ab_pf=1.0)
                else:
                    einsums.permute("j,i <- i,j", tmp, self.t[f"A{k}"], c_pf=0.0, a_pf=1.0)
                einsums.linalg.axpby(s, tmp, beta, self.t["acc"])

    def step(self, a):
        for k, (op, s, beta) in enumerate(self.gens):
            tmp = a[f"A{k}"] @ a[f"B{k}"] if op == "einsum" else a[f"A{k}"].T.copy()
            a["acc"] = s * tmp + beta * a["acc"]

    def describe(self):
        return super().describe() + f" n={self.n} gens={self.gens} deferred={self.deferred}"


class BigScratch(Motif):
    """A graph-owned intermediate above FreeInsertion's fixed 1 MiB floor.

    ``big = A @ B`` is an outer-product-shaped GEMM (inner extent 1 to 3), so
    a 1.2 MiB buffer costs well under a millisecond to fill; two GEMV readers
    then reduce it into caller-owned vectors. ``deferred`` picks a declared
    shell (Materialization allocates it) or an eager ``create_zero_tensor``
    (FreeInsertion then pairs its own Materialize before the first writer).
    """

    kind = "big"

    def __init__(self, rng, dtype, tag):
        super().__init__(rng, dtype, tag)
        itemsize = np.dtype(dtype).itemsize
        need = int(1.15 * (1 << 20) / itemsize)
        m = int(math.isqrt(need)) + int(rng.integers(1, 9))
        self.m = m
        self.k = int(rng.integers(1, 4))
        self.deferred = bool(rng.random() < 0.5)
        self.readers = 1 + int(rng.random() < 0.5)
        self.np["A"] = _rand(rng, (m, self.k), dtype)
        self.np["B"] = _rand(rng, (self.k, m), dtype)
        self.np["x"] = _rand(rng, (m,), dtype)
        self.np["v"] = _rand(rng, (m,), dtype)
        self.np["u"] = _rand(rng, (m,), dtype)
        self.cpf = float(rng.choice([0.0, 1.0]))

    def build(self, g, target):
        self.t = {k: _tensor(f"{self.tag}_{k}", v) for k, v in self.np.items()}
        mk = g.declare_zero_tensor if self.deferred else g.create_zero_tensor
        big = mk(f"{self.tag}_big", [self.m, self.m], intermediate=True, dtype=self.dtype)
        self._keep = [big]
        with cg.capture(target):
            einsums.einsum("ij <- ik ; kj", big, self.t["A"], self.t["B"], c_pf=0.0, ab_pf=1.0)
            einsums.einsum("i <- ij ; j", self.t["v"], big, self.t["x"], c_pf=self.cpf, ab_pf=0.5)
            if self.readers > 1:
                einsums.einsum("j <- ij ; i", self.t["u"], big, self.t["x"], c_pf=self.cpf, ab_pf=-0.5)

    def step(self, a):
        big = a["A"] @ a["B"]
        a["v"] = self.cpf * a["v"] + 0.5 * (big @ a["x"])
        if self.readers > 1:
            a["u"] = self.cpf * a["u"] - 0.5 * (big.T @ a["x"])

    def describe(self):
        return super().describe() + f" m={self.m} k={self.k} deferred={self.deferred} readers={self.readers}"


_TILED = {"float64": einsums.TiledRuntimeTensorD, "float32": einsums.TiledRuntimeTensorF,
          "complex128": einsums.TiledRuntimeTensorZ, "complex64": einsums.TiledRuntimeTensorC}


def _grid_axis(rng):
    return [int(rng.integers(1, 4)) for _ in range(int(rng.integers(1, 4)))]


def _offsets(axis):
    return list(np.cumsum([0] + axis)[:-1])


class _TiledArray:
    """A tiled operand: its grid, which tiles are stored, and the dense value (absent tiles zero)."""

    def __init__(self, rng, grid, dtype, sparse):
        self.grid = grid
        tiles = list(itertools.product(*[range(len(ax)) for ax in grid]))
        self.present = {t for t in tiles if not sparse or rng.random() < 0.65}
        dense = _rand(rng, tuple(sum(ax) for ax in grid), dtype)
        mask = np.zeros(dense.shape, dtype=bool)
        offs = [_offsets(ax) for ax in grid]
        for t in self.present:
            sl = tuple(slice(offs[d][t[d]], offs[d][t[d]] + grid[d][t[d]]) for d in range(len(grid)))
            mask[sl] = True
        self.dense = np.where(mask, dense, 0).astype(np.dtype(dtype))

    def make(self, name, dtype):
        t = _TILED[dtype](name, self.grid)
        for tile in self.present:
            t.add_tile(list(tile))
        t.materialize()
        offs = [_offsets(ax) for ax in self.grid]
        for tile in self.present:
            sl = tuple(slice(offs[d][tile[d]], offs[d][tile[d]] + self.grid[d][tile[d]]) for d in range(len(self.grid)))
            np.asarray(t.tile_view(list(tile)))[...] = self.dense[sl]
        return t


def _gather(t, dtype):
    sizes, offs = t.tile_sizes(), t.tile_offsets()
    out = np.zeros(tuple(sum(ax) for ax in sizes), dtype=np.dtype(dtype))
    for tile in itertools.product(*[range(len(ax)) for ax in sizes]):
        if t.has_tile(list(tile)):
            sl = tuple(slice(offs[d][tile[d]], offs[d][tile[d]] + sizes[d][tile[d]]) for d in range(len(sizes)))
            out[sl] = np.asarray(t.tile_view(list(tile)))
    return out


class TiledChain(Motif):
    """Tiled contraction, then a mix of tiled scale, axpy and permute over the result.

    Shared letters get one partition on every operand (TiledExpansion declines
    a misaligned pair), tiles are randomly absent, and the tail optionally goes
    through a graph-owned tiled intermediate, which has no tiles at pass time
    and so exercises the pass's predicted tile sets.
    """

    kind = "tiled"

    def __init__(self, rng, dtype, tag):
        super().__init__(rng, dtype, tag)
        r, k, c = _grid_axis(rng), _grid_axis(rng), _grid_axis(rng)
        sparse = bool(rng.random() < 0.5)
        self.arr = {"A": _TiledArray(rng, [r, k], dtype, sparse), "B": _TiledArray(rng, [k, c], dtype, sparse),
                    "C": _TiledArray(rng, [r, c], dtype, sparse), "Y": _TiledArray(rng, [r, c], dtype, sparse),
                    "E": _TiledArray(rng, [c, r], dtype, sparse)}
        self.np = {k: v.dense.copy() for k, v in self.arr.items()}
        self.cpf, self.ab = float(rng.choice([0.0, 1.0, 0.5])), _scalar(rng)
        self.tail = [str(op) for op in rng.choice(["scale", "axpy", "permute"], size=int(rng.integers(1, 4)))]
        self.scal = [_scalar(rng) for _ in self.tail]
        self.via_scratch = bool(rng.random() < 0.3)
        self.grid_rc = [r, c]

    def build(self, g, target):
        self.t = {k: v.make(f"{self.tag}_{k}", self.dtype) for k, v in self.arr.items()}
        with cg.capture(target):
            C = self.t["C"]
            einsums.einsum("ij <- ik ; kj", C, self.t["A"], self.t["B"], c_pf=self.cpf, ab_pf=self.ab)
            src = C
            if self.via_scratch:
                self._scr = g.declare_zero_tiled_tensor(f"{self.tag}_T", self.grid_rc, intermediate=True, dtype=self.dtype)
                einsums.einsum("ij <- ik ; kj", self._scr, self.t["A"], self.t["B"], c_pf=0.0, ab_pf=1.0)
                src = self._scr
            for op, s in zip(self.tail, self.scal):
                if op == "scale":
                    einsums.linalg.scale(s, C)
                elif op == "axpy":
                    einsums.linalg.axpy(s, src, self.t["Y"])
                else:
                    einsums.permute("j,i <- i,j", self.t["E"], src, c_pf=0.5, a_pf=s)

    def step(self, a):
        a["C"] = self.cpf * a["C"] + self.ab * (a["A"] @ a["B"])
        src = "C"
        if self.via_scratch:
            a["_T"] = a["A"] @ a["B"]
            src = "_T"
        for op, s in zip(self.tail, self.scal):
            if op == "scale":
                a["C"] = s * a["C"]
            elif op == "axpy":
                a["Y"] = a["Y"] + s * a[src]
            else:
                a["E"] = 0.5 * a["E"] + s * a[src].T

    def observe(self):
        return {k: _gather(t, self.dtype) for k, t in self.t.items()}

    def expected(self, runs):
        out = super().expected(runs)
        out.pop("_T", None)
        return out

    def describe(self):
        return super().describe() + (f" grids={ {k: v.grid for k, v in self.arr.items()} } cpf={self.cpf}"
                                     f" tail={list(zip(self.tail, self.scal))} scratch={self.via_scratch}")


class TiledAcrossLoop(Motif):
    """A tiled tensor X written on one side of a loop boundary and read on the other.

    X is produced before the loop or in its body; the body reads it when the
    producer is outside, and the parent reads it after the loop when the
    producer is inside (each also drawn on its own). A body-local scale of the
    body's own target gives TiledExpansion something it may expand beside the
    shared tensor, and ``big`` moves everything to a grid whose contractions are
    over the pass's node budget, so the producer can stay opaque while the
    elementwise ops beside it expand. The motif places its own loop, so it is
    never itself put inside one.
    """

    kind = "tiled_loop"

    def __init__(self, rng, dtype, tag):
        super().__init__(rng, dtype, tag)
        big = bool(rng.random() < 0.3)
        r, k, c = ([[1] * 17] * 3) if big else (_grid_axis(rng), _grid_axis(rng), _grid_axis(rng))
        sparse = bool(rng.random() < 0.5)
        self.arr = {"A": _TiledArray(rng, [r, k], dtype, sparse), "F": _TiledArray(rng, [k, c], dtype, sparse),
                    "G": _TiledArray(rng, [r, c], dtype, sparse), "X": _TiledArray(rng, [r, c], dtype, False),
                    "Y": _TiledArray(rng, [r, c], dtype, sparse), "E": _TiledArray(rng, [c, r], dtype, sparse)}
        self.np = {k: v.dense.copy() for k, v in self.arr.items()}
        self.grid_rc = [r, c]
        self.big = big
        self.iters = int(rng.integers(1, 4))
        self.prod_side = str(rng.choice(["before", "body"]))
        self.prod_op = str(rng.choice(["einsum", "axpy"]))
        # A graph-owned X is overwritten by an einsum; an axpy into it would carry
        # its value from one execution into the next, which the oracle does not model.
        self.scratch = bool(rng.random() < 0.3) and self.prod_op == "einsum"
        if self.scratch:
            self.np["X"] = np.zeros_like(self.np["X"])
        self.cpf = 0.0 if self.scratch else float(rng.choice([0.0, 1.0, 0.5]))
        self.body_reader = self.prod_side == "before" or bool(rng.random() < 0.5)
        self.after_reader = self.prod_side == "body" or bool(rng.random() < 0.5)
        self.s = [_scalar(rng) for _ in range(4)]

    def build(self, g, target):
        self.t = {k: v.make(f"{self.tag}_{k}", self.dtype) for k, v in self.arr.items() if not (k == "X" and self.scratch)}
        if self.scratch:
            self._x = g.declare_zero_tiled_tensor(f"{self.tag}_X", self.grid_rc, intermediate=True, dtype=self.dtype)
        X = self._x if self.scratch else self.t["X"]

        def produce():
            if self.prod_op == "einsum":
                einsums.einsum("ij <- ik ; kj", X, self.t["A"], self.t["F"], c_pf=self.cpf, ab_pf=self.s[0])
            else:
                einsums.linalg.axpy(self.s[0], self.t["G"], X)

        if self.prod_side == "before":
            with cg.capture(g):
                produce()
        body = g.add_loop(f"{self.tag}_loop", self.iters, lambda it, c=self.iters: it < c - 1)
        with cg.capture(body):
            if self.prod_side == "body":
                produce()
            if self.body_reader:
                einsums.linalg.axpy(self.s[1], X, self.t["Y"])
            einsums.linalg.scale(self.s[2], self.t["Y"])
        if self.after_reader:
            with cg.capture(g):
                einsums.permute("j,i <- i,j", self.t["E"], X, c_pf=0.5, a_pf=self.s[3])

    def step(self, a):
        def produce(x):
            if self.prod_op == "einsum":
                return self.cpf * x + self.s[0] * (a["A"] @ a["F"])
            return x + self.s[0] * a["G"]

        if self.prod_side == "before":
            a["X"] = produce(a["X"])
        for _ in range(self.iters):
            if self.prod_side == "body":
                a["X"] = produce(a["X"])
            if self.body_reader:
                a["Y"] = a["Y"] + self.s[1] * a["X"]
            a["Y"] = self.s[2] * a["Y"]
        if self.after_reader:
            a["E"] = 0.5 * a["E"] + self.s[3] * a["X"].T

    def observe(self):
        return {k: _gather(t, self.dtype) for k, t in self.t.items()}

    def expected(self, runs):
        out = super().expected(runs)
        if self.scratch:
            out.pop("X", None)
        return out

    def describe(self):
        return super().describe() + (f" big={self.big} iters={self.iters} producer={self.prod_op}@{self.prod_side}"
                                     f" body_reader={self.body_reader} after_reader={self.after_reader}"
                                     f" scratch={self.scratch} cpf={self.cpf} s={self.s}")


class DiskLoad(Motif):
    """A captured disk read placed after unrelated compute.

    The dataset is written before capture, so every execution loads the same
    values, and IOPrefetch moves the read to the front since nothing else
    touches its destination. Inside a loop the read is invariant, so
    IOPrefetch hoists it out of the body.
    """

    kind = "io"

    def __init__(self, rng, dtype, tag):
        super().__init__(rng, dtype, tag)
        shape = (int(rng.integers(2, 6)), int(rng.integers(2, 6)))
        self.np["X"] = _rand(rng, shape, dtype)   # the file's contents
        self.np["Y"] = np.zeros(shape, dtype=np.dtype(dtype))
        self.np["Z"] = _rand(rng, shape, dtype)
        self.np["W"] = _rand(rng, shape, dtype)
        self.s, self.a = _scalar(rng), _scalar(rng)
        self.path = os.path.join(tempfile.gettempdir(), f"einsums_respass_{os.getpid()}_{tag}.etn")

    def build(self, g, target):
        self.t = {k: _tensor(f"{self.tag}_{k}", v) for k, v in self.np.items()}
        einsums.io.write(self.path, "X", self.t["X"])
        with cg.capture(target):
            einsums.linalg.scale(self.s, self.t["W"])
            einsums.linalg.axpby(1.0, self.t["Z"], 0.5, self.t["W"])
            einsums.io.read(self.path, "X", self.t["Y"])
            einsums.linalg.axpy(self.a, self.t["Y"], self.t["Z"])

    def step(self, a):
        a["W"] = self.s * a["W"]
        a["W"] = a["Z"] + 0.5 * a["W"]
        a["Y"] = a["X"].copy()
        a["Z"] = a["Z"] + self.a * a["Y"]

    def cleanup(self):
        try:
            os.remove(self.path)
        except OSError:
            pass


class DiskRoundTrip(Motif):
    """A dataset written by the graph and read back by the graph.

    The file starts out holding stale values, so a read that runs before the
    write loads the wrong numbers. Nothing but the file connects the two nodes.
    """

    kind = "io_roundtrip"

    def __init__(self, rng, dtype, tag):
        super().__init__(rng, dtype, tag)
        shape = (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
        self.np["T"] = _rand(rng, shape, dtype)
        self.np["Y"] = np.zeros(shape, dtype=np.dtype(dtype))
        self.s = _scalar(rng) * 3.0
        self.path = os.path.join(tempfile.gettempdir(), f"einsums_respass_{os.getpid()}_{tag}.etn")

    def build(self, g, target):
        self.t = {k: _tensor(f"{self.tag}_{k}", v) for k, v in self.np.items()}
        einsums.io.write(self.path, "T", self.t["T"])  # stale: the unscaled values
        with cg.capture(target):
            einsums.linalg.scale(self.s, self.t["T"])
            einsums.io.write(self.path, "T", self.t["T"])
            einsums.io.read(self.path, "T", self.t["Y"])

    def step(self, a):
        a["T"] = self.s * a["T"]
        a["Y"] = a["T"].copy()

    cleanup = DiskLoad.cleanup


_SPACES = (("occ", "o", 4.0), ("virt", "v", 8.0), ("aux", "x", 16.0))


class SpaceChain(Motif):
    """A contraction chain whose inputs are annotated with index spaces.

    ``C[i,x] = A[i,a] B[a,x]`` into a graph-owned intermediate, then
    ``D[i,j] (+)= C[i,x] E[x,j]``. SpacePropagation fills in C's spaces;
    the ``conflict`` variant annotates B's first axis as occupied, which the
    registry declares disjoint from virtual, so CrossSpaceValidation reports it.
    """

    kind = "spaces"

    def __init__(self, rng, dtype, tag):
        super().__init__(rng, dtype, tag)
        o, v, x = (int(rng.integers(2, 5)) for _ in range(3))
        self.ov = (o, v, x)
        # 'b' puts an occupied slot on B's contracted letter, which the registry
        # declares disjoint from virtual (SpacePropagation then declines C); 'e'
        # puts a virtual slot where C's inferred aux slot meets E, a relation the
        # registry never declared, so the finding only appears once C is inferred.
        self.conflict = str(rng.choice(["none", "none", "b", "e"]))
        self.np["A"] = _rand(rng, (o, v), dtype)
        self.np["B"] = _rand(rng, (v, x), dtype)
        self.np["E"] = _rand(rng, (x, o), dtype)
        self.np["D"] = _rand(rng, (o, o), dtype)
        self.cpf = float(rng.choice([0.0, 1.0]))

    def build(self, g, target):
        self.t = {k: _tensor(f"{self.tag}_{k}", v) for k, v in self.np.items()}
        o, v, x = self.ov
        C = g.create_zero_tensor(f"{self.tag}_C", [o, x], intermediate=True, dtype=self.dtype)
        self._keep = [C]
        with cg.capture(target):
            einsums.einsum("ix <- ia ; ax", C, self.t["A"], self.t["B"], c_pf=0.0, ab_pf=1.0)
            einsums.einsum("ij <- ix ; xj", self.t["D"], C, self.t["E"], c_pf=self.cpf, ab_pf=1.0)
        cg.annotate(self.t["A"], ("occ", "virt"), graph=g)
        cg.annotate(self.t["B"], ("occ" if self.conflict == "b" else "virt", "aux"), graph=g)
        cg.annotate(self.t["E"], ("virt" if self.conflict == "e" else "aux", "occ"), graph=g)

    def step(self, a):
        a["D"] = self.cpf * a["D"] + (a["A"] @ a["B"]) @ a["E"]

    def describe(self):
        return super().describe() + f" dims={self.ov} conflict={self.conflict} cpf={self.cpf}"


class InplaceMerge(Motif):
    """Graph-owned scratch dying at an element-wise consumer, the shape InplaceOptimization merges.

    ``S_i = A_i B_i`` into scratch, then ``T_i = f(S_i)`` by a product, a quotient or an axpby,
    one node per member or one grouped node for all of them, then ``R_i (+)= T_i C_i`` into the
    caller's tensors. Each S_i dies at the consumer and each T_i is pure-written there, so every
    member is a merge. Draws the pass must decline ride along: a read of T_0 ahead of the
    consumer, which sees the T_0 the last replay left
    (``test_inplace_optimization_keeps_a_destination_read_before_its_writer``); a consumer reading
    S_0 through a view; and a grouped axpby whose first member reads the next member's
    destination (``test_inplace_optimization_keeps_a_member_whose_destination_an_earlier_member_reads``).
    Scratch a hazard reads before its writer is created eagerly, since only then does it hold a
    defined value; the rest is created or declared without a zero. Never in a loop body: the pass
    leaves a graph holding control flow alone, and a body sees the parent's scratch as operands
    rather than as its own intermediates, so a looped draw would only ever be declined.
    """

    kind = "inplace"

    def __init__(self, rng, dtype, tag):
        super().__init__(rng, dtype, tag)
        n = int(rng.integers(2, 6))
        self.n = n
        self.count = int(rng.integers(1, 4))
        self.op = str(rng.choice(["product", "division", "axpby"]))
        self.grouped = self.count > 1 and rng.random() < 0.6
        self.alphas = [_scalar(rng) for _ in range(self.count)]
        self.cpf = float(rng.choice([0.0, 1.0]))
        self.read_early = bool(rng.random() < 0.2)
        self.view_src = not self.grouped and rng.random() < 0.15
        self.cross = self.grouped and self.op == "axpby" and rng.random() < 0.3
        self.declared = not (self.read_early or self.cross) and rng.random() < 0.4
        for i in range(self.count):
            self.np[f"A{i}"] = _rand(rng, (n, n), dtype)
            self.np[f"B{i}"] = _rand(rng, (n, n), dtype)
            self.np[f"C{i}"] = _rand(rng, (n, n), dtype)
            self.np[f"D{i}"] = (2.0 + rng.random((n, n))).astype(np.dtype(dtype))
            self.np[f"R{i}"] = _rand(rng, (n, n), dtype)
        if self.read_early or self.cross:
            self.np["RE"] = _rand(rng, (n, n), dtype)

    def build(self, g, target):
        self.t = {k: _tensor(f"{self.tag}_{k}", v) for k, v in self.np.items()}
        n, count = self.n, self.count

        def scratch(name):
            if self.declared:
                return g.declare_tensor(f"{self.tag}_{name}", [n, n], intermediate=True, dtype=self.dtype)
            return g.create_zero_tensor(f"{self.tag}_{name}", [n, n], intermediate=True, dtype=self.dtype)

        S = [scratch(f"S{i}") for i in range(count)]
        T = [scratch(f"T{i}") for i in range(count)]
        E = scratch("E") if self.read_early or self.cross else None
        self._keep = S + T + ([E] if E is not None else [])
        t = self.t
        la = einsums.linalg
        with cg.capture(target):
            for i in range(count):
                einsums.einsum("ij <- ik ; kj", S[i], t[f"A{i}"], t[f"B{i}"], c_pf=0.0, ab_pf=1.0)
            if self.read_early:
                la.axpby(1.0, T[0], 0.0, E)
            src = [cg.view(S[0], [(0, n), (0, n)])] + S[1:] if self.view_src else S
            D = [t[f"D{i}"] for i in range(count)]
            if self.grouped:
                zeros = [0.0] * count
                if self.op == "product":
                    la.grouped_direct_product(self.alphas, src, D, zeros, T)
                elif self.op == "division":
                    la.grouped_direct_division(self.alphas, src, D, zeros, T)
                elif self.cross:
                    la.grouped_axpby([1.0] + self.alphas, [T[1]] + src, [0.0] + zeros, [E] + T)
                else:
                    la.grouped_axpby(self.alphas, src, zeros, T)
            else:
                for i in range(count):
                    if self.op == "product":
                        la.direct_product(self.alphas[i], src[i], D[i], 0.0, T[i])
                    elif self.op == "division":
                        la.direct_division(self.alphas[i], src[i], D[i], 0.0, T[i])
                    else:
                        la.axpby(self.alphas[i], src[i], 0.0, T[i])
            for i in range(count):
                einsums.einsum("ij <- ik ; kj", t[f"R{i}"], T[i], t[f"C{i}"], c_pf=self.cpf, ab_pf=1.0)
            if E is not None:
                la.axpby(1.0, E, 0.0, t["RE"])

    def step(self, a):
        zero = np.zeros((self.n, self.n), dtype=np.dtype(self.dtype))
        S = [a[f"A{i}"] @ a[f"B{i}"] for i in range(self.count)]
        if self.read_early:
            a["_E"] = a.get("_T0", zero).copy()
        if self.cross:
            a["_E"] = a.get("_T1", zero).copy()
        for i in range(self.count):
            if self.op == "product":
                a[f"_T{i}"] = self.alphas[i] * S[i] * a[f"D{i}"]
            elif self.op == "division":
                a[f"_T{i}"] = self.alphas[i] * S[i] / a[f"D{i}"]
            else:
                a[f"_T{i}"] = self.alphas[i] * S[i]
        for i in range(self.count):
            a[f"R{i}"] = self.cpf * a[f"R{i}"] + a[f"_T{i}"] @ a[f"C{i}"]
        if self.read_early or self.cross:
            a["RE"] = a["_E"].copy()

    def describe(self):
        return super().describe() + (f" n={self.n} count={self.count} op={self.op} grouped={self.grouped}"
                                     f" read_early={self.read_early} view_src={self.view_src} cross={self.cross}"
                                     f" declared={self.declared} cpf={self.cpf}")


class GpuGemm(Motif):
    """A float32 GEMM chain large enough for GPUPlacement's cost model to offload.

    MPS runs float32 only, so the motif ignores the program dtype. The middle
    tensor either belongs to the caller or is a graph-owned intermediate, and
    the chain gives TransferElimination a device round trip to remove.
    """

    kind = "gpu"

    def __init__(self, rng, dtype, tag):
        super().__init__(rng, "float32", tag)
        n = int(rng.choice([256, 288, 320]))
        self.n = n
        self.np["A"] = _rand(rng, (n, n), "float32")
        self.np["B"] = _rand(rng, (n, n), "float32")
        self.np["D"] = _rand(rng, (n, n), "float32")
        self.np["C"] = _rand(rng, (n, n), "float32")
        self.np["x"] = _rand(rng, (n,), "float32")
        self.np["y"] = _rand(rng, (n,), "float32")
        self.owned = bool(rng.random() < 0.5)
        self.gemv = bool(rng.random() < 0.5)
        # A host permute overwriting T between the two device GEMMs: the
        # device copy of T is then stale, which is what TransferElimination's
        # residency simulation notices and TransferInsertion's does not.
        self.cpu_write = bool(rng.random() < 0.5)
        self.np["X"] = _rand(rng, (n, n), "float32") / np.float32(math.sqrt(n))
        self.cpf = float(rng.choice([0.0, 1.0]))
        if not self.owned:
            self.np["T"] = np.zeros((n, n), dtype=np.float32)

    def build(self, g, target):
        self.t = {k: _tensor(f"{self.tag}_{k}", v) for k, v in self.np.items()}
        if self.owned:
            T = g.declare_zero_tensor(f"{self.tag}_T", [self.n, self.n], intermediate=True, dtype="float32")
            self._keep = [T]
        else:
            T = self.t["T"]
        with cg.capture(target):
            einsums.einsum("ij <- ik ; kj", T, self.t["A"], self.t["B"], c_pf=0.0, ab_pf=1.0 / self.n)
            if self.cpu_write:
                einsums.permute("j,i <- i,j", T, self.t["X"], c_pf=0.0, a_pf=1.0)
            einsums.einsum("ij <- ik ; kj", self.t["C"], T, self.t["D"], c_pf=self.cpf, ab_pf=1.0)
            if self.gemv:
                einsums.einsum("i <- ij ; j", self.t["y"], T, self.t["x"], c_pf=self.cpf, ab_pf=1.0)

    def step(self, a):
        T = (a["A"].astype(np.float64) @ a["B"]) / self.n
        if self.cpu_write:
            T = a["X"].T.astype(np.float64)
        if not self.owned:
            a["T"] = T.astype(np.float32)
        a["C"] = (self.cpf * a["C"] + T @ a["D"]).astype(np.float32)
        if self.gemv:
            a["y"] = (self.cpf * a["y"] + T @ a["x"]).astype(np.float32)

    def describe(self):
        return super().describe() + (f" n={self.n} owned={self.owned} gemv={self.gemv} cpf={self.cpf}"
                                     f" cpu_write={self.cpu_write}")


MOTIFS = {m.kind: m for m in (StreamJK, LayoutChain, ScratchReuse, BigScratch, TiledChain, TiledAcrossLoop, DiskLoad,
                              DiskRoundTrip, SpaceChain, GpuGemm, InplaceMerge)}

#: The motif each pass needs in order to fire.
MOTIF_FOR_PASS = {
    "TiledExpansion": "tiled", "ScratchPrivatization": "scratch", "LayoutAssignment": "layout",
    "StreamContractionFusion": "stream", "IOPrefetch": "io", "FreeInsertion": "big",
    "SpacePropagation": "spaces", "CrossSpaceValidation": "spaces", "ScalingAnalysis": "spaces",
    "GPUPlacement": "gpu", "TransferInsertion": "gpu", "TransferElimination": "gpu", "StreamAssignment": "gpu",
}

#: Motifs the default corpus draws from.
CORPUS_KINDS = ("stream", "layout", "scratch", "big", "tiled", "tiled_loop", "io", "io_roundtrip", "spaces", "gpu")

#: Motifs that can sit inside a loop body. The stream and GPU motifs stay at
#: top level only to keep the runtime down.
_LOOPABLE = {"layout", "scratch", "big", "tiled", "io", "spaces"}


class Program:
    """Two to four motifs in one graph, some inside loop bodies, with one executor choice."""

    def __init__(self, rng, dtype, label, kinds=None, n_motifs=None, loops=True):
        if kinds is None:
            n = n_motifs or int(rng.integers(2, 5))
            kinds = [str(k) for k in rng.choice(CORPUS_KINDS, size=n, replace=False)]
        self.label = label
        self.motifs = []
        for j, kind in enumerate(kinds):
            m = MOTIFS[kind](rng, dtype, f"{label}_{j}_{next(_uid)}")
            if loops and kind in _LOOPABLE and rng.random() < 0.35:
                m.loop = int(rng.integers(2, 4))
            self.motifs.append(m)
        # ScratchPrivatization rewrites only graphs with an installed executor,
        # which is how the default pipeline builds it; most trials install one.
        self.executor = bool(rng.random() < 0.6)
        self.registry = None

    def build(self):
        g = cg.Graph(self.label)
        if any(m.kind == "spaces" for m in self.motifs):
            reg = cg.SpaceRegistry()
            ids = {name: reg.register_space(cg.index_space(name, sym, ext)) for name, sym, ext in _SPACES}
            reg.declare_disjoint(ids["occ"], ids["virt"])
            reg.declare_less(ids["occ"], ids["virt"])
            g.set_space_registry(reg)
            self.registry = reg
        if self.executor:
            g.set_executor(cg.DataflowExecutor())
        for m in self.motifs:
            target = g
            if m.loop:
                target = g.add_loop(f"{m.tag}_loop", m.loop, lambda it, c=m.loop: it < c - 1)
                if self.executor:
                    target.set_executor(cg.DataflowExecutor())
            m.build(g, target)
        return g

    def cleanup(self):
        for m in self.motifs:
            m.cleanup()

    def describe(self):
        return f"executor={self.executor} motifs=[{'; '.join(m.describe() for m in self.motifs)}]"


def compare(prog, runs, where):
    """Every caller-visible tensor of every motif against its numpy oracle."""
    for m in prog.motifs:
        exp = m.expected(runs)
        got = m.observe()
        rtol, atol = _DTYPE_TOL[m.dtype]
        for name, value in got.items():
            want = exp[name]
            scale = max(1.0, float(np.max(np.abs(want)))) if want.size else 1.0
            if not np.allclose(value, want, rtol=rtol, atol=atol * scale):
                err = float(np.max(np.abs(value - want)))
                raise AssertionError(
                    f"{where}: {m.kind} tensor {name} disagrees with numpy (max abs err {err:.3g}, scale {scale:.3g})\n"
                    f"motif={m.describe()}\nprogram={prog.describe()}")


def draw_order(rng, available, neighbours=True):
    """A shuffled pass list: every target this build has, plus a random handful of neighbours."""
    order = [p for p in TARGET_PASSES if p in available] + ["Materialization"]
    if neighbours:
        k = int(rng.integers(0, len(NEIGHBOUR_PASSES) + 1))
        order += [str(p) for p in rng.choice(NEIGHBOUR_PASSES, size=k, replace=False)]
    rng.shuffle(order)
    return order


def run_trial(prog, order, runs=1, record=True):
    """Build @p prog, apply @p order one pass at a time, close with Materialization, execute and compare."""
    g = prog.build()
    fired = []
    try:
        for name in order:
            hit = apply_pass(g, name)
            if record:
                STATS[name]["ran"] += 1
                STATS[name]["fired"] += int(hit)
            if hit:
                fired.append(name)
        _apply_one_pass(g, "Materialization")
        for _ in range(runs):
            g.execute()
        compare(prog, runs, f"order={order} runs={runs} fired={fired}")
    finally:
        prog.cleanup()
    if record:
        for kind in {m.kind for m in prog.motifs}:
            MOTIF_STATS[kind] = MOTIF_STATS.get(kind, 0) + 1
    return fired


def run_default(prog, runs=1, level=None):
    """The same program through ``default_pass_manager()`` or ``graph.optimize(level)``."""
    g = prog.build()
    try:
        if level is None:
            g.apply(cg.default_pass_manager())
        else:
            g.optimize(level)
        for _ in range(runs):
            g.execute()
        compare(prog, runs, f"default level={level} runs={runs}")
    finally:
        prog.cleanup()
