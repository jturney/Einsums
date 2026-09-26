# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Random pass order over the resource, backend and distributed passes.

Every shuffled arm before this one drew from the structural passes. The twenty
passes the default pipeline runs after them (TiledExpansion, the planning and
lifetime passes, the space diagnostics, the GPU and the distributed blocks) had
only ever run in the default order. This shard shuffles all of them, plus
Materialization and a random handful of their pipeline neighbours, one pass per
``PassManager``, closes with Materialization, and compares every caller-visible
tensor against numpy.

The pool corpus cannot make these passes act, so the programs come from motifs
in ``_resource_pass_motifs``, each shaped to make one pass fire. The census
guards at the bottom draw their own trials, so they hold whichever subset of the
file is selected, and they fail when a pass stops firing: a pass that is
shuffled and never fires has not been tested.

What this build can and cannot exercise, established by reading each pass:

  * The distributed passes need more than one rank. The mock Comm backend
    hard-wires ``comm::world_size()`` to 1, and DistributionPlanning,
    InputSlicing, SUMMAExpansion, CommunicationInsertion and
    CommunicationScheduling all return early on a single rank.
    CommunicationElimination has no rank guard, but its only input is an
    Allreduce node, which only CommunicationInsertion creates and Python has no
    way to capture. They are still shuffled, since running a no-op in every
    position checks that it stays one; a placeholder records the gap.
  * The GPU passes are compiled in when a GPU backend is (MPS on this machine),
    and GPUPlacement offloads float32 GEMMs and GEMVs the cost model favours.
    GPUDiagnostics is read-only and has no Python class, so nothing from Python
    can observe it doing anything; another placeholder records that.

Defects the shuffle found are pinned below, each reduced to the smallest
deterministic program and kept as the guard for its fix. The corpus comment in
``_resource_pass_motifs`` that steered the generator around each one names it.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import tempfile
import textwrap

# The shard is meaningless without the per-pass graph check, and ctest sets
# this for every test; a manual run gets it too.
os.environ.setdefault("EINSUMS_PASS_VERIFY", "1")

import numpy as np
import pytest

import einsums
import einsums.graph as cg
import einsums._core.graph as _G

import _resource_pass_motifs as R
from _fuzz_diff_common import _apply_one_pass
from _sanitizer_scaling import fuzz_seeds

_HERE = os.path.dirname(os.path.abspath(__file__))
_DTYPES = ["float64", "complex128", "float32", "complex64"]
_AVAILABLE = R.default_pipeline_pass_names()


def _gpu_offloads():
    """Whether GPUPlacement offloads a float32 GEMM in this build, on this machine.

    The GPU block of the default pipeline exists only when a backend or the
    mock is compiled in, and GPUPlacement declines without a usable device, so
    the firing guards ask the pass itself rather than a build flag.
    """
    if "GPUPlacement" not in _AVAILABLE:
        return False
    n = 256
    A = einsums.create_zero_tensor("probe_A", [n, n], dtype="float32")
    B = einsums.create_zero_tensor("probe_B", [n, n], dtype="float32")
    C = einsums.create_zero_tensor("probe_C", [n, n], dtype="float32")
    g = cg.Graph("gpu_probe")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
    return bool(_apply_one_pass(g, "GPUPlacement"))


_GPU_LIVE = _gpu_offloads()


def _trial(seed, dtype, label, kinds=None):
    rng = np.random.default_rng(seed)
    prog = R.Program(rng, dtype, label, kinds=kinds)
    order = R.draw_order(rng, _AVAILABLE)
    runs = 1 + int(rng.random() < 0.4)
    return R.run_trial(prog, order, runs=runs)


# ──────────────────────────────────────────────────────────────────────────
# The shuffled arms
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("seed", fuzz_seeds(800))
def test_fuzz_resource_random_pipeline(seed, dtype):
    """Two to four motifs, some in loop bodies, under a shuffled order; replayed on some seeds."""
    _trial(410_000 + seed, dtype, f"rp{seed}")


@pytest.mark.parametrize("kind", R.CORPUS_KINDS)
@pytest.mark.parametrize("seed", fuzz_seeds(50))
def test_fuzz_resource_random_pipeline_single_motif(seed, kind):
    """One motif alone, so the pass it targets meets every order with nothing else in the way."""
    dtype = _DTYPES[seed % len(_DTYPES)]
    _trial(420_000 + 97 * seed + R.CORPUS_KINDS.index(kind), dtype, f"rp1{kind}{seed}", kinds=[kind])


# ──────────────────────────────────────────────────────────────────────────
# Census guards: a shuffled pass that never fires has been shuffled, not tested.
# ──────────────────────────────────────────────────────────────────────────

# Fixed rather than scaled with the sweep, since a rate needs trials; a
# sanitizer run still gets a smaller census, just not one of four.
_GUARD_SEEDS = range(max(80, len(fuzz_seeds(240))))

#: Per pass: the motif whose presence the rate is measured against, and a floor
#: well under what the corpus gives today (in brackets), so a floor trips when a
#: generator stops producing the shape rather than on ordinary drift.
_FLOORS = {
    "TiledExpansion": ("tiled", 0.6),            # [0.89]
    "ScratchPrivatization": ("scratch", 0.25),   # [0.55]
    "LayoutAssignment": ("layout", 0.25),        # [0.49]
    "StreamContractionFusion": ("stream", 0.5),  # [fires on stream and big]
    "IOPrefetch": ("io", 0.8),                   # [1.0]
    "FreeInsertion": ("big", 0.8),               # [1.0]
    "SpacePropagation": ("spaces", 0.25),        # [0.51]
    "CrossSpaceValidation": ("spaces", 0.1),     # [0.26]
    "ScalingAnalysis": (None, 0.8),              # [0.98 of all trials]
}

_GPU_FLOORS = {
    "GPUPlacement": ("gpu", 0.5),                # [1.0 and more: big and stream offload too]
    "TransferInsertion": ("gpu", 0.3),           # [0.59]
    "StreamAssignment": ("gpu", 0.08),           # [0.20]
}


def _census(seeds, base, kinds=None):
    with_motif = {}
    fired = {}
    trials = 0
    for seed in seeds:
        rng = np.random.default_rng(base + seed)
        dtype = _DTYPES[seed % len(_DTYPES)]
        prog = R.Program(rng, dtype, f"guard{base}_{seed}", kinds=kinds)
        order = R.draw_order(rng, _AVAILABLE)
        present = {m.kind for m in prog.motifs}
        hits = R.run_trial(prog, order, runs=1 + int(rng.random() < 0.4), record=False)
        trials += 1
        for kind in present:
            with_motif[kind] = with_motif.get(kind, 0) + 1
        for name in hits:
            fired[name] = fired.get(name, 0) + 1
    return trials, with_motif, fired


def test_resource_passes_fire_under_a_shuffled_order():
    trials, with_motif, fired = _census(_GUARD_SEEDS, 430_000)
    floors = dict(_FLOORS)
    if _GPU_LIVE:
        floors.update(_GPU_FLOORS)
    for name, (kind, floor) in floors.items():
        base = trials if kind is None else with_motif.get(kind, 0)
        assert base >= 0.2 * trials, f"only {base} of {trials} programs carried a {kind} motif: {with_motif}"
        assert fired.get(name, 0) >= floor * base, (
            f"{name} fired on only {fired.get(name, 0)} of {base} programs carrying its motif ({kind})")


# ──────────────────────────────────────────────────────────────────────────
# The in-place motif. InplaceOptimization is a neighbour pass, drawn into some
# orders only, so its motif has arms of its own whose orders always hold it.
# ──────────────────────────────────────────────────────────────────────────


def _inplace_trial(seed, record=True):
    rng = np.random.default_rng(seed)
    dtype = _DTYPES[seed % len(_DTYPES)]
    kinds = ["inplace"] + ([str(rng.choice(R.CORPUS_KINDS))] if rng.random() < 0.4 else [])
    prog = R.Program(rng, dtype, f"ip{seed}", kinds=kinds)
    order = R.draw_order(rng, _AVAILABLE)
    if "InplaceOptimization" not in order:
        order.insert(int(rng.integers(0, len(order) + 1)), "InplaceOptimization")
    runs = 1 + int(rng.random() < 0.5)
    return R.run_trial(prog, order, runs=runs, record=record)


@pytest.mark.parametrize("seed", fuzz_seeds(120))
def test_fuzz_resource_inplace_motif(seed):
    """The in-place motif, alone or beside another motif, under a shuffled order that runs InplaceOptimization."""
    _inplace_trial(460_000 + seed)


def test_inplace_optimization_fires_on_its_motif():
    """InplaceOptimization merges on a floor share of the in-place motif's trials.

    A fifth of the draws carry a hazard the pass must decline on, and some orders put
    Materialization's zero-initialization of declared scratch ahead of it, so the floor sits
    well under the measured rate (0.81).
    """
    seeds = range(max(80, len(fuzz_seeds(120))))
    fired = sum("InplaceOptimization" in _inplace_trial(470_000 + seed, record=False) for seed in seeds)
    assert fired >= 0.5 * len(seeds), f"InplaceOptimization fired on only {fired} of {len(seeds)} in-place trials"


def test_transfer_elimination_fires_on_a_host_write_between_device_gemms():
    """Placeholder: TransferElimination has nothing to remove in the GPU corpus.

    The one shape the GPU motif gave it was a host permute overwriting a tensor
    between two device GEMMs. TransferInsertion used to keep that tensor
    device-resident across the host write and append a final DeviceToHost for
    it, which TransferElimination's simulation (it does track host writes) saw
    was redundant and removed. TransferInsertion now tracks host writes too
    (pinned in ``test_transfer_insertion_uploads_a_tensor_the_host_rewrote``),
    so it no longer emits that transfer, and on these 150 GPU-only programs
    TransferElimination fires on none.

    The replacement test needs a program whose transfers become redundant
    AFTER TransferInsertion places them, for instance a later pass removing or
    moving a device node (a shuffled StreamContractionFusion after
    TransferInsertion over a stream motif beside a looped ``big`` one is the
    one shape the mixed census has shown), reduced to a deterministic program
    that asserts TransferElimination fires and the numbers still match numpy.
    """
    if not _GPU_LIVE:
        pytest.skip("no GPU offload in this build or on this machine")
    trials, with_motif, fired = _census(range(150), 440_000, kinds=["gpu"])
    assert trials == 150
    assert fired.get("TransferElimination", 0) == 0, (
        f"TransferElimination fired on {fired.get('TransferElimination', 0)} of {trials} GPU programs; "
        "restore a firing floor for it and replace this placeholder")


# ──────────────────────────────────────────────────────────────────────────
# Passes this build cannot make fire. Each asserts the current behaviour and
# says what a real test would need.
# ──────────────────────────────────────────────────────────────────────────


def test_distributed_passes_are_no_ops_on_a_single_rank():
    """Placeholder: the distributed block never fires on a serial or mock-MPI build.

    Without MPI, ``comm::world_size()`` returns a hard-wired 1
    (libs/Einsums/Comm/src/Runtime.cpp), and DistributionPlanning,
    InputSlicing, SUMMAExpansion, CommunicationInsertion and
    CommunicationScheduling all return false on a single rank.
    CommunicationElimination has no rank guard, but it only deletes a repeated
    Allreduce and nothing reachable from Python creates one. Python exposes no
    ProcessGrid and no rank query either.

    The replacement test needs an MPI build launched on four ranks (SUMMA wants
    a square grid): the same motif corpus with a DistributedLayout motif (a
    deferred output above DistributionPlanning's 64 MiB threshold, or that
    threshold lowered, fed by a GEMM whose A operand is pre-allocated for
    InputSlicing), compared rank by rank against the replicated result. It
    belongs beside DistributedIntegration.cpp, which already runs under mpirun.
    """
    distributed = [p for p in R.DISTRIBUTED_PASSES if p in _AVAILABLE]
    if not distributed:
        pytest.skip("this build has no distributed block in the default pipeline")
    trials, _, fired = _census(range(60), 450_000)
    assert trials == 60
    for name in distributed:
        assert fired.get(name, 0) == 0, f"{name} fired on a single rank; replace this placeholder with a real test"


def test_gpu_diagnostics_is_unobservable_from_python():
    """Placeholder: GPUDiagnostics never reports a change and Python cannot read its counters.

    The pass is read-only by design (``run`` always returns false) and has no
    Python class, so the shuffled arm can only check that it leaves the graph
    alone. A real test would bind ``GPUDiagnostics`` with its ``gpu_nodes``,
    ``h2d_transfers``, ``d2h_transfers`` and ``peak_device_bytes`` getters and
    compare them with the transfer nodes TransferInsertion left in the graph.
    """
    if "GPUDiagnostics" not in _AVAILABLE:
        pytest.skip("this build has no GPU block in the default pipeline")
    assert not hasattr(einsums._core.graph, "GPUDiagnostics")
    _, _, fired = _census(range(40), 460_000, kinds=["gpu"])
    assert fired.get("GPUDiagnostics", 0) == 0


# ──────────────────────────────────────────────────────────────────────────
# Defects the shuffle found, each reduced to a deterministic program.
# ──────────────────────────────────────────────────────────────────────────


def _run_isolated(code):
    """Run ``code`` in a fresh interpreter, so a crash or a read of freed memory is an outcome."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_HERE, env.get("PYTHONPATH", "")])
    env.setdefault("EINSUMS_PASS_VERIFY", "1")
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, env=env,
                          timeout=300)


def _kinds(g):
    return [(n["kind"], n.get("label", "")) for n in json.loads(g.to_json())["nodes"]]


def test_disk_read_stays_after_a_disk_write_of_the_same_dataset():
    """A captured read of a dataset the same graph wrote stays behind the write.

    Defends against the capture recording a DiskRead with no inputs and a
    DiskWrite with no outputs, so that nothing tied the two to the file they
    share. IOPrefetch's earliest-legal-slot scan (IOPrefetch.cpp,
    ``prefetch_within``) only checked tensor producers and touches of the
    destination, and slid the read to the front, where it loaded the stale
    contents. Reorder alone did the same, and the DataflowExecutor ran the two
    unordered with no pass at all; ``default_pass_manager()`` and O2 returned
    the stale values.
    """
    path = os.path.join(tempfile.gettempdir(), f"einsums_respass_{os.getpid()}_roundtrip.etn")
    try:
        T = einsums.asarray(np.ones((2, 3)), name="rt_T")
        einsums.io.write(path, "T", T)  # the stale contents
        Y = einsums.create_zero_tensor("rt_Y", [2, 3], dtype="float64")
        g = cg.Graph("disk_roundtrip")
        with cg.capture(g):
            einsums.linalg.scale(5.0, T)
            einsums.io.write(path, "T", T)
            einsums.io.read(path, "T", Y)
        _apply_one_pass(g, "IOPrefetch")
        g.execute()
        np.testing.assert_allclose(np.asarray(Y), 5.0)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_io_prefetch_hoist_resets_the_node_id():
    """Hoisting a loop-invariant disk read out of a body gives the hoisted node a fresh id.

    Defends against ``hoist_reads_from_body`` (IOPrefetch.cpp) copying the
    body's node and inserting it into the parent without resetting its id to
    ``unassigned_node_id``, so that it collided with the parent's own
    numbering. The PassManager's ``settle_node_ids`` threw naming IOPrefetch,
    or, when the ids happened not to collide yet, whichever later pass issued
    the clashing id. The default pipeline did not reach it only because
    LoopInvariantHoisting lifts the read first.
    """
    path = os.path.join(tempfile.gettempdir(), f"einsums_respass_{os.getpid()}_hoist.etn")
    try:
        einsums.io.write(path, "X", einsums.asarray(np.arange(6.0).reshape(2, 3), name="hx_X"))
        Y = einsums.create_zero_tensor("hx_Y", [2, 3], dtype="float64")
        Z = einsums.create_zero_tensor("hx_Z", [2, 3], dtype="float64")
        g = cg.Graph("io_hoist")
        body = g.add_loop("l", 2, lambda it: it < 1)
        with cg.capture(body):
            einsums.io.read(path, "X", Y)
            einsums.linalg.axpy(1.0, Y, Z)
        _apply_one_pass(g, "IOPrefetch")
        g.execute()
        np.testing.assert_allclose(np.asarray(Z), 2 * np.arange(6.0).reshape(2, 3))
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_contraction_planning_uses_the_first_chains_dtype():
    """Two GEMM chains of different dtypes: each is priced and rebuilt with its own.

    Defends against ContractionPlanning::run reading ``element_size`` and
    ``dtype`` once per scan from ``chains[0][0].output_tid`` and using them for
    every chain it restructures, so that the float32 chain's intermediate and
    its two Gemm executors were built for float64 and the executor threw at the
    first replay. Whether the pass restructures is the cost model's decision,
    which Python cannot pin, so the case asserts only that C matches numpy
    whichever way it decides.
    """
    rng = np.random.default_rng(0)
    small, n = 3, 320

    def mk(name, shape, dt):
        return einsums.asarray(rng.standard_normal(shape).astype(dt), name=name)

    g = cg.Graph("mixed_chains")
    P, Q, S = (mk(f"mc_{c}", (small, small), "float64") for c in "PQS")
    O = einsums.create_zero_tensor("mc_O", [small, small], dtype="float64")
    W = g.declare_zero_tensor("mc_W", [small, small], intermediate=True, dtype="float64")
    A, B, D = (mk(f"mc_{c}", (n, n), "float32") for c in "ABD")
    C = einsums.create_zero_tensor("mc_C", [n, n], dtype="float32")
    T = g.declare_zero_tensor("mc_T", [n, n], intermediate=True, dtype="float32")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", W, P, Q)
        einsums.einsum("ij <- ik ; kj", O, W, S)
        einsums.einsum("ij <- ik ; kj", T, A, B)
        einsums.einsum("ij <- ik ; kj", C, T, D)
    _apply_one_pass(g, "ContractionPlanning")
    _apply_one_pass(g, "Materialization")
    g.execute()
    ref = np.asarray(A).astype(np.float64) @ np.asarray(B) @ np.asarray(D)
    np.testing.assert_allclose(np.asarray(C), ref, rtol=1e-3, atol=1e-3 * np.max(np.abs(ref)))


def test_gpu_gemm_into_freed_scratch_writes_the_old_buffer():
    """A device GEMM into an eager scratch that FreeInsertion frees computes into the live buffer on replay.

    Defends against ``resolve_device_ptr`` (Graph/GpuDispatch.cpp) returning,
    on unified memory, ``TensorHandle::data_ptr``, the snapshot taken at
    registration, rather than ``live_host_ptr``. FreeInsertion's Free and its
    paired Materialize give the scratch a new buffer on the second replay, so
    the device GEMM wrote freed memory and the host permute reading the scratch
    saw the fresh zeros. ``default_pass_manager()`` and O2 reached it on this
    program. Run in a child process because the old failure wrote memory the
    allocator had already reclaimed.
    """
    if not _GPU_LIVE:
        pytest.skip("no GPU offload in this build or on this machine")
    result = _run_isolated(
        """
        import numpy as np, einsums, einsums.graph as cg
        from _fuzz_diff_common import _apply_one_pass
        m = 520
        rng = np.random.default_rng(0)
        a = rng.standard_normal((m, 1)).astype("float32")
        b = rng.standard_normal((1, m)).astype("float32")
        A = einsums.asarray(a, name="A"); B = einsums.asarray(b, name="B")
        P = einsums.create_zero_tensor("P", [m, m], dtype="float32")
        g = cg.Graph("gpu_freed")
        big = g.create_zero_tensor("big", [m, m], intermediate=True, dtype="float32")
        with cg.capture(g):
            einsums.einsum("ij <- ik ; kj", big, A, B)
            einsums.permute("j,i <- i,j", P, big, c_pf=1.0, a_pf=1.0)
        assert _apply_one_pass(g, "GPUPlacement")
        assert _apply_one_pass(g, "FreeInsertion")
        g.execute()
        g.execute()
        np.testing.assert_allclose(np.asarray(P), 2 * (a @ b).T, rtol=1e-4, atol=1e-4)
        """
    )
    assert result.returncode == 0, f"the child failed (exit {result.returncode}):\n{result.stderr[-3000:]}"


def test_hoisted_producer_of_freed_eager_scratch_leaves_the_body_stale():
    """LoopInvariantHoisting then FreeInsertion: on replay the body reads the scratch's live buffer.

    Defends against the body going stale: the producer of an eager graph-owned
    scratch is lifted out of the loop, so it runs in the parent against the
    buffer FreeInsertion's paired Materialize reallocates on every replay,
    while the body's readers kept resolving the buffer captured into the body.
    A deferred scratch was not affected, so the eager release/rematerialize
    path was what left the body's stand-in behind (the stand-in resync is
    ``Graph::resync_slot_storage``, keyed on ``StorageBase::generation``). Left
    alone the allocator often hands the freed block straight back, and a stale
    read then happens to be right; the numpy allocation between the two
    replays takes that block, so the Materialize has to move the scratch and
    the check is deterministic. Run in a child process because the old failure
    read memory the scratch no longer owned.
    """
    result = _run_isolated(
        """
        import numpy as np, einsums, einsums.graph as cg
        from _fuzz_diff_common import _apply_one_pass
        m = 400
        rng = np.random.default_rng(0)
        a = rng.standard_normal((m, 1)); b = rng.standard_normal((1, m)); x = rng.standard_normal(m)
        bad, held = 0, []
        for trial in range(3):
            A = einsums.asarray(a, name="A"); B = einsums.asarray(b, name="B"); X = einsums.asarray(x, name="x")
            v = einsums.create_zero_tensor("v", [m], dtype="float64")
            g = cg.Graph("lih_free")
            big = g.create_zero_tensor("big", [m, m], intermediate=True, dtype="float64")
            body = g.add_loop("l", 3, lambda it: it < 2)
            with cg.capture(body):
                einsums.einsum("ij <- ik ; kj", big, A, B)
                einsums.einsum("i <- ij ; j", v, big, X, c_pf=1.0)
            assert _apply_one_pass(g, "LoopInvariantHoisting")
            assert _apply_one_pass(g, "FreeInsertion")
            g.execute()
            held.append(np.ones(m * m))  # takes the block the Free returned
            g.execute()
            bad += not np.allclose(np.asarray(v), 6 * ((a @ b) @ x))
        assert bad == 0, f"{bad} of 3 replays read the stale buffer"
        """
    )
    assert result.returncode == 0, f"the child failed (exit {result.returncode}):\n{result.stderr[-3000:]}"


def _tiled_from(name, grid, value):
    """A float64 tiled tensor over @p grid with every tile stored, holding @p value."""
    t = einsums.TiledRuntimeTensorD(name, grid)
    for tile in itertools.product(*[range(len(ax)) for ax in grid]):
        t.add_tile(list(tile))
    t.materialize()
    offs = [np.cumsum([0] + ax)[:-1] for ax in grid]
    for tile in itertools.product(*[range(len(ax)) for ax in grid]):
        sl = tuple(slice(offs[d][tile[d]], offs[d][tile[d]] + grid[d][tile[d]]) for d in range(len(grid)))
        np.asarray(t.tile_view(list(tile)))[...] = value[sl]
    return t


def _apply_order(g, order):
    """Apply @p order one pass at a time ("default" is the whole default pipeline), then Materialization."""
    for name in order:
        if name == "default":
            g.apply(cg.default_pass_manager())
        else:
            _apply_one_pass(g, name)
    _apply_one_pass(g, "Materialization")


#: Orders the cross-graph tiled pins run under. Reorder right after TiledExpansion
#: is what exposes a missing ordering edge; the default pipeline twice meets the
#: hoisted form of whatever the first run moved.
_CROSS_GRAPH_ORDERS = [
    ["TiledExpansion"],
    ["TiledExpansion", "Reorder"],
    ["LoopInvariantHoisting", "TiledExpansion", "Reorder"],
    ["TiledExpansion", "LoopInvariantHoisting", "TiledExpansion"],
    ["default"],
    ["default", "default"],
]

#: One-element tiles, 17 to an axis: a contraction over this grid is 17**3 = 4913
#: tile combinations, over TiledExpansion's default budget of 4096, so it stays
#: the opaque tiled op, while an elementwise op over its 289 tiles expands.
_OVER_BUDGET = [[1] * 17, [1] * 17]


@pytest.mark.parametrize("order", [["LoopInvariantHoisting", "TiledExpansion"],
                                   ["TiledExpansion", "LoopInvariantHoisting", "TiledExpansion"],
                                   ["default", "default"]], ids=",".join)
def test_tiled_consumer_of_a_hoisted_tiled_producer_expands_to_nothing(order):
    """LoopInvariantHoisting then TiledExpansion: a body permute of a tiled scratch keeps its terms.

    Defends against a hoist that stranded the body's consumer: once the
    scratch's producer sat in the parent as the opaque tiled einsum, which
    TiledExpansion leaves unexpanded, the body permute that reads it WAS
    expanded, against a predicted tile set seeded from the scratch's stored
    tiles, of which a deferred shell has none, and the body kept only the
    leftover scale of E. TiledExpansion used to plan each graph on its own, so
    a body never learned that its operand's producer lived one level up. It now
    leaves a tiled tensor that one graph writes and another reads opaque in
    both, so LoopInvariantHoisting hoists the tiled producer again and the
    numbers hold in every order.
    """
    grid = [[2], [2]]
    rng = np.random.default_rng(0)
    a, b, e = (rng.standard_normal((2, 2)) for _ in range(3))
    A, B, E = _tiled_from("th_A", grid, a), _tiled_from("th_B", grid, b), _tiled_from("th_E", grid, e)
    g = cg.Graph("tiled_hoist")
    T = g.declare_zero_tiled_tensor("th_T", grid, intermediate=True, dtype="float64")
    body = g.add_loop("l", 2, lambda it: it < 1)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", T, A, B, c_pf=0.0, ab_pf=1.0)
        einsums.permute("j,i <- i,j", E, T, c_pf=0.5, a_pf=2.0)
    if order[0] == "LoopInvariantHoisting":
        lih = _G.LoopInvariantHoisting()
        pm = cg.PassManager()
        pm.add(lih)
        assert g.apply(pm)
        assert lih.num_hoisted == 1, "the tiled producer is loop invariant and should leave the body"
        order = order[1:]
    _apply_order(g, order)
    g.execute()
    expected = e.copy()
    for _ in range(2):
        expected = 0.5 * expected + 2.0 * (a @ b).T
    np.testing.assert_allclose(np.asarray(E.tile_view([0, 0])), expected, atol=1e-12)


@pytest.mark.parametrize("order", _CROSS_GRAPH_ORDERS, ids=",".join)
def test_tiled_reader_after_a_loop_sees_the_tiles_its_body_creates(order):
    """A parent permute after a loop reads the tiled scratch the loop body fills.

    Defends against the parent being planned before, and apart from, the body:
    the scratch has no tiles at pass time and the parent held no writer of it,
    so the permute expanded against an empty tile set and E kept only
    ``0.5 * E``. Seen as a max error of 2.7 under TiledExpansion alone and 11
    under the default pipeline.
    """
    grid = [[2, 1], [2, 1]]
    rng = np.random.default_rng(1)
    a, b, e = (rng.standard_normal((3, 3)) for _ in range(3))
    A, B, E = _tiled_from("ra_A", grid, a), _tiled_from("ra_B", grid, b), _tiled_from("ra_E", grid, e)
    g = cg.Graph("tiled_reader_after_loop")
    T = g.declare_zero_tiled_tensor("ra_T", grid, intermediate=True, dtype="float64")
    body = g.add_loop("l", 2, lambda it: it < 1)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", T, A, B, c_pf=0.0, ab_pf=1.0)
    with cg.capture(g):
        einsums.permute("j,i <- i,j", E, T, c_pf=0.5, a_pf=2.0)
    _apply_order(g, order)
    g.execute()
    np.testing.assert_allclose(R._gather(E, "float64"), 0.5 * e + 2.0 * (a @ b).T, atol=1e-12)


@pytest.mark.parametrize("order", _CROSS_GRAPH_ORDERS, ids=",".join)
def test_loop_body_reading_an_opaque_parent_producer_keeps_its_terms(order):
    """A body permute reads a tiled scratch the parent writes with a contraction too big to expand.

    Defends against the body predicting its operand without the parent's
    writer, with no hoist involved: the producer stays the opaque tiled op, the
    scratch has no stored tiles, and the body permute expanded into nothing but
    the scale of E (max error 36 under TiledExpansion alone).
    """
    n = 17
    rng = np.random.default_rng(2)
    a, b, e = (rng.standard_normal((n, n)) for _ in range(3))
    A, B = _tiled_from("op_A", _OVER_BUDGET, a), _tiled_from("op_B", _OVER_BUDGET, b)
    E = _tiled_from("op_E", _OVER_BUDGET, e)
    g = cg.Graph("tiled_opaque_parent_producer")
    T = g.declare_zero_tiled_tensor("op_T", _OVER_BUDGET, intermediate=True, dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", T, A, B, c_pf=0.0, ab_pf=1.0)
    body = g.add_loop("l", 2, lambda it: it < 1)
    with cg.capture(body):
        einsums.permute("j,i <- i,j", E, T, c_pf=0.5, a_pf=2.0)
    _apply_order(g, order)
    g.execute()
    expected = e.copy()
    for _ in range(2):
        expected = 0.5 * expected + 2.0 * (a @ b).T
    np.testing.assert_allclose(R._gather(E, "float64"), expected, atol=1e-10)


@pytest.mark.parametrize("order", _CROSS_GRAPH_ORDERS, ids=",".join)
def test_expanded_parent_writer_stays_ordered_before_a_loop_that_reads_it(order):
    """A parent axpy into X, then a loop whose body contracts X too big to expand.

    Defends against a missing dependency edge: the parent's axpy expanded into
    per-tile writes naming tile ids, while the Loop node, through its body's
    opaque contraction, names X as a whole. Nothing ordered the two, and
    Reorder ran the loop first (max error 43 with Reorder after TiledExpansion,
    38 under the default pipeline).
    """
    n = 17
    rng = np.random.default_rng(3)
    a, x0, f, e = (rng.standard_normal((n, n)) for _ in range(4))
    A, X = _tiled_from("oe_A", _OVER_BUDGET, a), _tiled_from("oe_X", _OVER_BUDGET, x0)
    F, E = _tiled_from("oe_F", _OVER_BUDGET, f), _tiled_from("oe_E", _OVER_BUDGET, e)
    g = cg.Graph("tiled_parent_writer_edge")
    with cg.capture(g):
        einsums.linalg.axpy(1.5, A, X)
    body = g.add_loop("l", 2, lambda it: it < 1)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", E, X, F, c_pf=1.0, ab_pf=1.0)
    _apply_order(g, order)
    g.execute()
    np.testing.assert_allclose(R._gather(E, "float64"), e + 2.0 * ((x0 + 1.5 * a) @ f), atol=1e-10)


def test_default_pipeline_reports_a_contraction_over_disjoint_spaces():
    """The default pipeline reports a contraction over disjoint spaces it also zeroes.

    DeltaElimination.hpp says reducing such a contraction to its prefactor and
    CrossSpaceValidation reporting it as an error are "one observation with two
    responses, and running both is the intended combination". Defends against
    the default order running DeltaElimination second and CrossSpaceValidation
    after Materialization, by which time the offending contraction was gone, so
    ``explain()`` carried no finding and the only trace was DeltaElimination's
    own "reduced 1 contraction(s) over disjoint spaces" line.
    """
    reg = cg.SpaceRegistry()
    ids = {name: reg.register_space(cg.index_space(name, sym, ext)) for name, sym, ext in R._SPACES}
    reg.declare_disjoint(ids["occ"], ids["virt"])
    rng = np.random.default_rng(0)
    A = einsums.asarray(rng.standard_normal((2, 3)), name="ds_A")
    B = einsums.asarray(rng.standard_normal((3, 4)), name="ds_B")
    C = einsums.create_zero_tensor("ds_C", [2, 4], dtype="float64")
    g = cg.Graph("disjoint")
    g.set_space_registry(reg)
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)
    cg.annotate(A, ("occ", "virt"), graph=g)
    cg.annotate(B, ("occ", "aux"), graph=g)  # 'a' is virtual on A, occupied on B
    pm = cg.default_pass_manager()
    g.apply(pm)
    assert "CrossSpaceValidation: error" in pm.explain()


def test_transfer_insertion_uploads_a_tensor_the_host_rewrote():
    """A host write between two device GEMMs is followed by an upload of what it wrote.

    Defends against TransferInsertion::run setting an output's residency to
    Device after a GPU node and never lowering it when a CPU node writes the
    tensor. After ``T = A B`` on the device and a host permute into T, the
    second device GEMM reading T got no HostToDevice, and the final
    DeviceToHost(T) it appended would copy the device's stale T over the host
    result. Unified memory (MPS) makes both transfers synchronisation points
    only, so the numbers here are right either way and the check is on the
    transfer nodes; on a discrete device both are wrong answers. With the fix
    TransferElimination has nothing left to remove on this shape (see the
    placeholder above).
    """
    if not _GPU_LIVE:
        pytest.skip("no GPU offload in this build or on this machine")
    n = 256
    rng = np.random.default_rng(0)
    A, B, D, X = (einsums.asarray(rng.standard_normal((n, n)).astype("float32"), name=f"ti_{c}") for c in "ABDX")
    T = einsums.create_zero_tensor("ti_T", [n, n], dtype="float32")
    C = einsums.create_zero_tensor("ti_C", [n, n], dtype="float32")
    g = cg.Graph("transfer_after_host_write")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", T, A, B)
        einsums.permute("j,i <- i,j", T, X, c_pf=0.0, a_pf=1.0)
        einsums.einsum("ij <- ik ; kj", C, T, D)
    assert _apply_one_pass(g, "GPUPlacement")
    assert _apply_one_pass(g, "TransferInsertion")
    labels = [label for _, label in _kinds(g)]
    permute_at = next(i for i, label in enumerate(labels) if label.startswith("permute"))
    assert "H2D(ti_T)" in labels[permute_at:], labels
    assert "D2H(ti_T)" not in labels[permute_at:], labels
