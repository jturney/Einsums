# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
"""Profile eager calls vs a captured ComputeGraph on two workloads.

--workload batch (default): N independent small matrix multiplications, the
shape where graph optimization shines - the GEMMBatching pass collapses the
captured loop into batched kernels, while the eager loop pays per-call
dispatch. This generates the README performance figure.

--workload fock: the J/K/G Fock build shared with the other profile_*
drivers in this directory. Kernel-dominated with nothing to fuse, it shows
graph capture adding NO overhead when the passes find no structure
(speedup ~1.0x) - the honest baseline companion to the batch numbers.

Run from the repo root with the python package built (EINSUMS_BUILD_PYTHON=ON):

    PYTHONPATH=build/lib python devtools/profiling/profile_graph.py \
        --workload batch --plot docs/sphinx/_static/index-images/eager_vs_graph.png
"""

import argparse
import math
import platform
import timeit

import numpy as np

import einsums
import einsums.graph as cg


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs, m):
    if len(xs) < 2:
        return 0.0
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def make_workload(norbs, rng):
    g = einsums.asarray(rng.standard_normal((norbs, norbs, norbs, norbs)))
    D = einsums.asarray(rng.standard_normal((norbs, norbs)))
    J = einsums.zeros((norbs, norbs))
    K = einsums.zeros((norbs, norbs))
    G = einsums.zeros((norbs, norbs))
    return g, D, J, K, G


def eager_iteration(g, D, J, K, G):
    einsums.einsum("mn <- mnls ; ls", J, g, D, c_pf=0.0, ab_pf=2.0)
    einsums.einsum("mn <- mlns ; ls", K, g, D, c_pf=0.0, ab_pf=1.0)
    einsums.linalg.axpby(1.0, J, 0.0, G)   # G = J   (J already carries the 2x)
    einsums.linalg.axpy(-1.0, K, G)        # G -= K


def profile_size(norbs, trials, seed=0):
    rng = np.random.default_rng(seed)
    g, D, J, K, G = make_workload(norbs, rng)

    def refresh_density():
        np.asarray(D)[...] = rng.standard_normal((norbs, norbs))

    # Eager: every iteration re-dispatches each call.
    def eager_once():
        refresh_density()
        eager_iteration(g, D, J, K, G)

    eager_once()  # warm-up (plan caches, allocators)
    eager_times = timeit.repeat(eager_once, repeat=trials, number=1)

    # Graph: captured once, optimized once, replayed per iteration.
    graph = cg.Graph(f"fock_{norbs}")
    with cg.capture(graph):
        einsums.einsum("mn <- mnls ; ls", J, g, D, c_pf=0.0, ab_pf=2.0)
        einsums.einsum("mn <- mlns ; ls", K, g, D, c_pf=0.0, ab_pf=1.0)
        einsums.linalg.axpby(1.0, J, 0.0, G)
        einsums.linalg.axpy(-1.0, K, G)
    graph.apply(cg.default_pass_manager())

    def graph_once():
        refresh_density()
        graph.execute()

    graph_once()  # warm-up
    graph_times = timeit.repeat(graph_once, repeat=trials, number=1)

    em, gm = mean(eager_times), mean(graph_times)
    print(f"n={norbs:4d}  eager {em * 1e3:9.3f} ms (sd {stdev(eager_times, em) * 1e3:7.3f})"
          f"  graph {gm * 1e3:9.3f} ms (sd {stdev(graph_times, gm) * 1e3:7.3f})"
          f"  speedup {em / gm:5.2f}x")
    return em, gm


def make_plot(sizes, eager_ms, graph_ms, path, xlabel="orbitals"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    ax.plot(sizes, eager_ms, "o-", color="#888888", label="eager API")
    ax.plot(sizes, graph_ms, "o-", color="#1f77b4", label="ComputeGraph (captured + optimized)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("time per iteration (ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    # Points sitting near the bottom of the axis get their speedup label
    # ABOVE the marker so it never collides with the x-axis; higher points
    # keep the label below.
    low_cutoff = 2.0 * min(graph_ms)
    for x, e, g in zip(sizes, eager_ms, graph_ms):
        above = g <= low_cutoff
        ax.annotate(f"{round(e / g):.0f}x", (x, g), textcoords="offset points",
                    xytext=(0, 8 if above else -16), ha="center", fontsize=9, color="#1f77b4")
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")
    print(f"generated on: {platform.platform()}, {platform.processor() or platform.machine()}")


def profile_batch(dim, count, trials, seed=0):
    rng = np.random.default_rng(seed)
    As = [einsums.asarray(rng.standard_normal((dim, dim))) for _ in range(count)]
    Bs = [einsums.asarray(rng.standard_normal((dim, dim))) for _ in range(count)]
    Cs = [einsums.zeros((dim, dim)) for _ in range(count)]

    def eager_once():
        for a, b, c in zip(As, Bs, Cs):
            einsums.einsum("ij <- ik ; kj", c, a, b)

    graph = cg.Graph(f"batch_{dim}")
    with cg.capture(graph):
        for a, b, c in zip(As, Bs, Cs):
            einsums.einsum("ij <- ik ; kj", c, a, b)
    graph.apply(cg.default_pass_manager())

    # Sub-millisecond timings are noisy on a desktop OS (scheduler moves,
    # P/E-core migration): aggregate ~100 ms of work per sample, INTERLEAVE
    # the eager and graph samples so both see the same machine conditions,
    # and report the minimum across samples (the standard least-interference
    # estimator) so regenerated figures are stable run to run.
    eager_once()      # warm-up
    graph.execute()   # warm-up
    number = max(1, int(0.100 // max(timeit.timeit(eager_once, number=1), 1e-6)))
    eager_samples, graph_samples = [], []
    for _ in range(trials):
        eager_samples.append(timeit.timeit(eager_once, number=number) / number)
        graph_samples.append(timeit.timeit(graph.execute, number=number) / number)
    eager_t, graph_t = min(eager_samples), min(graph_samples)

    print(f"dim={dim:4d} x{count}  eager {eager_t * 1e3:9.3f} ms  graph {graph_t * 1e3:9.3f} ms"
          f"  (min of {trials} interleaved samples x {number} iters)  speedup {eager_t / graph_t:5.2f}x")
    return eager_t, graph_t


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workload", choices=["batch", "fock"], default="batch")
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="batch: matrix dims (default 8 16 32 64 128); fock: orbital counts (default 20..100)")
    ap.add_argument("--count", type=int, default=200, help="batch: number of matrix multiplications")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--plot", type=str, default=None, help="write a PNG to this path")
    args = ap.parse_args()

    if args.workload == "batch":
        sizes = args.sizes or [8, 16, 32, 64, 128]
        results = [profile_batch(n, args.count, args.trials) for n in sizes]
        xlabel = f"matrix dimension ({args.count} independent multiplications)"
    else:
        sizes = args.sizes or [20, 40, 60, 80, 100]
        results = [profile_size(n, args.trials) for n in sizes]
        xlabel = "orbitals"

    eager_ms = [e * 1e3 for e, _ in results]
    graph_ms = [g * 1e3 for _, g in results]

    if args.plot:
        make_plot(sizes, eager_ms, graph_ms, args.plot, xlabel)


if __name__ == "__main__":
    main()
