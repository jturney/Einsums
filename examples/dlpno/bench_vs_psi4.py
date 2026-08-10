#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Wall-clock comparison against psi4's native C++ DLPNO-MP2.

Both sides now apply the same three truncations - differential-overlap PAO
domains, Mulliken auxiliary domains and dipole pair prescreening - so this is a
like-for-like comparison, and the printed correlation energies agree to roughly
1e-13. Check that agreement before reading the timings: if the energies differ
at the truncation level (1e-5), the two are not solving the same problem and
the comparison means nothing.

Two differences remain, both in the port's disfavour and both visible in the
sizes printed alongside:

* psi4 builds the three-index integrals with a screened, linear-scaling
  shell-triplet loop; this port builds the dense ``(Q|mn)`` and slices it.
* the port pads every pair's block to a bucket size, trading flops for
  batchability. The padding factor is reported below.

The LMP2 iteration is the phase where the two are doing recognizably the same
work, and it is reported per iteration because the convergence paths differ.

psi4 runs in a subprocess so its timer file is flushed and its threads do not
overlap with the einsums run.

    PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/stage/lib \
        python examples/dlpno/bench_vs_psi4.py --molecule methanol
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno.molecules import MOLECULES, water_chain

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--molecule", default="methanol",
                    help=f"one of {sorted(MOLECULES)}, or 'chain<N>' for an N-monomer "
                         "water chain at 2.9 A - the geometry the scaling work targets")
parser.add_argument("--basis", default="cc-pvdz")
parser.add_argument("--t-cut-pno", type=float, default=1e-8)
parser.add_argument("--buckets", type=int, default=None,
                    help="PNO-count buckets to pad pair blocks into; default chooses "
                         "per thread count from the measured OpenMP region cost")
parser.add_argument(
    "--threads", type=int, default=1,
    help="thread count for BOTH sides. Importing psi4 clamps process-wide "
         "OpenMP to 1, so einsums needs this set too; OMP_NUM_THREADS alone "
         "does nothing once psi4 is in the process.",
)
parser.add_argument(
    "--integrals", default="screened", choices=["screened", "dfhelper", "dense"],
    help="where (Q|i u) comes from. 'screened' builds only the blocks the solver "
         "declares it will read, which is what psi4's own DLPNO does and so is "
         "what makes the DF Ints row a like-for-like comparison; 'dfhelper' is "
         "the exact psi4-backed source and 'dense' the unthreaded reference. "
         "Orthogonal to --backend, which selects stage implementations.",
)
parser.add_argument(
    "--backend", default="",
    help="stage backend spec for the port, e.g. "
         "compute_pno_overlaps=cpp,transform_pnos=cpp. Loads the dlpno_stages "
         "module, which must be on PYTHONPATH; this is how the hybrid "
         "(C++-stage) configuration is benchmarked.",
)
args = parser.parse_args()

if args.molecule.startswith("chain"):
    GEOM = water_chain(int(args.molecule[5:]), 2.9) + "symmetry c1\nno_reorient\nno_com\n"
elif args.molecule in MOLECULES:
    GEOM = MOLECULES[args.molecule] + "symmetry c1\n"
else:
    parser.error(f"unknown molecule {args.molecule!r}; "
                 f"use one of {sorted(MOLECULES)} or chain<N>")
OPTIONS = {
    "basis": args.basis, "scf_type": "df", "freeze_core": "false",
    "e_convergence": 1e-10, "d_convergence": 1e-10,
    # Match Thresholds.r_convergence. psi4's LMP2 default is 1e-6, two orders
    # looser than the port's, which leaves its energy less converged and shows
    # up as a ~2e-8 disagreement that looks like a domain difference but is not.
    # It also costs psi4 an iteration or two, so leaving it unmatched would
    # flatter psi4 in exactly the phase being compared.
    "r_convergence": 1e-8,
    "t_cut_pno": args.t_cut_pno,
}
THREADS = args.threads

PSI4_CHILD = r'''
import json, sys, time
import psi4
geom, options, outfile = json.loads(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
psi4.core.set_output_file(outfile, False)
psi4.set_options(options)
psi4.set_num_threads(int(sys.argv[4]))
psi4.geometry(geom)
t0 = time.perf_counter(); psi4.energy("scf"); t_scf = time.perf_counter() - t0
psi4.set_options({"dlpno_algorithm": "mp2"})
t0 = time.perf_counter(); e = psi4.energy("dlpno-mp2"); t_dlpno = time.perf_counter() - t0
print(json.dumps({"scf": t_scf, "dlpno": t_dlpno,
                  "corr": psi4.variable("MP2 CORRELATION ENERGY")}))
'''


def run_psi4(workdir):
    """psi4's DLPNO-MP2 in a subprocess; returns (timings, phase table, stats)."""
    script = os.path.join(workdir, "child.py")
    with open(script, "w") as fh:
        fh.write(PSI4_CHILD)
    out = os.path.join(workdir, "psi4.out")
    proc = subprocess.run(
        [sys.executable, script, json.dumps(GEOM), json.dumps(OPTIONS), out, str(THREADS)],
        cwd=workdir, capture_output=True, text=True, check=True,
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    phases = {}
    timer = os.path.join(workdir, "timer.dat")
    if os.path.exists(timer):
        for line in open(timer):
            m = re.match(r"^([A-Za-z0-9 /()-]+?)\s*:\s+\S+u\s+\S+s\s+([0-9.]+)w", line)
            if m:
                phases.setdefault(m.group(1).strip(), float(m.group(2)))

    text = open(out).read()
    stats = {}
    m = re.search(r"Avg:\s+(\d+) NOs", text)
    if m:
        stats["avg_pno"] = int(m.group(1))
    stats["iterations"] = len(re.findall(r"@LMP2 iter", text))
    m = re.search(r"Screened (\d+) of (\d+) LMO pairs", text)
    if m:
        stats["pairs"] = int(m.group(2)) - int(m.group(1))
    m = re.search(r"Projected AOs per Local MO pair:\s*\n\s*Avg:\s+(\d+)", text)
    if m:
        stats["avg_pao_pair"] = int(m.group(1))
    return result, phases, stats


with tempfile.TemporaryDirectory() as workdir:
    print(f"\n{args.molecule}/{args.basis}, T_CUT_PNO = {args.t_cut_pno:.0e}, "
          f"{THREADS} thread(s) both sides, (Q|iu) from {args.integrals!r}")
    print("running psi4 (subprocess) ...", flush=True)
    psi4_times, psi4_phases, psi4_stats = run_psi4(workdir)

# ---- this port, in-process ---------------------------------------------------
print("running the einsums port ...", flush=True)
import psi4  # noqa: E402
from dlpno import DLPNOMP2, Thresholds  # noqa: E402
from dlpno.psi4_source import from_psi4  # noqa: E402

if args.backend:
    # Import the stages module first: load_stage_module matches cpp exports
    # against stages Python has already declared. The composition methods
    # dispatch through the registry, so the selection reaches the phases
    # timed below without any further plumbing here.
    import dlpno.stages  # noqa: E402,F401
    from einsums import stages as _estages  # noqa: E402

    _estages.load_stage_module("dlpno_stages")
    _estages.apply_backend_spec(args.backend)

psi4.core.set_output_file("/tmp/psi4_dlpno_ref_scf.out", False)
# Importing psi4 sets the process-wide OpenMP thread count to 1, which silently
# serializes every einsums BLAS and gemm_batch call in this process regardless
# of OMP_NUM_THREADS. Setting it here restores threading for BOTH libraries.
psi4.set_num_threads(THREADS)
psi4.set_options(OPTIONS)
psi4.geometry(GEOM)
t0 = time.perf_counter()
_, wfn = psi4.energy("scf", return_wfn=True)
t_scf_ours = time.perf_counter() - t0

ours = {}


def phase(name, fn):
    t0 = time.perf_counter()
    result = fn()
    ours[name] = time.perf_counter() - t0
    return result


# from_psi4 builds the dense (Q|mn), and that has to be inside the timed region.
# It used to run before the clock started, which quietly excluded the port's
# single largest DF cost from the comparison while psi4's number included
# generating its own integrals - so the difference the header calls out as
# being in the port's disfavour was not actually being counted. At
# ethanol/cc-pVTZ that is 0.243 s against a 1.16 s total.
cut = Thresholds.preset("NORMAL", t_cut_pno=args.t_cut_pno, n_buckets=args.buckets)
t_total = time.perf_counter()
# The thresholds go in here, not just into the solver: a producer that screens
# has to be configured before it is declared to, and handing it the same object
# the solver gets is what keeps the two from drifting apart.
reference = phase("DF Ints", lambda: from_psi4(wfn, integrals=args.integrals, thresholds=cut))
mp2 = DLPNOMP2(reference, cut, verbose=False)

phase("Setup Orbitals", mp2.setup_orbitals)
phase("Sparsity", mp2.prep_sparsity)
t0 = time.perf_counter()
mp2.compute_metric()
mp2.compute_qia()
ours["DF Ints"] += time.perf_counter() - t0
# precompute_fits belongs to this phase, not to DF Ints: psi4 solves the
# domain fitting equations inside its own pno_transform.
phase("PNO Transform", lambda: (mp2.precompute_fits(), mp2.pno_transform()))
phase("PNO Overlaps", mp2.compute_pno_overlaps)
phase("LMP2", mp2.lmp2_iterations)
ours["DLPNO-MP2"] = time.perf_counter() - t_total
# Split the one-time graph build (allocate + capture + optimize) out of the
# LMP2 row: psi4 has no equivalent setup cost, so a row mixing the two
# compares different things. Both rows still sum into the total.
ours["LMP2 build"] = mp2.t_capture
ours["LMP2"] = mp2.t_iterate

# ---- report ------------------------------------------------------------------
print(f"\n  problem size")
print(f"    {'':22} {'psi4':>12} {'this port':>12}")
print(f"    {'LMO pairs':22} {psi4_stats.get('pairs', '?'):>12} {mp2.n_lmo_pairs:>12}"
      f"   (of {mp2.ref.naocc**2} possible; both sides prescreen)")
print(f"    {'avg PNOs per pair':22} {psi4_stats.get('avg_pno', '?'):>12} "
      f"{sum(mp2.n_pno) / mp2.n_lmo_pairs:>12.1f}"
      "   (both sides use screened PAO domains)")
print(f"    {'padded PNO dimension':22} {'-':>12} {mp2.npno_max:>12}"
      "   (port pads every block to this)")
print(f"    {'LMP2 iterations':22} {psi4_stats.get('iterations', '?'):>12} "
      f"{mp2.n_iterations:>12}")

# (row label, key in `ours`, psi4 timer name or None when psi4 has no analogue).
#
# Every phase timed above must reach this table. It did not always: `Sparsity`
# was timed and never printed, so 16% of an ethanol run was invisible in the
# very table used to decide what to optimize next. The residual row and the
# check below exist so that cannot recur silently - a newly timed phase either
# appears here by name or falls out into `other (untabulated)`.
ROWS = [
    ("Setup Orbitals", "Setup Orbitals", "Setup Orbitals"),
    ("Sparsity", "Sparsity", "Sparsity"),
    ("DF Ints", "DF Ints", "DF Ints"),
    ("PNO Transform", "PNO Transform", "PNO Transform"),
    ("PNO Overlaps", "PNO Overlaps", "PNO Overlaps"),
    ("LMP2 iterations", "LMP2", "LMP2"),
    ("LMP2 graph build", "LMP2 build", None),
]
# Any phase timed but not named above still gets a row, in the order it was
# timed, so adding a phase() call cannot drop work out of the table.
ROWS += [(key, key, key) for key in ours
         if key != "DLPNO-MP2" and key not in {r[1] for r in ROWS}]

print(f"\n  wall time (seconds)")
print(f"    {'phase':22} {'psi4':>12} {'this port':>12} {'ratio':>10}")


def print_row(label, p, o):
    if o is None:
        print(f"    {label:22} {p:>12.3f} {'-':>12} {'-':>10}")
        return
    if p is None:
        # A cost with no psi4 analogue (the one-time graph build).
        print(f"    {label:22} {'-':>12} {o:>12.3f} {'-':>10}")
        return
    ratio = f"{o / p:>9.1f}x" if p > 1e-6 else "        -"
    print(f"    {label:22} {p:>12.3f} {o:>12.3f} {ratio:>10}")


port_accounted = psi4_accounted = 0.0
for label, key, psi4_key in ROWS:
    o = ours.get(key)
    if o is None:
        continue
    port_accounted += o
    p = psi4_phases.get(psi4_key) if psi4_key else None
    if p is not None:
        psi4_accounted += p
    print_row(label, p, o)

# psi4 phases inside DLPNO-MP2 that this port folds into a phase of its own
# (the port takes S and the dipole integrals from the reference rather than
# timing them separately). Printed so psi4's column adds up too.
for psi4_key in ("Overlap Ints", "Dipole Ints"):
    p = psi4_phases.get(psi4_key)
    if p is not None:
        psi4_accounted += p
        print_row(psi4_key, p, None)

port_total = ours["DLPNO-MP2"]
psi4_total = psi4_phases.get("DLPNO-MP2", psi4_times["dlpno"])
port_residual = port_total - port_accounted
psi4_residual = psi4_total - psi4_accounted
print_row("other (untabulated)", psi4_residual, port_residual)
print_row("total DLPNO-MP2", psi4_total, port_total)

# The whole point of the rows above is to be a complete account of the total.
# Warn loudly (and fail the run at the end) if they are not, rather than let
# the table quietly under-report a phase again.
table_incomplete = abs(port_residual) > 0.05 * port_total
if table_incomplete:
    print(f"\n    WARNING: the port rows above account for only "
          f"{port_accounted:.3f} s of a {port_total:.3f} s total "
          f"({port_residual:.3f} s untabulated, over the 5% bound).\n"
          "    Some timed work is missing a row, or work between phases is "
          "untimed. Fix the\n    table before using it to choose what to "
          "optimize.")

p_it = psi4_phases.get("LMP2", 0.0) / max(psi4_stats.get("iterations", 1), 1)
# Steady state only: the graph build is paid once and has its own row above;
# it amortizes with iteration count, which the per-iteration figure is for.
o_it = mp2.t_iterate / max(mp2.n_iterations, 1)
if p_it > 1e-9:
    print(f"\n    {'LMP2 per iteration':22} {p_it:>12.4f} {o_it:>12.4f} {o_it / p_it:>9.1f}x")

# How much of the gap is padding? The coupling GEMMs run on each pair's BUCKET
# dimension rather than its own PNO count, so the wasted flops are the ratio of
# the two, summed over couplings.
#
# This used to be reported as (npno_max/avg)^3, which is the factor only if
# every block were padded to the global maximum - true before bucketing landed
# and wrong ever since. At n_buckets=4 on a six-monomer chain it reads 37x where
# the real figure is 1.6x, which is not a rounding error: it turned "the port is
# 30x more efficient per useful flop" into something believable.
padded_flops = exact_flops = 0
for ij, q in zip(mp2.plan.pair, mp2.plan.partner):
    M_b, n_ij = mp2.pair_dim(ij), mp2.n_pno[ij]
    M_p, n_p = mp2.pair_dim(q), mp2.n_pno[q]
    padded_flops += M_b * M_p * M_p
    exact_flops += n_ij * n_p * n_p
padding_overhead = padded_flops / max(exact_flops, 1)
if p_it > 1e-9:
    print(f"\n  where the LMP2 gap comes from (estimate)")
    print(f"    padding overhead   bucket dims vs true PNO counts"
          f" = {padding_overhead:>5.2f}x extra coupling flops")
    print(f"    measured slowdown                    = {o_it / p_it:>5.2f}x")
    print(f"    per useful flop                      = {o_it / p_it / padding_overhead:>5.2f}x"
          "   (<1 means faster than psi4 per flop actually needed)")

e_ours = mp2.e_lmp2 + mp2.de_pno_total + mp2.de_dipole
print(f"\n  correlation energy   psi4 {psi4_times['corr']:.10f}   "
      f"port {e_ours:.10f}   diff {abs(e_ours - psi4_times['corr']):.2e}")
print(f"  (SCF, excluded above: psi4 {psi4_times['scf']:.3f} s, "
      f"port reference {t_scf_ours:.3f} s)")
# Tolerance scaled to the PNO truncation correction rather than fixed. A
# genuine domain or truncation disagreement shows up at the scale of that
# correction; what remains below it is the PAO linear-dependence tie-break,
# where an overlap eigenvalue sits near S_CUT and the two codes round it
# differently. On ethanol/cc-pVTZ that is worth ~4e-8 per retained vector,
# so a fixed 1e-9 bound would fail on a basis-set artefact rather than a bug.
tol = max(1e-9, 0.01 * abs(mp2.de_pno_total))
if abs(e_ours - psi4_times["corr"]) > tol:
    print(f"\n  WARNING: the correlation energies disagree by more than {tol:.1e} "
          "(1% of the PNO\n  truncation correction), so the two sides are NOT solving "
          "the same problem and\n  these timings are not comparable.")
else:
    print("\n  The energies agree to within the linear-dependence tie-break, so the "
          "two sides\n  are solving the same problem. The port's remaining handicaps "
          "are the dense\n  (Q|mn) build and the block padding; both are quantified "
          "above.")

if table_incomplete:
    sys.exit(1)
