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
  batchability (see bench_batching.py). The padding factor is reported below.

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
parser.add_argument("--buckets", type=int, default=4,
                    help="PNO-count buckets to pad pair blocks into")
parser.add_argument(
    "--threads", type=int, default=1,
    help="thread count for BOTH sides. Importing psi4 clamps process-wide "
         "OpenMP to 1, so einsums needs this set too; OMP_NUM_THREADS alone "
         "does nothing once psi4 is in the process.",
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
          f"{THREADS} thread(s) both sides")
    print("running psi4 (subprocess) ...", flush=True)
    psi4_times, psi4_phases, psi4_stats = run_psi4(workdir)

# ---- this port, in-process ---------------------------------------------------
print("running the einsums port ...", flush=True)
import psi4  # noqa: E402
from dlpno import DLPNOMP2, Thresholds  # noqa: E402
from dlpno.psi4_source import from_psi4  # noqa: E402

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
t_total = time.perf_counter()
reference = phase("DF Ints", lambda: from_psi4(wfn))
mp2 = DLPNOMP2(reference, Thresholds.preset("NORMAL", t_cut_pno=args.t_cut_pno, n_buckets=args.buckets), verbose=False)

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

print(f"\n  wall time (seconds)")
print(f"    {'phase':22} {'psi4':>12} {'this port':>12} {'ratio':>10}")
for label, key in [("Setup Orbitals", "Setup Orbitals"), ("DF Ints", "DF Ints"),
                   ("PNO Transform", "PNO Transform"), ("PNO Overlaps", "PNO Overlaps"),
                   ("LMP2", "LMP2"), ("total DLPNO-MP2", "DLPNO-MP2")]:
    p = psi4_phases.get(key)
    o = ours.get(key)
    if p is None or o is None:
        continue
    ratio = f"{o / p:>9.1f}x" if p > 1e-6 else "        -"
    print(f"    {label:22} {p:>12.3f} {o:>12.3f} {ratio:>10}")

p_it = psi4_phases.get("LMP2", 0.0) / max(psi4_stats.get("iterations", 1), 1)
# Steady state only: graph capture and optimization are paid once, and at ten
# iterations they dominate the LMP2 total. psi4 has no equivalent setup cost,
# so folding ours into a per-iteration figure would compare different things.
o_it = mp2.t_iterate / max(mp2.n_iterations, 1)
print(f"\n    {'graph capture+optimize':22} {'-':>12} {mp2.t_capture:>12.3f}"
      "   (one-time, not in the per-iteration figure)")
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
