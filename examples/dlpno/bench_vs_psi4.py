#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Wall-clock comparison against psi4's native C++ DLPNO-MP2 and DLPNO-CCSD.

Both sides apply the same three truncations - differential-overlap PAO domains,
Mulliken auxiliary domains and dipole pair prescreening - so this is a
like-for-like comparison, and the printed correlation energies agree to roughly
1e-13 for MP2 and to convergence tolerance for CCSD. Check that agreement before
reading the timings: if the energies differ at the truncation level (1e-5), the
two are not solving the same problem and the comparison means nothing.

Two differences remain, both in the port's disfavour and both visible in the
sizes printed alongside:

* psi4 builds the three-index integrals with a screened, linear-scaling
  shell-triplet loop; this port builds the dense ``(Q|mn)`` and slices it. For
  ``--method ccsd`` that handicap is unavoidable rather than a default: only the
  dense source serves the ``(Q|i j)`` and ``(Q|u v)`` the CC integral layer
  declares.
* the port pads every pair's block to a bucket size, trading flops for
  batchability. The padding factor is reported below for MP2.

The iteration is the phase where the two are doing recognizably the same work,
and it is reported per iteration because the convergence paths differ.

**Rows are exclusive on the port's side and made exclusive on psi4's.** A phase
that contains another billed phase is charged only for its own part, so the rows
sum to the total rather than over-counting it. The one-time graph builds are the
exception: printed, but kept out of the compared total, because psi4 has no
analogue and a one-time cost mixed into a steady-state comparison misprices both. psi4's CC timers nest the other
way - ``Refined Pair Prescreening`` contains ``PNO-LMP2 Iterations``, which is
why psi4's own rows sum to more than its total - so the table subtracts to
recover psi4's exclusive time before comparing.

psi4 runs in a subprocess so its timer file is flushed and its threads do not
overlap with the einsums run.

    PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/stage/lib \
        python examples/dlpno/bench_vs_psi4.py --molecule methanol
    PYTHONPATH=... python examples/dlpno/bench_vs_psi4.py \
        --method ccsd --molecule chain6 --threads 10
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
parser.add_argument(
    "--method", default="mp2", choices=["mp2", "ccsd", "ccsd(t)"],
    help="which DLPNO method to compare. 'ccsd' runs the whole prescreening "
         "cascade and the T1-dressed CC iteration on both sides and prices "
         "them phase against phase; 'ccsd(t)' adds the triples, which psi4's "
         "own budget makes the majority of the run at benchmark scale. Only "
         "the dense integral source serves either, because the CC layers "
         "declare (Q|i j) and (Q|u v).")
parser.add_argument("--molecule", default="methanol",
                    help=f"one of {sorted(MOLECULES)}, or 'chain<N>' for an N-monomer "
                         "water chain at 2.9 A - the geometry the scaling work targets")
parser.add_argument("--basis", default="cc-pvdz")
parser.add_argument(
    "--t-cut-pno", type=float, default=None,
    help="PNO occupation cutoff for BOTH sides. Defaults to 1e-8 for mp2, the "
         "value the MP2 comparisons were calibrated at; for ccsd it defaults to "
         "leaving the preset alone, because the CC branch's NORMAL is 3.33e-7 "
         "and pinning it to the MP2 number would benchmark a different method.")
parser.add_argument("--buckets", type=int, default=None,
                    help="PNO-count buckets to pad pair blocks into; default chooses "
                         "per thread count from the measured OpenMP region cost")
parser.add_argument(
    "--threads", type=int, default=1,
    help="thread count for BOTH sides. Importing psi4 takes the process-wide "
         "OpenMP count over, setting it to OMP_NUM_THREADS if that was exported "
         "and to 1 if not; this overrides whichever it picked.",
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
    "--in-core-memory", default=None,
    help="in-core budget for the port's chunked phases, e.g. '8 GB'. Defaults "
         "to what the machine actually has free, NOT to --memory: psi4 can "
         "honour a grant larger than the machine because it spills to disk, "
         "and the port has no disk path (design decision 10), so handing it "
         "psi4's number authorises an allocation that pages instead of one "
         "that refuses. The port refuses rather than paging when it does not "
         "fit, so a run that reports its requirement here is a result, not a "
         "failure.")
parser.add_argument(
    "--memory", default="24 GB",
    help="memory ceiling for psi4. Its DLPNO memory estimate scales with the "
         "thread count, so the default 500 MB aborts a threaded cc-pVTZ run "
         "that a serial one completes.")
parser.add_argument(
    "--buffer-size", default=None,
    help="einsums buffer ceiling (--einsums:buffer-size). Left alone by "
         "default. The coupled-cluster runs used to force this to 2GB, because "
         "every view's dims and strides were charged against a ceiling meant "
         "for contraction workspace and a captured iteration holds hundreds of "
         "thousands of views; shape metadata now uses ShapeVector, so the stock "
         "default is enough and this is only an escape hatch.")
parser.add_argument(
    "--backend", default="",
    help="stage backend spec OVERRIDE, e.g. compute_pno_integrals=python. "
         "The compiled backends are selected automatically whenever the "
         "dlpno_stages module is importable (the main build puts it in "
         "build/lib), so the hybrid configuration is the default and this "
         "flag exists to benchmark away from it. EINSUMS_STAGE_BACKEND "
         "does the same from the environment.",
)
args = parser.parse_args()

if args.molecule.startswith("chain"):
    GEOM = water_chain(int(args.molecule[5:]), 2.9) + "symmetry c1\nno_reorient\nno_com\n"
elif args.molecule in MOLECULES:
    GEOM = MOLECULES[args.molecule] + "symmetry c1\n"
else:
    parser.error(f"unknown molecule {args.molecule!r}; "
                 f"use one of {sorted(MOLECULES)} or chain<N>")
# Before the first compute call, which is what einsums::initialize reads.
import einsums.rc  # noqa: E402

if args.buffer_size is not None:
    einsums.rc.buffer_size = args.buffer_size

T_CUT_PNO = args.t_cut_pno
if T_CUT_PNO is None and args.method == "mp2":
    T_CUT_PNO = 1e-8
if args.method != "mp2":
    # Only the dense source serves (Q|i j) and (Q|u v); see decision 5 of the
    # design. That is a real handicap in this table rather than a detail - it is
    # a row psi4 wins on, and a screened (Q|u v) source is a psi4-side patch.
    args.integrals = "dense"

OPTIONS = {
    "basis": args.basis, "scf_type": "df", "freeze_core": "false",
    "e_convergence": 1e-10, "d_convergence": 1e-10,
    # Match Thresholds.r_convergence. psi4's LMP2 default is 1e-6, two orders
    # looser than the port's, which leaves its energy less converged and shows
    # up as a ~2e-8 disagreement that looks like a domain difference but is not.
    # It also costs psi4 an iteration or two, so leaving it unmatched would
    # flatter psi4 in exactly the phase being compared.
    "r_convergence": 1e-8,
}
if T_CUT_PNO is not None:
    OPTIONS["t_cut_pno"] = T_CUT_PNO
if args.method == "ccsd":
    OPTIONS["pno_convergence"] = "NORMAL"
THREADS = args.threads

PSI4_CHILD = r'''
import json, sys, time
import psi4
geom, options, outfile = json.loads(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
method, memory = sys.argv[5], sys.argv[6]
psi4.core.set_output_file(outfile, False)
psi4.set_memory(memory)
psi4.set_options(options)
psi4.set_num_threads(int(sys.argv[4]))
psi4.geometry(geom)
t0 = time.perf_counter(); psi4.energy("scf"); t_scf = time.perf_counter() - t0
psi4.set_options({"dlpno_algorithm": method})
t0 = time.perf_counter()
e = psi4.energy("dlpno-" + method)
t_dlpno = time.perf_counter() - t0
psivar = {"mp2": "MP2 CORRELATION ENERGY",
          "ccsd": "CCSD CORRELATION ENERGY"}.get(method, "CCSD(T) CORRELATION ENERGY")
print(json.dumps({"scf": t_scf, "dlpno": t_dlpno,
                  "corr": psi4.variable(psivar)}))
'''


def run_psi4(workdir):
    """psi4's DLPNO-MP2 in a subprocess; returns (timings, phase table, stats)."""
    script = os.path.join(workdir, "child.py")
    with open(script, "w") as fh:
        fh.write(PSI4_CHILD)
    out = os.path.join(workdir, "psi4.out")
    proc = subprocess.run(
        [sys.executable, script, json.dumps(GEOM), json.dumps(OPTIONS), out,
         str(THREADS), args.method, args.memory],
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
    stats["cc_iterations"] = len(re.findall(r"@LCCSD iter", text))
    stats["t_iterations"] = len(re.findall(r"@LCCSD\(T\) iter", text))
    m = re.search(r"Number of \(Unique\) Local MO triplets:\s*(\d+)", text)
    if m:
        stats["triplets"] = int(m.group(1))
    tno = re.findall(r"Natural Orbitals per Local MO triplet:\s*\n\s*Avg:\s+(\d+)", text)
    if tno:
        stats["avg_tno"] = int(tno[-1])
    m = re.search(r"Screened (\d+) of (\d+) LMO pairs", text)
    if m:
        stats["pairs"] = int(m.group(2)) - int(m.group(1))
    m = re.search(r"Projected AOs per Local MO pair:\s*\n\s*Avg:\s+(\d+)", text)
    if m:
        stats["avg_pao_pair"] = int(m.group(1))
    return result, phases, stats


with tempfile.TemporaryDirectory() as workdir:
    pno = f"{T_CUT_PNO:.0e}" if T_CUT_PNO is not None else "preset NORMAL"
    print(f"\nDLPNO-{args.method.upper()}, {args.molecule}/{args.basis}, "
          f"T_CUT_PNO = {pno}, {THREADS} thread(s) both sides, "
          f"(Q|iu) from {args.integrals!r}")
    print("running psi4 (subprocess) ...", flush=True)
    psi4_times, psi4_phases, psi4_stats = run_psi4(workdir)

# ---- this port, in-process ---------------------------------------------------
print("running the einsums port ...", flush=True)
import psi4  # noqa: E402
from dlpno import DLPNOMP2, Thresholds  # noqa: E402
from dlpno.ccsd import DLPNOCCSD  # noqa: E402
from dlpno.triples import DLPNOCCSDT  # noqa: E402
from dlpno.psi4_source import from_psi4  # noqa: E402

if args.backend:
    # Importing dlpno.stages declares the stages AND auto-loads the compiled
    # backends when dlpno_stages is importable, so all the flag has left to do
    # is override that default. The composition methods dispatch through the
    # registry, so the selection reaches the phases timed below without any
    # further plumbing here.
    import dlpno.stages  # noqa: E402,F401
    from einsums import stages as _estages  # noqa: E402

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
#: Depth of billed calls currently on the stack, and the time each has already
#: charged to a nested row. Rows are EXCLUSIVE - a phase that contains another
#: billed phase is charged only for the part that is its own - so the rows sum
#: to the total instead of over-counting it. psi4's own timers nest the other
#: way, which is why its CC rows sum to more than its total; where a psi4 row
#: contains another, the table below subtracts to get psi4's exclusive time.
_billing_stack = []


def bill(obj, name, method_name):
    """Charge every call of ``obj.method_name`` to the row ``name``.

    Wrapping rather than calling the steps one at a time from here, because the
    coupled-cluster cascade runs eleven of them in an order ``prescreen_pairs``
    already defines. Spelling that order out a second time in the benchmark is
    how a benchmark and the thing it measures drift apart; and several of the
    steps run more than once, which a linear script would have to know about.
    """
    original = getattr(obj, method_name)

    def timed(*a, **kw):
        t0 = time.perf_counter()
        _billing_stack.append(0.0)
        try:
            return original(*a, **kw)
        finally:
            elapsed = time.perf_counter() - t0
            nested = _billing_stack.pop()
            ours[name] = ours.get(name, 0.0) + elapsed - nested
            if _billing_stack:
                _billing_stack[-1] += elapsed

    setattr(obj, method_name, timed)


preset_kwargs = {"n_buckets": args.buckets}
if T_CUT_PNO is not None:
    preset_kwargs["t_cut_pno"] = T_CUT_PNO


def _bytes(text):
    """'24 GB' -> bytes, matching how psi4 reads the same string."""
    number, _, unit = text.strip().partition(" ")
    scale = {"": 1, "B": 1, "KB": 2 ** 10, "MB": 2 ** 20, "GB": 2 ** 30,
             "TB": 2 ** 40}[unit.strip().upper()]
    return int(float(number) * scale)


def _available_bytes(margin=0.5):
    """What the machine will actually hand back, from ``memory_pressure``.

    Deliberately not ``--memory``. That grant is psi4's, and psi4 honours one
    larger than the machine has by spilling to disk; the port has no disk path,
    so the same number would authorise an allocation that pages rather than one
    that refuses - which is the failure this budget exists to turn into an
    error message.

    The margin is half rather than something closer to one because this budget
    is a PEAK dial and not only a refusal threshold: the chunked phases pack
    their work up to it, so a chunk grows until it fills whatever it is given.
    Measured on ethanol/cc-pVTZ, a 4 GiB budget peaks the process at 7.8 GB and
    a 16 GiB budget at 17.6 GB, for the same answer in the same time. Half of
    free memory leaves room for everything already resident and for the fact
    that a phase is billed against this number while the previous phase's
    stores are still alive.
    """
    try:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                   capture_output=True, text=True).stdout)
        out = subprocess.run(["memory_pressure"],
                             capture_output=True, text=True).stdout
        free = re.search(r"free percentage:\s*(\d+)%", out).group(1)
        return int(total * int(free) / 100 * margin)
    except (ValueError, AttributeError, OSError):
        # Not macOS, or the tools moved. sysconf is POSIX and reports the same
        # quantity; falling back beats crashing a benchmark over a budget the
        # user can always state explicitly with --in-core-memory.
        return int(os.sysconf("SC_AVPHYS_PAGES")
                   * os.sysconf("SC_PAGE_SIZE") * margin)


if args.method != "mp2":
    if args.in_core_memory:
        IN_CORE = _bytes(args.in_core_memory)
    else:
        IN_CORE = _available_bytes()
        print(f"in-core budget {IN_CORE / 2**30:.1f} GiB, from free memory "
              f"(psi4 is separately granted {args.memory}, which it can exceed "
              f"by spilling and the port cannot)")
    preset_kwargs["in_core_memory"] = IN_CORE

cut = Thresholds.preset("NORMAL", method="mp2" if args.method == "mp2" else "cc",
                        **preset_kwargs)

t_total = time.perf_counter()
# The thresholds go in here, not just into the solver: a producer that screens
# has to be configured before it is declared to, and handing it the same object
# the solver gets is what keeps the two from drifting apart.
reference = phase("DF Ints", lambda: from_psi4(wfn, integrals=args.integrals, thresholds=cut))

if args.method == "mp2":
    calc = mp2 = DLPNOMP2(reference, cut, verbose=False)
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
    # compares different things. The build is excluded from the compared total.
    ours["LMP2 build"] = mp2.t_capture
    ours["LMP2"] = mp2.t_iterate
else:
    triples = args.method == "ccsd(t)"
    calc = mp2 = (DLPNOCCSDT if triples else DLPNOCCSD)(reference, cut, verbose=False)
    # The row names are psi4's timer names, so the two columns line up without
    # a translation table. Every step of compute_energy is charged to one.
    for row, name in (
            ("Setup Orbitals", "setup_orbitals"),
            ("Setup Orbitals", "compute_doi"),
            ("Sparsity", "prep_sparsity"),
            ("DF Ints", "compute_metric"),
            ("DF Ints", "compute_qia"),
            ("DF Ints", "compute_qij"),
            ("DF Ints", "compute_qab"),
            ("Initial Pair Prescreening", "semicanonical_pair_energies"),
            ("PNO Transform", "precompute_fits"),
            ("PNO Transform", "pno_transform"),
            ("PNO-LMP2 Iterations", "pno_lmp2_iterations"),
            ("Compute PNOs (CCSD)", "recompute_pnos"),
            ("PNO Integrals", "compute_pno_integrals"),
            ("PNO Overlaps", "compute_pno_overlaps"),
            ("LCCSD", "lccsd_iterations"),
    ) + ((
            # psi4's own timer names again. Triples Sparsity and TNO transform
            # each run three times and LCCSD(T0) twice; bill() accumulates, so
            # each row is the whole of that phase across the cascade, which is
            # what psi4's timer reports too.
            ("Triples Sparsity", "triples_sparsity"),
            ("TNO transform", "tno_transform"),
            ("LCCSD(T0)", "compute_lccsd_t0"),
            ("Sort Triplets", "sort_triplets"),
            ("LCCSD(T) Iterations", "lccsd_t_iterations"),
    ) if triples else ()):
        bill(mp2, row, name)
    mp2.compute_energy(method="ccsd(t)") if triples else mp2.compute_energy()
    ours["DLPNO-CCSD"] = time.perf_counter() - t_total
    # Split the one-time plan and capture out of the LCCSD row, as the MP2 path
    # does: psi4 has no equivalent, so a row mixing the two compares different
    # things. The build is excluded from the compared total.
    # The triples path drops the CCSD solver once it has the amplitudes, so
    # its statistics come from the snapshot _release_ccsd keeps.
    stats_cc = (mp2.ccsd_stats if triples else
                dict(t_plan=mp2.lccsd.t_plan, t_capture=mp2.lccsd.t_capture,
                     t_iterate=mp2.lccsd.t_iterate,
                     iterations=mp2.lccsd.n_iterations,
                     nodes=mp2.lccsd.num_nodes(),
                     lmp2_iterations=mp2.lmp2.n_iterations))
    ours["LCCSD build"] = stats_cc["t_plan"] + stats_cc["t_capture"]
    ours["LCCSD"] -= ours["LCCSD build"]
    if triples:
        ours["DLPNO-CCSD(T)"] = ours.pop("DLPNO-CCSD")

# ---- report ------------------------------------------------------------------
print(f"\n  problem size")
print(f"    {'':22} {'psi4':>12} {'this port':>12}")
# For ccsd psi4 prints both of these partway through its cascade - the pair
# count before the crude pass eliminates any, the PNO count at MP2-level
# cutoffs - while the port's are the post-rebuild values the CC iteration
# actually runs on. Comparing them would read as a disagreement where there is
# none, so psi4's column is a dash and the note says which point each is from.
_staged = args.method != "mp2"
print(f"    {'LMO pairs':22} {'-' if _staged else psi4_stats.get('pairs', '?'):>12} "
      f"{mp2.n_lmo_pairs:>12}"
      + (f"   (of {mp2.ref.naocc**2} possible, after crude elimination)"
         if _staged else
         f"   (of {mp2.ref.naocc**2} possible; both sides prescreen)"))
print(f"    {'avg PNOs per pair':22} "
      f"{'-' if _staged else psi4_stats.get('avg_pno', '?'):>12} "
      f"{sum(mp2.n_pno) / mp2.n_lmo_pairs:>12.1f}"
      + ("   (rebuilt at CC cutoffs)" if _staged else
         "   (both sides use screened PAO domains)"))
print(f"    {'padded PNO dimension':22} {'-':>12} {mp2.npno_max:>12}"
      "   (port pads every block to this)")
if args.method == "mp2":
    print(f"    {'LMP2 iterations':22} {psi4_stats.get('iterations', '?'):>12} "
          f"{mp2.n_iterations:>12}")
else:
    print(f"    {'strong / weak pairs':22} {'-':>12} "
          f"{f'{len(mp2.ij_to_i_j_strong)} / {len(mp2.ij_to_i_j_weak)}':>12}")
    # psi4's LMP2-inside-CC iterations are not labelled the way its standalone
    # ones are, so its column is a dash rather than a wrong zero.
    print(f"    {'LMP2 iterations':22} {psi4_stats.get('iterations') or '-':>12} "
          f"{stats_cc['lmp2_iterations']:>12}")
    print(f"    {'LCCSD iterations':22} {psi4_stats.get('cc_iterations', '?'):>12} "
          f"{stats_cc['iterations']:>12}")
    print(f"    {'captured CC nodes':22} {'-':>12} {stats_cc['nodes']:>12}"
          "   (replayed once per iteration)")
    if triples:
        print(f"    {'LMO triplets':22} {psi4_stats.get('triplets', '?'):>12} "
              f"{mp2.n_lmo_triplets:>12}"
              f"   ({sum(mp2.is_strong_triplet)} strong, at the (T) cutoffs)")
        print(f"    {'avg TNOs per triplet':22} "
              f"{psi4_stats.get('avg_tno', '?'):>12} "
              f"{sum(mp2.n_tno) / max(mp2.n_lmo_triplets, 1):>12.1f}")
        print(f"    {'(T) iterations':22} "
              f"{psi4_stats.get('t_iterations', '?'):>12} "
              f"{mp2.lccsd_t.n_iterations:>12}")

# (row label, key in `ours`, psi4 timer name or None when psi4 has no analogue).
#
# Every phase timed above must reach this table. It did not always: `Sparsity`
# was timed and never printed, so 16% of an ethanol run was invisible in the
# very table used to decide what to optimize next. The residual row and the
# check below exist so that cannot recur silently - a newly timed phase either
# appears here by name or falls out into `other (untabulated)`.
if args.method == "mp2":
    ROWS = [
        ("Setup Orbitals", "Setup Orbitals", "Setup Orbitals"),
        ("Sparsity", "Sparsity", "Sparsity"),
        ("DF Ints", "DF Ints", "DF Ints"),
        ("PNO Transform", "PNO Transform", "PNO Transform"),
        ("PNO Overlaps", "PNO Overlaps", "PNO Overlaps"),
        ("LMP2 iterations", "LMP2", "LMP2"),
    ]
else:
    # psi4's "Refined Pair Prescreening" CONTAINS its "PNO-LMP2 Iterations", so
    # its rows sum to more than its total. The port's are exclusive, so psi4's
    # PNO-transform time is the difference between the two - which is what makes
    # the columns comparable row by row and both of them add up.
    refined = psi4_phases.get("Refined Pair Prescreening")
    if refined is not None:
        psi4_phases["PNO Transform"] = (
            refined - psi4_phases.get("PNO-LMP2 Iterations", 0.0))
    ROWS = [
        ("Setup Orbitals", "Setup Orbitals", "Setup Orbitals"),
        ("Sparsity", "Sparsity", "Sparsity"),
        ("DF Ints", "DF Ints", "DF Ints"),
        ("Initial Prescreening", "Initial Pair Prescreening",
         "Initial Pair Prescreening"),
        ("PNO Transform", "PNO Transform", "PNO Transform"),
        ("PNO-LMP2 iterations", "PNO-LMP2 Iterations", "PNO-LMP2 Iterations"),
        ("Compute PNOs (CCSD)", "Compute PNOs (CCSD)", "Compute PNOs (CCSD)"),
        ("PNO Integrals", "PNO Integrals", "PNO Integrals"),
        ("PNO Overlaps", "PNO Overlaps", "PNO Overlaps"),
        ("LCCSD iterations", "LCCSD", "LCCSD"),
    ]
    if args.method == "ccsd(t)":
        ROWS += [
            ("Triples Sparsity", "Triples Sparsity", "Triples Sparsity"),
            ("TNO transform", "TNO transform", "TNO transform"),
            ("LCCSD(T0)", "LCCSD(T0)", "LCCSD(T0)"),
            ("Sort Triplets", "Sort Triplets", "Sort Triplets"),
            ("LCCSD(T) iterations", "LCCSD(T) Iterations",
             "LCCSD(T) Iterations"),
        ]
# Any phase timed but not named above still gets a row, in the order it was
# timed, so adding a phase() call cannot drop work out of the table.
TOTAL = {"mp2": "DLPNO-MP2", "ccsd": "DLPNO-CCSD"}.get(
    args.method, "DLPNO-CCSD(T)")
BUILD_KEYS = ("LMP2 build", "LCCSD build")
ROWS += [(key, key, key) for key in ours
         if key != TOTAL and key not in BUILD_KEYS
         and key not in {r[1] for r in ROWS}]

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

# The one-time graph builds are excluded from the compared total: psi4 has no
# analogue, so a row mixing a one-time cost into a steady-state comparison
# misprices both. They are printed under the total instead, and the
# per-iteration figures below are what they amortize into.
port_total = ours[TOTAL]
psi4_total = psi4_phases.get(TOTAL, psi4_times["dlpno"])
port_build = sum(ours.get(key, 0.0) for key in BUILD_KEYS)
port_steady = port_total - port_build
port_residual = port_steady - port_accounted
psi4_residual = psi4_total - psi4_accounted
print_row("other (untabulated)", psi4_residual, port_residual)
print_row("total " + TOTAL, psi4_total, port_steady)
for _key in BUILD_KEYS:
    if _key in ours:
        print_row(_key.replace("build", "graph build"), None, ours[_key])
if port_build > 0.0:
    print("    (one-time graph builds are excluded from the port's total; "
          "they amortize\n     with iteration count)")

# The whole point of the rows above is to be a complete account of the total.
# Warn loudly (and fail the run at the end) if they are not, rather than let
# the table quietly under-report a phase again.
table_incomplete = abs(port_residual) > 0.05 * port_steady
if table_incomplete:
    print(f"\n    WARNING: the port rows above account for only "
          f"{port_accounted:.3f} s of a {port_steady:.3f} s build-excluded "
          f"total ({port_residual:.3f} s untabulated, over the 5% bound).\n"
          "    Some timed work is missing a row, or work between phases is "
          "untimed. Fix the\n    table before using it to choose what to "
          "optimize.")

if args.method != "mp2":
    p_it = (psi4_phases.get("LCCSD", 0.0)
            / max(psi4_stats.get("cc_iterations", 0) or 1, 1))
    o_it = stats_cc["t_iterate"] / max(stats_cc["iterations"], 1)
    if p_it > 1e-9:
        print(f"\n    {'LCCSD per iteration':22} {p_it:>12.4f} {o_it:>12.4f} "
              f"{o_it / p_it:>9.1f}x")
    nodes = stats_cc["nodes"]
    print(f"\n  where the LCCSD time goes")
    print(f"    {nodes} captured nodes, {o_it * 1e6 / max(nodes, 1):.2f} us per "
          f"node per iteration")
    print(f"    graph build {ours['LCCSD build']:.3f} s, paid once and "
          f"excluded from the compared total above")
    print(f"    the correctness cut emits one operation per plan record. The "
          f"campaign's first\n    lever is grouping records by shape class into "
          f"batched calls, which is what took\n    LMP2 from 754 dispatches an "
          f"iteration to 13.")
    if triples:
        # The (T) row deserves its own per-iteration figure, because the two
        # sides do not take the same NUMBER of iterations and the row total
        # therefore compares two different amounts of work. psi4 updates its
        # amplitudes in place, so a term sees this pass's neighbours; the port
        # is Jacobi and sees last pass's. Same fixed point, different rate.
        p_t = (psi4_phases.get("LCCSD(T) Iterations", 0.0)
               / max(psi4_stats.get("t_iterations", 0) or 1, 1))
        o_t = mp2.lccsd_t.t_iterate / max(mp2.lccsd_t.n_iterations, 1)
        if p_t > 1e-9:
            print(f"\n    {'(T) per iteration':22} {p_t:>12.4f} {o_t:>12.4f} "
                  f"{o_t / p_t:>9.1f}x")
            print(f"    {'(T) iteration count':22} "
                  f"{psi4_stats.get('t_iterations', '?'):>12} "
                  f"{mp2.lccsd_t.n_iterations:>12}"
                  "   (in-place vs Jacobi)")

    e_ours = mp2.e_corr
    want = psi4_times["corr"]
    print(f"\n  correlation energy   psi4 {want:.10f}   port {e_ours:.10f}   "
          f"diff {abs(e_ours - want):.2e}")
    print(f"  (SCF, excluded above: psi4 {psi4_times['scf']:.3f} s, "
          f"port reference {t_scf_ours:.3f} s)")
    if abs(e_ours - want) > max(1e-8, 1e-6 * abs(want)):
        print("\n  WARNING: the correlation energies disagree, so the two sides "
              "are NOT solving\n  the same problem and these timings are not "
              "comparable.")
    else:
        print("\n  The energies agree, so the two sides are solving the same "
              "problem.")
    sys.exit(1 if table_incomplete else 0)

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
