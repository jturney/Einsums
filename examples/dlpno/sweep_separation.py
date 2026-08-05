#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Pair prescreening through the separation where it actually decides something.

The two dimer geometries in ``molecules.py`` are the ends of a range, not a
test of it. At 2.9 A psi4 keeps all 100 pairs and so does the port; at 12 A
every inter-monomer pair fails both criteria by orders of magnitude and the
discarded energy is 2e-7 Eh. Either would still pass with a comparison
inverted at the boundary, a swapped threshold, or a dipole bound wrong by a
factor of a few, because no pair is anywhere near a threshold in either.

Walking the O-O distance puts pairs across the thresholds a few at a time.
Three things become visible that the endpoints hide:

* whether the port drops the *same* pairs as psi4 while the count is changing,
  rather than only when the answer is all or nothing;
* whether the two survival criteria (differential overlap ``t_cut_do_ij``, and
  the dipole energy bound ``t_cut_pre``, combined with ``or``) hand over to
  each other cleanly - overlap dies off faster with distance, so there is a
  window where the dipole bound is the only thing keeping a pair;
* whether ``de_dipole`` is right when it is large enough to matter. At 12 A it
  is 2e-7 Eh, so its correctness is currently untested at any magnitude a
  chemist would notice.

Run with psi4 on the path, as bench_vs_psi4.py describes:

    PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/stage/lib \
        python examples/dlpno/sweep_separation.py
"""

import argparse
import os
import sys

import re
import time

import psi4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno import DLPNOMP2, Thresholds
from dlpno.molecules import water_dimer_at
from dlpno.psi4_source import from_psi4

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--distances", type=float, nargs="+",
    default=[2.9, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 12.0],
    help="O-O separations in angstrom",
)
parser.add_argument("--basis", default="cc-pvdz")
parser.add_argument("--t-cut-pno", type=float, default=1e-8)
parser.add_argument("--buckets", type=int, default=4)
parser.add_argument("--threads", type=int, default=1)
args = parser.parse_args()


def psi4_screened_pairs(path):
    """``(screened, total)`` from psi4's own prescreening report.

    psi4 prints this rather than setting a variable, so the output file is the
    only place the count exists. The last match is this geometry's DLPNO run;
    the canonical MP2 above it does not screen.
    """
    last = None
    with open(path) as fh:
        for line in fh:
            m = re.search(r"Screened (\d+) of (\d+) LMO pairs", line)
            if m:
                last = (int(m.group(1)), int(m.group(2)))
    return last if last is not None else (-1, -1)

psi4.set_options({
    "basis": args.basis,
    "scf_type": "df",
    "mp2_type": "df",
    "freeze_core": "false",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    "r_convergence": 1e-8,
})
psi4.set_num_threads(args.threads)

print(f"water dimer, {args.basis}, T_CUT_PNO = {args.t_cut_pno:.0e}\n")
header = (f"{'R(O-O)':>7}  {'kept':>9}  {'psi4 kept':>9}  {'de_dipole':>14}  "
          f"{'E_corr(port)':>15}  {'vs psi4':>9}  {'t_port':>7}  {'t_psi4':>7}")
print(header)
print("-" * len(header))

rows = []
for r in args.distances:
    psi4.core.clean()
    # One file per point, so the screening report parsed below is this
    # geometry's and not a previous one's.
    out = f"/tmp/psi4_dlpno_sweep_{r:.2f}.out"
    psi4.core.set_output_file(out, False)
    mol = psi4.geometry(water_dimer_at(r) + "symmetry c1\nno_reorient\nno_com\n")

    # Canonical DF-MP2 first: the truncation error is only meaningful against
    # the untruncated answer for this geometry.
    psi4.set_options({"dlpno_algorithm": "mp2"})
    _, wfn = psi4.energy("mp2", return_wfn=True)
    df_mp2 = psi4.variable("MP2 CORRELATION ENERGY")

    psi4.set_options({"t_cut_pno": args.t_cut_pno})
    t0 = time.perf_counter()
    psi4.energy("dlpno-mp2")
    t_psi4 = time.perf_counter() - t0
    psi4_dlpno = psi4.variable("MP2 CORRELATION ENERGY")
    psi4_dropped, psi4_total = psi4_screened_pairs(out)

    reference = from_psi4(wfn, localization="BOYS")
    cut = Thresholds.preset("NORMAL", t_cut_pno=args.t_cut_pno, n_buckets=args.buckets)
    port = DLPNOMP2(reference, cut, use_diis=True, verbose=False)
    t0 = time.perf_counter()
    port.compute_energy(optimize=True)
    t_port = time.perf_counter() - t0

    naocc = reference.naocc
    kept = len(port.ij_to_i_j)
    total = naocc * naocc
    print(f"{r:7.2f}  {kept:4d}/{total:<4d}  {psi4_total - psi4_dropped:4d}/{psi4_total:<4d}  "
          f"{port.de_dipole:14.10f}  {port.e_corr:15.10f}  "
          f"{abs(port.e_corr - psi4_dlpno):9.2e}  {t_port:6.2f}s  {t_psi4:6.2f}s")
    rows.append((r, kept, total, port.de_dipole, port.e_corr, psi4_dlpno, t_port))

print()
# The columns that matter are the middle ones: a sweep where "kept" never
# changes has not tested screening, and one where the port and psi4 disagree
# while it changes has found something the endpoints could not.
changed = {k for (_, k, _, _, _, _, _) in rows}
print(f"distinct pair counts across the sweep: {len(changed)} ({sorted(changed)})")
worst = max(abs(e - p) for (_, _, _, _, e, p, _) in rows)
print(f"worst disagreement with psi4 DLPNO-MP2: {worst:.2e}")
biggest_dipole = max(abs(d) for (_, _, _, d, _, _, _) in rows)
print(f"largest de_dipole correction exercised: {biggest_dipole:.3e} Eh")

# Screening is supposed to buy time, not just accuracy. Pairs and wall clock
# should fall together; if they do not, the saved pairs were not where the
# work was.
first, last = rows[0], rows[-1]
print(f"pairs {first[1]} -> {last[1]} ({100 * last[1] / first[1]:.0f}%), "
      f"port {first[6]:.2f}s -> {last[6]:.2f}s ({100 * last[6] / first[6]:.0f}%)")
