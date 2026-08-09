#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Does the locality actually pay off? A water chain of growing length.

Every other case in this directory is a fixed molecule, so none of them can
show the property DLPNO exists for: pair count grows as the square of the
system size while the pairs close enough to survive prescreening grow only
linearly, so the kept *fraction* has to fall as the chain lengthens. A method
that reproduces psi4's energies but keeps every pair is not doing DLPNO, it is
doing MP2 in a local basis, and the fixed geometries cannot tell the two apart.

Reported per chain length:

* ``kept/total`` and the kept fraction - the locality claim itself. Linear
  scaling means kept/total falls roughly as 1/n.
* the same counts from psi4, so a divergence in *which* pairs are dropped
  shows up as the chain grows rather than only at one separation.
* ``de_dipole``, the estimated energy of everything discarded, which grows
  with the number of dropped pairs and is the error being traded for the
  speed.
* wall clock for both, which is what the trade is for.

The monomers share one orientation and a uniform spacing, so pair distance is
a clean function of index difference. That is not a hydrogen-bonded chain and
the energies are not physically interesting; the locality structure is.

    PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/stage/lib \\
        python examples/dlpno/sweep_chain.py --lengths 2 3 4 5
"""

import argparse
import os
import re
import sys
import time

import psi4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno import DLPNOMP2, Thresholds
from dlpno.molecules import water_chain
from dlpno.psi4_source import from_psi4

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--lengths", type=int, nargs="+", default=[2, 3, 4, 5],
                    help="number of water monomers in the chain")
parser.add_argument("--spacing", type=float, default=2.9,
                    help="O-O distance between neighbouring monomers, angstrom")
parser.add_argument("--basis", default="cc-pvdz")
parser.add_argument("--t-cut-pno", type=float, default=1e-8)
parser.add_argument("--buckets", type=int, default=None,
                    help="PNO-count buckets to pad pair blocks into; default chooses per thread count")
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--skip-psi4-dlpno", action="store_true",
                    help="skip psi4's own DLPNO run; the port still needs psi4 for integrals")
args = parser.parse_args()

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


def psi4_screened_pairs(path):
    """``(screened, total)`` from psi4's own prescreening report."""
    last = None
    with open(path) as fh:
        for line in fh:
            m = re.search(r"Screened (\d+) of (\d+) LMO pairs", line)
            if m:
                last = (int(m.group(1)), int(m.group(2)))
    return last if last is not None else (-1, -1)


print(f"water chain, {args.spacing} A spacing, {args.basis}, "
      f"T_CUT_PNO = {args.t_cut_pno:.0e}\n")
header = (f"{'n':>3}  {'atoms':>5}  {'occ':>4}  {'kept/total':>12}  {'frac':>6}  "
          f"{'psi4 kept':>10}  {'de_dipole':>13}  {'t_port':>7}  {'t_psi4':>7}")
print(header)
print("-" * len(header))

rows = []
for n in args.lengths:
    psi4.core.clean()
    out = f"/tmp/psi4_dlpno_chain_{n}.out"
    psi4.core.set_output_file(out, False)
    psi4.geometry(water_chain(n, args.spacing) + "symmetry c1\nno_reorient\nno_com\n")

    _, wfn = psi4.energy("mp2", return_wfn=True)

    psi4_kept = "-"
    t_psi4 = float("nan")
    if not args.skip_psi4_dlpno:
        psi4.set_options({"dlpno_algorithm": "mp2", "t_cut_pno": args.t_cut_pno})
        t0 = time.perf_counter()
        # ref_wfn, so this converges no SCF of its own. Without it the timed
        # region is psi4's SCF *plus* its correlation while the port's is
        # correlation alone, and at ten threads the SCF is up to 84% of it -
        # which silently turned a 2.4x deficit into a reported 1.1x lead.
        psi4.energy("dlpno-mp2", ref_wfn=wfn)
        t_psi4 = time.perf_counter() - t0
        dropped, tot = psi4_screened_pairs(out)
        psi4_kept = f"{tot - dropped}/{tot}"

    # from_psi4 is inside the port's clock. It builds the dense (Q|mn), which
    # is work psi4 does too - screened, inside the run just timed - so leaving
    # it out is the same bias in the other direction. It does not thread, and
    # at n=6 it is 0.36 s.
    cut = Thresholds.preset("NORMAL", t_cut_pno=args.t_cut_pno, n_buckets=args.buckets)
    t0 = time.perf_counter()
    reference = from_psi4(wfn, localization="BOYS")
    port = DLPNOMP2(reference, cut, use_diis=True, verbose=False)
    port.compute_energy(optimize=True)
    t_port = time.perf_counter() - t0

    naocc = reference.naocc
    kept, total = len(port.ij_to_i_j), naocc * naocc
    print(f"{n:3d}  {3 * n:5d}  {naocc:4d}  {kept:5d}/{total:<6d}  "
          f"{kept / total:6.2f}  {psi4_kept:>10}  {port.de_dipole:13.10f}  "
          f"{t_port:6.2f}s  {t_psi4:6.2f}s")
    rows.append((n, naocc, kept, total, t_port))

print()
# Linear scaling shows up as the kept fraction falling; if it holds at 1.0 the
# chain is still short enough that every pair is close to every other, and the
# test has not reached the regime it was written for.
for (n, _, kept, total, _) in rows:
    print(f"  n={n}: kept {kept}/{total} = {kept / total:.3f}")
if len(rows) > 1:
    first, last = rows[0], rows[-1]
    grow_total = last[3] / first[3]
    grow_kept = last[2] / first[2]
    grow_time = last[4] / first[4]
    print(f"\nfrom n={first[0]} to n={last[0]}: total pairs x{grow_total:.1f}, "
          f"kept x{grow_kept:.1f}, wall clock x{grow_time:.1f}")
    print("(kept growing more slowly than total is the locality; equal growth means "
          "the chain is too short to screen)")
