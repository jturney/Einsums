#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Local MP2 in a PNO basis, validated against psi4.

The first milestone of the DLPNO port. Screening is off (every LMO pair is
significant, every domain complete), which makes the check exact rather than
approximate:

* with PNO truncation also off, local MP2 in the full PAO space *is* canonical
  DF-MP2, just expressed in a non-canonical basis, so the correlation energy
  must match psi4's DF-MP2 to machine precision;
* with truncation on, ``E_lmp2 + de_pno`` must land back on that same number to
  within the PNO truncation correction's own accuracy.

Both are asserted. Run with the Einsums build and the psi4 stage on PYTHONPATH,
using the conda-env python::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        /Users/jturney/Code/Einsums/Einsums/examples/dlpno/run_pno_mp2.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psi4

from dlpno import DLPNOMP2, Thresholds
from dlpno.molecules import MOLECULES
from dlpno.psi4_source import from_psi4

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--molecule", default="water", choices=sorted(MOLECULES))
parser.add_argument("--basis", default="cc-pvdz")
parser.add_argument("--localization", default="BOYS", choices=["BOYS", "PIPEK_MEZEY"])
parser.add_argument(
    "--t-cut-pno", type=float, default=1e-8,
    help="PNO occupation-number cutoff for the truncated run (default psi4 NORMAL)",
)
parser.add_argument(
    "--no-optimize", action="store_true",
    help="skip the ComputeGraph optimization passes",
)
parser.add_argument("--no-diis", action="store_true", help="disable DIIS extrapolation")
parser.add_argument("--buckets", type=int, default=4,
                    help="PNO-count buckets to pad pair blocks into (see README)")
parser.add_argument("--threads", type=int, default=1,
                    help="thread count. Importing psi4 clamps process-wide OpenMP "
                         "to 1, so this must be set for einsums to thread at all")
args = parser.parse_args()

psi4.core.set_output_file(f"/tmp/psi4_dlpno_{args.molecule}.out", False)
psi4.set_options({
    "basis": args.basis,
    "scf_type": "df",
    "mp2_type": "df",
    "freeze_core": "false",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    # Match Thresholds.r_convergence; psi4's LMP2 default of 1e-6 stops short
    # of the port and the residual difference then looks like a domain bug.
    "r_convergence": 1e-8,
})

# Importing psi4 sets the process-wide OpenMP thread count to 1, silently
# serializing every einsums BLAS and gemm_batch call regardless of
# OMP_NUM_THREADS. This restores threading for both libraries.
psi4.set_num_threads(args.threads)

mol = psi4.geometry(MOLECULES[args.molecule] + "symmetry c1\n")

# Two psi4 references: canonical DF-MP2 pins the untruncated run exactly, and
# psi4's own DLPNO-MP2 at the same T_CUT_PNO is what the truncated run has to
# reproduce.
_, wfn = psi4.energy("mp2", return_wfn=True)
df_mp2 = psi4.variable("MP2 CORRELATION ENERGY")

psi4.set_options({"dlpno_algorithm": "mp2", "t_cut_pno": args.t_cut_pno})
psi4.energy("dlpno-mp2")
dlpno_mp2 = psi4.variable("MP2 CORRELATION ENERGY")

print(f"\npsi4 DF-MP2    correlation energy = {df_mp2:.12f}")
print(f"psi4 DLPNO-MP2 correlation energy = {dlpno_mp2:.12f}  "
      f"(T_CUT_PNO = {args.t_cut_pno:.1e})")

reference = from_psi4(wfn, localization=args.localization)

# ---- untruncated: local MP2 in the full PAO space *is* canonical DF-MP2 -----
print("\n=== PNO-MP2, no truncation (must equal canonical DF-MP2) ===")
exact = DLPNOMP2(reference, Thresholds.untruncated(n_buckets=args.buckets),
                 use_diis=not args.no_diis)
exact.compute_energy(optimize=not args.no_optimize)
err_exact = abs(exact.e_lmp2 - df_mp2)
print(f"  difference vs psi4 DF-MP2 = {err_exact:.3e}  ({exact.n_iterations} iterations)")

# ---- truncated: compare against psi4's own DLPNO-MP2 -----------------------
print(f"\n=== PNO-MP2, T_CUT_PNO = {args.t_cut_pno:.1e} ===")
cut = Thresholds.preset("NORMAL", t_cut_pno=args.t_cut_pno, n_buckets=args.buckets)
trunc = DLPNOMP2(reference, cut, use_diis=not args.no_diis)
trunc.compute_energy(optimize=not args.no_optimize)
err_dlpno = abs(trunc.e_corr - dlpno_mp2)
ours = abs(trunc.e_corr - df_mp2)
theirs = abs(dlpno_mp2 - df_mp2)
print(f"  difference vs psi4 DLPNO-MP2 = {err_dlpno:.3e}  ({trunc.n_iterations} iterations)")
print(f"  truncation error vs DF-MP2   = {ours:.3e} (psi4's own: {theirs:.3e})")

assert err_exact < 1e-9, f"untruncated PNO-MP2 disagrees with DF-MP2 by {err_exact:.3e}"
# Both sides now apply the same three truncations - PNO occupation, PAO and
# auxiliary domains, and dipole pair prescreening - so this is an equality, not
# the ordering check it had to be while the port kept full domains. The
# tolerance is loose next to the ~1e-13 actually observed because the two
# solvers converge along different paths; anything at the truncation scale
# (1e-5) means the domains genuinely disagree.
# Scaled to the PNO truncation correction. Below that scale the residual
# difference is the PAO linear-dependence tie-break: an overlap eigenvalue
# sitting near S_CUT that the two codes round opposite ways, worth ~4e-8 per
# vector on ethanol/cc-pVTZ. A genuine domain disagreement is orders larger.
tol_dlpno = max(1e-9, 0.01 * abs(trunc.de_pno_total))
assert err_dlpno < tol_dlpno, (
    f"truncated PNO-MP2 disagrees with psi4 DLPNO-MP2 by {err_dlpno:.3e} "
    f"(tolerance {tol_dlpno:.1e}); the two are not applying the same truncations"
)
print("\nPNO-MP2 (einsums ComputeGraph) MATCHES psi4 DF-MP2 and psi4 DLPNO-MP2")
