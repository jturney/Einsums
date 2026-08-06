#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Run PNO-MP2 from a saved reference, with no psi4 in the process.

``run_pno_mp2.py`` is the real validation: it builds the reference, runs psi4's
own DF-MP2 and DLPNO-MP2, and checks the port against both. This runs the same
port against a reference frozen to disk by ``dump_reference.py``, checking it
against the psi4 energies recorded at dump time.

That makes it a smoke test rather than a validation - the reference is fixed, so
it cannot catch anything upstream of ``Reference`` - but it needs nothing except
numpy and einsums, so it runs anywhere the library builds.

    python examples/dlpno/run_pno_mp2_offline.py examples/dlpno/fixtures/water-ccpvdz.npz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno.mp2 import DLPNOMP2
from dlpno.reference_io import load_reference
from dlpno.thresholds import Thresholds

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("fixture", help=".npz written by dump_reference.py")
parser.add_argument("--buckets", type=int, default=4)
parser.add_argument("--no-diis", action="store_true")
parser.add_argument("--no-optimize", action="store_true")
parser.add_argument("--quiet", action="store_true")
args = parser.parse_args()

reference, extras = load_reference(args.fixture)
energies, meta = extras["energies"], extras["metadata"]

print(f"fixture: {args.fixture}")
if meta:
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(meta.items())))
print(f"  nbf = {reference.nbf}, naux = {reference.naux}, active occ = {reference.naocc}")

df_mp2 = energies.get("psi4_df_mp2")
dlpno_mp2 = energies.get("psi4_dlpno_mp2")
t_cut_pno = float(meta.get("t_cut_pno", 1e-8))

failures = []

print("\n=== PNO-MP2, no truncation (must equal canonical DF-MP2) ===")
exact = DLPNOMP2(reference, Thresholds.untruncated(n_buckets=args.buckets),
                 verbose=not args.quiet, use_diis=not args.no_diis)
exact.compute_energy(optimize=not args.no_optimize)
if df_mp2 is not None:
    err_exact = abs(exact.e_lmp2 - df_mp2)
    print(f"  difference vs recorded psi4 DF-MP2 = {err_exact:.3e}  "
          f"({exact.n_iterations} iterations)")
    if err_exact >= 1e-9:
        failures.append(f"untruncated PNO-MP2 disagrees with DF-MP2 by {err_exact:.3e}")

print(f"\n=== PNO-MP2, T_CUT_PNO = {t_cut_pno:.1e} ===")
cut = Thresholds.preset("NORMAL", t_cut_pno=t_cut_pno, n_buckets=args.buckets)
trunc = DLPNOMP2(reference, cut, verbose=not args.quiet, use_diis=not args.no_diis)
trunc.compute_energy(optimize=not args.no_optimize)
if dlpno_mp2 is not None:
    err_dlpno = abs(trunc.e_corr - dlpno_mp2)
    # Same tolerance rationale as run_pno_mp2.py: scaled to the PNO truncation
    # correction, because below that scale the difference is the PAO
    # linear-dependence tie-break rather than a domain disagreement.
    tol = max(1e-9, 0.01 * abs(trunc.de_pno_total))
    print(f"  difference vs recorded psi4 DLPNO-MP2 = {err_dlpno:.3e}  "
          f"({trunc.n_iterations} iterations)")
    if err_dlpno >= tol:
        failures.append(f"truncated PNO-MP2 disagrees with psi4 DLPNO-MP2 by "
                        f"{err_dlpno:.3e} (tolerance {tol:.1e})")

if failures:
    for message in failures:
        print(f"\nFAIL: {message}")
    sys.exit(1)

print("\nPNO-MP2 (offline fixture) MATCHES the recorded psi4 energies")
