#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Replay every saved reference in a directory, with no psi4 in the process.

The whole psi4-free suite in one command, which is what makes it usable as a
smoke test after a change to the port:

    PYTHONPATH=/path/to/Einsums/build/lib python examples/dlpno/run_fixtures.py

Each fixture is run at both its untruncated and its truncated thresholds and
checked against the psi4 energies recorded in it. Exits non-zero if any
disagrees, so it can be wired into a script.
"""

import argparse
from dataclasses import replace
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno.mp2 import DLPNOMP2
from dlpno.triples import DLPNOCCSDT
from dlpno.reference_io import load_reference
from dlpno.thresholds import Thresholds

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("fixtures", nargs="*",
                    help=f"fixture files; default is every .npz in {DEFAULT_DIR}")
parser.add_argument("--buckets", type=int, default=None,
                    help="PNO-count buckets to pad pair blocks into; default chooses per thread count")
parser.add_argument("--no-diis", action="store_true")
parser.add_argument("--no-optimize", action="store_true")
parser.add_argument("-k", "--filter", default="",
                    help="only run fixtures whose name contains this substring")
parser.add_argument("--backend", default="",
                    help="stage backend spec override, e.g. compute_pno_overlaps=python. "
                         "Implies --stages. The compiled backends are selected "
                         "automatically when dlpno_stages is importable")
parser.add_argument("--method", default="mp2", choices=["mp2", "ccsd", "ccsd(t)"],
                    help="which recorded energies to check. 'ccsd' and 'ccsd(t)' need "
                         "fixtures written by dump_reference.py --with-cc, and skip any "
                         "fixture that does not carry them")
parser.add_argument("--stages", action="store_true",
                    help="run the phases through einsums.stages and print the per-stage "
                         "timing table. The energies must not move: that they do not is "
                         "what makes the table trustworthy")
args = parser.parse_args()

if args.stages or args.backend:
    from einsums import stages
    if args.backend:
        # Importing dlpno.stages declares the stages AND auto-loads the
        # compiled backends when dlpno_stages is importable; the flag then
        # only overrides that default.
        import dlpno.stages  # noqa: F401
        stages.apply_backend_spec(args.backend)
else:
    stages = None


def run(mp2, label):
    """Run one calculation, optionally inside a stage session, and return it."""
    if stages is None:
        mp2.compute_energy(optimize=not args.no_optimize)
        return None
    session = stages.session(label)
    with session.capture():
        mp2.compute_energy(optimize=not args.no_optimize, session=session)
    session.run()
    return session

paths = args.fixtures or sorted(glob.glob(os.path.join(DEFAULT_DIR, "*.npz")))
paths = [p for p in paths if args.filter in os.path.basename(p)]
if not paths:
    parser.error(f"no fixtures matched (looked in {DEFAULT_DIR})")

# Untruncated has to reproduce canonical DF-MP2 exactly; truncated is scaled to
# the PNO truncation correction, for the reason run_dlpno_mp2.py explains.
UNTRUNCATED_TOL = 1e-9

_second = {'mp2': 'truncated'}.get(args.method, args.method)
print(f"{'fixture':<32} {'untruncated':>13} {_second:>13} {'time':>8}")
print("-" * 70)

failures = []
for path in paths:
    name = os.path.basename(path)
    started = time.perf_counter()
    reference, extras = load_reference(path)
    energies, meta = extras["energies"], extras["metadata"]
    t_cut_pno = float(meta.get("t_cut_pno", 1e-8))

    exact = DLPNOMP2(reference, Thresholds.untruncated(n_buckets=args.buckets),
                     verbose=False, use_diis=not args.no_diis)
    run(exact, f"{name}:untruncated")
    err_exact = abs(exact.e_lmp2 - energies["psi4_df_mp2"])
    if err_exact >= UNTRUNCATED_TOL:
        failures.append(f"{name}: untruncated off by {err_exact:.3e}")

    trunc = DLPNOMP2(reference, Thresholds.preset("NORMAL", t_cut_pno=t_cut_pno,
                                                  n_buckets=args.buckets),
                     verbose=False, use_diis=not args.no_diis)
    session = run(trunc, f"{name}:truncated")
    err_trunc = abs(trunc.e_corr - energies["psi4_dlpno_mp2"])
    tol_trunc = max(1e-9, 0.01 * abs(trunc.de_pno_total))
    if err_trunc >= tol_trunc:
        failures.append(f"{name}: truncated off by {err_trunc:.3e} (tolerance {tol_trunc:.1e})")

    if args.method != "mp2":
        want_key = ("psi4_dlpno_ccsd" if args.method == "ccsd"
                    else "psi4_dlpno_ccsd_t")
        if want_key not in energies:
            print(f" {name:<31} {'-':>13} {'(no CC reference)':>13}"
                  "   regenerate with dump_reference.py --with-cc")
            continue
        # The CC branch has its own preset table, so this is NOT the MP2
        # t_cut_pno the fixture was written at; dump_reference pins both sides
        # to 3.33e-7 for exactly this reason.
        cut = Thresholds.preset("NORMAL", method="cc", n_buckets=args.buckets)
        cc = DLPNOCCSDT(reference, replace(cut, t0_approximation=False),
                        verbose=False, use_diis=not args.no_diis)
        cc.compute_energy(method="ccsd(t)")
        got = (cc.e_lccsd + cc.de_weak + cc.de_lmp2_eliminated + cc.de_dipole
               + cc.de_pno_total) if args.method == "ccsd" else cc.e_corr
        err_cc = abs(got - energies[want_key])
        # Looser than the MP2 gates and for a reason that is the method's, not
        # the port's: a coupled-cluster energy is a fixed point two codes reach
        # along different DIIS trajectories, and the (T) iteration adds a
        # second one. See the M6 entry in DESIGN-ccsd.md.
        tol_cc = 1e-8 if args.method == "ccsd(t)" else 1e-9
        if err_cc >= tol_cc:
            failures.append(f"{name}: {args.method} off by {err_cc:.3e} "
                            f"(tolerance {tol_cc:.1e})")
        elapsed = time.perf_counter() - started
        mark = " " if err_cc < tol_cc else "X"
        print(f"{mark}{name:<31} {err_exact:>13.3e} {err_cc:>13.3e} {elapsed:>7.1f}s")
        continue

    elapsed = time.perf_counter() - started
    mark = " " if err_exact < UNTRUNCATED_TOL and err_trunc < tol_trunc else "X"
    print(f"{mark}{name:<31} {err_exact:>13.3e} {err_trunc:>13.3e} {elapsed:>7.1f}s")
    if session is not None:
        print()
        print(session.report())
        print()

print("-" * 70)
if failures:
    for message in failures:
        print(f"FAIL {message}")
    sys.exit(1)
print(f"all {len(paths)} fixtures match their recorded psi4 energies")
