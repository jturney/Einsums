#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Run every fixture through both backends of a stage and compare the energies.

The acceptance test for a promoted stage, and the regression test for anything
that regenerates one. Requires the compiled stage module; see cpp/CMakeLists.txt
for how to build it out of tree, and the "Two backends" section of README.md.

    PYTHONPATH=/path/to/Einsums/build/lib:/path/to/build-dlpno-stages \
        python examples/dlpno/check_backends.py

Two things about how this is written are the point rather than incidental.

**It compares at full precision and expects zero.** Both backends emit the same
GEMMs in the same order into the same BLAS, and only the operand-list
construction differs, which is index arithmetic with no floating-point content.
Agreement merely to tolerance would mean something reordered, so a loose
comparison here would hide the failure it exists to catch.

**A zero disagreement is also what a backend that never ran produces**, which is
why ``--prove`` exists. It reports the name of a returned tensor from each
backend: the Python implementation names per-class tensors by the class tuple
and the C++ one by the class ordinal, so the output itself says which code made
it. Use it whenever a passing result would otherwise be indistinguishable from a
backend that silently was not selected.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from einsums import stages

import dlpno.stages  # noqa: F401  (declares the stages so the loader can match)
from dlpno.ccsd import DLPNOCCSD
from dlpno.mp2 import DLPNOMP2
from dlpno.reference_io import load_reference
from dlpno.thresholds import Thresholds

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--stage", default="compute_pno_overlaps", help="stage to compare")
parser.add_argument(
    "--method", default="mp2", choices=["mp2", "cc"],
    help="which calculation to compare through. 'cc' runs the DLPNO-CCSD "
         "prescreening cascade, which is the only path where the PNO "
         "transform's trace and energy criteria are live at all: the MP2 "
         "branch passes zero for both, so an mp2 comparison cannot exercise "
         "them and a backend could disagree there undetected")
parser.add_argument("--module", default="dlpno_stages", help="compiled stage module")
parser.add_argument("--buckets", type=int, default=None,
                    help="PNO-count buckets to pad pair blocks into; default chooses per thread count")
parser.add_argument("-k", "--filter", default="")
parser.add_argument("--tol", type=float, default=1e-13,
                    help="disagreement treated as a failure (default: exact)")
parser.add_argument("--prove", action="store_true",
                    help="report which implementation produced the result, since a zero "
                         "disagreement is also what a backend that never ran produces")
args = parser.parse_args()

stages.load_stage_module(args.module)


def run(path, backend, label):
    stages.select(**{args.stage: backend})
    ref, extras = load_reference(path)
    t_cut_pno = float(extras["metadata"].get("t_cut_pno", 1e-8))
    if args.method == "cc":
        # No session: the CC cascade is not registered as stages yet, and the
        # phases dispatch through the registry anyway, so the selected backend
        # runs either way.
        cc = DLPNOCCSD(ref, Thresholds.preset("NORMAL", method="cc",
                                              n_buckets=args.buckets), verbose=False)
        cc.compute_energy()
        return cc
    thresholds = (Thresholds.untruncated(n_buckets=args.buckets) if label == "untrunc"
                  else Thresholds.preset("NORMAL", t_cut_pno=t_cut_pno, n_buckets=args.buckets))
    mp2 = DLPNOMP2(ref, thresholds, verbose=False)
    session = stages.session(f"{label}-{backend}")
    with session.capture():
        mp2.compute_energy(session=session)
    session.run()
    return mp2


paths = [p for p in sorted(glob.glob(os.path.join(DEFAULT_DIR, "*.npz")))
         if args.filter in os.path.basename(p)]
if not paths:
    parser.error(f"no fixtures matched (looked in {DEFAULT_DIR})")

print(f"comparing backends of {args.stage!r} through {args.method}\n")
# The CC cascade has no untruncated mode worth running here: with every
# threshold off there is nothing to prescreen and the PNO criteria this is
# checking are switched off again.
LABELS = ("trunc",) if args.method == "cc" else ("untrunc", "trunc")
FIELDS = {"mp2": ("e_lmp2", "e_corr", "de_pno_total"),
          "cc": ("e_lmp2", "e_corr", "de_pno_total", "de_weak", "de_lmp2_eliminated")}[args.method]
worst, failures = 0.0, []
for path in paths:
    name = os.path.basename(path)
    for label in LABELS:
        py, cpp = run(path, "python", label), run(path, "cpp", label)
        delta = max(abs(getattr(py, f) - getattr(cpp, f)) for f in FIELDS)
        worst = max(worst, delta)
        ok = delta <= args.tol
        if args.method == "cc":
            # A PNO count that moves is a truncation decision that disagreed,
            # which no energy tolerance would catch reliably: the criteria are
            # data dependent, so one extra PNO can be worth less than the
            # tolerance and still means the two backends decided differently.
            counts = "same" if py.n_pno == cpp.n_pno else "DIFFER"
            ok = ok and py.n_pno == cpp.n_pno
            extra = f"  PNO counts {counts}"
        else:
            ok = ok and py.n_iterations == cpp.n_iterations
            extra = f"  iters {py.n_iterations}/{cpp.n_iterations}"
        if not ok:
            failures.append(f"{name} {label}: max|dE| = {delta:.3e}{extra}")
        print(f"{'OK ' if ok else 'XX '}{name:<32} {label:<8} max|dE| = {delta:.3e}{extra}")
        if args.prove and args.method == "mp2":
            print(f"     python produced {py._S_cls[0].name!r}, cpp produced {cpp._S_cls[0].name!r}")

print(f"\nworst disagreement: {worst:.3e}")
if failures:
    for message in failures:
        print(f"FAIL {message}")
    sys.exit(1)
print("both backends agree on every fixture")
