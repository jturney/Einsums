#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Print every fixture's energies at full precision, for bit-identity checks.

``run_fixtures.py`` reports the error against psi4, which is the right check for
"is the method correct" and the wrong one for "did my refactor move anything".
A refactor that changes the tenth digit passes the first and should fail the
second. This prints ``repr`` so the comparison is textual and exact:

    python examples/dlpno/dump_energies.py > /tmp/before.txt
    ... refactor ...
    python examples/dlpno/dump_energies.py > /tmp/after.txt
    diff /tmp/before.txt /tmp/after.txt && echo BIT-IDENTICAL

Needs no compiled stage module, which is the point: it verifies pure-Python
restructuring, where there is no second implementation to compare against and
the only available oracle is the code as it was five minutes ago.

Worth running after *each* step of a multi-step refactor rather than only at the
end. Doing that during the M3 split caught two mistakes while each was still one
edit deep.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno.mp2 import DLPNOMP2
from dlpno.reference_io import load_reference
from dlpno.thresholds import Thresholds

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--buckets", type=int, default=None,
                    help="PNO-count buckets to pad pair blocks into; default chooses per thread count")
parser.add_argument("-k", "--filter", default="")
args = parser.parse_args()

paths = [p for p in sorted(glob.glob(os.path.join(DEFAULT_DIR, "*.npz")))
         if args.filter in os.path.basename(p)]

for path in paths:
    name = os.path.basename(path)
    reference, extras = load_reference(path)
    t_cut_pno = float(extras["metadata"].get("t_cut_pno", 1e-8))
    for label, thresholds in (
        ("untrunc", Thresholds.untruncated(n_buckets=args.buckets)),
        ("trunc", Thresholds.preset("NORMAL", t_cut_pno=t_cut_pno, n_buckets=args.buckets)),
    ):
        mp2 = DLPNOMP2(reference, thresholds, verbose=False)
        mp2.compute_energy()
        print(f"{name}:{label:<8} {mp2.e_lmp2!r} {mp2.e_corr!r} "
              f"{mp2.de_pno_total!r} {mp2.de_dipole!r} {mp2.n_iterations}")
