#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""What a ``DFHelper``-backed integral source costs, against the exact one.

Needs psi4, which is why it is a script rather than one of the tests: the dense
source is the oracle precisely because it needs nothing, and a test that
imported psi4 to check it would give that property up.

Two questions, both answered in numbers rather than prose:

* how far does the ``J^1/2 . J^-1/2`` round trip move the integrals, and does
  that survive into the correlation energy,
* what does the threading and screening buy, against what it costs serially.

    python examples/dlpno/check_integral_sources.py --molecule ethanol --basis cc-pvtz --threads 10
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psi4

from dlpno import integrals
from dlpno.molecules import MOLECULES, water_chain
from dlpno.mp2 import DLPNOMP2
from dlpno.psi4_source import DFHelperSource, from_psi4, raw_three_index
from dlpno.thresholds import Thresholds

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--molecule", default="water")
parser.add_argument("--basis", default="cc-pvdz")
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--schwarz", type=float, default=0.0,
                    help="Schwarz cutoff for the DFHelper source; 0 disables screening")
parser.add_argument("--t-cut-pno", type=float, default=1e-8)
args = parser.parse_args()

geom = (water_chain(int(args.molecule[5:]), 2.9) + "symmetry c1\nno_reorient\nno_com\n"
        if args.molecule.startswith("chain") else MOLECULES[args.molecule] + "symmetry c1\n")

psi4.core.be_quiet()
psi4.set_num_threads(args.threads)
psi4.set_options({"basis": args.basis, "scf_type": "df", "freeze_core": "false",
                  "e_convergence": 1e-10, "d_convergence": 1e-10})
mol = psi4.geometry(geom)
_, wfn = psi4.energy("scf", return_wfn=True)
reference = from_psi4(wfn)

primary = wfn.basisset()
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "", "RIFIT", args.basis)


def run(source, label):
    """One full calculation on one source, timing only the integral phase."""
    mp2 = DLPNOMP2(reference, Thresholds.preset("NORMAL", t_cut_pno=args.t_cut_pno),
                   verbose=False, integral_source=source)
    mp2.setup_orbitals(); mp2.compute_doi(); mp2.prep_sparsity(); mp2.compute_metric()
    t0 = time.perf_counter()
    mp2.compute_qia()
    elapsed = time.perf_counter() - t0
    mp2.precompute_fits(); mp2.pno_transform(); mp2.compute_pno_overlaps()
    mp2.lmp2_iterations(optimize=True)
    e_corr = mp2.e_lmp2 + mp2.de_pno_total + mp2.de_dipole
    print(f"  {label:<34} compute_qia {elapsed:7.3f} s   e_corr {e_corr:.12f}   "
          f"threshold {source.screening_threshold:.0e}")
    return np.asarray(mp2.q_ia, copy=True), e_corr, elapsed


print(f"\n{args.molecule}/{args.basis}, {args.threads} thread(s), "
      f"nbf={primary.nbf()} naux={aux.nbf()}\n")

# The dense source's real cost is split across two phases and only half of it
# lands in compute_qia: from_psi4 already built (Q|mn), and compute_qia only
# transforms what it was handed. DFHelper does both inside compute_qia, so
# timing the two phases against each other flatters it by the whole integral
# build. Price that build separately and add it back.
t0 = time.perf_counter()
raw_three_index(primary, aux)
t_build = time.perf_counter() - t0
print(f"  (dense (Q|mn) build, charged to DF Ints rather than compute_qia: {t_build:.3f} s)\n")

q_dense, e_dense, t_dense = run(integrals.DenseSource(reference.eri_3index), "DenseSource (exact)")
q_dfh, e_dfh, t_dfh = run(
    DFHelperSource(primary, aux, schwarz_cutoff=args.schwarz, nthreads=args.threads),
    "DFHelperSource (raw AO)")

scale = np.abs(q_dense).max()
print(f"\n  integrals:  max abs deviation {np.abs(q_dfh - q_dense).max():.3e}"
      f"   relative {np.abs(q_dfh - q_dense).max() / scale:.3e}")
print(f"  energy:     {abs(e_dfh - e_dense):.3e} Hartree")
fair_dense = t_dense + t_build
print(f"  integrals available:  dense {fair_dense:.3f} s (build {t_build:.3f} + transform {t_dense:.3f})"
      f"   vs DFHelper {t_dfh:.3f} s"
      f"   -> {fair_dense / t_dfh:.2f}x {'faster' if t_dfh < fair_dense else 'SLOWER'}")
print("\n  The energy deviation is the number that matters: untruncated DLPNO-MP2 has to")
print("  reproduce canonical DF-MP2 to 1e-13, and anything this source loses to the")
print("  metric round trip is spent out of that budget before the calculation starts.")
