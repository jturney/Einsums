#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Dump a DLPNO reference wavefunction to a psi4-free ``.npz`` fixture.

Run this once on a machine with psi4; :mod:`dlpno.reference_io` reads the result
anywhere. The psi4 DF-MP2 and DLPNO-MP2 correlation energies are recorded in the
file, so a replay can check itself against them without psi4 present.

    python examples/dlpno/dump_reference.py --molecule water --basis cc-pvdz \
        --out examples/dlpno/fixtures/water-ccpvdz.npz

The grid is drained into the file, so this is for fixture-sized molecules.
Check the reported size before adding one to the repository.
"""

import argparse
import os
import sys

import psi4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno.molecules import MOLECULES
from dlpno.psi4_source import from_psi4
from dlpno.reference_io import save_reference

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--molecule", default="water", choices=sorted(MOLECULES))
parser.add_argument("--basis", default="cc-pvdz")
parser.add_argument("--localization", default="BOYS", choices=["BOYS", "PIPEK_MEZEY"])
parser.add_argument("--t-cut-pno", type=float, default=1e-8,
                    help="threshold psi4's own DLPNO-MP2 reference is taken at")
parser.add_argument("--out", required=True, help="destination .npz")
args = parser.parse_args()

psi4.core.set_output_file(f"/tmp/psi4_dump_{args.molecule}.out", False)
psi4.set_options({
    "basis": args.basis,
    "scf_type": "df",
    "mp2_type": "df",
    "freeze_core": "false",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    "r_convergence": 1e-8,
})

mol = psi4.geometry(MOLECULES[args.molecule] + "symmetry c1\n")

_, wfn = psi4.energy("mp2", return_wfn=True)
df_mp2 = psi4.variable("MP2 CORRELATION ENERGY")

psi4.set_options({"dlpno_algorithm": "mp2", "t_cut_pno": args.t_cut_pno})
psi4.energy("dlpno-mp2")
dlpno_mp2 = psi4.variable("MP2 CORRELATION ENERGY")

reference = from_psi4(wfn, localization=args.localization)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
save_reference(
    reference,
    args.out,
    energies={"psi4_df_mp2": df_mp2, "psi4_dlpno_mp2": dlpno_mp2, "scf": wfn.energy()},
    metadata={"molecule": args.molecule, "basis": args.basis,
              "localization": args.localization, "t_cut_pno": args.t_cut_pno,
              "psi4_version": psi4.__version__},
)

size_mb = os.path.getsize(args.out) / (1024 * 1024)
print(f"wrote {args.out}  ({size_mb:.2f} MiB)")
print(f"  nbf = {reference.nbf}, naux = {reference.naux}, active occ = {reference.naocc}")
print(f"  psi4 DF-MP2    = {df_mp2:.12f}")
print(f"  psi4 DLPNO-MP2 = {dlpno_mp2:.12f}  (T_CUT_PNO = {args.t_cut_pno:.1e})")
