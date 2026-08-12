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
import re
import os
import sys

import numpy as np
import psi4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno.molecules import MOLECULES, water_chain, water_dimer_at
from dlpno.psi4_source import from_psi4
from dlpno.reference_io import save_reference

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--molecule", default="water",
                    help=f"one of {sorted(MOLECULES)}, 'chain<N>' for an N-monomer water "
                         "chain at 2.9 A, or 'dimer@<R>' for the dimer at an O-O distance "
                         "of R angstrom")
parser.add_argument("--basis", default="cc-pvdz")
parser.add_argument("--localization", default="BOYS", choices=["BOYS", "PIPEK_MEZEY"])
parser.add_argument("--update-energies", metavar="FIXTURE",
                    help="add reference energies to an EXISTING fixture without "
                         "touching its buffers. The molecule, basis, localization and "
                         "frozen-core setting come from the fixture's own metadata, so "
                         "the psi4 run matches what produced it. Preferred over a full "
                         "regeneration when only the energies are new: rewriting the "
                         "buffers means a fresh SCF, which moves every recorded number "
                         "in its last bits and invalidates any dump_energies baseline "
                         "taken before it.")
parser.add_argument("--with-cc", action="store_true",
                    help="also record psi4's DLPNO-CCSD and DLPNO-CCSD(T) correlation "
                         "energies and the classification counts, at the CC branch's "
                         "own NORMAL preset. Costs a full CCSD(T) per fixture, which is "
                         "why it is opt-in rather than the default")
parser.add_argument("--freeze-core", action="store_true",
                    help="freeze core orbitals; otherwise they are correlated with a "
                         "scaled PNO threshold, which is psi4's default")
parser.add_argument("--t-cut-pno", type=float, default=1e-8,
                    help="threshold psi4's own DLPNO-MP2 reference is taken at")
parser.add_argument("--out", help="destination .npz")
args = parser.parse_args()


def resolve_geometry(name):
    """Same spelling the bench scripts accept, so a fixture can be made of
    anything they can profile."""
    if name.startswith("chain"):
        return water_chain(int(name[len("chain"):]), 2.9) + "symmetry c1\nno_reorient\nno_com\n"
    if name.startswith("dimer@"):
        return water_dimer_at(float(name[len("dimer@"):])) + "symmetry c1\nno_reorient\nno_com\n"
    if name not in MOLECULES:
        parser.error(f"unknown molecule '{name}'; use one of {sorted(MOLECULES)}, "
                     "chain<N>, or dimer@<R>")
    return MOLECULES[name] + "symmetry c1\n"



def update_energies(path):
    """Recompute the reference energies of an existing fixture, in place.

    Every array except the ``energy_`` and ``meta_`` keys is copied through
    untouched, so the reference buffers - and therefore every energy the port
    computes from them - stay bit for bit what they were.

    That is the whole point of having this rather than a regeneration. Writing
    a fixture afresh runs a new SCF, and a new SCF is a different converged
    solution in its last bits, so every recorded number moves and any
    ``dump_energies`` baseline taken before it becomes incomparable. Adding a
    reference energy should not cost that.

    The psi4 numbers do come from a fresh SCF, which is a different solution
    from the frozen one - but only to convergence tolerance (1e-10), where the
    CC gates run at 1e-8 to 1e-9.
    """
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = {k[len("meta_"):]: str(arrays[k]) for k in arrays if k.startswith("meta_")}

    molecule = meta["molecule"]
    psi4.core.set_output_file(f"/tmp/psi4_update_{molecule}.out", False)
    psi4.set_options({
        "basis": meta["basis"], "scf_type": "df", "e_convergence": 1e-10,
        "d_convergence": 1e-10, "r_convergence": 1e-8,
        "freeze_core": meta.get("freeze_core", "False") == "True",
    })
    psi4.geometry(resolve_geometry(molecule))
    _, wfn = psi4.energy("mp2", return_wfn=True)

    fresh = {"psi4_df_mp2": psi4.variable("MP2 CORRELATION ENERGY"),
             "scf": wfn.energy()}
    psi4.set_options({"dlpno_algorithm": "mp2",
                      "t_cut_pno": float(meta.get("t_cut_pno", 1e-8))})
    psi4.energy("dlpno-mp2")
    fresh["psi4_dlpno_mp2"] = psi4.variable("MP2 CORRELATION ENERGY")

    # t_cut_pno persists and would otherwise override the CC branch's NORMAL;
    # see the note on the main path.
    psi4.set_options({"dlpno_algorithm": "ccsd(t)", "pno_convergence": "NORMAL",
                      "t_cut_pno": 3.33e-7})
    psi4.energy("dlpno-ccsd(t)")
    fresh.update({
        "psi4_dlpno_ccsd": psi4.variable("CCSD CORRELATION ENERGY"),
        "psi4_dlpno_ccsd_t": psi4.variable("CCSD(T) CORRELATION ENERGY"),
        "psi4_dlpno_t0": psi4.variable("DLPNO SEMICANONICAL (T0) ENERGY"),
        "psi4_dlpno_screened_triplets": psi4.variable("DLPNO SCREENED TRIPLETS ENERGY"),
        "psi4_dlpno_weak_pairs": psi4.variable("DLPNO LMP2 WEAK PAIR ENERGY"),
        "psi4_dlpno_eliminated": psi4.variable("DLPNO SC-LMP2 PAIR ENERGY"),
        "psi4_dlpno_pno_trunc": psi4.variable("DLPNO PNO TRUNCATION ERROR"),
    })
    for key, value in fresh.items():
        arrays[f"energy_{key}"] = np.array(value, dtype=np.float64)

    text = open(f"/tmp/psi4_update_{molecule}.out").read()
    for key, pattern in (("strong", r"Strong Pairs\s+=\s*(\d+)"),
                         ("weak", r"Weak Pairs\s+=\s*(\d+)"),
                         ("triplets", r"Number of \(Unique\) Local MO triplets:\s*(\d+)")):
        found = re.findall(pattern, text)
        if found:
            arrays[f"meta_count_{key}"] = np.array(str(found[-1]))

    np.savez_compressed(path, **arrays)
    print(f"updated {path}")
    print(f"    DLPNO-CCSD    {fresh['psi4_dlpno_ccsd']:.12f}")
    print(f"    DLPNO-CCSD(T) {fresh['psi4_dlpno_ccsd_t']:.12f}")


if args.update_energies:
    update_energies(args.update_energies)
    sys.exit(0)

if args.out is None:
    parser.error("--out is required unless --update-energies is given")

GEOM = resolve_geometry(args.molecule)

OUT = f"/tmp/psi4_dump_{args.molecule}.out"
psi4.core.set_output_file(OUT, False)
psi4.set_options({
    "basis": args.basis,
    "scf_type": "df",
    "mp2_type": "df",
    "freeze_core": "true" if args.freeze_core else "false",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    "r_convergence": 1e-8,
})

mol = psi4.geometry(GEOM)

_, wfn = psi4.energy("mp2", return_wfn=True)
df_mp2 = psi4.variable("MP2 CORRELATION ENERGY")

psi4.set_options({"dlpno_algorithm": "mp2", "t_cut_pno": args.t_cut_pno})
psi4.energy("dlpno-mp2")
dlpno_mp2 = psi4.variable("MP2 CORRELATION ENERGY")

energies = {"psi4_df_mp2": df_mp2, "psi4_dlpno_mp2": dlpno_mp2,
            "scf": wfn.energy()}
counts = {}
if args.with_cc:
    # The CC branch has its own preset table, so the thresholds move; recording
    # which is why ``pno_convergence`` goes into the metadata alongside the
    # numbers. ``dlpno-ccsd(t)`` gives the iterative correction, and psi4
    # publishes the semicanonical one alongside it, so both are kept - a
    # fixture that carried only the total could not tell a (T0) regression
    # from a (T) one.
    # ``t_cut_pno`` was pinned above for the MP2 reference, and it PERSISTS -
    # left alone it would silently override the CC branch's own NORMAL of
    # 3.33e-7 and record energies from a calculation no port run reproduces.
    # Set back explicitly rather than cleared, so both sides state the same
    # number and neither depends on which preset table psi4 consults.
    psi4.set_options({"dlpno_algorithm": "ccsd(t)", "pno_convergence": "NORMAL",
                      "t_cut_pno": 3.33e-7})
    psi4.energy("dlpno-ccsd(t)")
    energies.update({
        "psi4_dlpno_ccsd": psi4.variable("CCSD CORRELATION ENERGY"),
        "psi4_dlpno_ccsd_t": psi4.variable("CCSD(T) CORRELATION ENERGY"),
        "psi4_dlpno_t0": psi4.variable("DLPNO SEMICANONICAL (T0) ENERGY"),
        "psi4_dlpno_screened_triplets": psi4.variable("DLPNO SCREENED TRIPLETS ENERGY"),
        "psi4_dlpno_weak_pairs": psi4.variable("DLPNO LMP2 WEAK PAIR ENERGY"),
        "psi4_dlpno_eliminated": psi4.variable("DLPNO SC-LMP2 PAIR ENERGY"),
        "psi4_dlpno_pno_trunc": psi4.variable("DLPNO PNO TRUNCATION ERROR"),
    })
    text = open(OUT).read()
    for key, pattern in (("strong", r"Strong Pairs\s+=\s*(\d+)"),
                         ("weak", r"Weak Pairs\s+=\s*(\d+)"),
                         ("triplets", r"Number of \(Unique\) Local MO triplets:\s*(\d+)")):
        found = re.findall(pattern, text)
        if found:
            counts[key] = int(found[-1])

# Dense on purpose: a fixture is buffers, and save_reference cannot freeze a
# live integral source.
reference = from_psi4(wfn, integrals="dense", localization=args.localization,
                      freeze_core=args.freeze_core)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
save_reference(
    reference,
    args.out,
    energies=energies,
    metadata={"molecule": args.molecule, "basis": args.basis,
              "localization": args.localization, "t_cut_pno": args.t_cut_pno,
              "freeze_core": args.freeze_core, "psi4_version": psi4.__version__,
              **({"counts": counts, "pno_convergence": "NORMAL"} if args.with_cc else {})},
)

size_mb = os.path.getsize(args.out) / (1024 * 1024)
print(f"wrote {args.out}  ({size_mb:.2f} MiB)")
print(f"  nbf = {reference.nbf}, naux = {reference.naux}, active occ = {reference.naocc}")
print(f"  psi4 DF-MP2    = {df_mp2:.12f}")
print(f"  psi4 DLPNO-MP2 = {dlpno_mp2:.12f}  (T_CUT_PNO = {args.t_cut_pno:.1e})")
if args.with_cc:
    print(f"  psi4 DLPNO-CCSD    = {energies['psi4_dlpno_ccsd']:.12f}")
    print(f"  psi4 DLPNO-CCSD(T) = {energies['psi4_dlpno_ccsd_t']:.12f}   counts {counts}")
