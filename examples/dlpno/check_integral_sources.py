#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""What each three-index integral source costs, and what it gives up.

Needs psi4, which is why it is a script rather than one of the tests: the dense
source is the oracle precisely because it needs nothing, and a test that
imported psi4 to check it would give that property up.

Three sources, checked against each other on one wavefunction in one process:

* ``DenseSource`` builds a whole ``(Q|mn)`` with ``ao_eri`` and transforms it.
  Exact, unthreaded, and the oracle.
* ``DFHelperSource`` builds the same AO integrals threaded and Schwarz screened,
  then runs the same transform. Exact when the cutoff is zero.
* ``ScreenedQiaSource`` builds only the blocks the solver declares it will read.
  Domain-restricted, so it is checked twice: at its untruncated settings, where
  it has to reproduce the oracle on every declared block, and at psi4's own
  NORMAL settings, where the deviation IS the approximation and the number to
  report is how it compares with the PNO truncation error it sits underneath.

    python examples/dlpno/check_integral_sources.py --molecule ethanol --basis cc-pvtz --threads 10

``--sweep`` runs the exactness gate over every geometry the fixtures cover,
which is the differential test a change to the screened builder has to pass.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psi4

from dlpno import integrals
from dlpno.molecules import MOLECULES, water_chain, water_dimer_at
from dlpno.mp2 import DLPNOMP2
from dlpno.psi4_source import DFHelperSource, ScreenedQiaSource, from_psi4, raw_three_index
from dlpno.thresholds import Thresholds

# The geometries the frozen fixtures were made from. --sweep rebuilds them live
# rather than loading the .npz, because the comparison has to happen in one
# process against one wavefunction: psi4's threaded SCF and localization are not
# bit-reproducible run to run, so two processes disagree at 5e-12 in the
# coefficients before any integral is built.
FIXTURE_CASES = [
    ("water", "cc-pvdz", False),
    ("water", "cc-pvtz", False),
    ("water", "cc-pvdz", True),
    ("water-dimer", "cc-pvdz", False),
    ("water-dimer-far", "cc-pvdz", False),
    ("methanol", "cc-pvdz", False),
]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--molecule", default="water")
parser.add_argument("--basis", default="cc-pvdz")
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--schwarz", type=float, default=0.0,
                    help="Schwarz cutoff for the DFHelper source; 0 disables screening")
parser.add_argument("--t-cut-pno", type=float, default=1e-8)
parser.add_argument("--sweep", action="store_true",
                    help="run the exactness gate over every fixture geometry and exit "
                         "non-zero if any fails")
args = parser.parse_args()


def resolve_geometry(name):
    if name.startswith("chain"):
        return water_chain(int(name[len("chain"):]), 2.9) + "symmetry c1\nno_reorient\nno_com\n"
    if name.startswith("dimer@"):
        return water_dimer_at(float(name[len("dimer@"):])) + "symmetry c1\nno_reorient\nno_com\n"
    if name not in MOLECULES:
        parser.error(f"unknown molecule '{name}'; use one of {sorted(MOLECULES)}, "
                     "chain<N>, or dimer@<R>")
    return MOLECULES[name] + "symmetry c1\n"


def build_wavefunction(molecule, basis, freeze_core):
    psi4.core.clean_options()
    psi4.core.be_quiet()
    psi4.set_num_threads(args.threads)
    psi4.set_options({"basis": basis, "scf_type": "df",
                      "freeze_core": "true" if freeze_core else "false",
                      "e_convergence": 1e-10, "d_convergence": 1e-10})
    mol = psi4.geometry(resolve_geometry(molecule))
    _, wfn = psi4.energy("scf", return_wfn=True)
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "", "RIFIT", basis)
    return wfn, wfn.basisset(), aux


def run(reference, source, cut, label=None):
    """One full calculation on one source, timing only the integral phase."""
    mp2 = DLPNOMP2(reference, cut, verbose=False, integral_source=source)
    mp2.setup_orbitals(); mp2.compute_doi(); mp2.prep_sparsity(); mp2.compute_metric()
    t0 = time.perf_counter()
    mp2.compute_qia()
    elapsed = time.perf_counter() - t0
    mp2.precompute_fits(); mp2.pno_transform(); mp2.compute_pno_overlaps()
    mp2.lmp2_iterations(optimize=True)
    e_corr = mp2.e_lmp2 + mp2.de_pno_total + mp2.de_dipole
    if label is not None:
        print(f"  {label:<34} compute_qia {elapsed:7.3f} s   e_corr {e_corr:.12f}   "
              f"threshold {source.screening_threshold:.0e}")
    return np.asarray(mp2.q_ia, copy=True), e_corr, elapsed


def exactness_gate(molecule, basis, freeze_core, verbose=True):
    """The differential test: untruncated screened source against the oracle.

    Switching all three of the screened source's tolerances off makes its basis
    function domains the whole basis and its refit the identity, so every block
    it writes must be the oracle's to round-off. It still writes only the
    declared blocks - the demand comes from the pair list, which the solver's
    own thresholds decide, not from these tolerances - which is why the
    comparison is masked.

    This is the gate that catches an indexing or scatter bug, which is the
    failure mode a per-atom builder has: the port's domain lists are sorted and
    psi4's blocks come back in the caller's own ordering, so a disagreement
    about ordering shows up as a small energy error rather than a crash.
    """
    wfn, primary, aux = build_wavefunction(molecule, basis, freeze_core)
    reference = from_psi4(wfn, integrals="dense", freeze_core=freeze_core)
    cut = Thresholds.preset("NORMAL", t_cut_pno=args.t_cut_pno)
    exact = Thresholds.untruncated()

    screened = ScreenedQiaSource(primary, aux, reference.S,
                                 t_cut_clmo=exact.t_cut_clmo, t_cut_cpao=exact.t_cut_cpao,
                                 ao_ints_tol=exact.ao_ints_tol, bp_refit=False,
                                 nthreads=args.threads)
    q_dense, e_dense, _ = run(reference, integrals.DenseSource(reference.eri_3index), cut)
    q_screened, e_screened, _ = run(reference, screened, cut)

    mask = screened.declared_mask()
    scale = max(np.abs(q_dense).max(), 1.0)
    err = np.abs(q_screened[mask] - q_dense[mask]).max() / scale
    de = abs(e_screened - e_dense)
    ok = err < 1e-13 and de < 1e-11
    if verbose:
        core = "frozen core" if freeze_core else "all electron"
        print(f"  {molecule}/{basis} ({core}): {mask.sum()} of {mask.size} entries declared "
              f"({100.0 * mask.sum() / mask.size:.1f}%), max rel error {err:.2e}, "
              f"energy {de:.2e} Eh  {'ok' if ok else 'FAILED'}")
    return ok


if args.sweep:
    print("\nexactness gate: screened source with every tolerance off, against the dense oracle\n")
    failures = [case for case in FIXTURE_CASES if not exactness_gate(*case)]
    if failures:
        print(f"\n{len(failures)} of {len(FIXTURE_CASES)} geometries FAILED")
        sys.exit(1)
    print(f"\nall {len(FIXTURE_CASES)} geometries reproduce the oracle on every declared block")
    sys.exit(0)

wfn, primary, aux = build_wavefunction(args.molecule, args.basis, False)
cut = Thresholds.preset("NORMAL", t_cut_pno=args.t_cut_pno)
reference = from_psi4(wfn, integrals="dense")

print(f"\n{args.molecule}/{args.basis}, {args.threads} thread(s), "
      f"nbf={primary.nbf()} naux={aux.nbf()}\n")

# The dense source's real cost is split across two phases and only half of it
# lands in compute_qia: from_psi4 already built (Q|mn), and compute_qia only
# transforms what it was handed. The other two do both inside compute_qia, so
# timing the phases against each other flatters them by the whole integral
# build. Price that build separately and add it back.
t0 = time.perf_counter()
raw_three_index(primary, aux)
t_build = time.perf_counter() - t0
print(f"  (dense (Q|mn) build, charged to DF Ints rather than compute_qia: {t_build:.3f} s)\n")

q_dense, e_dense, t_dense = run(reference, integrals.DenseSource(reference.eri_3index), cut,
                                "DenseSource (exact)")
q_dfh, e_dfh, t_dfh = run(
    reference, DFHelperSource(primary, aux, schwarz_cutoff=args.schwarz, nthreads=args.threads),
    cut, "DFHelperSource (raw AO)")
screened = ScreenedQiaSource(primary, aux, reference.S, t_cut_clmo=cut.t_cut_clmo,
                             t_cut_cpao=cut.t_cut_cpao, ao_ints_tol=cut.ao_ints_tol,
                             nthreads=args.threads)
q_scr, e_scr, t_scr = run(reference, screened, cut, "ScreenedQiaSource (NORMAL)")

scale = np.abs(q_dense).max()
mask = screened.declared_mask()
print(f"\n  DFHelper vs dense")
print(f"    integrals:  max abs {np.abs(q_dfh - q_dense).max():.3e}"
      f"   relative {np.abs(q_dfh - q_dense).max() / scale:.3e}")
print(f"    energy:     {abs(e_dfh - e_dense):.3e} Hartree")
print(f"  screened vs dense, on the {100.0 * mask.sum() / mask.size:.1f}% of entries it declares")
print(f"    integrals:  max abs {np.abs(q_scr[mask] - q_dense[mask]).max():.3e}"
      f"   relative {np.abs(q_scr[mask] - q_dense[mask]).max() / scale:.3e}")
print(f"    energy:     {abs(e_scr - e_dense):.3e} Hartree, "
      f"{abs(e_scr - e_dense) / abs(e_dense):.2e} of the correlation energy")
print(f"    {screened.describe()}")

fair_dense = t_dense + t_build
print(f"\n  integrals available:")
print(f"    dense      {fair_dense:.3f} s (build {t_build:.3f} + transform {t_dense:.3f})")
print(f"    DFHelper   {t_dfh:.3f} s   {fair_dense / t_dfh:.2f}x the dense path")
print(f"    screened   {t_scr:.3f} s   {fair_dense / t_scr:.2f}x the dense path, "
      f"{t_dfh / t_scr:.2f}x DFHelper")

print("\n  The two deviations mean different things. DFHelper's is round-off: with the")
print("  cutoff at zero it is the same integrals in a different order, and untruncated")
print("  DLPNO-MP2 still has to reproduce canonical DF-MP2 to 1e-13. The screened")
print("  source's is the domain approximation itself, the same one psi4's DLPNO makes,")
print("  and it is only defensible while it stays well under the PNO truncation error.")
