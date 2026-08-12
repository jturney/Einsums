#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Does the port classify pairs the way psi4 does, while the classification moves?

The coupled-cluster analogue of ``sweep_separation.py``, and it exists for the
reason that one does: an energy comparison at a single geometry is blind to the
thing most likely to be wrong.

DLPNO-CCSD sorts every pair into one of four buckets - never formed (dipole
screening), eliminated by the crude semicanonical pass, weak, or strong - and
each boundary is a threshold comparison that can be inverted, off by a factor,
or applied to the wrong quantity without any of it showing in a total. It does
not show because each bucket's energy is added back as an estimate: a pair
misclassified as weak contributes its converged LMP2 energy instead of its
coupled-cluster one, and at the boundary those two numbers agree to several
digits by construction. The counts do not have that property. They are
integers, and they either match psi4's or they do not.

Two modes, because the clause in the design's Validation section has two halves.

**The sweep** (the default) walks the water dimer's O-O separation through the
boundaries and compares every count and every energy term against psi4 at each
point. Walking is the whole idea: at 2.9 A every pair is strong and at 12 A
every inter-monomer pair is gone, so both endpoints pass with a comparison
inverted at the boundary. What has to be exercised is the range where the
counts are changing a few at a time.

**The fixture census** (``--fixtures``) runs the cascade over the saved
references and reports which classes each one populates, so the claim that the
fixture set covers every class is data rather than an assumption. It needs no
psi4.

**What the sweep found, and what it is not.** Every count matches psi4 at every
separation, and every energy does too except at R(O-O) = 4.0, where the port is
9.6e-8 high. That one is not a coupled-cluster difference and not a
classification difference: it is already 9.3e-8 in the MP2 energy the cascade
produces, the pair counts agree exactly, and the PNO counts are identical. It is
the PAO linear-dependence tie-break that
:func:`~dlpno.tensors.orthocanonicalizer` documents - psi4 removes near-dependent
directions with a partial Cholesky and this port with canonical
orthogonalization, and at this geometry one PAO overlap eigenvalue sits just
below the default ``s_cut`` of 1e-8, so the two keep different directions.
Lowering ``s_cut`` one decade takes the MP2 difference to 5e-13 and the PNO
count does not move. A point that exceeds 1e-9 here is therefore a
tie-break candidate before it is a bug, and the run flags it as one; what would
be a bug is a count mismatch, and there are none.

    PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/stage/lib \
        python examples/dlpno/sweep_pairs_cc.py
    PYTHONPATH=/path/to/Einsums/build/lib \
        python examples/dlpno/sweep_pairs_cc.py --fixtures
"""

import argparse
import gc
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno.ccsd import DLPNOCCSD
from dlpno.thresholds import Thresholds

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

#: The classes a configuration can populate, and how to count each from a
#: finished calculation. "screened" is the pairs the dipole bound never formed,
#: which is the difference between all LMO pairs and the ones that survived it.
CLASSES = {
    "screened": lambda cc: cc.n_dipole_screened,
    "eliminated": lambda cc: cc.n_eliminated,
    "weak": lambda cc: len(cc.ij_to_i_j_weak),
    "strong": lambda cc: len(cc.ij_to_i_j_strong),
}


def counts_of(cc):
    return {name: int(fn(cc)) for name, fn in CLASSES.items()}


def instrument(cc):
    """Record the two counts the calculation does not otherwise keep.

    ``prep_sparsity`` and ``eliminate_pairs`` both narrow the pair list in
    place, so by the time the cascade finishes the sizes they started from are
    gone. Wrapping them is less invasive than making the calculation carry two
    integers no other caller wants.
    """
    cc.n_dipole_screened = 0
    cc.n_eliminated = 0
    naocc = cc.ref.naocc

    original_prep = cc.prep_sparsity
    original_eliminate = cc.eliminate_pairs

    def prep(*a, **kw):
        out = original_prep(*a, **kw)
        cc.n_dipole_screened = naocc * naocc - cc.n_lmo_pairs - cc.n_eliminated
        return out

    def eliminate(e_pair):
        before = cc.n_lmo_pairs
        out = original_eliminate(e_pair)
        cc.n_eliminated = before - cc.n_lmo_pairs
        return out

    cc.prep_sparsity = prep
    cc.eliminate_pairs = eliminate
    return cc


# -- the fixture census ------------------------------------------------------


def census(paths, preset):
    """Which classes each saved reference populates."""
    from dlpno.reference_io import load_reference

    print(f"\n=== fixture census, {preset} ===\n")
    header = f"  {'fixture':<32} " + " ".join(f"{c:>11}" for c in CLASSES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    populated = {name: [] for name in CLASSES}
    for path in paths:
        reference, _ = load_reference(path)
        cc = instrument(DLPNOCCSD(reference, Thresholds.preset(preset, method="cc"),
                                  verbose=False))
        # The cascade alone: the CC iteration does not reclassify anything, and
        # skipping it makes the census cheap enough to run on every fixture.
        cc.compute_energy(method="mp2")
        counts = counts_of(cc)
        name = os.path.basename(path)
        print(f"  {name:<32} " + " ".join(f"{counts[c]:>11d}" for c in CLASSES))
        for c, n in counts.items():
            if n:
                populated[c].append(name)

    print()
    missing = [c for c, who in populated.items() if not who]
    for c in CLASSES:
        who = populated[c]
        if who:
            print(f"  {c:<12} populated by {len(who)} fixture(s), "
                  f"first {who[0]}")
        else:
            print(f"  {c:<12} POPULATED BY NOTHING")
    if missing:
        print(f"\nFAIL no fixture exercises: {', '.join(missing)}")
        print("Add the one molecule that does rather than loosening anything.")
        return 1
    print("\nEvery classification bucket is exercised by at least one fixture.")
    return 0


# -- the separation sweep ----------------------------------------------------


def psi4_counts_from(path):
    """psi4's own classification report, which has no psivar."""
    text = open(path).read()
    patterns = {
        "eliminated": r"Eliminated Pairs \(SC-LMP2\)\s*=\s*(\d+)",
        "surviving": r"Surviving Pairs\s+=\s*(\d+)",
        "weak": r"Weak Pairs\s+=\s*(\d+)",
        "strong": r"Strong Pairs\s+=\s*(\d+)",
    }
    out = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = int(match.group(1))
    return out


def sweep(distances, basis, preset, threads, memory):
    import psi4

    from dlpno.molecules import water_dimer_at
    from dlpno.psi4_source import from_psi4

    psi4.set_memory(memory)
    psi4.set_options({
        "basis": basis,
        "scf_type": "df",
        "freeze_core": "false",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
        "r_convergence": 1e-8,
        "pno_convergence": preset,
        "dlpno_algorithm": "ccsd",
    })
    psi4.set_num_threads(threads)

    print(f"\n=== water dimer separation sweep, {basis}, {preset} ===\n")
    header = (f"  {'R(O-O)':>7}  {'screened':>17}  {'eliminated':>17}  "
              f"{'weak':>13}  {'strong':>13}  {'E(CCSD) port':>16}  {'vs psi4':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    failures = []
    tiebreaks = []
    seen = {name: set() for name in CLASSES}
    for r in distances:
        psi4.core.clean()
        out = f"/tmp/psi4_dlpno_cc_sweep_{r:.2f}.out"
        psi4.core.set_output_file(out, False)
        psi4.geometry(water_dimer_at(r) + "symmetry c1\nno_reorient\nno_com\n")

        _, wfn = psi4.energy("scf", return_wfn=True)
        psi4.energy("dlpno-ccsd")
        want_e = psi4.variable("CCSD CORRELATION ENERGY")
        want = psi4_counts_from(out)

        # Dense, because only that source serves the (Q|i j) and (Q|u v) the CC
        # integral layer declares; see decision 5.
        reference = from_psi4(wfn, localization="BOYS", integrals="dense")
        cc = instrument(DLPNOCCSD(reference, Thresholds.preset(preset, method="cc"),
                                  verbose=False))
        cc.compute_energy()
        got = counts_of(cc)
        for name, n in got.items():
            seen[name].add(n)

        # psi4 reports the pairs the dipole bound screened as the complement of
        # its surviving count, before the crude pass removes any more.
        want_screened = None
        if "surviving" in want and "eliminated" in want:
            want_screened = (reference.naocc ** 2 - want["surviving"]
                             - want["eliminated"])
        want_full = dict(want, screened=want_screened)

        cells = []
        for name in CLASSES:
            expect = want_full.get(name)
            mark = "" if expect is None or expect == got[name] else "X"
            shown = "-" if expect is None else str(expect)
            cells.append(f"{mark}{got[name]}/{shown}".rjust(
                17 if name in ("screened", "eliminated") else 13))
            if expect is not None and expect != got[name]:
                failures.append(f"R={r:.2f} {name}: psi4 {expect} vs port {got[name]}")

        delta = abs(cc.e_corr - want_e)
        if delta >= max(1e-8, 1e-6 * abs(want_e)):
            failures.append(f"R={r:.2f} E(CCSD): psi4 {want_e:.12f} vs port "
                            f"{cc.e_corr:.12f} ({delta:.2e})")
        elif delta >= 1e-9:
            tiebreaks.append(
                f"R={r:.2f} is {delta:.2e} from psi4, above the 1e-9 the rest "
                f"of the sweep reaches. Check s_cut before suspecting the "
                f"equations: see the module docstring.")
        print(f"  {r:7.2f}  " + "  ".join(cells)
              + f"  {cc.e_corr:16.12f}  {delta:9.2e}", flush=True)
        # A captured solver holds its graphs, its stores and every operand they
        # reference until it is dropped, which at sweep scale is gigabytes. The
        # release is explicit rather than left to whenever the name is rebound,
        # so a point's memory goes back before the next point's is taken.
        #
        # This used to be load-bearing for a second reason that no longer
        # applies: shape metadata was charged against the buffer ceiling, so a
        # live graph's views exhausted it outright. That is fixed in the library
        # (ShapeVector), and this is now ordinary hygiene rather than a
        # workaround.
        del cc
        gc.collect()

    print()
    for name in CLASSES:
        values = sorted(seen[name])
        moved = "moves" if len(values) > 1 else "CONSTANT"
        print(f"  {name:<12} {moved} across the sweep: {values}")
    constant = [n for n in CLASSES if len(seen[n]) == 1]
    if constant:
        print(f"\n  Note: {', '.join(constant)} never changed, so this sweep did "
              f"not test its boundary. Widen --distances.")

    if tiebreaks:
        print()
        for message in tiebreaks:
            print(f"  NOTE {message}")

    if failures:
        print()
        for message in failures:
            print(f"FAIL {message}")
        return 1
    print("\nEvery classification count matches psi4 at every separation, and "
          "every energy\nis within the linear-dependence tie-break.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", action="store_true",
                        help="run the psi4-free fixture census instead of the sweep")
    parser.add_argument("--distances", type=float, nargs="+",
                        default=[2.9, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 12.0],
                        help="O-O separations in angstrom")
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--preset", default="NORMAL",
                        choices=["LOOSE", "NORMAL", "TIGHT", "VERY_TIGHT"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory", default="24 GB")
    args = parser.parse_args()

    if args.fixtures:
        return census(sorted(glob.glob(os.path.join(FIXTURES, "*.npz"))),
                      args.preset)
    return sweep(args.distances, args.basis, args.preset, args.threads,
                 args.memory)


if __name__ == "__main__":
    sys.exit(main())
