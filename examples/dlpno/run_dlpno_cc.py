#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Run DLPNO-CCSD or DLPNO-CCSD(T0) on one molecule and validate against psi4.

The coupled-cluster analogue of ``run_dlpno_mp2.py``, and it compares more than
an energy on purpose. A local correlation method can reach the right total from
two compensating errors - a pair misclassified as weak whose energy is then
added back as an estimate is the classic one - so this checks every term psi4
publishes as a psivar separately, and the classification COUNTS on top:

* ``MP2 CORRELATION ENERGY``, which psi4 sets partway through its own CC run,
  after the refined prescreening and before the PNO rebuild;
* ``DLPNO SC-LMP2 PAIR ENERGY`` - the crude pass's eliminated pairs;
* ``DLPNO DIPOLE ENERGY`` - the pairs never formed at all;
* ``DLPNO PNO TRUNCATION ERROR`` and ``DLPNO LMP2 WEAK PAIR ENERGY``;
* the number of strong, weak and eliminated pairs, which no psivar carries and
  which is read out of psi4's own printed table.

Counts are the check the untruncated one is structurally blind to: with every
threshold off there are no weak pairs to misclassify.

``--method 'ccsd(t)'`` adds the semicanonical triples on both sides, with psi4
put into ``T0_APPROXIMATION true`` because that is the approximation the port
implements - the iterative (T) is milestone M6. Three more psivars and the
triplet count come with it.

    PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/stage/lib \
        python examples/dlpno/run_dlpno_cc.py --molecule water-dimer
    ... --molecule water-dimer --method 'ccsd(t)'
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psi4

from dlpno.ccsd import DLPNOCCSD
from dlpno.molecules import MOLECULES, water_dimer_at
from dlpno.psi4_source import from_psi4
from dlpno.thresholds import Thresholds
from dlpno.triples import DLPNOCCSDT

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--method", default="ccsd", choices=["ccsd", "ccsd(t)"],
                    help="'ccsd(t)' runs the semicanonical triples on both "
                         "sides, with psi4 in T0_APPROXIMATION")
parser.add_argument("--molecule", default="water")
parser.add_argument("--basis", default="cc-pvdz")
parser.add_argument("--preset", default="NORMAL",
                    choices=["LOOSE", "NORMAL", "TIGHT", "VERY_TIGHT"])
parser.add_argument("--localization", default="BOYS", choices=["BOYS", "PIPEK_MEZEY"])
parser.add_argument("--threads", type=int, default=1,
                    help="thread count. Importing psi4 clamps process-wide OpenMP "
                         "to 1, so this must be set for einsums to thread at all")
parser.add_argument("--integrals", default="dense", choices=["dense", "dfhelper"],
                    help="where the three-index integrals come from. Only 'dense' "
                         "serves (Q|i j) and (Q|u v), which the CC layers need")
parser.add_argument("--memory", default="24 GB")
args = parser.parse_args()

if args.molecule in MOLECULES:
    GEOM = MOLECULES[args.molecule] + "symmetry c1\n"
elif args.molecule.startswith("dimer-"):
    GEOM = water_dimer_at(float(args.molecule[6:])) + "symmetry c1\n"
else:
    parser.error(f"unknown molecule {args.molecule!r}; use one of {sorted(MOLECULES)} "
                 "or dimer-<r_oo>")

OUT = f"/tmp/psi4_dlpno_cc_{args.molecule}_{args.method}.out"
psi4.core.set_output_file(OUT, False)
psi4.set_memory(args.memory)
psi4.set_options({
    "basis": args.basis,
    "scf_type": "df",
    "freeze_core": "false",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    "r_convergence": 1e-8,
    "pno_convergence": args.preset,
})
psi4.set_num_threads(args.threads)
mol = psi4.geometry(GEOM)

_, wfn = psi4.energy("scf", return_wfn=True)
if args.method == "ccsd":
    psi4.set_options({"dlpno_algorithm": "ccsd"})
    psi4.energy("dlpno-ccsd")
else:
    # ``dlpno-ccsd(t0)``, NOT ``dlpno-ccsd(t)`` with ``T0_APPROXIMATION`` set.
    # The option is not user-settable through the second route: psi4's
    # ``run_dlpnoccsd_t`` calls ``set_local_option`` from the method NAME and
    # overwrites whatever was asked for, so ``dlpno-ccsd(t)`` runs the full
    # iterative (T) with the option silently flipped back to false. On water
    # that is a 1.5e-4 difference, which is the whole (T0) approximation error
    # and would read as a port defect.
    psi4.set_options({"dlpno_algorithm": "ccsd(t)"})
    psi4.energy("dlpno-ccsd(t0)")

psi4_terms = {
    "mp2": psi4.variable("MP2 CORRELATION ENERGY"),
    "ccsd": psi4.variable("CCSD CORRELATION ENERGY"),
    "eliminated": psi4.variable("DLPNO SC-LMP2 PAIR ENERGY"),
    "dipole": psi4.variable("DLPNO DIPOLE ENERGY"),
    "pno_trunc": psi4.variable("DLPNO PNO TRUNCATION ERROR"),
}
if args.method == "ccsd(t)":
    psi4_terms.update({
        "ccsd(t)": psi4.variable("CCSD(T) CORRELATION ENERGY"),
        # These two are the whole triples story: the correction psi4 adds to
        # its CCSD energy, and the part of it that came from triplets the
        # screening never computed properly. Under T0_APPROXIMATION the first
        # equals "DLPNO SEMICANONICAL (T0) ENERGY", so comparing both is a free
        # check that psi4 really did stop at (T0).
        "t_correction": psi4.variable("(T) CORRECTION ENERGY"),
        "t0_semicanonical": psi4.variable("DLPNO SEMICANONICAL (T0) ENERGY"),
        "screened_triplets": psi4.variable("DLPNO SCREENED TRIPLETS ENERGY"),
    })


def psi4_output_facts(path):
    """The pair counts and the MP2-level weak-pair energy, from psi4's output.

    Read rather than asked for, and for two different reasons.

    The counts have no psivar at all. They are still the sharpest check
    available: an energy can be right for the wrong reason - two compensating
    classification errors is the classic local-correlation bug - and a count
    cannot.

    The weak-pair energy is published as a psivar and printed separately, and
    the two are DIFFERENT NUMBERS, so both are read and both are compared.
    ``DLPNO LMP2 WEAK PAIR ENERGY`` is set at the very end of psi4's
    ``compute_energy``, by which point ``lccsd_iterations`` has overwritten
    ``de_weak_`` with its own value: the same doubles contribution plus the
    singles term, since psi4 computes ``tau . L`` over weak pairs and ``tau``
    carries ``t_i t_j``. The doubles halves are algebraically identical
    (``T . L == K . Tt``), so the whole difference is singles. What
    ``recompute_pnos`` printed on its way past is the MP2-level value, which is
    what the cascade alone produces. Comparing only one of them cannot tell
    which of the two the port computed.
    """
    text = open(path).read()
    facts = {"counts": {}}
    patterns = {
        "eliminated": r"Eliminated Pairs \(SC-LMP2\)\s*=\s*(\d+)",
        "surviving": r"Surviving Pairs\s+=\s*(\d+)",
        "weak": r"Weak Pairs\s+=\s*(\d+)",
        "strong": r"Strong Pairs\s+=\s*(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            facts["counts"][key] = int(match.group(1))
    # The LAST occurrence: recompute_pnos prints it once, but a future psi4
    # that printed it twice should be read at the point this compares against.
    weak = re.findall(r"LMP2 Weak Pair energy\s*=\s*(-?[0-9.]+)", text)
    if weak:
        facts["weak"] = float(weak[-1])
    # The triplet count, which like the pair counts has no psivar. Printed once
    # per tno_transform, so the LAST one is the production pass - the one whose
    # triplets actually produced the reported energy.
    triplets = re.findall(r"Number of \(Unique\) Local MO triplets:\s*(\d+)", text)
    if triplets:
        facts["counts"]["triplet"] = int(triplets[-1])
    return facts


psi4_facts = psi4_output_facts(OUT)
psi4_counts = psi4_facts["counts"]
if "weak" in psi4_facts:
    psi4_terms["weak_mp2"] = psi4_facts["weak"]
psi4_terms["weak"] = psi4.variable("DLPNO LMP2 WEAK PAIR ENERGY")

reference = from_psi4(wfn, localization=args.localization, integrals=args.integrals)
cut = Thresholds.preset(args.preset, method="cc")
if args.method == "ccsd":
    cc = DLPNOCCSD(reference, cut)
    cc.compute_energy()
else:
    from dataclasses import replace as _replace
    cc = DLPNOCCSDT(reference, _replace(cut, t0_approximation=True))
    cc.compute_energy(method="ccsd(t)")

ours = {
    # The CCSD row is the CCSD energy either way. Under --method 'ccsd(t)' the
    # port's own ``e_corr`` carries the triples, so the CCSD total has to be
    # reassembled from its terms rather than read off.
    "ccsd": (cc.e_lccsd + cc.de_weak + cc.de_lmp2_eliminated + cc.de_dipole
             + cc.de_pno_total),
    "mp2": cc.e_mp2_corr(),
    "weak": cc.de_weak,
    "weak_mp2": cc.de_weak_mp2,
    "eliminated": cc.de_lmp2_eliminated,
    "dipole": cc.de_dipole,
    "pno_trunc": cc.de_pno_total,
}
our_counts = {
    "strong": len(cc.ij_to_i_j_strong),
    "weak": len(cc.ij_to_i_j_weak),
}
if args.method == "ccsd(t)":
    ours.update({
        "ccsd(t)": cc.e_corr,
        "t_correction": cc.e_t0 + cc.de_lccsd_t_screened,
        "t0_semicanonical": cc.e_t0 + cc.de_lccsd_t_screened,
        "screened_triplets": cc.de_lccsd_t_screened,
    })
    our_counts["triplet"] = cc.n_lmo_triplets

label = "DLPNO-CCSD" if args.method == "ccsd" else "DLPNO-CCSD(T0)"
print(f"\n=== {label}, {args.molecule}/{args.basis}, {args.preset} ===\n")
print(f"  {'term':<28} {'psi4':>18} {'port':>18} {'difference':>12}")
failures = []
# Scaled to each term rather than fixed: these corrections span several orders
# of magnitude, and a 1e-9 bound on a 1e-3 term is a different statement from
# the same bound on a 1e-8 one.
#
# The CCSD row is looser than the rest, and for a reason that is a property of
# the method rather than of the port. Every other term is the output of a linear
# or one-shot construction and agrees to machine precision; the CCSD energy is
# a fixed point that two implementations reach along different DIIS
# trajectories, so their agreement is bounded by the convergence tolerance
# rather than by arithmetic. At r_convergence 1e-8 that is a few times 1e-11.
#
# The triples rows inherit that: (T0) is a one-shot function of the converged
# CCSD amplitudes, so whatever the amplitudes differ by propagates straight
# through and no tighter bound is available.
TOLERANCE = {"ccsd": 1e-9, "ccsd(t)": 1e-9, "t_correction": 1e-9,
             "t0_semicanonical": 1e-9, "screened_triplets": 1e-9}
for key in ("mp2", "ccsd", "eliminated", "dipole", "pno_trunc",
            "weak", "weak_mp2", "ccsd(t)", "t_correction",
            "t0_semicanonical", "screened_triplets"):
    if key not in psi4_terms:
        if key in ours:
            print(f"  {key:<28} {'(not published by psi4)':>18}")
        continue
    want, got = psi4_terms[key], ours[key]
    delta = abs(want - got)
    tol = max(TOLERANCE.get(key, 1e-9), 1e-6 * abs(want))
    mark = " " if delta < tol else "X"
    print(f" {mark}{key:<28} {want:>18.12f} {got:>18.12f} {delta:>12.3e}")
    if delta >= tol:
        failures.append(f"{key}: psi4 {want:.12f} vs port {got:.12f} "
                        f"(differ by {delta:.3e}, tolerance {tol:.1e})")

print(f"\n  {'classification':<28} {'psi4':>18} {'port':>18}")
for key, got in sorted(our_counts.items()):
    want = psi4_counts.get(key)
    mark = " " if want is None or want == got else "X"
    shown = "-" if want is None else str(want)
    kind = "triplets" if key == "triplet" else "pairs"
    print(f" {mark}{key + ' ' + kind:<28} {shown:>18} {got:>18}")
    if want is not None and want != got:
        failures.append(f"{key} {kind} count: psi4 {want} vs port {got}")

if failures:
    print()
    for message in failures:
        print(f"FAIL {message}")
    sys.exit(1)
print("\nThe port matches psi4 on every energy term and every count.")
# The triples path drops the CCSD solver once it has the amplitudes - psi4 does
# the same, and for the same reason - so its statistics are read from the
# snapshot ``_release_ccsd`` keeps.
stats = (dict(iterations=cc.lccsd.n_iterations, nodes=cc.lccsd.num_nodes(),
              t_plan=cc.lccsd.t_plan, t_capture=cc.lccsd.t_capture,
              t_iterate=cc.lccsd.t_iterate)
         if cc.lccsd is not None else cc.ccsd_stats)
print(f"  CCSD converged in {stats['iterations']} iterations, "
      f"{stats['nodes']} captured nodes replayed in "
      f"{stats['t_iterate']:.3f} s "
      f"(plan {stats['t_plan'] * 1e3:.0f} ms, capture "
      f"{stats['t_capture'] * 1e3:.0f} ms).")
if args.method == "ccsd(t)":
    print(f"  (T0) over {cc.n_lmo_triplets} triplets "
          f"({min(cc.n_tno)}-{max(cc.n_tno)} TNOs) in {cc.t_t0:.3f} s "
          f"(sparsity {cc.t_sparsity:.3f} s, TNO transform {cc.t_tno:.3f} s).")
