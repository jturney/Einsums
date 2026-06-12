#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Timing harness for MintsHelper.mo_bra_half_transform_einsums.

Baseline + speedup tracker for the integral-direct bra half-transform. Times the
two half-transforms that feed the exact CC blocks:

    HT_oo = bra_half(C_o, C_o)   (o, o, nbf, nbf)
    HT_ov = bra_half(C_o, C_v)   (o, v, nbf, nbf)   <- the bigger one

Use --threads to compare serial vs parallel once OpenMP lands, and --basis or
--mol to scale the system. Reports wall time per transform, taking the best of
--reps runs.

Run::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        examples/psi4-bridge/cc_half_transform_bench.py --basis cc-pvdz --threads 1
"""
import argparse
import time

import numpy as np
import psi4
import einsums
import einsums._core  # noqa: F401  force pybind type registration (bug-916)

GEOMETRIES = {
    # Benzene: spatially extended, so Schwarz screening has something to skip.
    "benzene": """
        C  0.000  1.396  0.000
        C  1.209  0.698  0.000
        C  1.209 -0.698  0.000
        C  0.000 -1.396  0.000
        C -1.209 -0.698  0.000
        C -1.209  0.698  0.000
        H  0.000  2.479  0.000
        H  2.147  1.240  0.000
        H  2.147 -1.240  0.000
        H  0.000 -2.479  0.000
        H -2.147 -1.240  0.000
        H -2.147  1.240  0.000
        symmetry c1
    """,
    "water": "O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n",
    # Four waters strung ~6 Å apart along x: spatially extended, so most
    # cross-monomer shell-pairs are negligible and Schwarz screening bites.
    "wchain": """
        O  0.000  0.000  0.000
        H  0.757  0.586  0.000
        H -0.757  0.586  0.000
        O  6.000  0.000  0.000
        H  6.757  0.586  0.000
        H  5.243  0.586  0.000
        O 12.000  0.000  0.000
        H 12.757  0.586  0.000
        H 11.243  0.586  0.000
        O 18.000  0.000  0.000
        H 18.757  0.586  0.000
        H 17.243  0.586  0.000
        symmetry c1
    """,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", default="benzene", choices=list(GEOMETRIES))
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-12, help="INTS_TOLERANCE (Schwarz screening threshold)")
    args = ap.parse_args()

    psi4.core.set_output_file("/tmp/psi4_ht_bench.out", False)
    psi4.set_num_threads(args.threads)
    psi4.set_options({"basis": args.basis, "scf_type": "pk", "ints_tolerance": args.tol})
    mol = psi4.geometry(GEOMETRIES[args.mol])
    e_scf, wfn = psi4.energy("scf", return_wfn=True)

    mints = psi4.core.MintsHelper(wfn.basisset())
    nbf, nocc = wfn.nso(), wfn.nalpha()
    nvir = nbf - nocc
    C = np.asarray(wfn.Ca())
    Co = psi4.core.Matrix.from_array(np.ascontiguousarray(C[:, :nocc]))
    Cv = psi4.core.Matrix.from_array(np.ascontiguousarray(C[:, nocc:]))

    print(f"mol={args.mol} basis={args.basis} nbf={nbf} nocc={nocc} nvir={nvir} threads={args.threads} tol={args.tol:.0e}")

    def bench(label, fn):
        best = float("inf")
        chk = None
        for _ in range(args.reps):
            t0 = time.perf_counter()
            out = fn()
            dt = time.perf_counter() - t0
            best = min(best, dt)
            chk = float(np.linalg.norm(np.asarray(out)))
        print(f"  {label:18} best {best:8.3f} s   (||.||={chk:.6e})")
        return best

    bench("HT_oo (o,o,n,n)", lambda: mints.mo_bra_half_transform_einsums(Co, Co))
    bench("HT_ov (o,v,n,n)", lambda: mints.mo_bra_half_transform_einsums(Co, Cv))


if __name__ == "__main__":
    main()
