#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""DF-MP2 correlation energy written in numpy-style einsums.

Same memory-optimal, pair-driven DF-MP2 as df_mp2_energy.py (never forms the
O(o^2 v^2) four-index (ia|jb) — the point of density fitting), but the compute
is expressed with einsums' numpy-like surface instead of explicit linalg calls.
Every operator still dispatches to einsums' own BLAS-level routines — NOT numpy —
so the math stays on einsums code paths and would compose inside cg.capture.

Side-by-side with the explicit form (see df_mp2_energy.py):

    explicit (df_mp2_energy.py)                 numpy-style (this file)
    -----------------------------------------   -----------------------------------------
    einsums.einsum("ab <- Qa ; Qb", I, Bi, Bj)  I = Bi.T @ Bj        # .T view + @ (gemm)
    permute + axpby + axpby  -> K               K = 2.0 * I - I.T    # scalar*, -, .T
    axpby + axpby            -> denom           W = Dbase + (e_i + e_j)   # scalar add
    element_transform(1/x) + direct_product     T = I / W            # / = Hadamard divide
    dot(K, T)                                   einsums.linalg.dot(K, T)    (reduction)

The numpy-like abilities exercised here:
    * einsums.asarray(...)          ingest a computed numpy array as an einsums tensor
    * A + scalar / A - scalar      scalar add/subtract (einsums.linalg.shift)
    * einsums.zeros(shape)          accumulator allocation
    * A.T                           zero-copy transpose view
    * A @ B                         matmul -> einsums gemm
    * 2.0 * A, A - B, A * B, A / B  scalar scaling, subtraction, element-wise product/quotient
    * A.shape / A.ndim              numpy-parity attributes

One op has no operator spelling and stays explicit: the full reduction
(einsums.linalg.dot). The elementwise quotient now has the '/' operator
(direct_division), so building the amplitude denominator needs no reciprocal callback.

numpy itself appears only to ingest psi4 data and read scalar orbital energies —
never for tensor math. Checked against psi4's own DF-MP2.

Pass ``--profile [FILE]`` to have einsums write a profile report at exit
(default file: df_mp2_numpy_style_profile.txt).

Run with the in-tree Einsums build and the psi4 stage on PYTHONPATH, using the
conda-env Python (which has numpy/psi4's deps)::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        /Users/jturney/Code/Einsums/Einsums/examples/psi4-bridge/df_mp2_numpy_style.py

The two PYTHONPATH entries are the Einsums build's ``lib`` (provides ``einsums``)
and psi4's ``stage/lib`` (provides ``psi4``); adjust if your trees live elsewhere.
"""
import argparse
import os

import numpy as np
import einsums  # loads einsums._core (registers types); the runtime is NOT yet up

# Profiling is configured through einsums.rc, which must be set BEFORE the
# runtime initializes (that happens on the first compute use below). So parse
# args and set rc here, while only the bindings — not the runtime — are loaded.
_parser = argparse.ArgumentParser(description="DF-MP2 correlation energy in numpy-style einsums.")
_parser.add_argument(
    "--profile", nargs="?", const="df_mp2_numpy_style_profile.txt", default=None, metavar="FILE",
    help="write an einsums profile report to FILE at exit "
         "(default: df_mp2_numpy_style_profile.txt in the launch directory)",
)
_args = _parser.parse_args()
if _args.profile is not None:
    # Resolve to an absolute path now: psi4 chdir's into its scratch directory,
    # so a relative profiler-filename would otherwise land there.
    _report_path = os.path.abspath(_args.profile)
    einsums.rc.profile_detailed = False
    einsums.rc.profile_filename = _report_path
    print(f"einsums profile report -> {_report_path} (written at exit)")

from einsums import linalg as la  # touching linalg initializes the runtime (reads einsums.rc)
import psi4

psi4.core.set_output_file("/tmp/psi4_df_mp2.out", False)
psi4.set_options({
    "basis": "cc-pvdz", "scf_type": "df", "mp2_type": "df",
    "freeze_core": "false", "e_convergence": 1e-10, "d_convergence": 1e-10,
})

mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n")

# psi4's own DF-MP2 (reference); the wfn carries the canonical SCF orbitals.
_, wfn = psi4.energy("mp2", return_wfn=True)
escf = psi4.variable("SCF TOTAL ENERGY")
ref_corr = psi4.variable("MP2 CORRELATION ENERGY")
print(f"psi4  SCF = {escf:.10f}   DF-MP2 corr = {ref_corr:.10f}")

# ---- ingest psi4 data into einsums tensors (the only numpy: data I/O) -------
primary = wfn.basisset()
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "", "RIFIT", primary.name())
C = wfn.Ca()
nocc = wfn.nalpha()
nvir = wfn.nmo() - nocc
eps = np.asarray(wfn.epsilon_a())
eo = eps[:nocc]                                  # occupied orbital energies (scalars)
ev_np = eps[nocc:]                               # virtual orbital energies (numpy, scalars)

dft = psi4.core.DFTensor(primary, aux, C, nocc, nvir)
B = dft.Qov_einsums()                            # dense rank-3 RuntimeTensor (naux, nocc, nvir)
print(f"B (Q|ov) shape = {B.shape}   nocc = {nocc}  nvir = {nvir}")

# Constant denominator base D_ab = -e_a - e_b, ingested with einsums.asarray from
# a one-time numpy outer sum. This is setup/data-prep, not tensor math in the loop.
Dbase = einsums.asarray(-ev_np[:, None] - ev_np[None, :], name="-ea-eb")
assert Dbase.shape == (nvir, nvir)

# ---- pair-driven DF-MP2, tensor math via numpy-style einsums operators ------
e_corr = 0.0
for i in range(nocc):
    Bi = B[:, i, :]                              # (naux, nvir) zero-copy view
    for j in range(i, nocc):
        Bj = B[:, j, :]
        I = Bi.T @ Bj                            # I_ab = sum_Q B^Q_ia B^Q_jb  (gemm)
        K = 2.0 * I - I.T                        # K_ab = 2 I_ab - I_ba
        W = Dbase + (eo[i] + eo[j])              # denom = (e_i+e_j) - e_a - e_b  (scalar add)
        T = I / W                                # T_ab = I_ab / denom  (Hadamard divide)
        e_pair = float(la.dot(K, T))             # sum_ab K_ab T_ab  (reduction)
        e_corr += e_pair if i == j else 2.0 * e_pair   # i<=j symmetry

print(f"einsums DF-MP2 corr = {e_corr:.10f}")
print(f"difference vs psi4  = {abs(e_corr - ref_corr):.2e}")
assert abs(e_corr - ref_corr) < 1e-6, "DF-MP2 correlation energy disagrees with psi4"
print("DF-MP2 (numpy-style einsums, pair-driven, no numpy in the compute) MATCHES psi4")
