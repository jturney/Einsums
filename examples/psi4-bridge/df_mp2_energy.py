#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Density-fitted MP2 correlation energy computed entirely in einsums.

End-to-end demonstration of the bridge feeding a real correlated method, done the
way DF-MP2 is actually run *and* with every tensor operation expressed in einsums
(no numpy in the compute loop). The compact 3-index B^Q_{ia} stays resident; we
loop over occupied pairs and never form the O(o^2 v^2) four-index (ia|jb):

    per pair (i <= j):
        I_ab   = sum_Q B^Q_{ia} B^Q_{jb}                 einsums.einsum  (GEMM)
        K_ab   = 2 I_ab - I_ba                           permute + axpby
        w_ab   = 1 / (e_i + e_j - e_a - e_b)             outer_sum + axpby + element_transform
        t_ab   = I_ab * w_ab                             direct_product  (Hadamard)
        e_pair = sum_ab K_ab t_ab                        dot             (reduction)

    E_corr = sum_{i<=j} (2 - d_ij) e_pair

Checked against psi4's own DF-MP2. numpy appears only to ingest psi4 data into
einsums tensors and to read scalar orbital energies — never for tensor math.

Pass ``--profile [FILE]`` to have einsums write a profile report at exit
(default file: df_mp2_profile.txt). The report shows the per-pair einsum GEMM
dominating, with axpby/direct_product/dot for the energy assembly.

Run with the in-tree Einsums build and the psi4 stage on PYTHONPATH, using the
conda-env Python (which has numpy/psi4's deps)::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        /Users/jturney/Code/Einsums/Einsums/examples/psi4-bridge/df_mp2_energy.py

    # with a profile report:
    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        /Users/jturney/Code/Einsums/Einsums/examples/psi4-bridge/df_mp2_energy.py --profile

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
_parser = argparse.ArgumentParser(description="DF-MP2 correlation energy in einsums.")
_parser.add_argument(
    "--profile", nargs="?", const="df_mp2_profile.txt", default=None, metavar="FILE",
    help="write an einsums profile report to FILE at exit "
         "(default: df_mp2_profile.txt in the launch directory)",
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

dft = psi4.core.DFTensor(primary, aux, C, nocc, nvir)
B = dft.Qov_einsums()                                       # dense rank-3 RuntimeTensor
                                                            # (naux, nocc, nvir) -- sliceable
ev = einsums.create_zero_tensor("ev", [nvir], dtype="float64")
np.asarray(ev)[:] = eps[nocc:]                              # virtual orbital energies

# ---- constant scratch (built once, in einsums) ------------------------------
Dbase = einsums.create_zero_tensor("-ea-eb", [nvir, nvir], dtype="float64")
la.outer_sum(Dbase, [ev, ev], [-1.0, -1.0])                 # Dbase_ab = -e_a - e_b
ones = einsums.create_zero_tensor("ones", [nvir, nvir], dtype="float64")
la.element_transform(ones, lambda _: 1.0)                   # all-ones (for scalar add)
I = einsums.create_zero_tensor("I(ab)", [nvir, nvir], dtype="float64")
IT = einsums.create_zero_tensor("I(ba)", [nvir, nvir], dtype="float64")
K = einsums.create_zero_tensor("K(ab)", [nvir, nvir], dtype="float64")
W = einsums.create_zero_tensor("w(ab)", [nvir, nvir], dtype="float64")
T = einsums.create_zero_tensor("t(ab)", [nvir, nvir], dtype="float64")
recip = lambda x: 1.0 / x

# ---- pair-driven DF-MP2, all tensor ops in einsums --------------------------
e_corr = 0.0
for i in range(nocc):
    Bi = B[:, i, :]                                         # (naux, nvir) view
    for j in range(i, nocc):
        Bj = B[:, j, :]
        einsums.einsum("ab <- Qa ; Qb", I, Bi, Bj)          # I_ab = (ia|jb)
        einsums.permute("ab <- ba", IT, I)                  # IT = I^T
        la.axpby(2.0, I, 0.0, K)                            # K = 2 I
        la.axpby(-1.0, IT, 1.0, K)                          # K = 2 I - I^T
        la.axpby(1.0, Dbase, 0.0, W)                        # W = -e_a - e_b
        la.axpby(eo[i] + eo[j], ones, 1.0, W)               # W = (e_i+e_j) - e_a - e_b
        la.element_transform(W, recip)                      # W = 1 / denom
        la.direct_product(1.0, I, W, 0.0, T)                # T_ab = I_ab w_ab
        e_pair = float(la.dot(K, T))                        # sum_ab K_ab T_ab
        e_corr += e_pair if i == j else 2.0 * e_pair        # i<=j symmetry

print(f"einsums DF-MP2 corr = {e_corr:.10f}")
print(f"difference vs psi4  = {abs(e_corr - ref_corr):.2e}")
assert abs(e_corr - ref_corr) < 1e-6, "DF-MP2 correlation energy disagrees with psi4"
print("DF-MP2 (einsums, pair-driven, no numpy in the compute) MATCHES psi4")
