#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""DLPNO truncation thresholds, mirroring psi4's DLPNO options.

Names and defaults follow psi4's ``read_options.cc`` and the ``PNO_CONVERGENCE``
presets applied in ``DLPNO::common_init``, so a run here can be lined up against
a psi4 run option for option.
"""

from dataclasses import dataclass, replace

__all__ = ["Thresholds"]


@dataclass(frozen=True)
class Thresholds:
    """Truncation parameters for a DLPNO calculation.

    The defaults are psi4's, at ``PNO_CONVERGENCE NORMAL`` for MP2.
    """

    # PNO truncation
    t_cut_pno: float = 1e-8
    t_cut_pno_diag_scale: float = 3e-2
    t_cut_pno_core_scale: float = 1e-2
    min_pnos: int = 5

    # LMO / PAO / auxiliary domain truncation
    t_cut_do: float = 1e-2
    t_cut_do_ij: float = 1e-5
    t_cut_do_pre: float = 3e-2
    t_cut_mkn: float = 1e-3
    t_cut_pre: float = 1e-6
    t_cut_pairs: float = 1e-5
    t_cut_clmo: float = 1e-4
    t_cut_cpao: float = 1e-4

    #: Shell-pair screening on the AO integrals a three-index producer builds,
    #: psi4's ``DLPNO_AO_INTS_TOL``. Unlike everything else here this one is a
    #: screening rather than an approximation: the error goes to zero with it,
    #: and only a producer that builds its own integrals reads it at all.
    ao_ints_tol: float = 1e-10

    # linear algebra cutoffs
    s_cut: float = 1e-8
    f_cut: float = 1e-5

    #: How many PNO-count buckets the pair blocks are padded into, or None to
    #: choose per machine. Blocks must share a shape to batch, but padding
    #: everything to the global maximum wastes elements; a handful of buckets
    #: keeps the batches large while cutting most of the waste. 1 reproduces a
    #: single padded store.
    #:
    #: None is the default because the right count is not a property of the
    #: molecule. More buckets is cheaper per element and dearer per batched
    #: call, and what a call costs is what entering an OpenMP region costs on
    #: this machine at this thread count - zero serial, tens of microseconds on
    #: ten threads. Measured on ethanol/cc-pVTZ the best count runs 12, 8, 8, 4
    #: at 1, 2, 4 and 10 threads. See :meth:`DLPNOBase._choose_buckets`.
    #:
    #: Set an integer to pin it, which is what the calibration sweeps do.
    n_buckets: int | None = None

    #: Largest bucket count the automatic chooser will consider. Past this the
    #: shape classes outnumber the pairs and every batch is a handful of members.
    max_buckets: int = 16

    #: How many groups each PNO bucket's pairs are split into by the total width
    #: of their concatenated couplings. The residual folds all of a pair's
    #: couplings in one GEMM, and batching those across pairs needs them to agree
    #: on that width, so it is padded to the group's widest. One group per bucket
    #: is the fewest calls and the most padding (1.42x the flops at a six-monomer
    #: water chain); four costs 1.10x for sixteen calls against 32948 in the
    #: per-coupling form this replaced.
    n_width_groups: int = 4

    # iterative solver
    maxiter: int = 50
    e_convergence: float = 1e-10
    r_convergence: float = 1e-8
    diis_max_vecs: int = 6

    @classmethod
    def preset(cls, name="NORMAL", **overrides):
        """psi4's ``PNO_CONVERGENCE`` presets for the MP2 branch."""
        presets = {
            "LOOSE": dict(t_cut_pno=1e-7, t_cut_do=2e-2, t_cut_mkn=1e-3),
            "NORMAL": dict(t_cut_pno=1e-8, t_cut_do=1e-2, t_cut_mkn=1e-3),
            "TIGHT": dict(t_cut_pno=1e-9, t_cut_do=5e-3, t_cut_mkn=1e-3),
            "VERY_TIGHT": dict(t_cut_pno=1e-10, t_cut_do=5e-3, t_cut_mkn=1e-4),
        }
        key = name.upper()
        if key not in presets:
            raise ValueError(f"unknown PNO_CONVERGENCE preset {name!r}")
        return replace(cls(), **{**presets[key], **overrides})

    @classmethod
    def untruncated(cls, **overrides):
        """Every truncation switched off.

        With full PAO and auxiliary domains this makes local MP2 exactly
        equivalent to canonical DF-MP2, which is how the port is validated.
        """
        return replace(
            cls(),
            t_cut_pno=0.0,
            t_cut_pno_diag_scale=1.0,
            t_cut_pno_core_scale=1.0,
            min_pnos=0,
            f_cut=0.0,
            # Negative rather than zero: the domain tests are on absolute
            # values, so a zero threshold would still drop an orbital whose
            # overlap or population happened to vanish exactly.
            t_cut_do=-1.0,
            t_cut_do_ij=-1.0,
            t_cut_do_pre=-1.0,
            t_cut_mkn=-1.0,
            t_cut_pre=-1.0,
            t_cut_clmo=-1.0,
            t_cut_cpao=-1.0,
            # Zero rather than negative, unlike the two above: this one gates a
            # product against tol^2, so zero already admits everything.
            ao_ints_tol=0.0,
            **overrides,
        )
