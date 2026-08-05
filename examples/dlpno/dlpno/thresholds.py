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

    # linear algebra cutoffs
    s_cut: float = 1e-8
    f_cut: float = 1e-5

    #: How many PNO-count buckets the pair blocks are padded into. Blocks must
    #: share a shape to batch, but padding everything to the global maximum
    #: wastes cubic flops; a handful of buckets keeps the batches large while
    #: cutting most of the waste. 1 reproduces a single padded store.
    n_buckets: int = 4

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
            **overrides,
        )
