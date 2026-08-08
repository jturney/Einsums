#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Cross-boundary contracts for the DLPNO stages that get promoted to C++.

A contract is a stage's signature expressed entirely in types that can cross
into C++, plus the rule each field is compared by in a differential test. These
are validated at import: ``@contract`` checks every field against the closed
type list, so a structure that cannot cross says so here rather than two
milestones later in the generator.

Structure of arrays, not array of structures. ``_couplings`` used to be a dict
from pair to a list of six-tuples, which crosses as a vector of vectors of
tuples and arrives in C++ as a pointer chase per coupling. One flat array per
field, all the same length and indexed by coupling ordinal, is the same
information in one allocation each, and it turns the batched-GEMM operand loop
into a linear scan. At 32,948 couplings that is the difference between the C++
stage reproducing Python's object graph and replacing it, which is the entire
reason this stage is being promoted.
"""

from dataclasses import dataclass

from einsums.stages import TensorD, cmp, contract

__all__ = ["CouplingPlan", "PnoOverlaps"]


@contract(parallel=["pair", "partner", "cls", "dest_slot", "src_slot", "sign", "factor"])
@dataclass(frozen=True)
class CouplingPlan:
    """Which pairs couple to which partners, and where every block goes.

    Pure index bookkeeping: no floating-point array appears here, and nothing in
    it depends on the values of the integrals. That is what makes it the natural
    seam. The planning is where the method's *logic* lives and is cheap; the
    numerics it describes are where the time goes.

    Two layouts are described at once, because the residual's two halves want
    opposite groupings and the couplings are a bipartite graph, so no single
    order serves both. ``dest_slot`` is the pair-major position, ``src_slot``
    the partner-major one, and ``inv_perm`` maps partner-major back to
    pair-major for the transposed copy.

    Structure of arrays, chosen for the C++ side's benefit: a dict of lists of
    tuples would arrive there as a pointer chase per coupling, where the flat
    arrays make the operand loop a linear scan over 32,948 of them.
    """

    # -- per shape class -------------------------------------------------
    #: ``(bucket of the pair, bucket of the partner)`` for each class, in
    #: ascending order. The class ORDINAL is the identity everywhere else; this
    #: is only how a class recovers its two block dimensions.
    classes: list[tuple[int, int]] = cmp.exact()
    #: Columns in the pair-major layout, excluding the reserved zero slot.
    slots_pair: list[int] = cmp.exact()
    #: Columns in the partner-major layout, including the reserved zero slot.
    slots_partner: list[int] = cmp.exact()
    #: For each partner-major slot, the pair-major slot it came from. Padding
    #: points at the reserved slot past the end of the pair-major side, so a
    #: gather writes its whole destination and the padded columns are genuinely
    #: zero rather than stale.
    inv_perm: list[list[int]] = cmp.exact()

    # -- per coupling, all the same length -------------------------------
    pair: list[int] = cmp.exact()
    partner: list[int] = cmp.exact()
    #: Class ordinal, indexing :attr:`classes` and every other per-class list.
    cls: list[int] = cmp.exact()
    dest_slot: list[int] = cmp.exact()
    src_slot: list[int] = cmp.exact()
    #: Sign of the Fock element. Rides on the transposed copy only: ``S``
    #: appears on both sides of the congruence, so a sign in both would square
    #: away.
    sign: list[float] = cmp.exact()
    #: The Fock element itself. Its square root scales the pair-major copy.
    factor: list[float] = cmp.exact()

    #: Pairs with at least one coupling, ascending. Derivable from :attr:`pair`,
    #: carried because every consumer needs it and rebuilding a sorted unique
    #: set per call is work the planner already did.
    coupled_pairs: list[int] = cmp.exact()


@contract
@dataclass(frozen=True)
class PnoOverlaps:
    """The overlap tensors, one per shape class, in both layouts.

    ``close`` rather than ``exact`` because the two backends reach these through
    different orderings of the same GEMMs, so agreement is to floating-point
    tolerance rather than bit for bit.
    """

    #: Pair-major, ``(M_b, M_p * (slots_pair + 1))`` per class, already scaled
    #: by ``sqrt|factor|``.
    S_cls: list[TensorD] = cmp.close()
    #: Partner-major, ``(M_p, M_b, slots_partner)`` per class, carrying
    #: ``sqrt|factor|`` from the copy it was lifted from and the sign of its own.
    S_T: list[TensorD] = cmp.close()
