# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Permutation operators for the differential fuzzers: the table, the spelling, the oracle.

An operator is ``(groups, shape_index, chosen)``: ``groups`` is the letter
partition the spec spells, ``shape_index`` picks a row of ``P_SHAPES`` and
``chosen`` lists the letters in placeholder order, so ``chosen[p]`` is the
letter that stands for placeholder ``p`` of that row.

Shared rather than copied because the expansion rows are the oracle. Two
fuzzers transcribing them twice would be two oracles, and the one that drifted
would pass against whichever pass happened to agree with it.
"""

from __future__ import annotations

import numpy as np

# Permutation operators, as SHAPES with their expansions written out.
#
# Each entry is (groups over placeholder positions, [(substitution, sign)]).
# A substitution maps placeholder p to placeholder ``perm[p]``, so the term
# ``(ij)`` of ``P(x0/x1 x2)`` is ``(1, 0, 2)``.
#
# Written out rather than computed, deliberately. An oracle that re-derived the
# coset representatives would only restate whatever the implementation does, and
# the representative is a CONVENTION: a coset holds permutations of both
# parities, so a wrong choice is still a valid antisymmetrizer and agrees with
# this one on every term that has the within-group symmetry the operator
# presumes. These rows are the literature's, transcribed by hand.
P_SHAPES = [
    # P(x0/x1) = 1 - (01)
    ([[0], [1]],
     [((0, 1), 1.0), ((1, 0), -1.0)]),
    # P(x0/x1 x2) = 1 - (01) - (02)
    ([[0], [1, 2]],
     [((0, 1, 2), 1.0), ((1, 0, 2), -1.0), ((2, 1, 0), -1.0)]),
    # P(x0 x1/x2) = 1 - (02) - (12)
    ([[0, 1], [2]],
     [((0, 1, 2), 1.0), ((2, 1, 0), -1.0), ((0, 2, 1), -1.0)]),
    # P(x0/x1/x2) = the full antisymmetrizer on three letters
    ([[0], [1], [2]],
     [((0, 1, 2), 1.0), ((1, 0, 2), -1.0), ((2, 1, 0), -1.0),
      ((0, 2, 1), -1.0), ((1, 2, 0), 1.0), ((2, 0, 1), 1.0)]),
    # P(x0 x1/x2 x3) = 1 - (02) - (03) - (12) - (13) + (02)(13)
    ([[0, 1], [2, 3]],
     [((0, 1, 2, 3), 1.0), ((0, 2, 1, 3), -1.0), ((0, 3, 2, 1), -1.0),
      ((2, 1, 0, 3), -1.0), ((3, 1, 2, 0), -1.0), ((2, 3, 0, 1), 1.0)]),
]


def operator_prefix(op):
    """The ``P(a/bc) `` spelling, empty when there is no operator.

    Comma-free inside the parentheses, which both spec modes accept: a
    character-mode spec reads each letter of a group, and a comma-mode spec
    whose letters are all single characters reads a comma-free group the same
    way. A comma inside the operator would instead put a character-mode
    operand into multi-character mode.
    """
    if op is None:
        return ""
    groups, _, _ = op
    return "P(" + "/".join("".join(g) for g in groups) + ") "


def apply_operator(op, c_idx, base, magnitude=False):
    """The antisymmetrized base result, by the table above.

    ``magnitude`` sums every term with a positive sign, which is the scale a
    floating-point error bound over the antisymmetrized sum is relative to: the
    signed sum can cancel to zero from terms that are not.
    """
    if op is None:
        return base
    _, shape_index, chosen = op
    acc = np.zeros_like(base)
    for perm, sign in P_SHAPES[shape_index][1]:
        sub    = {chosen[p]: chosen[perm[p]] for p in range(len(chosen))}
        term_c = [sub.get(x, x) for x in c_idx]
        axes   = [c_idx.index(letter) for letter in term_c]
        acc    = acc + (abs(sign) if magnitude else sign) * base.transpose(np.argsort(axes))
    return acc
