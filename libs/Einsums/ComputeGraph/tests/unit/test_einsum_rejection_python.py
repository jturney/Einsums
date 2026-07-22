# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Malformed / unsupported einsum specs must be REJECTED, not silently
miscomputed. Mirrors the intent of numpy's ``test_einsum_errors``: an output
label that appears in neither input, ellipsis broadcasting (unsupported by
einsums), and a link index whose extent disagrees between the operands all have
to raise rather than produce a wrong answer. Checked on both the eager
string-dispatch path and the graph capture/execute path.
"""
from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg


def _mk(nm, arr):
    t = einsums.create_zero_tensor(nm, list(arr.shape), dtype="float64")
    if arr.size:
        np.asarray(t)[...] = arr
    return t


def _run(mode, spec, C, A, B):
    if mode == "eager":
        einsums.einsum(spec, C, A, B)
    else:
        g = cg.Graph("rej")
        with cg.capture(g):
            einsums.einsum(spec, C, A, B)
        g.execute()


# (label, spec, a_shape, b_shape, c_shape). The C shape is only a plausible
# placeholder - the spec itself is what must be rejected.
_INVALID = [
    ("output label absent from inputs", "ijx <- ij ; jk", (2, 3), (3, 4), (2, 4, 5)),
    ("ellipsis unsupported",            "...ik <- ...ij ; jk", (2, 3, 4), (4, 5), (2, 3, 5)),
    ("link extent mismatch",            "ij <- ik ; kj", (2, 3), (4, 5), (2, 5)),
]


@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("label,spec,ash,bsh,csh", _INVALID, ids=[x[0] for x in _INVALID])
def test_invalid_einsum_spec_rejected(label, spec, ash, bsh, csh, mode):
    A = _mk("A", np.ones(ash))
    B = _mk("B", np.ones(bsh))
    C = _mk("C", np.zeros(csh))
    # std::invalid_argument -> ValueError, std::out_of_range -> IndexError,
    # raw throws -> RuntimeError; any of these is an acceptable rejection.
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        _run(mode, spec, C, A, B)
