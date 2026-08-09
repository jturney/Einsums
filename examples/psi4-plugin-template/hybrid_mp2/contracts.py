#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Cross-boundary contracts for the stages that get promoted to C++.

A contract is a stage's return type expressed entirely in types that can
cross into C++, plus the rule each field is compared by in a differential
test. ``@contract`` validates the fields at import, so a structure that
cannot cross says so here rather than at generation time.
"""

from dataclasses import dataclass

from einsums.stages import TensorD, cmp, contract

__all__ = ["Mp2Energy"]


@contract
@dataclass(frozen=True)
class Mp2Energy:
    """What the MP2 numerics stage produces.

    Both fields are length-1 tensors rather than floats, and that is a rule
    worth internalizing before writing your own contract: a captured stage
    returns before its graphs execute, so a computed ``float`` output would be
    read too early, and the contract validator refuses it. The stage writes
    its reductions into scalar tensors instead, and the caller reads them
    after the stage has run.
    """

    #: Total MP2 correlation energy, as a graph-written scalar tensor.
    e_corr: TensorD = cmp.close()
    #: Opposite-spin component; same-spin is ``e_corr - e_os``, subtracted by
    #: the caller host-side.
    e_os: TensorD = cmp.close()
