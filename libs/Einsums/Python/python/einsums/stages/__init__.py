#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Write a method in Python, measure it, and promote the slow stages to C++.

A **stage** is a free function registered with :func:`stage`. It has a Python
backend always, and a C++ backend once a compiled stage module is loaded.
Which one runs is always an explicit choice, never a fallback, so a build
glitch cannot quietly change which code computed a published number.

A **session** runs stages against a sequence of graph segments and reports what
each one actually cost::

    from einsums import stages
    from einsums.stages import stage, contract, cmp, TensorD

    @stage
    def prep_sparsity(ref: Reference) -> Domains:
        ...

    s = stages.session("dlpno", passes=cg.default_pass_manager())
    with s.capture():
        dom = prep_sparsity(ref)
        pno = pno_transform(dom, ...)
    s.run()
    print(s.report())

Nothing here requires C++ to be useful. Stage decomposition and the timing
table are how a developer finds out which stages are worth promoting, which is
the question that has to be answered before any C++ gets written.

The world-identity machinery that :func:`load_stage_module` relies on lives in
:mod:`einsums.sealed`, because two mapped copies of libEinsums are a fact about
the process rather than about stages.
"""

from ._contract import (
    ComparisonRule,
    ContractError,
    TensorC,
    TensorD,
    TensorF,
    TensorZ,
    cmp,
    comparison_rules,
    contract,
    index,
    is_contract,
    validate_signature,
)
from ._registry import (
    Stage,
    StageError,
    apply_backend_spec,
    get_stage,
    registered_stages,
    select,
    select_default,
    selected_backend,
    stage,
)
from ._session import Session, StageTiming, active_session, session

__all__ = [
    # contracts
    "ComparisonRule",
    "ContractError",
    "TensorC",
    "TensorD",
    "TensorF",
    "TensorZ",
    "cmp",
    "comparison_rules",
    "contract",
    "index",
    "is_contract",
    "validate_signature",
    # registry and selection
    "Stage",
    "StageError",
    "apply_backend_spec",
    "get_stage",
    "registered_stages",
    "select",
    "select_default",
    "selected_backend",
    "stage",
    # session
    "Session",
    "StageTiming",
    "active_session",
    "session",
]
