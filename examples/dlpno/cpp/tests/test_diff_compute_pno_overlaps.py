#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Scaffolded once by 'python -m einsums.stages promote'; this file is yours.

"""Differential test for the ``compute_pno_overlaps`` stage: python against cpp.

Fill in ``make_inputs``. Until you do, this test skips - visibly, naming the
one thing it needs - rather than passing and looking like coverage.

The comparison uses the contract's own ``cmp`` rules, so a field declared
``up_to_sign`` is compared that way here without this file knowing which field
that is. That is the point of putting the rules on the contract.
"""

import pytest

from einsums import stages

import dlpno.stages  # noqa: F401  (registers the python backend)


def make_inputs():
    """Return the keyword arguments to call the stage with.

    TODO: build inputs small enough for CI. The other path is
    ``stages.record(names=["compute_pno_overlaps"], out_dir=...)`` on a real run,
    which serializes inputs and outputs for replay and needs no synthetic case.
    """
    return None


@pytest.fixture(scope="module")
def stage_module():
    # Idempotent: another test file in the same pytest process may have
    # loaded the module already, and the registry (rightly) refuses a second
    # cpp registration for a stage.
    if "cpp" in stages.get_stage("compute_pno_overlaps").backends:
        return None
    return stages.load_stage_module("dlpno_stages")


def test_compute_pno_overlaps_backends_agree(stage_module):
    inputs = make_inputs()
    if inputs is None:
        pytest.skip("make_inputs() is not filled in yet")

    from einsums.stages import compare

    st = stages.get_stage("compute_pno_overlaps")

    st.select("python")
    expected = st.call(**inputs)
    st.select("cpp")
    actual = st.call(**inputs)

    compare.assert_contract_equal(expected, actual)
