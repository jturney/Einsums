#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Tests for ``einsums.stages``: contracts, decoration-time validation, selection, sessions."""

from dataclasses import dataclass
from typing import Annotated

import pytest

import einsums  # noqa: F401  (loads the runtime)
import einsums.graph as cg
from einsums import stages
from einsums.stages import (
    ContractError,
    StageError,
    TensorD,
    cmp,
    contract,
    index,
    stage,
)


# ----------------------------------------------------------------------
# Contract types
# ----------------------------------------------------------------------
@contract
@dataclass(frozen=True)
class Basis:
    X: list[TensorD] = cmp.up_to_sign()
    n: list[int] = cmp.exact()
    e_tot: TensorD = cmp.close(atol=1e-10)


def test_contract_records_comparison_rules():
    rules = stages.comparison_rules(Basis)
    assert rules["X"].kind == "up_to_sign"
    assert rules["n"].kind == "exact"
    assert rules["e_tot"].kind == "close"
    assert rules["e_tot"].atol == 1e-10


def test_contract_defaults_close_for_tensors_and_exact_otherwise():
    @contract
    @dataclass(frozen=True)
    class Bare:
        t: TensorD = None
        k: int = 0

    rules = stages.comparison_rules(Bare)
    assert rules["t"].kind == "close"
    assert rules["k"].kind == "exact"


def test_contract_requires_frozen_dataclass():
    with pytest.raises(ContractError, match="frozen"):

        @contract
        @dataclass
        class Mutable:
            k: int = 0

    with pytest.raises(ContractError, match="dataclass"):

        @contract
        class NotADataclass:
            pass


def test_contract_rejects_a_field_that_cannot_cross():
    with pytest.raises(ContractError, match="cannot cross"):

        @contract
        @dataclass(frozen=True)
        class HasDict:
            d: dict = None


# ----------------------------------------------------------------------
# Decoration-time signature validation
# ----------------------------------------------------------------------
def test_stage_accepts_the_closed_type_list():
    @stage
    def ok(
        a: TensorD,
        k: int,
        i: Annotated[int, index],
        x: float,
        flag: bool,
        label: str,
        many: list[TensorD],
        pair: tuple[int, TensorD],
        basis: Basis,
    ) -> Basis: ...

    assert "ok" in stages.registered_stages()


def test_stage_rejects_an_unannotated_parameter():
    with pytest.raises(ContractError, match="no annotation"):

        @stage
        def bad(a) -> None: ...


def test_stage_rejects_a_missing_return_annotation():
    with pytest.raises(ContractError, match="no return annotation"):

        @stage
        def bad_ret(a: TensorD): ...


def test_stage_rejects_a_type_outside_the_closed_list():
    with pytest.raises(ContractError, match="cannot cross"):

        @stage
        def bad_type(a: dict) -> None: ...


def test_stage_rejects_varargs():
    with pytest.raises(ContractError, match=r"\*args"):

        @stage
        def bad_varargs(*args: int) -> None: ...


def test_stage_rejects_a_returned_computed_float():
    """A captured stage returns before its graph runs, so a computed float cannot exist."""
    with pytest.raises(ContractError, match="rank-0"):

        @stage
        def bad_scalar(a: TensorD) -> float: ...


def test_a_float_input_is_fine():
    @stage
    def cutoffs(a: TensorD, thresh: float) -> None: ...

    assert "cutoffs" in stages.registered_stages()


def test_returned_ints_stay_legal_because_shapes_are_known_at_capture():
    @stage
    def counts(a: TensorD) -> list[int]: ...

    assert "counts" in stages.registered_stages()


def test_a_returned_contract_is_checked_with_the_stricter_output_rules():
    @contract
    @dataclass(frozen=True)
    class HasComputedFloat:
        total: float = 0.0

    # Legal as an input...
    @stage
    def takes_it(c: HasComputedFloat) -> None: ...

    # ...and not as an output.
    with pytest.raises(ContractError, match="rank-0"):

        @stage
        def returns_it(a: TensorD) -> HasComputedFloat: ...


def test_promotable_false_skips_validation():
    @stage(promotable=False)
    def checkpoint(anything, at_all: dict) -> dict: ...

    assert stages.get_stage("checkpoint").promotable is False


def test_a_contract_defined_after_the_stage_validates_at_first_call():
    """Under string annotations the decorator sees names; a later contract defers."""

    @stage
    def later(a: TensorD) -> "DefinedBelow": ...

    assert stages.get_stage("later")._validated is False

    @contract
    @dataclass(frozen=True)
    class DefinedBelow:
        t: TensorD = None

    # Resolvable now, but only from this module's namespace; the deferred retry
    # happens on call, which is where get_type_hints can see it.
    globals()["DefinedBelow"] = DefinedBelow
    later(None)
    assert stages.get_stage("later")._validated is True


# ----------------------------------------------------------------------
# Registry and selection
# ----------------------------------------------------------------------
def test_selecting_an_unregistered_backend_raises_at_selection_time():
    @stage
    def only_python(a: TensorD) -> None: ...

    with pytest.raises(StageError, match="no cpp backend"):
        stages.select(only_python="cpp")


def test_selecting_an_unknown_stage_raises():
    with pytest.raises(StageError, match="no stage named"):
        stages.select(nonexistent_stage="python")


def test_select_default_is_loud_about_stages_that_lack_the_backend():
    with pytest.raises(StageError, match="have no cpp backend"):
        stages.select_default("cpp")


def test_two_different_functions_with_one_name_are_rejected():
    """Selection addresses stages by bare name, so the name has to be unique."""

    def site_a():
        @stage
        def collide(a: TensorD) -> None: ...

    def site_b():
        @stage
        def collide(a: TensorD) -> None: ...

    site_a()
    with pytest.raises(StageError, match="already registered"):
        site_b()


def test_redecorating_the_same_function_is_allowed():
    """Re-importing a stage module must not error; that is the reload path."""

    def make():
        @stage
        def reloadable(a: TensorD) -> None: ...

    make()
    make()
    assert "reloadable" in stages.registered_stages()


def test_backend_spec_parsing():
    @stage
    def spec_target(a: TensorD) -> None: ...

    stages.apply_backend_spec("spec_target=python")
    assert stages.selected_backend("spec_target") == "python"

    with pytest.raises(StageError, match="malformed"):
        stages.apply_backend_spec("spec_target=")


# ----------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------
def test_a_stage_called_outside_a_session_runs_eagerly():
    calls = []

    @stage
    def outside(a: TensorD) -> None:
        calls.append("ran")

    outside(None)
    assert calls == ["ran"]


def test_session_captures_into_one_segment_and_splits_only_on_eager():
    A = einsums.RuntimeTensorD("A", [4, 4])
    B = einsums.RuntimeTensorD("B", [4, 4])
    C = einsums.RuntimeTensorD("C", [4, 4])
    A.set_all(1.0)
    B.set_all(2.0)
    C.set_all(0.0)

    @stage
    def first(x: TensorD, y: TensorD, out: TensorD) -> None:
        einsums.linalg.gemm(1.0, x, y, 0.0, out)

    @stage
    def second(x: TensorD, y: TensorD, out: TensorD) -> None:
        einsums.linalg.gemm(1.0, x, y, 1.0, out)

    s = stages.session("two_stages")
    with s.capture():
        first(A, B, C)
        second(A, B, C)

    assert len(s.segments) == 1, "captured stages must share a segment"
    assert s.segments[0].num_nodes() == 2
    s.run()

    # 4x4 of ones times twos = 8 per element, applied twice.
    assert C[0, 0] == pytest.approx(16.0)
    assert all(not r.split for r in s.timings)


def test_an_eager_stage_forces_a_segment_boundary():
    A = einsums.RuntimeTensorD("A", [4, 4])
    C = einsums.RuntimeTensorD("C", [4, 4])
    A.set_all(1.0)
    C.set_all(0.0)
    seen = []

    @stage
    def before(x: TensorD, out: TensorD) -> None:
        einsums.linalg.gemm(1.0, x, x, 0.0, out)

    @stage(eager=True)
    def peek(out: TensorD) -> None:
        # Runs outside capture, so the pending segment must already have run.
        seen.append(out[0, 0])

    @stage
    def after(x: TensorD, out: TensorD) -> None:
        einsums.linalg.gemm(1.0, x, x, 1.0, out)

    s = stages.session("split")
    with s.capture():
        before(A, C)
        peek(C)
        after(A, C)
    s.run()

    assert len(s.segments) == 2
    assert seen == [4.0], "the eager stage must see materialized inputs"
    assert C[0, 0] == pytest.approx(8.0)
    assert [r.split for r in s.timings] == [False, True, False]


def test_report_names_every_stage_and_flags_splits():
    A = einsums.RuntimeTensorD("A", [2, 2])
    A.set_all(1.0)

    @stage
    def captured(x: TensorD) -> None:
        einsums.linalg.gemm(1.0, x, x, 0.0, x)

    @stage(eager=True, promotable=False)
    def eager_only(x: TensorD) -> None: ...

    s = stages.session("report")
    with s.capture():
        captured(A)
        eager_only(A)
    s.run()

    text = s.report()
    assert "captured" in text
    assert "eager_only" in text
    assert "SPLIT" in text
    assert "python-only" in text


def test_run_while_capturing_is_refused():
    s = stages.session("bad_order")
    with pytest.raises(RuntimeError, match="still capturing"):
        with s.capture():
            s.run()


def test_nested_sessions_are_refused():
    outer = stages.session("outer")
    inner = stages.session("inner")
    with pytest.raises(RuntimeError, match="already capturing"):
        with outer.capture():
            with inner.capture():
                pass


def test_session_uses_a_pipeline_and_one_stage_per_segment():
    s = stages.session("shape")

    @stage(eager=True)
    def split_here() -> None: ...

    with s.capture():
        split_here()
    assert isinstance(s.pipeline, cg.Pipeline)
    assert len(s.segments) == 2


def test_a_bad_env_var_says_it_came_from_the_env(monkeypatch):
    """Env selection is applied lazily, so the error must name its source."""
    import einsums.stages._registry as reg

    monkeypatch.setenv("EINSUMS_STAGE_BACKEND", "not_a_stage=cpp")
    monkeypatch.setattr(reg._registry, "_env_applied", False)

    @stage
    def env_probe(a: TensorD) -> None: ...

    with pytest.raises(StageError, match="EINSUMS_STAGE_BACKEND"):
        stages.select(env_probe="python")


# ----------------------------------------------------------------------
# contract=False: on the promotion path, contract not stated yet
# ----------------------------------------------------------------------
def test_contract_false_skips_validation_but_stays_promotable():
    @stage(contract=False)
    def not_yet_cut(mp2) -> None: ...

    st = stages.get_stage("not_yet_cut")
    assert st.promotable is True, "contract=False is a debt, not a retirement"
    assert st.contracted is False


def test_contract_false_is_reported_as_debt_not_as_python_only():
    """The two states must not read the same: one is owed work, one is finished."""

    @stage(eager=True, contract=False)
    def owes_a_contract(mp2) -> None: ...

    @stage(eager=True, promotable=False)
    def never_promoted(mp2) -> None: ...

    s = stages.session("debt")
    with s.capture():
        owes_a_contract(None)
        never_promoted(None)
    s.run()

    text = s.report()
    assert "NO CONTRACT" in text
    assert "python-only" in text
    assert "have not stated a contract yet: owes_a_contract" in text
    assert "never_promoted" not in text.split("have not stated a contract yet")[1]


def test_an_all_eager_session_does_not_blame_the_profiler():
    """Every row's execute time is wall time, so per-node timing was never asked for."""

    @stage(eager=True, contract=False)
    def only_eager(x) -> None: ...

    s = stages.session("all_eager")
    with s.capture():
        only_eager(None)
    s.run()
    assert "profiler" not in s.report()


def test_functools_wraps_over_a_stage_says_what_it_did():
    """wraps copies __annotations__, silently erasing the contract underneath.

    The resulting error is "parameter has no annotation", which is true and
    gives no hint at the cause. This was hit for real while contracting the
    DLPNO overlap stage.
    """
    import functools

    def unannotated(a, b):
        return None

    with pytest.raises(ContractError, match="functools.wraps"):

        @stage
        @functools.wraps(unannotated)
        def wrapped(a: TensorD, b: TensorD) -> None: ...
