# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python-side smoke tests for einsums.profile.

Mirrors what the C++ ProfileAnnotations / ProfileConsumer / ProfileRingBuffer
suites cover, but through the pybind surface:

  * ``section(...)`` context manager push/pops correctly (including nesting
    and exception-on-exit).
  * Each ``annotate`` overload (str / int / float) is dispatched correctly.
  * ``annotate_dims`` (Python helper) emits ``key.<i>`` entries.
  * ``mem_alloc`` / ``mem_free`` flow through without raising.
  * ``flush`` is a no-op when there is no pending data and is callable
    repeatedly.
  * ``print_report`` and ``export_json`` run to completion and the JSON file
    parses.
  * ``current_thread_id`` is stable across calls on the same thread.
  * ``set_thread_name`` does not raise.
  * Overhead and counter accessors return non-negative values that increase
    monotonically across push/pop pairs.
"""

from __future__ import annotations

import json
import pytest

import einsums.profile as prof


# ──────────────────────────────────────────────────────────────────────────
# section() context manager
# ──────────────────────────────────────────────────────────────────────────


def test_section_push_pop_balance():
    """A single section must increment push and pop counts by exactly one."""
    push_before = prof.total_push_count()
    pop_before = prof.total_pop_count()
    with prof.section("balance"):
        pass
    assert prof.total_push_count() == push_before + 1
    assert prof.total_pop_count() == pop_before + 1


def test_section_nesting_balance():
    """Nested sections push/pop in LIFO order."""
    push_before = prof.total_push_count()
    pop_before = prof.total_pop_count()
    with prof.section("outer"):
        with prof.section("inner-a"):
            pass
        with prof.section("inner-b"):
            with prof.section("inner-b-leaf"):
                pass
    assert prof.total_push_count() == push_before + 4
    assert prof.total_pop_count() == pop_before + 4


class _Sentinel(Exception):
    pass


def test_section_pops_on_exception():
    """contextmanager.__exit__ must call pop() even when the body raises."""
    push_before = prof.total_push_count()
    pop_before = prof.total_pop_count()
    with pytest.raises(_Sentinel):
        with prof.section("raises"):
            raise _Sentinel("boom")
    assert prof.total_push_count() == push_before + 1
    assert prof.total_pop_count() == pop_before + 1


def test_section_accepts_location_kwargs():
    """file/line/func kwargs are accepted (mirrors LabeledSection capture)."""
    with prof.section("loc", file=__file__, line=42, func="test_loc"):
        pass


# ──────────────────────────────────────────────────────────────────────────
# annotate overloads
# ──────────────────────────────────────────────────────────────────────────


def test_annotate_string_overload():
    with prof.section("ann-str"):
        prof.annotate("mode", "gemm")


def test_annotate_int_overload():
    with prof.section("ann-int"):
        prof.annotate("M", 512)
        prof.annotate("M", -1)  # ints can be negative


def test_annotate_float_overload():
    with prof.section("ann-float"):
        prof.annotate("alpha", 1.25)
        prof.annotate("beta", 0.0)


def test_annotate_dims_helper():
    """annotate_dims is a Python helper that fans out into per-axis ints."""
    with prof.section("ann-dims"):
        prof.annotate_dims("shape", [64, 32, 16])
        prof.annotate_dims("empty", [])  # zero-axis tensors


# ──────────────────────────────────────────────────────────────────────────
# Memory annotations
# ──────────────────────────────────────────────────────────────────────────


def test_mem_alloc_and_free():
    with prof.section("mem"):
        prof.mem_alloc(8 * 1024)
        prof.mem_free(8 * 1024)


# ──────────────────────────────────────────────────────────────────────────
# Drain / report / export
# ──────────────────────────────────────────────────────────────────────────


def test_flush_is_idempotent():
    prof.flush()
    prof.flush()
    prof.flush()


def test_print_report_runs(capfd):
    with prof.section("report-target"):
        prof.annotate("kind", "gemm")
    prof.flush()
    prof.print_report()
    out, _ = capfd.readouterr()
    assert "report-target" in out


def test_print_report_detailed_runs(capfd):
    with prof.section("detailed-target"):
        pass
    prof.flush()
    prof.print_report(detailed=True)
    out, _ = capfd.readouterr()
    assert "detailed-target" in out


def test_export_json_round_trip(tmp_path):
    with prof.section("export"):
        prof.annotate("M", 8)
        prof.annotate("alpha", 2.5)
    prof.flush()

    out = tmp_path / "profile.json"
    result = prof.export_json(str(out))

    assert result is not None
    assert out.exists()
    # Server-side JSON is per-thread newline-delimited or a single JSON tree;
    # either way it must parse.
    text = out.read_text()
    assert text
    # The implementation writes a JSON document — accept either a list or
    # an object at the top level. We don't pin the schema, only that it
    # parses and mentions the recorded section name somewhere.
    assert "export" in text
    json.loads(text)


# ──────────────────────────────────────────────────────────────────────────
# Thread metadata
# ──────────────────────────────────────────────────────────────────────────


def test_current_thread_id_stable():
    """Two calls on the same thread return the same id."""
    a = prof.current_thread_id()
    b = prof.current_thread_id()
    assert a == b
    assert a > 0  # platform-specific but never zero in practice


def test_set_thread_name_does_not_raise():
    prof.set_thread_name("pytest-worker")


# ──────────────────────────────────────────────────────────────────────────
# Overhead / counters
# ──────────────────────────────────────────────────────────────────────────


def test_counters_are_monotone_and_nonnegative():
    p_before = prof.total_push_count()
    q_before = prof.total_pop_count()
    with prof.section("counter-bump"):
        pass
    assert prof.total_push_count() >= p_before + 1
    assert prof.total_pop_count() >= q_before + 1
    assert prof.avg_push_overhead_ns() >= 0.0
    assert prof.avg_pop_overhead_ns() >= 0.0
