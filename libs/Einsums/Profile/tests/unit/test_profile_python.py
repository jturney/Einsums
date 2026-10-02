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

A build with ``EINSUMS_WITH_PROFILER=OFF`` keeps the whole surface callable but
records nothing, so the cases that read something back are skipped there and
replaced by ``test_disabled_build_records_nothing``. Everything else runs in
both configurations: that the API stays callable is the point of the no-op
build, not an accident of it.
"""

from __future__ import annotations

import json
import pytest

import einsums.profile as prof

recording = pytest.mark.skipif(
    not prof.available(),
    reason="requires EINSUMS_WITH_PROFILER=ON; this build records nothing",
)


# ──────────────────────────────────────────────────────────────────────────
# section() context manager
# ──────────────────────────────────────────────────────────────────────────


@recording
def test_section_push_pop_balance():
    """A single section must increment push and pop counts by exactly one."""
    push_before = prof.total_push_count()
    pop_before = prof.total_pop_count()
    with prof.section("balance"):
        pass
    assert prof.total_push_count() == push_before + 1
    assert prof.total_pop_count() == pop_before + 1


@recording
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


@recording
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
    """file/line/func kwargs are accepted (mirrors WAGGLE_ZONE capture)."""
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


@recording
def test_print_report_runs(capfd):
    with prof.section("report-target"):
        prof.annotate("kind", "gemm")
    prof.flush()
    prof.print_report()
    out, _ = capfd.readouterr()
    assert "report-target" in out


@recording
def test_print_report_detailed_runs(capfd):
    with prof.section("detailed-target"):
        pass
    prof.flush()
    prof.print_report(detailed=True)
    out, _ = capfd.readouterr()
    assert "detailed-target" in out


@recording
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
    # The implementation writes a JSON document, accept either a list or
    # an object at the top level. We don't pin the schema, only that it
    # parses and mentions the recorded section name somewhere.
    assert "export" in text
    json.loads(text)


# ──────────────────────────────────────────────────────────────────────────
# Thread metadata
# ──────────────────────────────────────────────────────────────────────────


def test_current_thread_id_stable():
    """Two calls on the same thread return the same id."""
    assert prof.current_thread_id() == prof.current_thread_id()


@recording
def test_current_thread_id_is_registered():
    """A recording build hands every thread a real id."""
    assert prof.current_thread_id() > 0  # platform-specific but never zero


def test_set_thread_name_does_not_raise():
    prof.set_thread_name("pytest-worker")


# ──────────────────────────────────────────────────────────────────────────
# Overhead / counters
# ──────────────────────────────────────────────────────────────────────────


@recording
def test_counters_are_monotone_and_nonnegative():
    p_before = prof.total_push_count()
    q_before = prof.total_pop_count()
    with prof.section("counter-bump"):
        pass
    assert prof.total_push_count() >= p_before + 1
    assert prof.total_pop_count() >= q_before + 1
    assert prof.avg_push_overhead_ns() >= 0.0
    assert prof.avg_pop_overhead_ns() >= 0.0


# ──────────────────────────────────────────────────────────────────────────
# EINSUMS_WITH_PROFILER=OFF
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(prof.available(), reason="requires EINSUMS_WITH_PROFILER=OFF")
def test_disabled_build_records_nothing(tmp_path, capfd):
    """The readers stay callable and stay empty, rather than raising.

    Everything above already established that the writers are callable here;
    this pins down what the readers say afterwards, so that a build with the
    profiler compiled out degrades to silence instead of to an error.
    """
    with prof.section("nothing"):
        prof.annotate("kind", "gemm")
        prof.mem_alloc(1024)
    prof.flush()

    assert prof.total_push_count() == 0
    assert prof.total_pop_count() == 0
    assert prof.avg_push_overhead_ns() == 0.0
    assert prof.avg_pop_overhead_ns() == 0.0
    assert prof.current_thread_id() == 0

    out_path = tmp_path / "profile.json"
    assert prof.export_json(str(out_path)) is None
    assert not out_path.exists()

    prof.print_report(detailed=True)
    out, _ = capfd.readouterr()
    assert out == ""


# ──────────────────────────────────────────────────────────────────────────
# Zones opened by the library nest under zones opened from Python
# ──────────────────────────────────────────────────────────────────────────


@recording
def test_library_zones_nest_under_a_python_section(tmp_path):
    """A section opened here and the zones cg::einsum opens inside libEinsums are one tree.

    The per-thread channel used to be reached through an inline function holding a
    thread_local, and every module is built with -fvisibility-inlines-hidden, so the
    bindings and the library each had their own channel for the thread. Their zones
    never nested: cg::einsum landed at the root of the report, beside the section
    that called it rather than under it.
    """
    import numpy as np

    import einsums

    def tensor(name, shape):
        t = einsums.create_zero_tensor(name, list(shape), dtype="float64")
        np.asarray(t)[...] = 1.0
        return t

    A, B, C = tensor("A", (3, 4)), tensor("B", (4, 5)), tensor("C", (3, 5))
    with prof.section("python section around einsum"):
        einsums.einsum("ij <- ik ; kj", C, A, B)
    prof.flush()

    out = prof.export_json(str(tmp_path / "profile.json"))
    tree = json.loads(open(out).read())

    def find(node, name):
        if node.get("name") == name:
            return node
        for child in node.get("children", []):
            found = find(child, name)
            if found is not None:
                return found
        return None

    def names_below(node):
        for child in node.get("children", []):
            yield child["name"]
            yield from names_below(child)

    section = next(filter(None, (find(root, "python section around einsum") for root in tree.values())), None)
    assert section is not None
    assert any(name.startswith("cg::einsum:") for name in names_below(section))


# ──────────────────────────────────────────────────────────────────────────
# The profile decorator
# ──────────────────────────────────────────────────────────────────────────


def _zone_calls(tmp_path, name):
    """Calls of every zone called ``name``, across threads, from an exported report."""
    prof.flush()
    out = prof.export_json(str(tmp_path / "profile.json"))
    tree = json.loads(open(out).read())

    def total(node):
        return (node.get("call_count", 0) if node.get("name") == name else 0) + sum(total(c) for c in node.get("children", []))

    return sum(total(root) for root in tree.values())


@recording
def test_decorator_records_each_call_under_the_qualified_name(tmp_path):
    @prof.profile
    def decorated_bare(x):
        return 2 * x

    before = _zone_calls(tmp_path, decorated_bare.__qualname__)
    assert decorated_bare(21) == 42
    assert decorated_bare(1) == 2
    assert _zone_calls(tmp_path, decorated_bare.__qualname__) == before + 2


@recording
def test_decorator_takes_a_zone_name(tmp_path):
    @prof.profile(name="decorated with a name")
    def decorated_named():
        return "ok"

    before = _zone_calls(tmp_path, "decorated with a name")
    assert decorated_named() == "ok"
    assert _zone_calls(tmp_path, "decorated with a name") == before + 1


@recording
def test_decorator_closes_its_zone_when_the_function_raises():
    @prof.profile
    def decorated_raises():
        raise _Sentinel("boom")

    push_before = prof.total_push_count()
    pop_before = prof.total_pop_count()
    with pytest.raises(_Sentinel):
        decorated_raises()
    assert prof.total_push_count() == push_before + 1
    assert prof.total_pop_count() == pop_before + 1


def test_decorator_keeps_the_function_s_identity():
    def original(a, b=2):
        """The docstring."""
        return a + b

    wrapped = prof.profile(original)
    assert wrapped.__name__ == "original"
    assert wrapped.__doc__ == "The docstring."
    assert wrapped.__wrapped__ is original
    assert wrapped(1) == 3


def test_decorator_on_methods():
    class Holder:
        @prof.profile
        def method(self, x):
            return x + 1

        @staticmethod
        @prof.profile
        def static(x):
            return x + 2

        @classmethod
        @prof.profile
        def klass(cls, x):
            return x + 3

    assert Holder().method(1) == 2
    assert Holder.static(1) == 3
    assert Holder.klass(1) == 4


def test_decorator_refuses_coroutines_and_generators():
    """A zone held open across an await or a yield would not close before zones opened after it."""

    async def coroutine():
        return 1

    def generator():
        yield 1

    async def async_generator():
        yield 1

    for func in (coroutine, generator, async_generator):
        with pytest.raises(TypeError, match="coroutine or generator"):
            prof.profile(func)
