# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Every exposed name in the graph module can actually be reached from Python.

``APIARY_EXPOSE`` is a promise, and nothing checked that it was kept. A type
once reached Python exposed and unbindable: the annotation was there, the class
appeared in the module, and no caller could construct one or be handed one, so
the capability behind it was not shipped and the tests that would have found
that were never written, because the surface looked complete.

Two questions, both asked of the module and of the generated ``.pyi`` rather
than of the headers, because what a caller has is the module.

* Every exposed class is CONSTRUCTIBLE, or is an enum with members, or the stub
  hands one over: a return annotation or an attribute. A parameter does not
  count, and that is the distinction the whole test turns on. A class named
  only as an argument is one nothing produces, so a caller who cannot build one
  cannot call that function either.
* No exposed signature carries a raw C++ type. pybind renders an unregistered
  type by its C++ spelling, so ``::``, a template's angle brackets, a reference
  or ``const`` in a signature is the shape of a binding that renders and does
  not work.
"""

from __future__ import annotations

import enum
import inspect
import pathlib
import re

import pytest

import einsums.graph as cg
import einsums._core.graph as core

#: Classes with no constructor and nothing that hands one over, each with the
#: reason it is nonetheless exposed. Every entry is load-bearing: removing one
#: makes the first test below fail, an entry that stops being needed makes the
#: same test fail, and a name the module no longer exposes makes the second
#: fail, so the table cannot rot in any direction.
EXPECTED_UNCONSTRUCTIBLE = {
    "Executor": "abstract base; exposed so a concrete executor's binding has a bound direct base",
    "OptimizerPass": "abstract base; exposed so a concrete pass's binding has a bound direct base, "
                     "without which PassManager.add refuses the pass as an unrelated type",
    "RegionRewrite": "abstract base between OptimizerPass and its region-rewriting clients, for the same reason",
}

# ``FactorizationProvider`` is the case this test was expected to have to excuse:
# abstract with no trampoline, so nothing in Python subclasses it. It is NOT
# listed, because it turns out to be reachable after all - the registry hands
# the providers back as a list, so a caller can be given one and read it. The
# staleness check below is what says so, and it is why the entry is absent
# rather than present and unused.


def _stub_path():
    path = pathlib.Path(cg.__file__).with_suffix(".pyi")
    if not path.exists():
        pytest.skip(f"no generated stub beside {cg.__file__}")
    return path


def _signature_lines(text):
    """Every ``def`` and every attribute annotation, docstrings excluded.

    The exclusion is the whole difficulty: the doc comments carry reST blocks
    and C++ examples, so a scan that read them would report ``cg::PassManager
    pm;`` as a leaked type. The triple quote is toggled per DELIMITER rather
    than per line, because a tracker that closed on the first line ending in one
    desynchronised on those blocks and then skipped every real signature until
    the next. An attribute is matched with a single colon, which is also what
    keeps a line like ``std::cerr << ...`` out.
    """
    out = []
    in_doc = False
    for line in text.splitlines():
        started_inside = in_doc
        if line.count('"""') % 2 == 1:
            in_doc = not in_doc
        if started_inside:
            continue
        stripped = line.strip()
        if stripped.startswith('"""'):
            continue
        if stripped.startswith("def ") or re.match(r"^[A-Za-z_]\w*\s*:(?!:)\s*\S", stripped):
            out.append(stripped)
    return out


def _handed_over_annotations(text):
    """The annotations through which a caller can be HANDED a value.

    A ``def``'s return type, and an attribute's type. Not a parameter, for the
    reason the module docstring gives.
    """
    out = []
    for line in _signature_lines(text):
        if line.startswith("def "):
            arrow = line.rfind("->")
            if arrow != -1:
                out.append(line[arrow + 2:])
            continue
        out.append(line.split(":", 1)[1])
    return out


def _exposed_classes():
    return {name: getattr(core, name)
            for name in dir(core)
            if not name.startswith("_") and inspect.isclass(getattr(core, name))}


def _constructible(cls):
    """Whether a caller can build one, whatever arguments it takes.

    pybind says "No constructor defined!" for a class with no exposed init and
    "incompatible constructor arguments" for one whose init wants something
    else, and only the first is unreachable.
    """
    try:
        cls()
        return True
    except TypeError as error:
        return "No constructor defined" not in str(error)
    except Exception:
        return True


def test_every_exposed_class_can_be_constructed_or_handed_over():
    stub = _stub_path().read_text()
    handed = _handed_over_annotations(stub)
    assert len(handed) > 100, f"only {len(handed)} annotations found; the stub parser is looking at the wrong thing"

    classes = _exposed_classes()
    assert classes, "the graph module exposes no classes at all, which means this test is looking in the wrong place"

    unreachable = []
    for name, cls in sorted(classes.items()):
        if issubclass(cls, enum.Enum):
            assert list(cls), f"{name} is an enum with no members"
            continue
        if _constructible(cls):
            continue
        if any(re.search(rf"\b{re.escape(name)}\b", annotation) for annotation in handed):
            continue
        unreachable.append(name)

    unexpected = [name for name in unreachable if name not in EXPECTED_UNCONSTRUCTIBLE]
    assert not unexpected, (
        "exposed and unbindable: a caller can neither construct these nor be handed one\n"
        + "\n".join(f"  {name}" for name in unexpected)
    )

    # The other direction, so an entry that stops being needed is noticed rather
    # than left standing as a reason for nothing.
    stale = [name for name in EXPECTED_UNCONSTRUCTIBLE if name in classes and name not in unreachable]
    assert not stale, f"the exception list excuses classes that are now reachable: {stale}"


def test_the_exception_list_names_types_that_still_exist():
    """A reason for a class nobody exposes any more is a reason nobody reads."""
    classes = _exposed_classes()
    missing = sorted(name for name in EXPECTED_UNCONSTRUCTIBLE if name not in classes)
    assert not missing, f"the exception list names classes the module no longer exposes: {missing}"


def test_no_signature_in_the_stub_carries_a_raw_cpp_type():
    """The failure mode this file is named for.

    pybind renders an unregistered type by its C++ spelling, so the stub is
    where a binding that compiled and cannot be called shows itself. The arrow
    is removed before the scan because it is the one place a signature
    legitimately holds an angle bracket.
    """
    lines = _signature_lines(_stub_path().read_text())
    assert len(lines) > 100, f"only {len(lines)} signatures found; the stub parser is looking at the wrong thing"

    cpp = re.compile(r"::|[<>]|\bconst\b|\bunsigned\b|&|\bstd\b")
    leaked = [line for line in lines if cpp.search(line.replace("->", " "))]
    assert not leaked, (
        "a signature in the generated stub carries a raw C++ type, which is how pybind renders one "
        "it has no binding for\n" + "\n".join(f"  {line}" for line in leaked[:20])
    )


def test_every_exposed_function_is_documented():
    """A binding with no doc is one nobody can use without reading the header.

    Weaker than the two above and cheap: the module's convention is that every
    exposed entity carries its header's doc comment, so an empty one means the
    annotation landed on a declaration the doc did not.
    """
    undocumented = []
    for name in dir(core):
        if name.startswith("_"):
            continue
        attr = getattr(core, name)
        if inspect.isclass(attr) or not callable(attr):
            continue
        if not (attr.__doc__ or "").strip():
            undocumented.append(name)
    assert not undocumented, f"exposed functions with no documentation: {undocumented}"
