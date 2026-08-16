# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""``einsums.rc`` against the option descriptors it is generated from.

There used to be three hand-kept copies of the option list: rc.py's fields, the
listing of ``--help`` in its comments, and a table in ``PyEinsumsMain.cpp``
mapping each field to a flag spelling. Nothing checked any of them against the
descriptors, and all three had drifted -- flags in a polarity the runtime no
longer printed, two options missing outright.

rc.py is now generated from the descriptors by a static parse of the headers,
and the binding layer builds argv by walking the registry. These tests hold the
two ends together: the generated file comes from the source, the registry comes
from the running process, and anything that makes them disagree is the drift
this arrangement exists to prevent.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

import einsums
import einsums.rc as rc
from einsums import _core

# A sanitizer build needs its runtime preloaded before the interpreter dlopens
# the instrumented _core, and macOS strips DYLD_* from a signed binary's
# environment, so a child interpreter cannot inherit it. Same reasoning, and
# same skip, as test_cli_options_python.py.
_sanitizer_active = any(os.environ.get(v) for v in ("ASAN_OPTIONS", "TSAN_OPTIONS", "LSAN_OPTIONS"))

requires_child_preload = pytest.mark.skipif(
    sys.platform == "darwin" and _sanitizer_active,
    reason="macOS strips DYLD_* for signed binaries, so a child cannot inherit the sanitizer runtime",
)


def registry():
    """Every registered option, keyed by the rc attribute it maps to."""
    return {opt["attribute"]: opt for opt in _core._registered_options()}


def generated():
    """Every option rc.py was generated from, keyed by attribute."""
    return {opt["attribute"]: opt for opt in rc._OPTIONS}


@pytest.fixture
def pristine_rc():
    """Restore every rc field, so a test may set whatever it likes."""
    saved = {name: getattr(rc, name) for name in rc.__annotations__}
    yield
    for name, value in saved.items():
        setattr(rc, name, value)


# -- the generated surface against the registry ------------------------------


def test_generated_options_are_exactly_the_registered_ones():
    assert set(generated()) == set(registry())


@pytest.mark.parametrize("attribute", sorted(generated()))
def test_generated_option_matches_the_registry(attribute):
    """Name, kind, type, and default, as parsed from the header and as the
    process holds them. A computed default is exempt from the value check: it
    is produced by a function at registration, so the header has nothing to
    compare against."""
    want = generated()[attribute]
    got = registry()[attribute]

    assert want["name"] == got["name"]
    assert want["kind"] == got["kind"]
    assert want["type"] == got["type"]
    assert want["category"] == got["category"]
    assert want["computed_default"] == got["computed_default"]
    if not want["computed_default"]:
        assert want["default"] == got["default"]


# -- the module's own fields against the generated surface -------------------


def test_every_option_has_a_field():
    for attribute in generated():
        assert hasattr(rc, attribute), f"einsums.rc is missing a field for {attribute}"


def test_annotations_are_the_options_plus_threads():
    # The annotations are what the binding layer reads to spot a field no
    # option claims, so they have to be exactly the option surface. `threads`
    # is the one setting on the page that is not an option: einsums has no flag
    # for it and the binding routes it through OMP_NUM_THREADS. Underscored
    # names are the generated manifest, skipped on both sides.
    public = {name for name in rc.__annotations__ if not name.startswith("_")}
    assert public == set(generated()) | {"threads"}


def test_help_listing_names_every_option():
    """rc.py's comment block used to be a pasted transcript of ``--help`` and
    went stale. It is rendered from the descriptors now, so every option is in
    it by construction - which is worth asserting, because losing that is how
    the pasted version failed."""
    source = einsums.rc.__file__
    with open(source, encoding="utf-8") as f:
        text = f.read()
    for opt in generated().values():
        assert f"--{opt['name']}" in text, f"{opt['name']} is missing from rc.py's option listing"


# -- the rc -> argv translation ----------------------------------------------


def sample_value(opt):
    """A value of the right shape for an option, for the round trip below."""
    if opt["kind"] == "flag":
        return True
    return {"str": "sample", "int": 7, "float": 1.5}[opt["type"]]


def test_every_field_reaches_a_registered_flag(pristine_rc):
    """Set every field, then check that each argv entry the binding produces
    names an option the parser knows. The parser resolves a token by its long
    name, so a token whose name is in the registry is a token it accepts."""
    for attribute, opt in generated().items():
        setattr(rc, attribute, sample_value(opt))

    names = {opt["name"] for opt in registry().values()}
    argv = _core._argv_from_rc()

    assert argv[0] == "einsums-python"
    emitted = set()
    for token in argv[1:]:
        assert token.startswith("--")
        name = token[2:].split("=", 1)[0]
        assert name in names, f"{token} names no registered option"
        emitted.add(name)

    assert emitted == names


def test_false_asks_for_the_negation(pristine_rc):
    """A flag that already defaults to on cannot be turned off by staying
    silent, so ``False`` has to pass the negated spelling rather than nothing."""
    for attribute, opt in generated().items():
        setattr(rc, attribute, False if opt["kind"] == "flag" else None)

    argv = set(_core._argv_from_rc()[1:])
    for opt in registry().values():
        if opt["kind"] == "flag":
            assert f"--{opt['negated_name']}" in argv
            assert f"--{opt['name']}" not in argv


def test_none_emits_nothing(pristine_rc):
    for attribute in generated():
        setattr(rc, attribute, None)
    assert _core._argv_from_rc() == ["einsums-python"]


def test_log_level_accepts_the_enum_and_a_plain_int(pristine_rc):
    for attribute in generated():
        setattr(rc, attribute, None)

    rc.log_level = rc.LogLevel.WARN
    assert "--einsums:log:level=3" in _core._argv_from_rc()

    rc.log_level = 3
    assert "--einsums:log:level=3" in _core._argv_from_rc()


def test_a_field_no_option_claims_is_refused(pristine_rc):
    """Silence is what let the old hand-written table drift. An annotated field
    the registry does not claim is a field describing an option that no longer
    exists, and it is now an error rather than a no-op."""
    rc.__annotations__["not_an_option"] = "bool | None"
    try:
        with pytest.raises(RuntimeError, match="not_an_option"):
            _core._argv_from_rc()
    finally:
        del rc.__annotations__["not_an_option"]


# -- end to end, in a child interpreter --------------------------------------


_CHILD = """
import sys
import einsums
import einsums.rc as rc
rc.profile_filename = sys.argv[1]
rc.profile_append = False
rc.debug_attach_debugger = False
{setting}
einsums._core._initialize_from_rc()
assert einsums._core._is_initialized()
"""


_SETTINGS_CHILD = """
import einsums
import einsums.rc as rc
rc.log_level = rc.LogLevel.INFO
rc.debug_attach_debugger = False
rc.profile_port = 20117
rc.row_major = True
einsums._core._initialize_from_rc()
"""


@requires_child_preload
def test_settings_reach_the_runtime():
    """The whole chain, in one child: an rc field becomes a flag, the parser
    takes it, and the value lands in the option's slot. The runtime logs every
    option at INFO on the way up, which is what makes that observable.

    ``debug_attach_debugger`` is the one worth naming: it defaults to on, so
    turning it off is only expressible as the generated negation, and a
    Python process that fails to turn it off hangs inside
    ``attach_debugger()`` instead of reporting a crash."""
    result = subprocess.run(
        [sys.executable, "-c", _SETTINGS_CHILD],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for expected in ('"debug-attach-debugger": false',
                     '"profile-port": 20117',
                     '"row-major": true'):
        assert expected in result.stderr, result.stderr


@requires_child_preload
@pytest.mark.parametrize(
    "setting,expect_report",
    [
        ("", True),  # None leaves the runtime's default, which is on
        ("rc.profile_report = False", False),
        ("rc.profile_report = True", True),
    ],
)
def test_profile_report_is_three_valued(tmp_path, setting, expect_report):
    """The regression that renaming the negated fields exposed: ``False`` has to
    turn a default-on setting off, which silence cannot express."""
    report = tmp_path / "profile.txt"
    result = subprocess.run(
        [sys.executable, "-c", _CHILD.format(setting=setting), str(report)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert report.is_file() == expect_report, result.stderr
