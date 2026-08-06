# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""The sealed-world handshake, as Python sees it.

``einsums._core`` publishes which libEinsums it bound to. A compiled stage
module publishes the same thing, and the loader refuses it when the two
disagree; that refusal is the whole point, because a stage module bound to a
different copy of the library gets a different graph-capture context and
silently drops the nodes it emits.

These tests cannot create a second world in-process, so they pin the pieces the
refusal is built from: that the identity is stable and self-consistent, that the
fingerprints are present, and that registration is genuinely scoped to one
library rather than being a global set anyone can satisfy.
"""

import einsums
from einsums import _core


def test_world_identity_is_stable_and_nonzero():
    # The identity is an address inside libEinsums. Zero would mean the lookup
    # failed, and a value that changed between calls would make the comparison
    # a stage module performs meaningless.
    assert _core.__einsums_world__ != 0
    assert _core._world_info()["identity"] == _core.__einsums_world__
    assert _core._world_info()["identity"] == _core._world_info()["identity"]


def test_world_info_carries_the_fields_a_mismatch_report_needs():
    info = _core._world_info()

    for key in ("config_fingerprint", "layout_fingerprint"):
        # A zero fingerprint would compare equal against a module that failed
        # to compute one, which is the one case that must never pass.
        assert info[key] != 0, key

    assert info["version"].startswith(f"{info['version_major']}.{info['version_minor']}.")
    assert info["compiler"] != "Unknown"
    assert info["compiler_major"] > 0

    # The path is the field that turns "wrong world" into an actionable
    # message, so it must name a real library rather than being blank.
    assert "Einsums" in info["library_path"]


def test_this_process_maps_exactly_one_library():
    libs = _core._mapped_einsums_libraries()
    if not libs:
        # Documented outcome on a platform that cannot enumerate images, not a
        # failure. Anything else here would be a false multi-world alarm.
        return

    assert len(libs) == 1, f"expected one world, found: {libs}"
    assert libs[0] == _core._world_info()["library_path"]


def test_registration_is_scoped_to_this_library():
    name = "test_world_python_stage"

    assert not _core._stage_module_registered(name)
    assert _core._register_stage_module(name) is True
    assert _core._stage_module_registered(name) is True

    # Re-registering reports "already there" instead of duplicating: a module
    # may legitimately be imported more than once.
    assert _core._register_stage_module(name) is False

    # A name nobody registered stays unregistered. This is the case that makes
    # the check meaningful: a stage module bound to a DIFFERENT libEinsums
    # registers into that copy's table, so asking this one looks exactly like
    # asking about a name that was never registered at all.
    assert not _core._stage_module_registered("stage_from_another_world")


def test_handshake_needs_no_running_runtime():
    # Deliberate: the handshake has to work at import time, before
    # einsums.rc is applied and the runtime starts. If this ever requires a
    # started runtime, a stage module could not be validated until after it had
    # already been used.
    assert isinstance(einsums.__name__, str)
    assert _core._world_info()["identity"] != 0
