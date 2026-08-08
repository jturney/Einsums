#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""World identity: which copy of libEinsums a thing is bound to, and whether that is the right one.

A **world** is one copy of libEinsums plus everything dynamically linked
against it. A process can hold several: a host application may embed its own
private copy while the developer imports a newer one. That is legal and
supported. The rule that makes it safe is that typed Einsums objects never
cross worlds, only neutral buffers do.

This module is the Python face of ``einsums::sealed``. It answers three
questions:

* Which library am I bound to? :func:`world`, :func:`world_identity`.
* How many copies are in this process? :func:`mapped_einsums_libraries`.
* Is this compiled stage module bound to *my* copy, and built against *my*
  headers? :func:`verify_stage_module`.

It lives outside :mod:`einsums.stages` on purpose. Two mapped copies of
libEinsums are a fact about the process rather than about stages, and the
pure-Python user who trips over a host version-skew warning has no stages and
should not have to import a package about them to read the diagnosis.
"""

import warnings as _warnings

from . import _core as _c

__all__ = [
    "WorldMismatch",
    "world",
    "world_identity",
    "mapped_einsums_libraries",
    "registered_stage_modules",
    "verify_stage_module",
    "register_host_probe",
    "host_versions",
    "warn_on_skew",
]


class WorldMismatch(RuntimeError):
    """A compiled module is bound to a different libEinsums, or built against different headers."""


def world() -> dict:
    """Identity and build fingerprint of the libEinsums this interpreter is bound to."""
    return _c._world_info()


def world_identity() -> int:
    """A value unique to this copy of libEinsums, for a cheap same-library test."""
    return _c.__einsums_world__


def mapped_einsums_libraries() -> list[str]:
    """Paths of every libEinsums mapped into this process.

    More than one entry means more than one world. An empty list means the
    platform offers no way to enumerate loaded images, not that there are none.
    """
    return list(_c._mapped_einsums_libraries())


def registered_stage_modules() -> list[str]:
    """Stage modules that registered against *this* libEinsums, in load order."""
    return list(_c._registered_stage_modules())


def _std_name(cplusplus: int) -> str:
    """``__cplusplus`` as the flag a reader would type to reproduce it."""
    return {
        199711: "c++98", 201103: "c++11", 201402: "c++14",
        201703: "c++17", 202002: "c++20", 202302: "c++23",
    }.get(int(cplusplus), str(cplusplus))


def _diagnosis() -> str:
    """The paths worth printing whenever a handshake fails."""
    libs = mapped_einsums_libraries()
    if len(libs) > 1:
        listed = "\n  ".join(libs)
        return f"\n\nThis process has {len(libs)} mapped libEinsums:\n  {listed}"
    if libs:
        return f"\n\nThis process has one mapped libEinsums: {libs[0]}"
    return ""


def verify_stage_module(module, *, name: str | None = None) -> None:
    """Raise unless *module* is bound to this libEinsums and built against these headers.

    Runs both detection mechanisms, cheapest and most trustworthy first.

    Registration by side effect is checked before anything else. A stage module
    built with the SDK macro calls ``register_stage_module`` into whichever
    libEinsums it actually resolved to, so a module that bound to a different
    copy wrote into that copy's table and this one sees nothing. That holds
    without trusting a pointer comparison to survive whatever the loader did,
    and without depending on the module exporting any attribute.

    The fingerprints are then compared to catch the single-world case: one
    library, but a module compiled against stale headers, a different BLAS
    integer width, or an opposite iterator-debug setting. Those corrupt rather
    than misbehave, and nothing about having one world protects against them.
    """
    name = name or getattr(module, "__name__", None)
    if not name:
        raise WorldMismatch("verify_stage_module needs a module or an explicit name")
    short = name.rsplit(".", 1)[-1]

    identity = getattr(module, "__einsums_world__", None)
    info = getattr(module, "__einsums_world_info__", None)

    if identity is None and info is None and not _c._stage_module_registered(short):
        raise WorldMismatch(
            f"{name} does not look like an Einsums stage module: it neither registered "
            f"against this libEinsums nor exports __einsums_world__. Build it with the SDK "
            f"handshake macro in its bindings.cpp." + _diagnosis()
        )

    if not _c._stage_module_registered(short):
        labels = ("einsums._core is bound to", f"{name} reports")
        pad = max(len(lbl) for lbl in labels)
        raise WorldMismatch(
            f"{name} is bound to a different copy of libEinsums than einsums._core.\n"
            f"Its registration went into that copy's table, so this one never saw it.\n"
            f"  {labels[0]:<{pad}}: {world().get('library_path', '<unknown>')}\n"
            f"  {labels[1]:<{pad}}: {(info or {}).get('library_path', '<unknown>')}"
            + _diagnosis()
            + "\n\nTyped Einsums objects cannot cross worlds. Rebuild the stage module "
            "against the same installation einsums._core came from."
        )

    if identity is not None and identity != world_identity():
        raise WorldMismatch(
            f"{name} registered here but reports a different world identity "
            f"({identity:#x} against {world_identity():#x}), which should be impossible. "
            f"Treat this as a corrupt build rather than a skew." + _diagnosis()
        )

    if info:
        mine = world()
        for key, what in (
            ("config_fingerprint", "build configuration"),
            ("layout_fingerprint", "type layout"),
        ):
            theirs = info.get(key)
            if theirs is not None and theirs != mine.get(key):
                std = info.get("cplusplus")
                std_note = f", -std={_std_name(std)}" if std else ""
                raise WorldMismatch(
                    f"{name} was compiled against headers whose {what} does not match this "
                    f"libEinsums ({key} {theirs:#x} against {mine.get(key):#x}).\n"
                    f"  library: {mine.get('library_path', '<unknown>')} "
                    f"({mine.get('version')}, {mine.get('compiler')})\n"
                    f"  module:  built against {info.get('version', '<unknown>')}, "
                    f"{info.get('compiler', '<unknown>')}{std_note}\n"
                    f"Stale headers, a different language level, a different BLAS integer "
                    f"width, or a debug/release mismatch all land here. Rebuild the module "
                    f"against the installed headers, with the flags the library was built with."
                )


# ----------------------------------------------------------------------
# Host skew warning
# ----------------------------------------------------------------------
_host_probes: dict[str, object] = {}


def register_host_probe(name: str, fn) -> None:
    """Register a probe reporting the Einsums version a host embeds.

    The probe is only ever called when the host is already imported. The bridge
    must not import a host to ask it a question, so a probe registers itself
    from inside the host's own interop module (``einsums.interop.psi4`` ships
    the psi4 one) rather than being hardcoded here. That keeps "psi4 is one
    instance of the pattern" true in the code and not only in the prose.

    Args:
        name: The host's top-level module name, checked against ``sys.modules``.
        fn: Called with no arguments, returns a version string or None.
    """
    _host_probes[name] = fn


def host_versions() -> dict[str, str]:
    """Embedded Einsums versions reported by hosts that are already imported."""
    import sys

    out = {}
    for name, fn in _host_probes.items():
        if name not in sys.modules:
            continue
        try:
            v = fn()
        except Exception:  # a probe must never break the caller
            continue
        if v:
            out[name] = v
    return out


_warned = False


def warn_on_skew() -> None:
    """Warn once if this process holds more than one world. Skew is legal, not an error."""
    global _warned
    if _warned:
        return
    libs = mapped_einsums_libraries()
    if len(libs) > 1:
        _warned = True
        hosts = host_versions()
        detail = f" Hosts reporting an embedded Einsums: {hosts}." if hosts else ""
        _warnings.warn(
            f"This process has {len(libs)} mapped libEinsums: {', '.join(libs)}."
            f"{detail} That is supported, but typed Einsums objects must not cross "
            f"between them; pass buffers instead.",
            RuntimeWarning,
            stacklevel=2,
        )
