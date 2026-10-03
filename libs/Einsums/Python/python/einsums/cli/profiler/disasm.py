# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Disassembling a profiled function from the program's binaries.

A zone records ``__func__``, the bare function name. ``nm`` lists each binary's symbols
twice, mangled and demangled in the same order, which finds the mangled name of a text
symbol whose demangled name is ``...name(``; ``objdump --disassemble=<mangled>`` then
decodes only that symbol. Disassembling a whole library to search it would take minutes
for a library the size of libEinsums.

The tools run as asyncio subprocesses, killed if the request is cancelled (the user moved
to another row), so a slow binary never holds the app.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

_TEXT_TYPES = frozenset("TtWw")
_cache: dict[tuple[str, str], str] = {}


def _tool(*names: str) -> str | None:
    return next((path for name in names if (path := shutil.which(name))), None)


def candidate_binaries(executable: str | Path, prefer: Sequence[str] = ()) -> list[Path]:
    """The executable, then shared libraries beside it and in ``../lib``.

    Libraries named for *prefer* (the profiled program's clients, as ``einsums`` names
    ``libEinsums``) come first: they hold the instrumented functions. Symlinks are skipped, so
    ``libX.so -> libX.so.1 -> libX.so.1.2`` is searched once.
    """
    wanted = tuple(f"lib{name.lower()}" for name in prefer if name)
    exe = Path(executable)
    libs: list[Path] = []
    for directory in (exe.parent, exe.parent.parent / "lib"):
        if directory.is_dir():
            libs += [
                entry
                for entry in directory.iterdir()
                if (entry.suffix in (".so", ".dylib") or ".so." in entry.name) and not entry.is_symlink()
            ]
    return [exe] + sorted(set(libs), key=lambda p: (not p.name.lower().startswith(wanted) if wanted else True, p.name))


async def _run(*cmd: str, stdin: str | None = None, timeout: float = 60.0) -> str | None:
    """The command's stdout, or None if it failed. Killed when cancelled or timed out."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(stdin.encode() if stdin is not None else None), timeout)
    except asyncio.TimeoutError:
        return None
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
    return out.decode(errors="replace") if proc.returncode == 0 else None


def match_symbol(mangled_listing: str, demangled_listing: str, func_name: str) -> str | None:
    """The mangled name of the first text symbol whose demangled name is ``[scope::]func_name(...)``.

    Both listings are ``nm --defined-only`` output of one binary, without and with ``-C``.
    """
    pattern = re.compile(rf"(^|::){re.escape(func_name)}(\(|$)")
    for raw, pretty in zip(mangled_listing.splitlines(), demangled_listing.splitlines()):
        raw_fields, pretty_fields = raw.split(maxsplit=2), pretty.split(maxsplit=2)
        if len(raw_fields) == 3 and len(pretty_fields) == 3 and raw_fields[1] in _TEXT_TYPES:
            if pattern.search(pretty_fields[2]):
                return raw_fields[2]
    return None


def strip_listing(objdump_output: str) -> str:
    """The symbol's block of an ``objdump -d`` listing, without the file and section headers."""
    lines = objdump_output.splitlines()
    start = next((i for i, line in enumerate(lines) if line.rstrip().endswith(">:")), len(lines))
    body = lines[start:]
    while body and not body[-1].strip():
        body.pop()
    return "\n".join(body)


async def disassemble(func_name: str, executable: str, prefer: Sequence[str] = ()) -> str:
    """The disassembly of *func_name*, or a sentence saying why there is none. *prefer* names the
    libraries to search first (see :func:`candidate_binaries`)."""
    key = (func_name, executable)
    if key in _cache:
        return _cache[key]
    if not executable or not Path(executable).is_file():
        return f"Executable not available locally: {executable or '(unknown)'}"
    nm, objdump = _tool("nm", "llvm-nm"), _tool("objdump", "llvm-objdump")
    if nm is None or objdump is None:
        return "Disassembly needs nm and objdump (or their llvm- versions) on PATH"
    # GNU objdump spells it --disassemble=SYM; LLVM's (macOS's objdump) --disassemble-symbols=SYM.
    cxxfilt = _tool("c++filt", "llvm-cxxfilt")
    is_llvm = "LLVM" in (await _run(objdump, "--version") or "")
    select = "--disassemble-symbols=" if is_llvm else "--disassemble="
    result = f"No symbol named {func_name} in the executable or the libraries beside it (it may have been inlined)"
    for binary in candidate_binaries(executable, prefer):
        mangled = await _run(nm, "--defined-only", str(binary))
        demangled = await _run(nm, "--defined-only", "-C", str(binary)) if mangled else None
        symbol = match_symbol(mangled or "", demangled or "", func_name)
        if symbol is None:
            continue
        # Not -C: GNU objdump then matches the selection against demangled names. Demangle after.
        listing = await _run(objdump, "-d", "--no-show-raw-insn", f"{select}{symbol}", str(binary))
        if listing and cxxfilt:
            listing = await _run(cxxfilt, stdin=listing) or listing
        if listing and (block := strip_listing(listing)):
            result = f"; {binary.name}\n{block}"
            break
    _cache[key] = result
    return result
