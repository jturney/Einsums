#!/usr/bin/env python3
#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Heuristic CMake variable-typo auditor.

Flags `${_foo}`-style references inside a function/macro body that are
never defined in scope. CMake silently expands undefined references to
the empty string, so a single-character typo (`${_conf}` instead of
`${_config}`) can disable a whole helper without producing any error.
This is a static check for that exact failure mode.

Usage:

    # Audit specific files (used by pre-commit):
    cmake_audit.py path/to/file.cmake CMakeLists.txt ...

    # Audit every CMake file under the project tree:
    cmake_audit.py --all

Exits non-zero when at least one suspicious reference is found.

What "defined" means here:
    function/macro parameters, `cmake_parse_arguments` outputs (including
    the empty-prefix idiom `cmake_parse_arguments("" ...)`), `set()`,
    `foreach()`, `list()` (both first-arg and last-arg output forms),
    `string()`, `get_filename_component`, `get_target_property`,
    `get_property`, `execute_process` output/result/error vars,
    `file()` query forms, `find_program/_library/_path/_file`,
    `cmake_path`, `separate_arguments`, `math(EXPR ...)`, the
    `einsums_get_target_property` wrapper, and top-level `set()` /
    `set(... CACHE ...)` declarations visible from any function in the
    same file. `string()` outputs are found by subcommand, since the
    output is not always the last argument (`string(REPLACE <m> <r> <out>
    <input>)`). A call to a function defined in the same file defines
    whatever it writes back: a parameter the function sets with
    `set(${param} ... PARENT_SCOPE)`, or with plain `set(${param} ...)`
    in a macro.

Known false-negative classes (not flagged):
    - References to vars defined in *other* files (CMake includes).
      Module-level globals from `Einsums_*` helpers are common; the tool
      doesn't follow include() / find_package() across files. If a
      project-wide reference is missing and CMake can't find it either,
      the build will fail loudly elsewhere; the static check is most
      valuable for the silent in-file typo case.
"""

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Regexes for parsing CMake source.
# ---------------------------------------------------------------------------

# Function/macro boundaries. CMake doesn't really nest these, so a flat
# scan is sufficient.
FN_OPEN  = re.compile(
    r"^\s*(function|macro)\s*\(\s*([A-Za-z_][\w]*)\s*([^)]*)\)",
    re.MULTILINE,
)
FN_CLOSE = re.compile(r"^\s*end(function|macro)\s*\(", re.MULTILINE)

# What we're hunting for: ${_word} references.
REF = re.compile(r"\$\{(_[A-Za-z][\w]*)\}")

# Comment stripping. The block-comment closer optionally accepts a
# leading `#` to handle the `#]==]` form some upstream CMake modules use.
LINE_COMMENT  = re.compile(r"#(?!\[).*$", re.MULTILINE)
BLOCK_COMMENT = re.compile(r"#\[(=*)\[.*?#?\]\1\]", re.DOTALL)

# Defined-name extractors. Each captures the variable that the command
# writes to.
SET_DEF        = re.compile(r"(?:^|\s|;)set\s*\(\s*([A-Za-z_][\w]*)", re.IGNORECASE)
FOREACH        = re.compile(r"(?:^|\s|;)foreach\s*\(\s*([A-Za-z_][\w]*)", re.IGNORECASE)
LIST_OUT_FIRST = re.compile(
    # Mutating list ops: first arg is the list being modified.
    r"(?:^|\s|;)list\s*\(\s*"
    r"(?:APPEND|PREPEND|INSERT|REMOVE_AT|REMOVE_ITEM|REMOVE_DUPLICATES"
    r"|REVERSE|SORT|TRANSFORM|FILTER|POP_BACK|POP_FRONT)\s+"
    r"([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
LIST_OUT_LAST = re.compile(
    # Query list ops: last token before the closing paren is the output.
    r"(?:^|\s|;)list\s*\(\s*"
    r"(?:LENGTH|GET|FIND|JOIN|SUBLIST)\s+"
    r"(?:[^\s)]+\s+)+([A-Za-z_][\w]*)\s*\)",
    re.IGNORECASE,
)
STRING_OUT = re.compile(
    r"(?:^|\s|;)string\s*\(\s*\w+\s+(?:[^\s)]+\s+)*?([A-Za-z_][\w]*)\s*\)",
    re.IGNORECASE,
)
PARSE_ARGS = re.compile(r"cmake_parse_arguments\s*\(\s*([A-Za-z_][\w]*)")
PARSE_ARGS_FULL = re.compile(
    # prefix + options + one_value + multi_value, supporting quoted,
    # ${...}-expanded, and bare-token list forms.
    r"cmake_parse_arguments\s*\(\s*"
    r'("[^"]*"|[A-Za-z_][\w]*)\s+'
    r'("[^"]*"|\$\{[^}]+\}|[\S]+)\s+'
    r'("[^"]*"|\$\{[^}]+\}|[\S]+)\s+'
    r'("[^"]*"|\$\{[^}]+\}|[\S]+)',
    re.DOTALL,
)
GET_FILENAME = re.compile(
    r"(?:^|\s|;)get_filename_component\s*\(\s*([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
GET_TARGET_PROP = re.compile(
    r"(?:^|\s|;)get_target_property\s*\(\s*([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
GET_PROPERTY = re.compile(
    r"(?:^|\s|;)get_property\s*\(\s*([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
EXECUTE_PROCESS_OUT = re.compile(
    r"OUTPUT_VARIABLE\s+([A-Za-z_][\w]*)"
    r"|RESULT_VARIABLE\s+([A-Za-z_][\w]*)"
    r"|ERROR_VARIABLE\s+([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
FILE_OUT = re.compile(
    r"(?:^|\s|;)file\s*\(\s*"
    r"(?:READ|GLOB|GLOB_RECURSE|RELATIVE_PATH|TO_CMAKE_PATH|TO_NATIVE_PATH"
    r"|MD5|SHA1|SHA256|SHA384|SHA512|TIMESTAMP|STRINGS|SIZE|REAL_PATH)\s+"
    r"(?:[^\s)]+\s+)*?([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
FIND_PROGRAM = re.compile(
    r"(?:^|\s|;)(?:find_program|find_library|find_path|find_file)\s*\(\s*"
    r"([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
CMAKE_PATH = re.compile(
    r"(?:^|\s|;)cmake_path\s*\(\s*"
    r"(?:GET|HAS_\w+|IS_\w+|RELATIVE_PATH|ABSOLUTE_PATH|NATIVE_PATH|CONVERT)\s+"
    r"(?:[^\s)]+\s+)*?([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
SEPARATE_ARGS = re.compile(
    r"(?:^|\s|;)separate_arguments\s*\(\s*([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
MATH_OUT = re.compile(
    r"(?:^|\s|;)math\s*\(\s*EXPR\s+([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
# Custom Einsums wrapper. If more wrappers in this codebase write to
# their first arg, add them here.
EINSUMS_GET = re.compile(
    r"(?:^|\s|;)einsums_get_target_property\s*\(\s*([A-Za-z_][\w]*)",
    re.IGNORECASE,
)


def strip_comments(text: str) -> str:
    text = BLOCK_COMMENT.sub("", text)
    text = LINE_COMMENT.sub("", text)
    return text


def split_functions(text: str):
    """Yield (fn_name, fn_args, body_text, line_offset) tuples."""
    pos = 0
    while True:
        m = FN_OPEN.search(text, pos)
        if not m:
            return
        body_start = m.end()
        m_close = FN_CLOSE.search(text, body_start)
        if not m_close:
            return
        body = text[body_start:m_close.start()]
        line_offset = text.count("\n", 0, body_start) + 1
        yield m.group(2), m.group(3), body, line_offset
        pos = m_close.end()


def command_args(text: str, open_paren: int) -> list[str]:
    """The arguments of the command whose `(` is at `open_paren`, split the
    way CMake splits them: quoted and bracket arguments are one token each
    (quotes stripped), unquoted ones end at whitespace, and parentheses
    nest. Stops at the matching `)`."""
    tokens: list[str] = []
    i, n, depth = open_paren + 1, len(text), 0
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == ")" and depth == 0:
            break
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            tokens.append(text[i + 1:j])
            i = j + 1
        elif c == "[" and (m := re.match(r"\[(=*)\[", text[i:])):
            close = text.find("]" + m.group(1) + "]", i + m.end())
            end = n if close == -1 else close
            tokens.append(text[i + m.end():end])
            i = end + len(m.group(1)) + 2
        else:
            j = i
            while j < n and not text[j].isspace():
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    if depth == 0:
                        break
                    depth -= 1
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


COMMAND_CALL = re.compile(r"(?:^|(?<=[\s;(]))([A-Za-z_][\w]*)\s*\(")
NAME = re.compile(r"[A-Za-z_][\w]*")

# Where string(<subcommand> ...) writes, as an index into the arguments
# after the subcommand; -1 is the last argument.
STRING_OUT_INDEX = {
    "APPEND": 0, "PREPEND": 0, "CONCAT": 0, "TIMESTAMP": 0, "UUID": 0,
    "JSON": 0, "MD5": 0, "SHA1": 0, "SHA224": 0, "SHA256": 0, "SHA384": 0,
    "SHA512": 0, "SHA3_224": 0, "SHA3_256": 0, "SHA3_384": 0, "SHA3_512": 0,
    "JOIN": 1, "LENGTH": 1, "STRIP": 1, "TOLOWER": 1, "TOUPPER": 1,
    "GENEX_STRIP": 1, "MAKE_C_IDENTIFIER": 1, "HEX": 1, "CONFIGURE": 1,
    "REPLACE": 2, "FIND": 2, "REPEAT": 2, "SUBSTRING": 3, "COMPARE": 3,
    "ASCII": -1, "RANDOM": -1,
}
# string(REGEX <mode> <regex> ...), as an index into all the arguments
# (REGEX is 0, the mode 1, the regex 2): MATCH and MATCHALL write the
# argument after the regex, REPLACE the one after the replacement.
STRING_REGEX_OUT_INDEX = {"MATCH": 3, "MATCHALL": 3, "REPLACE": 4}


def string_outputs(body: str) -> set[str]:
    names: set[str] = set()
    for m in COMMAND_CALL.finditer(body):
        if m.group(1).lower() != "string":
            continue
        args = command_args(body, m.end() - 1)
        if not args:
            continue
        sub = args[0].upper()
        out = None
        if sub == "REGEX" and len(args) > 1:
            k = STRING_REGEX_OUT_INDEX.get(args[1].upper())
            if k is not None and k < len(args):
                out = args[k]
        elif sub in STRING_OUT_INDEX:
            rest = args[1:]
            k = STRING_OUT_INDEX[sub]
            if rest and -len(rest) <= k < len(rest):
                out = rest[k]
        if out and NAME.fullmatch(out):
            names.add(out)
    return names


def function_outputs(text: str) -> dict[str, list[int]]:
    """For each function or macro defined in `text`, the positions of the
    parameters it writes back to its caller."""
    outputs: dict[str, list[int]] = {}
    for m in FN_OPEN.finditer(text):
        kind, name, params = m.group(1).lower(), m.group(2), NAME.findall(m.group(3))
        m_close = FN_CLOSE.search(text, m.end())
        body = text[m.end():m_close.start() if m_close else len(text)]
        written = []
        for index, param in enumerate(params):
            for s in re.finditer(r"(?:^|(?<=[\s;(]))set\s*(\()\s*\$\{" + re.escape(param) + r"\}", body, re.IGNORECASE):
                args = command_args(body, s.start(1))
                if kind == "macro" or "PARENT_SCOPE" in args:
                    written.append(index)
                    break
        if written:
            outputs[name.lower()] = written
    return outputs


def call_outputs(body: str, outputs: dict[str, list[int]]) -> set[str]:
    """Names defined in `body` by calls to the functions in `outputs`."""
    names: set[str] = set()
    for m in COMMAND_CALL.finditer(body):
        positions = outputs.get(m.group(1).lower())
        if not positions:
            continue
        args = command_args(body, m.end() - 1)
        for k in positions:
            if k < len(args) and NAME.fullmatch(args[k]):
                names.add(args[k])
    return names


def collect_defined(args_str: str, body: str):
    """Return (names_set, cmake_parse_arguments_prefixes_set) for a body."""
    names: set[str] = set()
    # Function args are positional bare names.
    for tok in re.findall(r"[A-Za-z_][\w]*", args_str):
        names.add(tok)
    for m in SET_DEF.finditer(body):
        names.add(m.group(1))
    for m in FOREACH.finditer(body):
        names.add(m.group(1))
    for m in LIST_OUT_FIRST.finditer(body):
        names.add(m.group(1))
    for m in LIST_OUT_LAST.finditer(body):
        names.add(m.group(1))
    for m in STRING_OUT.finditer(body):
        names.add(m.group(1))
    names |= string_outputs(body)
    for m in GET_FILENAME.finditer(body):
        names.add(m.group(1))
    for m in GET_TARGET_PROP.finditer(body):
        names.add(m.group(1))
    for m in GET_PROPERTY.finditer(body):
        names.add(m.group(1))
    for m in EXECUTE_PROCESS_OUT.finditer(body):
        for g in m.groups():
            if g:
                names.add(g)
    for m in FILE_OUT.finditer(body):
        names.add(m.group(1))
    for m in FIND_PROGRAM.finditer(body):
        names.add(m.group(1))
    for m in CMAKE_PATH.finditer(body):
        names.add(m.group(1))
    for m in SEPARATE_ARGS.finditer(body):
        names.add(m.group(1))
    for m in MATH_OUT.finditer(body):
        names.add(m.group(1))
    for m in EINSUMS_GET.finditer(body):
        names.add(m.group(1))

    # cmake_parse_arguments: first record the prefix(es) so we can accept
    # `${<prefix>_anything}` references when the keyword list is variable-
    # expanded (we can't statically resolve that).
    parse_prefixes: set[str] = set()
    for m in PARSE_ARGS.finditer(body):
        parse_prefixes.add(m.group(1))

    # When the keyword lists are inline literals, materialize the
    # `<prefix>_<keyword>` names directly. The empty-prefix idiom
    # `cmake_parse_arguments("" ...)` produces `_<KEYWORD>`-style
    # references, which can only be recognized by parsing the keyword
    # lists.
    for m in PARSE_ARGS_FULL.finditer(body):
        prefix_raw = m.group(1)
        prefix = "" if prefix_raw == '""' else prefix_raw.strip('"')
        for kw_raw in (m.group(2), m.group(3), m.group(4)):
            kw_str = kw_raw.strip('"')
            if kw_str.startswith("${"):
                continue
            for kw in re.split(r"[;\s]+", kw_str):
                if not kw:
                    continue
                if not re.fullmatch(r"[A-Za-z_][\w]*", kw):
                    continue
                names.add(f"{prefix}_{kw}" if prefix else f"_{kw}")

    return names, parse_prefixes


def collect_top_level_defined(text: str) -> set[str]:
    """`set()` (incl. CACHE) calls outside any function body are visible
    from inside every function in the same file. Variables defined
    inside one function are NOT visible from another, so we must
    actually remove function bodies, not just their open/close lines.
    """
    names: set[str] = set()
    # Splice out each function/macro body (and its open/close lines),
    # leaving only the top-level statements.
    pieces: list[str] = []
    pos = 0
    while True:
        m = FN_OPEN.search(text, pos)
        if not m:
            pieces.append(text[pos:])
            break
        pieces.append(text[pos:m.start()])
        m_close = FN_CLOSE.search(text, m.end())
        if not m_close:
            break  # Malformed, bail.
        pos = m_close.end()
    cleaned = "".join(pieces)
    for m in SET_DEF.finditer(cleaned):
        names.add(m.group(1))
    return names


def audit_file(path: Path):
    """Return list of (line_number, function_name, undefined_var_name)."""
    raw = path.read_text()
    text = strip_comments(raw)
    top_level = collect_top_level_defined(text)
    outputs = function_outputs(text)
    findings = []
    for fn_name, fn_args, body, line_offset in split_functions(text):
        defined, parse_prefixes = collect_defined(fn_args, body)
        defined |= top_level
        defined |= call_outputs(body, outputs)
        for m in REF.finditer(body):
            name = m.group(1)
            if name in defined:
                continue
            if any(name.startswith(p + "_") for p in parse_prefixes):
                continue
            line_in_body = body.count("\n", 0, m.start())
            findings.append((line_offset + line_in_body, fn_name, name))
    return findings


def find_all_cmake_files(root: Path):
    """Every .cmake and CMakeLists.txt under the project tree, excluding
    typical build / vendored / IDE-history directories."""
    skip_dirs = {
        "build", ".git", ".cache", "third_party", "external",
        ".einsums-studio", ".idea", ".vs", ".vscode", "node_modules",
    }
    # A directory holding a CMakeCache.txt is a build tree whatever it is
    # called (build-no-profile, build-asan, ...); its .cmake files are
    # generated or copied, not ours to audit.
    build_trees = {c.parent for c in root.rglob("CMakeCache.txt")}
    for p in root.rglob("*"):
        if any(part in skip_dirs or part.startswith("cmake-build-") for part in p.parts):
            continue
        if any(tree in p.parents for tree in build_trees):
            continue
        if not p.is_file():
            continue
        name = p.name
        if name == "CMakeLists.txt" or name.endswith(".cmake"):
            yield p


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="CMake files to audit (positional). If empty, use --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Audit every CMake file under the current directory tree.",
    )
    args = parser.parse_args(argv)

    if args.all:
        files = sorted(find_all_cmake_files(Path.cwd()))
    else:
        files = [f for f in args.files if f.suffix == ".cmake" or f.name == "CMakeLists.txt"]

    if not files:
        # Nothing to do (e.g. pre-commit invoked us with non-CMake files).
        return 0

    total_findings = 0
    files_with_findings = 0
    for f in files:
        try:
            findings = audit_file(f)
        except Exception as e:  # noqa: BLE001 (never fail-open on parse errors)
            print(f"ERROR parsing {f}: {e}", file=sys.stderr)
            total_findings += 1
            files_with_findings += 1
            continue
        if findings:
            files_with_findings += 1
            print(f"\n{f}:")
            for line, fn, name in findings:
                print(f"  line {line:5d}  in {fn}()  : ${{{name}}} is not defined in scope")
                total_findings += 1

    if total_findings:
        print(
            f"\n{total_findings} suspicious reference(s) across "
            f"{files_with_findings} file(s).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
