#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
#
# Phase-2 smoke runner. Drives each fixture under tests/fixtures/ through
# einsums-pybind and checks the IR dump for a hand-written set of
# substring assertions. Phase 3 will replace this with golden-output diffs
# once the emitter output stabilizes.
#
# Invocation:
#     run_smoke.sh <einsums-pybind-binary> <annotations-include-dir>

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <einsums-pybind-binary> <annotations-include-dir>" >&2
    exit 64
fi

readonly TOOL="$1"
readonly INCLUDE_DIR="$2"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FIXTURE_DIR="${SCRIPT_DIR}/fixtures"

run_tool() {
    # The tool may exit non-zero when a fixture pulls in a system header
    # we can't find with -nostdinc++ (the AST is still populated and the
    # IR dump is correct in that case, so we let the assertions decide).
    "${TOOL}" --dump-ir "$1" -- -std=c++20 -nostdinc++ "-I${INCLUDE_DIR}" 2>/dev/null || true
}

assert_contains() {
    local fixture="$1" pattern="$2" output="$3"
    if ! grep -qE -- "${pattern}" <<<"${output}"; then
        echo "FAIL ${fixture}: missing pattern: ${pattern}" >&2
        echo "--- output ---" >&2
        printf '%s\n' "${output}" >&2
        exit 1
    fi
}

assert_absent() {
    local fixture="$1" pattern="$2" output="$3"
    if grep -qE -- "${pattern}" <<<"${output}"; then
        echo "FAIL ${fixture}: forbidden pattern present: ${pattern}" >&2
        echo "--- output ---" >&2
        printf '%s\n' "${output}" >&2
        exit 1
    fi
}

# ---- simple_class.hpp ------------------------------------------------------
out="$(run_tool "${FIXTURE_DIR}/simple_class.hpp")"
assert_contains simple_class 'class einsums::fixture::Shape'                     "${out}"
assert_contains simple_class '@expose'                                           "${out}"
assert_contains simple_class '@rename\(PyShape\)'                                "${out}"
assert_contains simple_class '@nocopy'                                           "${out}"
assert_contains simple_class 'method Shape \[ctor\]'                             "${out}"
assert_contains simple_class 'method rank.* \[const\]'                           "${out}"
assert_contains simple_class '@rvp\(reference_internal\)'                        "${out}"
assert_contains simple_class 'method operator== \[op\]'                          "${out}"
assert_contains simple_class '@operator\(__eq__\)'                               "${out}"
assert_contains simple_class 'method get_dim'                                    "${out}"
assert_contains simple_class '@getter\(dim\)'                                    "${out}"
assert_contains simple_class '@setter\(dim\)'                                    "${out}"
assert_contains simple_class 'field public_field: int'                           "${out}"
assert_contains simple_class '@hide'                                             "${out}"
assert_contains simple_class 'property dim: int \[rw\]'                          "${out}"

# ---- free_functions.hpp ----------------------------------------------------
out="$(run_tool "${FIXTURE_DIR}/free_functions.hpp")"
assert_contains free_functions 'function einsums::fixture::add: int\(int a, int b\)' "${out}"
assert_contains free_functions 'py: \(a: int, b: int\) -> int'                       "${out}"
assert_contains free_functions 'function einsums::fixture::scaled_add'               "${out}"
assert_contains free_functions 'int scale = 1'                                       "${out}"
assert_contains free_functions 'scale: int = 1'                                      "${out}"
assert_contains free_functions '@rvp\(move\)'                                        "${out}"
assert_contains free_functions 'function einsums::fixture::heavy_compute'            "${out}"
assert_contains free_functions 'py: \(seed: float\) -> float'                        "${out}"
assert_contains free_functions '@release_gil'                                        "${out}"
assert_absent   free_functions 'should_not_appear'                                   "${out}"

# ---- enums.hpp -------------------------------------------------------------
out="$(run_tool "${FIXTURE_DIR}/enums.hpp")"
assert_contains enums 'enum class einsums::fixture::Layout'                      "${out}"
assert_contains enums 'RowMajor = 0'                                             "${out}"
assert_contains enums 'ColumnMajor = 1'                                          "${out}"
assert_contains enums 'enum einsums::fixture::Severity'                          "${out}"
assert_contains enums 'Info = 0'                                                 "${out}"
assert_contains enums 'Warning = 1'                                              "${out}"
assert_contains enums 'class einsums::fixture::Engine'                           "${out}"
assert_contains enums 'enum class einsums::fixture::Engine::State'               "${out}"
assert_contains enums 'py: int'                                                  "${out}"

# ---- namespace_module.hpp --------------------------------------------------
out="$(run_tool "${FIXTURE_DIR}/namespace_module.hpp")"
assert_contains namespace_module 'function einsums::fixture::graph::inherited'   "${out}"
assert_contains namespace_module 'submodule: graph$'                             "${out}"
assert_contains namespace_module 'function einsums::fixture::graph::overridden'  "${out}"
assert_contains namespace_module 'submodule: graph\.ops$'                        "${out}"
assert_contains namespace_module 'function einsums::fixture::top_level'          "${out}"

echo "OK: all Phase-2 fixtures pass"
