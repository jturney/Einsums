#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
#
# Phase-3 golden-output runner. Drives each fixture under tests/fixtures/
# through einsums-pybind in emitter mode and diffs the result against a
# committed .golden file. Differences are reported with `diff -u` so the
# test driver shows exactly which lines moved.
#
# Updating goldens:
#     <tool> --module <name> <fixture> -- -std=c++20 -nostdinc++ -I<include>
#         > tests/golden/<name>.cpp.golden
# Or run with REGEN=1 to overwrite goldens in place.
#
# Invocation:
#     run_golden.sh <einsums-pybind-binary> <annotations-include-dir>

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <einsums-pybind-binary> <annotations-include-dir>" >&2
    exit 64
fi

readonly TOOL="$1"
readonly INCLUDE_DIR="$2"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FIXTURE_DIR="${SCRIPT_DIR}/fixtures"
readonly GOLDEN_DIR="${SCRIPT_DIR}/golden"
readonly REGEN="${REGEN:-0}"

# ``module|fixture|golden`` triples.
readonly CASES=(
    "fixture_simple_class|simple_class.hpp|simple_class.cpp.golden"
    "fixture_free_functions|free_functions.hpp|free_functions.cpp.golden"
    "fixture_enums|enums.hpp|enums.cpp.golden"
    "fixture_templated_class|templated_class.hpp|templated_class.cpp.golden"
    "fixture_templated_function_bools|templated_function_bools.hpp|templated_function_bools.cpp.golden"
    "fixture_namespace_module|namespace_module.hpp|namespace_module.cpp.golden"
    "fixture_dtype_dispatcher|dtype_dispatcher.hpp|dtype_dispatcher.cpp.golden"
    "fixture_member_template|member_template.hpp|member_template.cpp.golden"
)

run_tool() {
    local module="$1" fixture="$2"
    "${TOOL}" --module "${module}" "${FIXTURE_DIR}/${fixture}" \
        -- -std=c++20 -nostdinc++ "-I${INCLUDE_DIR}" 2>/dev/null || true
}

tmp_actual="$(mktemp)"
tmp_diff="$(mktemp)"
trap 'rm -f "${tmp_actual}" "${tmp_diff}"' EXIT

# The emitter calls clang::format::getStyle("file", path_hint, "LLVM", ...) to
# locate a `.clang-format`. The tool's default path_hint is "generated.cpp"
# resolved against CWD. ctest's default working dir is the build tree; in
# out-of-source builds (the CI default) the build tree is a sibling of the
# source tree, so the upward search never reaches the project root's
# `.clang-format` and the emitter silently falls back to LLVM style (2-space).
# That produces a stable mismatch against goldens generated under the project
# style (4-space). Run from this script's directory — inside the source tree —
# so the upward search hits the project's `.clang-format` regardless of where
# the build tree lives.
cd "${SCRIPT_DIR}"

failures=0
for case in "${CASES[@]}"; do
    IFS='|' read -r module fixture golden <<<"${case}"
    run_tool "${module}" "${fixture}" > "${tmp_actual}"
    golden_path="${GOLDEN_DIR}/${golden}"

    if [[ "${REGEN}" == "1" ]]; then
        cp "${tmp_actual}" "${golden_path}"
        echo "REGEN ${golden}"
        continue
    fi

    if ! diff -u "${golden_path}" "${tmp_actual}" > "${tmp_diff}" 2>&1; then
        echo "FAIL ${golden}: emitter output drifted from committed golden" >&2
        cat "${tmp_diff}" >&2
        failures=$((failures + 1))
    fi
done

if (( failures != 0 )); then
    echo "${failures} golden test(s) failed" >&2
    exit 1
fi

echo "OK: all Phase-3 golden tests pass"
