#!/usr/bin/env bash
#
# count_tests.sh - Inventory the Einsums test suite.
#
# For C++ (Catch2) and Python (pytest) it reports:
#   * test FILES
#   * SOURCE cases   - what you see in the source (one macro / one def)
#   * ACTUAL cases   - what actually runs:
#                        C++    : TEMPLATE_TEST_CASE expanded per type
#                                 (Catch2 `--list-tests`, non-hidden only)
#                        Python : @pytest.mark.parametrize expanded and every
#                                 hypothesis / differential-fuzzer test counted
#                                 as one collected item (pytest --collect-only)
#
# The ACTUAL columns require a build: the compiled Catch2 binaries and the
# `einsums` python module under <build>/lib. Point at a build with $1 or
# BUILD_DIR (default: build). If the build is missing the ACTUAL columns
# show "n/a".
#
# Conventions (see CLAUDE.md):
#   C++    files: .cpp under any libs/**/tests/ directory
#   Python files: test_*_python.py under any libs/**/tests/ directory
# cmake/tests/*.cpp are compiler feature-probes, not unit tests, and are
# deliberately excluded.
#
# Usage: devtools/count_tests.sh [build-dir]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${1:-${BUILD_DIR:-build}}"

# Catch2 test-case-introducing macros.
CATCH_MACROS='TEST_CASE|TEMPLATE_TEST_CASE|TEMPLATE_TEST_CASE_SIG|TEMPLATE_LIST_TEST_CASE|TEMPLATE_PRODUCT_TEST_CASE|SCENARIO'

# --------------------------------------------------------------------------
# Files + source cases (static; always available)
# --------------------------------------------------------------------------
cpp_files=$(find libs -path '*/tests/*' -name '*.cpp' -type f | wc -l | tr -d ' ')
cpp_src=$(find libs -path '*/tests/*' -name '*.cpp' -type f -print0 \
    | xargs -0 grep -hE "^[[:space:]]*(${CATCH_MACROS})[[:space:]]*\(" 2>/dev/null \
    | wc -l | tr -d ' ')

py_files=$(find libs -path '*/tests/*' -name 'test_*_python.py' -type f | wc -l | tr -d ' ')
py_src=$(find libs -path '*/tests/*' -name 'test_*_python.py' -type f -print0 \
    | xargs -0 grep -hE '^[[:space:]]*def[[:space:]]+test_' 2>/dev/null \
    | wc -l | tr -d ' ')

# --------------------------------------------------------------------------
# Actual C++ cases: sum Catch2 `--list-tests` counts across built binaries.
# stdin is redirected from /dev/null (the debugger-attach probe otherwise
# swallows the caller's stdin) and the no-attach flag keeps it non-interactive.
# --------------------------------------------------------------------------
cpp_actual="n/a"
if [ -d "${BUILD_DIR}/libs" ]; then
    total=0
    while IFS= read -r bin; do
        n=$("${bin}" --list-tests --einsums:debug:no-attach-debugger </dev/null 2>/dev/null \
            | grep -oE '^[0-9]+ test cases?' | grep -oE '^[0-9]+' || true)
        [ -n "${n}" ] && total=$((total + n))
    done < <(find "${BUILD_DIR}/libs" -name '*_test' -type f -perm +111)
    cpp_actual="${total}"
fi

# --------------------------------------------------------------------------
# Actual Python cases: pytest collection count. The repo's pytest.ini sets
# `norecursedirs = tests`, which hides every libs/**/tests/ dir, so we
# override it (and restrict to the test_*_python.py convention).
# --------------------------------------------------------------------------
py_actual="n/a"
if [ -f "${BUILD_DIR}/lib/einsums/__init__.py" ]; then
    collected=$(PYTHONPATH="${BUILD_DIR}/lib" python -m pytest libs \
        -p no:cacheprovider \
        --override-ini="norecursedirs=.git .hypothesis __pycache__ build *.egg-info" \
        --override-ini="python_files=test_*_python.py" \
        --collect-only -q 2>/dev/null \
        | grep -oE '[0-9]+ tests? collected' | grep -oE '^[0-9]+' || true)
    py_actual="${collected:-0}"
fi

# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
sum() { # $1 $2 -> numeric sum, or "n/a" if either is non-numeric
    case "$1$2" in
        *[!0-9]*) echo "n/a" ;;
        *) echo $(( $1 + $2 )) ;;
    esac
}
total_files=$(sum "${cpp_files}" "${py_files}")
total_src=$(sum "${cpp_src}" "${py_src}")
total_actual=$(sum "${cpp_actual}" "${py_actual}")

printf '%s\n' "Einsums test inventory   (build: ${BUILD_DIR})"
printf '%s\n' "============================================"
printf '%-18s %8s %8s %8s\n' "" "files" "source" "actual"
printf '%-18s %8s %8s %8s\n' "C++ (Catch2)"    "${cpp_files}" "${cpp_src}" "${cpp_actual}"
printf '%-18s %8s %8s %8s\n' "Python (pytest)" "${py_files}"  "${py_src}"  "${py_actual}"
printf '%s\n' "--------------------------------------------"
printf '%-18s %8s %8s %8s\n' "Total" "${total_files}" "${total_src}" "${total_actual}"
printf '\n'
printf '%s\n' "source = macros / def test_ in source"
printf '%s\n' "actual = TEMPLATE expanded (C++, non-hidden) / parametrize+hypothesis collected (Python)"
