#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
#
# Hold a source tree to having no compile-time index code, with a ratchet for what is left.
#
# ComputeGraph takes its contractions as string specs, which passes can rewrite; the templated
# `Indices{...}` API cannot be rewritten and is being removed from the module. This check counts,
# per file, the constructs that belong to that API and compares the count with an allowlist of
# the files that still have them. It fails when
#
#   - a file not on the list has any, so nothing new comes in;
#   - a listed file has more than its entry, so a listed file cannot quietly grow;
#   - a listed file has fewer, or is gone, so the entry is lowered or deleted in the same change
#     that removed the code. The list can only shrink, and when it is empty the check forbids the
#     constructs outright.
#
# Run as:  cmake -DROOT=<dir> -DALLOWLIST=<file> -P <this>
#     or:  cmake -DSELF_TEST=ON -DWORK_DIR=<scratch dir> -P <this>
#
# The self test runs the check against small trees it writes itself and asserts each kind of
# failure is reported. It exists because the allowlist is generated with the same patterns the
# check uses: a pattern that stopped matching would produce a list that agrees with it, and the
# check would pass forever while catching nothing.
#
# Allowlist format, one entry per line, paths relative to ROOT, `#` starts a comment:
#
#   <count> <relative/path>

# What counts. Each is a CMake regex applied to the whole file, after `;` has been replaced (so
# list splitting cannot inflate a count) and a newline prepended (so a leading character class can
# match at the start of the file).
set(_ci_patterns
    # An index tuple. The class keeps identifiers that merely end in "Indices" (NodeIndices{) out.
    "[^A-Za-z0-9_]Indices{"
    "MAKE_INDEX"
    # The index namespace, by any spelling that reaches it.
    "einsums::index[^A-Za-z0-9_]"
    "using namespace index[^A-Za-z0-9_]"
    "[^A-Za-z0-9_:]index::[A-Za-z_]"
    # The templated entry points. The character-index kernel, tensor_algebra::detail::permute,
    # is not matched: it takes std::string indices and stays.
    "tensor_algebra::einsum[^A-Za-z0-9_]"
    "tensor_algebra::permute[^A-Za-z0-9_]"
    # transpose names no index at its call site but is written as permute(Indices{i, j}, ...).
    "tensor_algebra::transpose[^A-Za-z0-9_]"
    # Headers that carry the index machinery.
    "<Einsums/TensorAlgebra\\.hpp>"
    "<Einsums/TensorAlgebra/TensorAlgebra\\.hpp>"
    "<Einsums/TensorAlgebra/Permute\\.hpp>"
    "<Einsums/TensorAlgebra/Detail/Index\\.hpp>"
)

set(_ci_globs
    *.hpp
    *.h
    *.hh
    *.cpp
    *.cxx
    *.cc
    *.inl
    *.ipp
    *.cu
    *.hip
    *.rst
    *.md
)

# Count the matches in one file.
function(_ci_count_file path out_count)
  file(READ "${path}" _content)
  string(REPLACE ";" " " _content "${_content}")
  set(_content "\n${_content}")
  set(_n 0)
  foreach(_pattern IN LISTS _ci_patterns)
    string(REGEX MATCHALL "${_pattern}" _hits "${_content}")
    list(LENGTH _hits _k)
    math(EXPR _n "${_n} + ${_k}")
  endforeach()
  set(${out_count} ${_n} PARENT_SCOPE)
endfunction()

# The lines of one file that match, as "  <line>: <text>", for the failure report. The file is
# walked with string(FIND) rather than turned into a list: an unbalanced `[` in a line of C++
# stops CMake splitting a list at `;`, which would merge lines and shift every number after it.
function(_ci_describe_file path out_text)
  file(READ "${path}" _content)
  string(REPLACE ";" " " _content "${_content}")
  set(_text "")
  set(_number 0)
  while(NOT _content STREQUAL "")
    math(EXPR _number "${_number} + 1")
    string(FIND "${_content}" "\n" _end)
    if(_end EQUAL -1)
      set(_line "${_content}")
      set(_content "")
    else()
      string(SUBSTRING "${_content}" 0 ${_end} _line)
      math(EXPR _next "${_end} + 1")
      string(SUBSTRING "${_content}" ${_next} -1 _content)
    endif()
    foreach(_pattern IN LISTS _ci_patterns)
      if("\n${_line}" MATCHES "${_pattern}")
        string(STRIP "${_line}" _stripped)
        string(APPEND _text "    ${_number}: ${_stripped}\n")
        break()
      endif()
    endforeach()
  endwhile()
  set(${out_text} "${_text}" PARENT_SCOPE)
endfunction()

# Check ROOT against ALLOWLIST. Returns the problems found, one per list element, empty when clean.
# A problem's text must not contain `;`, which would split it into two elements.
function(_ci_check root allowlist out_problems)
  set(_problems "")

  # The allowlist, as parallel lists of paths and counts.
  set(_listed_paths "")
  set(_listed_counts "")
  if(EXISTS "${allowlist}")
    file(STRINGS "${allowlist}" _entries)
    foreach(_entry IN LISTS _entries)
      string(STRIP "${_entry}" _entry)
      if(_entry STREQUAL "" OR _entry MATCHES "^#")
        continue()
      endif()
      if(NOT _entry MATCHES "^([0-9]+)[ \t]+(.+)$")
        list(APPEND _problems "allowlist line is not '<count> <path>': ${_entry}")
        continue()
      endif()
      list(APPEND _listed_counts "${CMAKE_MATCH_1}")
      list(APPEND _listed_paths "${CMAKE_MATCH_2}")
    endforeach()
  endif()

  # The allowlist is not scanned; it names the patterns by design.
  file(RELATIVE_PATH _allowlist_rel "${root}" "${allowlist}")

  set(_globs "")
  foreach(_glob IN LISTS _ci_globs)
    list(APPEND _globs "${root}/${_glob}")
  endforeach()
  file(GLOB_RECURSE _files LIST_DIRECTORIES false ${_globs})
  list(SORT _files)

  set(_seen "")
  foreach(_file IN LISTS _files)
    file(RELATIVE_PATH _rel "${root}" "${_file}")
    if(_rel STREQUAL _allowlist_rel)
      continue()
    endif()
    _ci_count_file("${_file}" _count)
    list(FIND _listed_paths "${_rel}" _at)
    if(_at EQUAL -1)
      if(_count GREATER 0)
        _ci_describe_file("${_file}" _where)
        list(APPEND _problems "${_rel}: ${_count} compile-time index construct(s) in a file not on the allowlist. Use a string spec.\n${_where}")
      endif()
      continue()
    endif()
    list(APPEND _seen "${_rel}")
    list(GET _listed_counts ${_at} _allowed)
    if(_count GREATER _allowed)
      _ci_describe_file("${_file}" _where)
      list(APPEND _problems "${_rel}: ${_count} compile-time index construct(s), allowlist says ${_allowed}. The new ones need string specs.\n${_where}")
    elseif(_count LESS _allowed)
      if(_count EQUAL 0)
        list(APPEND _problems "${_rel}: now clean. Delete its allowlist entry.")
      else()
        list(APPEND _problems "${_rel}: ${_count} compile-time index construct(s), allowlist says ${_allowed}. Lower the entry to ${_count}.")
      endif()
    endif()
  endforeach()

  # A listed file that no longer exists is an entry nobody will ever lower.
  foreach(_path IN LISTS _listed_paths)
    list(FIND _seen "${_path}" _at)
    if(_at EQUAL -1)
      list(APPEND _problems "${_path}: on the allowlist but not found under ${root}. Delete its entry.")
    endif()
  endforeach()

  set(${out_problems} "${_problems}" PARENT_SCOPE)
endfunction()

# Self test: write a tree, check it, and require the named problem (or none). Each tree file is
# "<name>=<content>"; a file needing a semicolon spells it <SEMI>, since the files arrive as a list.
function(_ci_expect name tree_files allowlist_text expect_regex)
  set(_dir "${WORK_DIR}/${name}")
  file(REMOVE_RECURSE "${_dir}")
  foreach(_spec IN LISTS tree_files)
    if(NOT _spec MATCHES "^([^=]+)=(.*)$")
      message(FATAL_ERROR "self test '${name}': tree entry is not '<name>=<content>': ${_spec}")
    endif()
    string(REPLACE "<SEMI>" ";" _file_content "${CMAKE_MATCH_2}")
    file(WRITE "${_dir}/${CMAKE_MATCH_1}" "${_file_content}\n")
  endforeach()
  file(WRITE "${_dir}/allowlist.txt" "${allowlist_text}")
  _ci_check("${_dir}" "${_dir}/allowlist.txt" _problems)
  string(REPLACE ";" "\n" _report "${_problems}")
  if(expect_regex STREQUAL "")
    if(NOT _report STREQUAL "")
      message(FATAL_ERROR "self test '${name}': expected a clean tree, got:\n${_report}")
    endif()
  elseif(NOT _report MATCHES "${expect_regex}")
    message(FATAL_ERROR "self test '${name}': expected a problem matching '${expect_regex}', got:\n${_report}")
  endif()
endfunction()

if(SELF_TEST)
  if(NOT DEFINED WORK_DIR)
    message(FATAL_ERROR "check_compile_time_indices: SELF_TEST needs WORK_DIR")
  endif()

  # Every pattern is seen, each in a file of its own on an empty allowlist.
  set(_samples
      "a.cpp=einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B)"
      "b.cpp=MAKE_INDEX(q)"
      "c.cpp=using namespace einsums::index"
      "d.cpp=using namespace index"
      "e.cpp=auto x = index::i"
      "f.cpp=tensor_algebra::einsum(1.0, t, &C, 1.0, t, A, t, B)"
      "g.cpp=tensor_algebra::permute(t, &C, t, A)"
      "h.cpp=tensor_algebra::transpose(&C, A)"
      "i.hpp=#include <Einsums/TensorAlgebra.hpp>"
      "j.hpp=#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>"
      "k.hpp=#include <Einsums/TensorAlgebra/Permute.hpp>"
      "l.hpp=#include <Einsums/TensorAlgebra/Detail/Index.hpp>"
      "m.rst=   using namespace einsums::index"
  )
  foreach(_sample IN LISTS _samples)
    string(REGEX MATCH "^([^=]+)=" _ignored "${_sample}")
    string(REPLACE "." "_" _case "${CMAKE_MATCH_1}")
    _ci_expect("new_${_case}" "${_sample}" "" "${CMAKE_MATCH_1}: [0-9]+ compile-time index construct\\(s\\) in a file not on the allowlist")
  endforeach()

  # What must not count: look-alike identifiers, the kept kernel, string specs.
  _ci_expect(
    "clean" "ok.cpp=NodeIndices{.a = x} tensor_algebra::detail::permute<false, T>(b, c, C, a, d, A) cg::einsum(\"ij <- ik <SEMI> kj\", &C, A, B) reindex::f()"
    "" ""
  )

  # A semicolon must not split one construct into two, or two into more.
  _ci_expect("semicolons" "s.cpp=x<SEMI> einsum(Indices{i}<SEMI> y)" "1 s.cpp\n" "")

  # The report names the right line even after a line with an unbalanced bracket.
  _ci_expect("line_numbers" "n.cpp=auto a = b[0<SEMI>\nint x<SEMI>\neinsum(Indices{i})" "" "n.cpp: 1 .*\n    3: einsum\\(Indices")

  # The ratchet in both directions, and the entries nobody would otherwise clean up.
  _ci_expect("grew" "g.cpp=Indices{i} Indices{j}" "1 g.cpp\n" "g.cpp: 2 compile-time index construct\\(s\\), allowlist says 1")
  _ci_expect("shrank" "s.cpp=Indices{i}" "2 s.cpp\n" "s.cpp: 1 compile-time index construct\\(s\\), allowlist says 2. Lower the entry to 1")
  _ci_expect("cleaned" "c.cpp=cg::einsum(\"i <- i\", &C, A, B)" "1 c.cpp\n" "c.cpp: now clean. Delete its allowlist entry")
  _ci_expect("vanished" "a.cpp=int x" "3 gone.cpp\n" "gone.cpp: on the allowlist but not found")
  _ci_expect("malformed" "a.cpp=int x" "three a.cpp\n" "allowlist line is not '<count> <path>'")
  _ci_expect("matches" "m.cpp=Indices{i} // and MAKE_INDEX(x)" "# comment\n\n2 m.cpp\n" "")

  message(STATUS "check_compile_time_indices: self test passed")
  return()
endif()

if(NOT DEFINED ROOT OR NOT IS_DIRECTORY "${ROOT}")
  message(FATAL_ERROR "check_compile_time_indices: ROOT not set or not a directory: '${ROOT}'")
endif()
if(NOT DEFINED ALLOWLIST OR NOT EXISTS "${ALLOWLIST}")
  message(FATAL_ERROR "check_compile_time_indices: ALLOWLIST not set or missing: '${ALLOWLIST}'")
endif()

_ci_check("${ROOT}" "${ALLOWLIST}" _problems)
if(NOT _problems STREQUAL "")
  # NOTICE prints the report as written. FATAL_ERROR would reflow it into paragraphs, breaking the
  # quoted source lines, so it carries only the summary.
  list(LENGTH _problems _count)
  foreach(_problem IN LISTS _problems)
    message(NOTICE "${_problem}")
  endforeach()
  message(FATAL_ERROR "compile-time index check: ${_count} problem(s) under ${ROOT} against ${ALLOWLIST}")
endif()
message(STATUS "check_compile_time_indices: ${ROOT} matches ${ALLOWLIST}")
