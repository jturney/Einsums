#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
#
# Hold every executor lambda in ComputeGraph to capturing by value.
#
# A node's executor outlives the code that built it: it runs at every replay, after the graph
# has been returned from a function or stored in a container, and so moved. A reference capture
# names whatever the reference named when the lambda was made. When that is the graph, or
# anything stored inside it (its tensor table, a node, a handle), the moved graph's executor
# reads the moved-from object, and freed memory once that object is destroyed. `this` is the
# same hazard for the pass or graph that built the lambda. An executor that needs the graph at
# run time captures `graph.anchor()`, which follows the graph through moves.
#
# So the capture list of an `execute = [...]` lambda may not hold `this`, a default capture
# (`&` or `=`, which hide what they take), or any `&name`. A capture that is safe for a reason
# the scan cannot see goes in _ec_allowed below as "<path relative to ROOT> <capture>", with the
# reason; an entry that no longer matches anything fails the check, so the list only shrinks.
#
# Run as:  cmake -DROOT=<ComputeGraph module dir> -P <this>
#     or:  cmake -DSELF_TEST=ON -DWORK_DIR=<scratch dir> -P <this>
#
# The self test writes a tree with each kind of offending capture and asserts the check reports
# every one, so a pattern that stopped matching fails here instead of passing forever.

# SUMMAExpansion's process grid is the function-local static comm::ProcessGrid::default_grid()
# returns, alive for the whole process, so the reference cannot dangle.
set(_ec_allowed "src/Passes/SUMMAExpansion.cpp &grid")

set(_ec_pattern "execute[ \t\r\n]*=[ \t\r\n]*\\[")

# Split the capture list that starts just after the `[` at `start` in `text` into its top-level
# items, stripped. Brackets nest, so `v[0]` or `f(a, b)` inside an init-capture stays one item.
function(_ec_captures text start out_items)
  string(LENGTH "${text}" _length)
  set(_depth 0)
  set(_item "")
  set(_items "")
  set(_at ${start})
  while(_at LESS _length)
    string(SUBSTRING "${text}" ${_at} 1 _char)
    math(EXPR _at "${_at} + 1")
    if(_depth EQUAL 0 AND (_char STREQUAL "]" OR _char STREQUAL ","))
      string(STRIP "${_item}" _item)
      if(NOT _item STREQUAL "")
        list(APPEND _items "${_item}")
      endif()
      set(_item "")
      if(_char STREQUAL "]")
        break()
      endif()
      continue()
    endif()
    if(_char MATCHES "[[({]")
      math(EXPR _depth "${_depth} + 1")
    elseif(_char MATCHES "[])}]")
      math(EXPR _depth "${_depth} - 1")
    endif()
    string(APPEND _item "${_char}")
  endwhile()
  set(${out_items} "${_items}" PARENT_SCOPE)
endfunction()

# Report every forbidden capture under `root` as "<rel>:<line>: <capture>" in out_problems, and
# every _ec_allowed entry that matched as "<rel> <capture>" in out_used.
function(_ec_check root out_problems out_used)
  file(GLOB_RECURSE _files "${root}/src/*.cpp" "${root}/src/*.hpp" "${root}/include/*.hpp")
  set(_problems "")
  set(_used "")
  foreach(_path IN LISTS _files)
    file(READ "${_path}" _text)
    if(NOT _text MATCHES "${_ec_pattern}")
      continue()
    endif()
    file(RELATIVE_PATH _rel "${root}" "${_path}")
    set(_offset 0)
    set(_rest "${_text}")
    while(_rest MATCHES "${_ec_pattern}")
      set(_match "${CMAKE_MATCH_0}")
      string(FIND "${_rest}" "${_match}" _found)
      string(LENGTH "${_match}" _match_length)
      math(EXPR _list_start "${_offset} + ${_found} + ${_match_length}")
      _ec_captures("${_text}" ${_list_start} _items)

      math(EXPR _prefix_length "${_offset} + ${_found}")
      string(SUBSTRING "${_text}" 0 ${_prefix_length} _prefix)
      string(REGEX MATCHALL "\n" _newlines "${_prefix}")
      list(LENGTH _newlines _line)
      math(EXPR _line "${_line} + 1")

      foreach(_item IN LISTS _items)
        if(_item STREQUAL "this" OR _item STREQUAL "&" OR _item STREQUAL "=" OR _item MATCHES "^&")
          string(REGEX REPLACE "[ \t\r\n]+" " " _shown "${_item}")
          if("${_rel} ${_shown}" IN_LIST _ec_allowed)
            list(APPEND _used "${_rel} ${_shown}")
          else()
            list(APPEND _problems "${_rel}:${_line}: ${_shown}")
          endif()
        endif()
      endforeach()

      math(EXPR _offset "${_list_start}")
      string(SUBSTRING "${_text}" ${_offset} -1 _rest)
    endwhile()
  endforeach()
  set(${out_problems} "${_problems}" PARENT_SCOPE)
  set(${out_used} "${_used}" PARENT_SCOPE)
endfunction()

if(SELF_TEST)
  set(_ec_allowed "")
  file(REMOVE_RECURSE "${WORK_DIR}")
  file(
    WRITE "${WORK_DIR}/src/Passes/Clean.cpp"
    "node.execute = [anchor, v = values[0], n = f(a, b), self = *this]() {};\n"
    "auto const scan = [&graph](Node const &n) { return n.id; };\n"
  )
  _ec_check("${WORK_DIR}" _clean _unused)
  if(_clean)
    message(FATAL_ERROR "self test: by-value executor captures were reported: ${_clean}")
  endif()
  file(
    WRITE "${WORK_DIR}/src/Passes/Bad.cpp"
    "node.execute = [&graph]() {};\n"
    "node.execute =\n    [tokens,\n     &tensor_map]() {};\n"
    "node.execute = [this]() {};\n"
    "node.execute = [&]() {};\n"
    "node.execute = [=]() {};\n"
    "node.execute = [&ref = graph.tensors_map()]() {};\n"
  )
  _ec_check("${WORK_DIR}" _dirty _unused)
  set(_expected
      "src/Passes/Bad.cpp:1: &graph"
      "src/Passes/Bad.cpp:2: &tensor_map"
      "src/Passes/Bad.cpp:5: this"
      "src/Passes/Bad.cpp:6: &"
      "src/Passes/Bad.cpp:7: ="
      "src/Passes/Bad.cpp:8: &ref = graph.tensors_map()"
  )
  if(NOT _dirty STREQUAL _expected)
    message(FATAL_ERROR "self test: expected\n  ${_expected}\nbut the check reported\n  ${_dirty}")
  endif()
  set(_ec_allowed "src/Passes/Bad.cpp &graph")
  _ec_check("${WORK_DIR}" _dirty _used)
  if("src/Passes/Bad.cpp:1: &graph" IN_LIST _dirty OR NOT _used STREQUAL "src/Passes/Bad.cpp &graph")
    message(FATAL_ERROR "self test: an allowed capture was still reported, or not recorded as used")
  endif()
  message(STATUS "check_executor_captures self test passed")
  return()
endif()

_ec_check("${ROOT}" _problems _used)
foreach(_entry IN LISTS _ec_allowed)
  if(NOT _entry IN_LIST _used)
    list(APPEND _problems "allowed capture no longer present, remove it from _ec_allowed: ${_entry}")
  endif()
endforeach()
if(_problems)
  list(JOIN _problems "\n  " _text)
  message(
    FATAL_ERROR
      "An executor lambda captures by reference or captures this; it dangles once the graph moves. "
      "Capture by value, and graph.anchor() for the graph:\n  ${_text}"
  )
endif()
