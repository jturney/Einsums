#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
#
# Hold the optimization passes to asking one question about storage identity.
#
# Two relations make one buffer answer to several tensor ids: a view names its parent, and a slot
# redirect sends a merged-away tensor to the survivor's storage. Graph::buffer_of follows both;
# Graph::resolve_alias follows only the first. Every question a pass asks ("is this the same
# data", "who else writes this") is about buffers, and passes that asked resolve_alias instead
# counted one reader of a tensor two nodes read. So a pass must not call resolve_alias at all.
#
# Run as:  cmake -DROOT=<ComputeGraph module dir> -P <this>
#     or:  cmake -DSELF_TEST=ON -DWORK_DIR=<scratch dir> -P <this>
#
# The self test writes a tree with one offending call and asserts the check reports it, so a
# pattern that stopped matching fails here instead of passing forever.

set(_bi_pattern "resolve_alias[ \t]*\\(")

function(_bi_check root out_problems)
  file(GLOB _files "${root}/src/Passes/*.cpp" "${root}/src/Passes/*.hpp" "${root}/include/Einsums/ComputeGraph/Passes/*.hpp")
  set(_problems "")
  foreach(_path IN LISTS _files)
    file(STRINGS "${_path}" _lines)
    set(_number 0)
    foreach(_line IN LISTS _lines)
      math(EXPR _number "${_number} + 1")
      if(_line MATCHES "${_bi_pattern}")
        file(RELATIVE_PATH _rel "${root}" "${_path}")
        string(STRIP "${_line}" _stripped)
        list(APPEND _problems "${_rel}:${_number}: ${_stripped}")
      endif()
    endforeach()
  endforeach()
  set(${out_problems} "${_problems}" PARENT_SCOPE)
endfunction()

if(SELF_TEST)
  file(REMOVE_RECURSE "${WORK_DIR}")
  file(WRITE "${WORK_DIR}/src/Passes/Clean.cpp" "auto b = graph.buffer_of(id);\n")
  _bi_check("${WORK_DIR}" _clean)
  if(_clean)
    message(FATAL_ERROR "self test: a tree with no resolve_alias call was reported: ${_clean}")
  endif()
  file(WRITE "${WORK_DIR}/src/Passes/Bad.cpp" "auto b = graph.resolve_alias (id);\n")
  _bi_check("${WORK_DIR}" _dirty)
  if(NOT _dirty)
    message(FATAL_ERROR "self test: a pass calling resolve_alias was not reported")
  endif()
  message(STATUS "check_pass_buffer_identity self test passed")
  return()
endif()

_bi_check("${ROOT}" _problems)
if(_problems)
  list(JOIN _problems "\n  " _text)
  message(FATAL_ERROR "A pass asks Graph::resolve_alias, which misses slot redirects. Ask Graph::buffer_of:\n  ${_text}")
endif()
