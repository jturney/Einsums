#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
#
# Hold every optimization pass in the library to declaring the node features it understands.
#
# A pass that declares nothing is left unchecked, which is right for a pass written outside the
# library and wrong for one inside it: the declaration is what makes a pass leave alone a node
# carrying a feature it was never taught, and a pass added without one would meet the next
# feature the way seven passes met permutation operators. So every class in a pass header that
# derives from OptimizerPass directly must override understood_features. A RegionRewrite client
# inherits its base's declaration, which admits the region default.
#
# Run as:  cmake -DROOT=<ComputeGraph module dir> -P <this>
#     or:  cmake -DSELF_TEST=ON -DWORK_DIR=<scratch dir> -P <this>

function(_pf_count text pattern out)
  string(REGEX MATCHALL "${pattern}" _hits "${text}")
  list(LENGTH _hits _n)
  set(${out} ${_n} PARENT_SCOPE)
endfunction()

function(_pf_check root out_problems)
  file(GLOB _headers "${root}/include/Einsums/ComputeGraph/Passes/*.hpp")
  set(_problems "")
  foreach(_path IN LISTS _headers)
    file(READ "${_path}" _text)
    string(REPLACE ";" " " _text "${_text}")
    _pf_count("${_text}" "public[ \t\n]+OptimizerPass" _passes)
    _pf_count("${_text}" "understood_features\\(\\)[ \t\n]+const[ \t\n]+override" _declared)
    if(_declared LESS _passes)
      file(RELATIVE_PATH _rel "${root}" "${_path}")
      list(APPEND _problems "${_rel}: ${_passes} pass class(es), ${_declared} understood_features override(s)")
    endif()
  endforeach()
  set(${out_problems} "${_problems}" PARENT_SCOPE)
endfunction()

if(SELF_TEST)
  set(_dir "${WORK_DIR}/include/Einsums/ComputeGraph/Passes")
  file(REMOVE_RECURSE "${WORK_DIR}")
  file(WRITE "${_dir}/Declared.hpp"
       "class Declared : public OptimizerPass {\n  std::optional<NodeFeatures> understood_features() const override;\n};\n")
  _pf_check("${WORK_DIR}" _clean)
  if(_clean)
    message(FATAL_ERROR "self test: a pass that declares its features was reported: ${_clean}")
  endif()
  file(WRITE "${_dir}/Silent.hpp" "class Silent : public OptimizerPass {\n  bool run(Graph &) override;\n};\n")
  _pf_check("${WORK_DIR}" _dirty)
  if(NOT _dirty)
    message(FATAL_ERROR "self test: a pass that declares no features was not reported")
  endif()
  message(STATUS "check_pass_feature_declarations self test passed")
  return()
endif()

_pf_check("${ROOT}" _problems)
if(_problems)
  list(JOIN _problems "\n  " _text)
  message(FATAL_ERROR "Every library pass declares the node features it understands (OptimizerPass::understood_features):\n  ${_text}")
endif()
