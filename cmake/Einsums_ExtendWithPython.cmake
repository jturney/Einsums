#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_extend_with_python_headers target)
  target_link_libraries(${target} PRIVATE pybind11::headers)
  target_include_directories(${target} PRIVATE ${Python_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
endfunction()

function(einsums_extend_with_python target)
  if(APPLE)
    set_target_properties(${target} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif()

  if(UNIX AND NOT APPLE)
    target_link_libraries(${target} PRIVATE pybind11::module)
  endif()

  if(APPLE)
    set_target_properties(
      ${target} PROPERTIES PREFIX "" DEBUG_POSTFIX "" INSTALL_RPATH
                                                      "${CMAKE_INSTALL_RPATH};\@loader_path/${_ein_pymod_relDir}"
    )
  elseif(NOT MSVC)
    set_target_properties(
      ${target} PROPERTIES PREFIX "" DEBUG_POSTFIX "" INSTALL_RPATH
                                                      "${CMAKE_INSTALL_RPATH};\$ORIGIN/${_ein_pymod_relDir}"
    )
  else()
    set_target_properties(${target} PROPERTIES PREFIX "" DEBUG_POSTFIX "")
  endif()
endfunction()
