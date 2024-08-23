#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_add_executable name)
  set(options GPU EXCLUDE_FROM_ALL EXCLUDE_FROM_DEFAULT_BUILD INTERNAL_FLAGS NOLIBS UNITYT_BUILD)
  set(one_value_args
      INI
      FOLDER
      SOURCE_ROOT
      HEADER_ROOT
      SOURCE_GLOB
      HEADER_GLOB
      OUTPUT_SUFFIX
      INSTALL_SUFFIX
      LANGUAGE)
  set(multi_value_args SOURCES HEADERS AUXILIARY DEPENDENCIES COMPILE_FLAGS LINK_FLAGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT ${name}_LANGUAGE)
    set(${name}_LANGUAGE CXX)
  endif()

  if(NOT ${name}_SOURCE_ROOT)
    set(${name}_SOURCE_ROOT ".")
  endif()
  einsums_debug("Add executable ${name}: ${name}_SOURCE_ROOT: ${${name}_SOURCE_ROOT}")

  if(NOT ${name}_HEADER_ROOT)
    set(${name}_HEADER_ROOT ".")
  endif()
  einsums_debug("Add executable ${name}: ${name}_HEADER_ROOT: ${${name}_HEADER_ROOT}")

  einsums_add_library_sources_noglob(${name}_executable SOURCES "${${name}_SOURCES}")

  einsums_add_source_group(
    NAME
    ${name}
    CLASS
    "Source Files"
    ROOT
    ${${name}_SOURCE_ROOT}
    TARGETS
    ${${name}_executable_SOURCES})

  set(${name}_SOURCES ${${name}_executable_SOURCES})
  set(${name}_HEADERS ${${name}_executable_HEADERS})

  einsums_print_list("DEBUG" "Add executable ${name}: Sources for ${name}" ${name}_SOURCES)
  einsums_print_list("DEBUG" "Add executable ${name}: Headers for ${name}" ${name}_HEADERS)
  einsums_print_list("DEBUG" "Add executable ${name}: Dependencies for ${name}" ${name}_DEPENDENCIES)

  set(_target_flags)

  # add the executable build target
  if(${name}_EXCLUDE_FROM_ALL)
    set(exclude_from_all EXCLUDE_FROM_ALL TRUE)
  else()
    set(install_destination ${CMAKE_INSTALL_BINDIR})
    if(${name}_INSTALL_SUFFIX)
      set(install_destination ${${name}_INSTALL_SUFFIX})
    endif()
    set(_target_flags INSTALL INSTALL_FLAGS DESTINATION ${install_destination})
    if(MSVC)
      set(_target_flags
          ${_target_flags}
          INSTALL_PDB
          $<TARGET_PDB_FILE:{name}>
          DESTINATION
          ${install_destination}
          CONFIGURATIONS
          Debug
          RelWithDebInfo
          OPTIONAL)
    endif()
  endif()

  if(${name}_EXCLUDE_FROM_DEFAULT_BUILD)
    set(exclude_from_all ${exclude_from_all} EXCLUDE_FROM_DEFAULT_BUILD TRUE)
  endif()

  if(EINSUMS_WITH_HIP)
    foreach(source ${${name}_SOURCES})
      get_filename_component(extension ${source} EXT)
      if(${extension} STREQUAL ".hip")
        set_source_file_properties(${source} PROPERTIES LANGUAGE HIP)
      endif()
    endforeach()
  endif()

  set(target_has_gpu_sources FALSE)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    foreach(source ${${name}_SOURCES})
      get_filename_component(extension ${source} EXT)
      if(${extension} STREQUAL ".cu")
        einsums_add_nvhpc_cuda_flags(${source})
        set(target_has_gpu_sources TRUE)
      endif()
    endforeach()
  endif()

  einsums_info("Add executable ${name}: ${${name}_SOURCES}")
  add_executable(${name} ${${name}_SOURCES} ${${name}_HEADERS} ${${name}_AUXILIARY})

  if(${name}_OUTPUT_SUFFIX)
    if(MSVC)
      set_target_properties(
        ${name}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${EINSUMS_WITH_BINARY_DIR}/Release/bin/${${name}_OUTPUT_SUFFIX}"
                   RUNTIME_OUTPUT_DIRECTORY_DEBUG "${EINSUMS_WITH_BINARY_DIR}/Debug/bin/${${name}_OUTPUT_SUFFIX}"
                   RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL
                   "${EINSUMS_WITH_BINARY_DIR}/MinSizeRel/bin/${${name}_OUTPUT_SUFFIX}"
                   RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO
                   "${EINSUMS_WITH_BINARY_DIR}/RelWithDebInfo/bin/${${name}_OUTPUT_SUFFIX}")
    else()
      set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                               "${EINSUMS_WITH_BINARY_DIR}/bin/${${name}_OUTPUT_SUFFIX}")
    endif()
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" AND target_has_gpu_sources)
    set_target_properties(${name} PROPERTIES LINK_FLAGS "-cuda")
  endif()

  if(${name}_GPU)
    if(EINSUMS_WITH_HIP)
      set_target_properties(${name} PROPERTIES LANGUAGE HIP)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
      set_target_properties(${name} PROPERTIES LINK_FLAGS "-cuda")
    else()
      set_target_properties(${name} PROPERTIES LANGUAGE CUDA)
    endif()
  endif()

  set_target_properties(${name} PROPERTIES OUTPUT_NAME "${EINSUMS_WITH_EXECUTABLE_PREFIX}${name}")

  if(exclude_from_all)
    set_target_properties(${name} PROPERTIES ${exclude_from_all})
  endif()

  if(${${name}_NOLIBS})
    set(_target_flags ${_target_flags} NOLIBS)
  endif()

  if(${name}_INTERNAL_FLAGS)
    set(_target_flags ${_target_flags} INTERNAL_FLAGS)
  endif()

  if(${name}_UNITY_BUILD)
    set(_target_flags ${_target_flags} UNITY_BUILD)
  endif()

  einsums_setup_target(
    ${name}
    TYPE EXECUTABLE
    FOLDER ${${name}_FOLDER}
    COMPILE_FLAGS ${${name}_COMPILE_FLAGS}
    LINK_FLAGS ${${name}_LINK_FLAGS}
    DEPENDENCIES ${${name}_DEPENDENCIES} ${_target_flags})
endfunction()
