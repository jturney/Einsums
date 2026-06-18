#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(Einsums_ExportTargets)
include(Einsums_WriteModuleHeader)
include(Einsums_CodeCoverage)

#:
#: einsums_add_module
#:
#:    Define and integrate an Einsums **module** (``<libname>_<modulename>``) with generated headers,
#:    dependency wiring, IDE grouping, install/export rules, and optional config files.
#:
#:    A module is either an **OBJECT** library (when sources are provided) or an **INTERFACE** library
#:    (header‑only). It is linked into the parent library ``<libname>`` and can depend on other modules,
#:    external targets, and private objects.
#:
#:    **Signature**
#:    ``einsums_add_module(<libname> <modulename>
#:        [BASE_LIBNAME <name>]
#:        [CONFIG_FILES]
#:        [SOURCES <...>]
#:        [HEADERS <...>]
#:        [OBJECTS <...>]
#:        [PRIVATE_DEPENDENCIES <targets...>]
#:        [DEPENDENCIES <targets...>]
#:        [MODULE_DEPENDENCIES <libname_foo libname_bar ...>]
#:        [CMAKE_SUBDIRS <dirs...>]
#:    )``
#:
#:    **Positional arguments**
#:    - ``libname`` *(required)*:
#:      Logical parent library to which this module will be attached (e.g., ``einsums``).
#:    - ``modulename`` *(required)*:
#:      Module identifier (e.g., ``Tensor``). The created target is ``<libname>_<modulename>``.
#:
#:    **Options / Keywords**
#:    - ``BASE_LIBNAME <name>``:
#:      When generating installed header paths, use this base namespace instead of ``<libname>``.
#:      (Defaults to ``<libname>``.)
#:
#:    - ``CONFIG_FILES`` *(switch)*:
#:      Also emit project‑wide config headers (in addition to per‑module headers):
#:      - ``<build>/include/<PROJECT_NAME>/Config/Version.hpp`` from
#:        ``cmake/templates/ConfigVersion.hpp.in``.
#:      - ``<build>/include/<PROJECT_NAME>/Config/Defines.hpp`` using
#:        ``einsums_write_config_defines_file`` with the default namespace and the
#:        template ``cmake/templates/ConfigDefines.hpp.in``.
#:
#:    - ``SOURCES <...>``:
#:      Source file names **relative** to the module ``src`` root (see *Layout*).
#:    - ``HEADERS <...>``:
#:      Header file names **relative** to the module ``include`` root. These are used for grouping
#:      and for generating a convenience umbrella header.
#:    - ``OBJECTS <...>``:
#:      Extra object files/targets to link **privately** into the module.
#:
#:    - ``PRIVATE_DEPENDENCIES <targets...>``:
#:      Targets linked privately to ``<libname>_<modulename>``.
#:    - ``DEPENDENCIES <targets...>``:
#:      Targets linked with module’s public keyword (``PUBLIC`` for object modules, ``INTERFACE`` for
#:      header‑only modules).
#:    - ``MODULE_DEPENDENCIES <libname_foo ...>``:
#:      Other *modules* this one depends on. When ``EINSUMS_WITH_CHECK_MODULE_DEPENDENCIES``
#:      is ON, the function validates that listed modules belong to the same category set.
#:
#:    - ``CMAKE_SUBDIRS <dirs...>``:
#:      Subdirectories to add via ``add_subdirectory`` (e.g., nested components/tests).
#:
#:    **Layout & paths**
#:    - ``SOURCE_ROOT`` is fixed to ``<module dir>/src``
#:    - ``HEADER_ROOT`` is fixed to ``<module dir>/include``
#:    - ``SOURCES``/``HEADERS`` are **prepended** with these roots internally.
#:
#:    **Generated headers**
#:    - **Per‑module defines header**:
#:      ``<build>/include/<BASE_LIBNAME>/<modulename>/Defines.hpp`` created via
#:      ``einsums_write_config_defines_file(NAMESPACE <MODULENAME_UPPER>)``.
#:    - **Per‑module umbrella header**:
#:      ``<build>/include/<BASE_LIBNAME>/<modulename>.hpp`` created via
#:      ``einsums_write_module_header(NAMESPACE <MODULENAME_UPPER>)``.
#:    - The function also removes stale (“zombie”) generated headers under
#:      ``<build>/include`` that are no longer part of this module’s output.
#:
#:    **Target type**
#:    - If any ``SOURCES`` are provided → module is an **OBJECT** library; public keyword = ``PUBLIC``.
#:    - If no sources are provided → module is an **INTERFACE** library; public keyword = ``INTERFACE``.
#:
#:    **Target creation & wiring**
#:    - Creates ``add_library(<libname>_<modulename> <OBJECT|INTERFACE> ...)`` with sources/objects.
#:    - Appends code‑coverage flags via ``einsums_append_coverage_compiler_flags_to_target``
#:      using the module’s public keyword.
#:    - Links, in order:
#:      - ``MODULE_DEPENDENCIES`` (public/interface as appropriate),
#:      - ``DEPENDENCIES`` (public/interface),
#:      - ``PRIVATE_DEPENDENCIES`` (private),
#:      - Always links ``einsums_public_flags`` and ``einsums_base_libraries`` with the module’s
#:        public/interface keyword; if the module is **OBJECT**, also links
#:        ``einsums_private_flags`` privately.
#:    - Sets include dirs (build + install interfaces), including the generated include tree.
#:    - If precompiled headers are enabled (``EINSUMS_WITH_PRECOMPILED_HEADERS``), reuses
#:      ``einsums_precompiled_headers``.
#:    - For **OBJECT** modules, defines an export macro:
#:      Derives ``<MACRO_LIB_NAME>_EXPORTS`` from ``<libname>`` and applies it privately to the module.
#:
#:    **IDE grouping & properties**
#:    - Source groups are created for headers, sources, and generated files via
#:      ``einsums_add_source_group``.
#:    - For non‑interface modules:
#:      - Places target under folder ``Core/Modules/<LibnameCapitalized>``.
#:      - Enables ``POSITION_INDEPENDENT_CODE``.
#:      - Enables Unity build/markers when ``EINSUMS_WITH_UNITY_BUILD`` is ON.
#:      - On MSVC, configures compile PDB name/output directories for Debug/RelWithDebInfo.
#:
#:    **Install & export**
#:    - Installs the module target (library/archive/runtime) and exports it via
#:      ``einsums_export_internal_targets``.
#:    - Installs public **source headers** from the module’s ``include/`` and the **generated headers**
#:      from ``<build>/include/<BASE_LIBNAME>``.
#:    - On MSVC, installs compile PDBs for Debug/RelWithDebInfo (if present).
#:
#:    **Integration with parent library**
#:    - Links the parent ``<libname>`` **PUBLIC** to ``<libname>_<modulename>``.
#:    - Links the parent ``<libname>`` **PRIVATE** to any listed ``OBJECTS``.
#:
#:    **Book‑keeping & summary**
#:    - Marks the module enabled by appending ``<BASENAME_UPPER>_<modulename>`` to the cached list
#:      ``EINSUMS_ENABLED_MODULES``.
#:    - Optionally adds extra subdirectories from ``CMAKE_SUBDIRS``.
#:    - Prints a configuration summary via ``einsums_create_configuration_summary``.
#:
#:    **Environment/toggles referenced**
#:    - ``EINSUMS_WITH_CHECK_MODULE_DEPENDENCIES``, ``EINSUMS_WITH_UNITY_BUILD``,
#:      ``EINSUMS_WITH_PRECOMPILED_HEADERS``.
#:
#:    **Example**
#:    .. code-block:: cmake
#:
#:       # In einsums/modules/tensor/CMakeLists.txt
#:       einsums_add_module(einsums Tensor
#:         SOURCES
#:           tensor/core.cpp
#:           tensor/ops.cpp
#:         HEADERS
#:           einsums/Tensor/core.hpp
#:           einsums/Tensor/ops.hpp
#:         MODULE_DEPENDENCIES
#:           einsums_LinearAlgebra
#:         DEPENDENCIES
#:           fmt::fmt Eigen3::Eigen
#:         PRIVATE_DEPENDENCIES
#:           einsums_internal_helpers
#:         CMAKE_SUBDIRS tests
#:         CONFIG_FILES
#:       )
#:
#:    This creates the target ``einsums_Tensor`` (OBJECT lib), generates
#:    ``<build>/include/einsums/Tensor/Defines.hpp`` and ``<build>/include/einsums/Tensor.hpp``,
#:    wires dependencies, installs public/generated headers, exports/installs the target, and links
#:    it into the parent ``einsums`` library.
function(einsums_add_module libname modulename)
  # Retrieve arguments
  set(options CONFIG_FILES)
  set(one_value_args BASE_LIBNAME)
  set(multi_value_args
      SOURCES
      HEADERS
      OBJECTS
      PRIVATE_DEPENDENCIES
      DEPENDENCIES
      MODULE_DEPENDENCIES
      CMAKE_SUBDIRS
  )
  cmake_parse_arguments(
    ${modulename} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )
  if(${modulename}_UNPARSED_ARGUMENTS)
    message(
      AUTHOR_WARNING "Arguments were not used by the module: ${${modulename}_UNPARSED_ARGUMENTS}"
    )
  endif()

  if(NOT ${modulename}_BASE_LIBNAME)
    set(basename ${libname})
  else()
    set(basename ${${modulename}_BASE_LIBNAME})
  endif()

  include(Einsums_Message)
  include(Einsums_Option)

  string(TOUPPER ${libname} libname_upper)
  string(TOUPPER ${modulename} modulename_upper)
  string(TOUPPER ${basename} basename_upper)

  string(MAKE_C_IDENTIFIER ${libname_upper} libname_upper)
  string(MAKE_C_IDENTIFIER ${modulename_upper} modulename_upper)
  string(MAKE_C_IDENTIFIER ${basename_upper} basename_upper)

  # Mark the module as enabled (see einsums/libs/CMakeLists.txt)
  set(EINSUMS_ENABLED_MODULES
      ${EINSUMS_ENABLED_MODULES} ${basename_upper}_${modulename}
      CACHE INTERNAL "List of enabled einsums modules" FORCE
  )

  # Main directories of the module
  set(SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src")
  set(HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include")

  einsums_debug("Add module ${modulename}: SOURCE_ROOT: ${SOURCE_ROOT}")
  einsums_debug("Add module ${modulename}: HEADER_ROOT: ${HEADER_ROOT}")

  set(all_headers ${${modulename}_HEADERS})

  set(module_headers "")

  foreach(header IN LISTS all_headers)
    set(module_headers "${module_headers}#include <${header}>\n")
  endforeach()

  # Write full path for the sources files
  list(
    TRANSFORM ${modulename}_SOURCES
    PREPEND ${SOURCE_ROOT}/
            OUTPUT_VARIABLE
            sources
  )
  list(
    TRANSFORM ${modulename}_HEADERS
    PREPEND ${HEADER_ROOT}/
            OUTPUT_VARIABLE
            headers
  )

  # generate configuration header for this module
  set(config_header "${CMAKE_CURRENT_BINARY_DIR}/include/${basename}/${modulename}/Defines.hpp")
  einsums_write_config_defines_file(NAMESPACE ${modulename_upper} FILENAME ${config_header})
  set(module_header "${CMAKE_CURRENT_BINARY_DIR}/include/${basename}/${modulename}.hpp")
  einsums_write_module_header(NAMESPACE ${modulename_upper} FILENAME ${module_header})
  set(generated_headers ${generated_headers} ${config_header} ${module_header})

  if(${modulename}_CONFIG_FILES)
    # Version file
    set(global_config_file ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/Config/Version.hpp)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/ConfigVersion.hpp.in" "${global_config_file}"
      @ONLY
    )
    set(generated_headers ${generated_headers} ${global_config_file})
    # Global config defines file (different from the one for each module)
    set(global_config_file ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/Config/Defines.hpp)
    einsums_write_config_defines_file(
      TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/ConfigDefines.hpp.in" NAMESPACE default
      FILENAME "${global_config_file}"
    )
    set(generated_headers ${generated_headers} ${global_config_file})
  endif()

  # collect zombie generated headers
  file(GLOB_RECURSE zombie_generated_headers ${CMAKE_CURRENT_BINARY_DIR}/include/*.hpp)
  list(REMOVE_ITEM zombie_generated_headers ${generated_headers} ${compat_headers}
       ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/Config/ModulesEnabled.hpp
  )
  foreach(zombie_header IN LISTS zombie_generated_headers)
    einsums_warn("      Removing zombie generated header: ${zombie_header}")
    file(REMOVE ${zombie_header})
  endforeach()

  # list all specified headers
  foreach(header_file ${headers})
    einsums_debug(${header_file})
  endforeach(header_file)

  if(sources)
    set(module_is_interface_library FALSE)
  else()
    set(module_is_interface_library TRUE)
  endif()

  if(module_is_interface_library)
    set(module_library_type INTERFACE)
    set(module_public_keyword INTERFACE)
  else()
    set(module_library_type OBJECT)
    set(module_public_keyword PUBLIC)
  endif()

  # create library modules
  add_library(${libname}_${modulename} ${module_library_type} ${sources} ${${modulename}_OBJECTS})
  einsums_append_coverage_compiler_flags_to_target(
    ${libname}_${modulename} ${module_public_keyword}
  )

  if(EINSUMS_WITH_CHECK_MODULE_DEPENDENCIES)
    # verify that all dependencies are from the same module category
    foreach(dep ${${modulename}_MODULE_DEPENDENCIES})
      # consider only module dependencies, not other targets
      string(FIND ${dep} "${libname}_" find_index)
      if(${find_index} EQUAL 0)
        string(LENGTH "${libname}_" lib_length)
        string(SUBSTRING ${dep} ${lib_length} -1 dep) # cut off leading "einsums_"
        list(FIND _${libname}_modules ${dep} dep_index)
        if(${dep_index} EQUAL -1)
          einsums_error(
            "The module ${dep} should not be be listed in MODULE_DEPENDENCIES "
            "for module Einsums_${modulename}"
          )
        endif()
      endif()
    endforeach()
  endif()

  target_link_libraries(
    ${libname}_${modulename} ${module_public_keyword} ${${modulename}_MODULE_DEPENDENCIES}
  )
  target_link_libraries(
    ${libname}_${modulename} ${module_public_keyword} ${${modulename}_DEPENDENCIES}
  )
  target_link_libraries(${libname}_${modulename} PRIVATE ${${modulename}_PRIVATE_DEPENDENCIES})

  target_link_libraries(
    ${libname}_${modulename} ${module_public_keyword} einsums_public_flags einsums_base_libraries
  )

  target_include_directories(
    ${libname}_${modulename} ${module_public_keyword} $<BUILD_INTERFACE:${HEADER_ROOT}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include> $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  if(NOT module_is_interface_library)
    target_link_libraries(${libname}_${modulename} PRIVATE einsums_private_flags)
  endif()

  if(EINSUMS_WITH_PRECOMPILED_HEADERS)
    target_precompile_headers(${libname}_${modulename} REUSE_FROM einsums_precompiled_headers)
  endif()

  if(NOT module_is_interface_library)
    # Add underscores before uppercase letters, except the first one
    string(REGEX REPLACE "([A-Z])" "_\\1" transformed_string ${libname})
    # Remove the leading underscore if it exists
    string(REGEX REPLACE "^_" "" transformed_string ${transformed_string})
    # Convert to uppercase
    string(TOUPPER ${transformed_string} LIB_NAME)
    string(MAKE_C_IDENTIFIER ${LIB_NAME} MACRO_LIB_NAME)

    target_compile_definitions(${libname}_${modulename} PRIVATE ${MACRO_LIB_NAME}_EXPORTS)
  endif()

  einsums_add_source_group(
    NAME ${libname}_${modulename}
    ROOT ${HEADER_ROOT}/${libname}
    CLASS "Header Files"
    TARGETS ${headers}
  )
  einsums_add_source_group(
    NAME ${libname}_${modulename}
    ROOT ${SOURCE_ROOT}
    CLASS "Source Files"
    TARGETS ${sources}
  )

  if(${modulename}_CONFIG_FILES)
    einsums_add_source_group(
      NAME ${libname}_${modulename}
      ROOT ${CMAKE_CURRENT_BINARY_DIR}/include/${libname}
      CLASS "Generated Files"
      TARGETS ${generated_headers}
    )
  endif()
  einsums_add_source_group(
    NAME ${libname}_${modulename}
    ROOT ${CMAKE_CURRENT_BINARY_DIR}/include/${libname}
    CLASS "Generated Files"
    TARGETS ${config_header}
  )

  # capitalize string
  string(SUBSTRING ${libname} 0 1 first_letter)
  string(TOUPPER ${first_letter} first_letter)
  string(REGEX REPLACE "^.(.*)" "${first_letter}\\1" libname_cap "${libname}")

  if(NOT module_is_interface_library)
    set_target_properties(
      ${libname}_${modulename} PROPERTIES FOLDER "Core/Modules/${libname_cap}"
                                          POSITION_INDEPENDENT_CODE ON
    )
  endif()

  if(EINSUMS_WITH_UNITY_BUILD AND NOT module_is_interface_library)
    set_target_properties(${libname}_${modulename} PROPERTIES UNITY_BUILD ON)
    set_target_properties(
      ${libname}_${modulename} PROPERTIES UNITY_BUILD_CODE_BEFORE_INCLUDE
                                          "// NOLINTBEGIN(bugprone-suspicious-include)"
    )
    set_target_properties(
      ${libname}_${modulename} PROPERTIES UNITY_BUILD_CODE_AFTER_INCLUDE
                                          "// NOLINTEND(bugprone-suspicious-include)"
    )
  endif()

  if(MSVC)
    set_target_properties(
      ${libname}_${modulename}
      PROPERTIES COMPILE_PDB_NAME_DEBUG ${libname}_${modulename}d
                 COMPILE_PDB_NAME_RELWITHDEBINFO ${libname}_${modulename}
                 COMPILE_PDB_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/Debug
                 COMPILE_PDB_OUTPUT_DIRECTORY_RELWITHDEBINFO
                 ${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo
    )
  endif()

  install(
    TARGETS ${libname}_${modulename}
    EXPORT einsums_internal_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${modulename}
  )
  einsums_export_internal_targets(${libname}_${modulename})

  # Install the headers from the source
  install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT ${modulename}
  )

  # Installing the generated header files from the build dir
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/${basename}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT ${modulename}
  )

  # install PDB if needed
  if(MSVC)
    foreach(cfg DEBUG;RELWITHDEBINFO)
      einsums_get_target_property(_pdb_file ${libname}_${modulename} COMPILE_PDB_NAME_${cfg})
      einsums_get_target_property(
        _pdb_dir ${libname}_${modulename} COMPILE_PDB_OUTPUT_DIRECTORY_${cfg}
      )
      install(
        FILES ${_pdb_dir}/${_pdb_file}.pdb
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        CONFIGURATIONS ${cfg}
        OPTIONAL
      )
    endforeach()
  endif()

  # Link modules to their higher-level libraries
  target_link_libraries(${libname} PUBLIC ${libname}_${modulename})
  target_link_libraries(${libname} PRIVATE ${${modulename}_OBJECTS})

  foreach(dir ${${modulename}_CMAKE_SUBDIRS})
    add_subdirectory(${dir})
  endforeach(dir)

  include(Einsums_PrintSummary)
  einsums_create_configuration_summary("    Module configuration (${modulename}):" "${modulename}")

endfunction(einsums_add_module)
