#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_add_compile_test category name)
  set(options FAILURE_EXPECTED NOLIBS OBJECT)
  set(one_value_args SOURCE_ROOT FOLDER)
  set(multi_value_args SOURCES DEPENDENCIES)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(_additional_flags)
  if(${name}_NOLIBS)
    set(_additional_flags ${_additional_flags} NOLIBS)
  endif()
  if(${name}_OBJECT)
    set(_additional_flags ${_additional_flags} OBJECT)
  endif()

  string(REGEX REPLACE "\\." "_" test_name "${category}.${name}")

  if(${name}_OBJECT)
    einsums_add_library(
      ${test_name}
      SOURCE_ROOT ${${name}_SOURCE_ROOT}
      SOURCES ${${name}_SOURCES}
      EXCLUDE_FROM_ALL EXCLUDE_FROM_DEFAULT_BUILD
      FOLDER ${${name}_FOLDER}
      DEPENDENCIES ${${name}_DEPENDENCIES} ${_additional_flags}
    )
  else()
    einsums_add_executable(
      ${test_name}
      SOURCE_ROOT ${${name}_SOURCE_ROOT}
      SOURCES ${${name}_SOURCES}
      EXCLUDE_FROM_ALL EXCLUDE_FROM_DEFAULT_BUILD
      FOLDER ${${name}_FOLDER}
      DEPENDENCIES ${${name}_DEPENDENCIES} ${_additional_flags}
    )
  endif()

  add_test(
    NAME "${category}.${name}"
    COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR} --target ${test_name} --config
            $<CONFIGURATION>
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  )

  set_tests_properties("${category}.${name}" PROPERTIES LABELS "COMPILE_ONLY")

  if(${name}_FAILURE_EXPECTED)
    set_tests_properties("${category}.${name}" PROPERTIES WILL_FAIL TRUE)
  endif()
endfunction()

function(einsums_add_compile_test_target_dependencies category name)
  einsums_add_pseudo_target(${category}.${name})
  # make pseudo-targets depend on master pseudo-target
  einsums_add_pseudo_dependencies(${category} ${category}.${name})
endfunction()

# To add test to the category root as in tests/regressions/ with correct name
function(einsums_add_test_and_deps_compile_test category subcategory name)
  if("${subcategory}" STREQUAL "")
    einsums_add_compile_test(tests.${category} ${name} ${ARGN})
    einsums_add_compile_test_target_dependencies(tests.${category} ${name})
  else()
    einsums_add_compile_test(tests.${category}.${subcategory} ${name} ${ARGN})
    einsums_add_compile_test_target_dependencies(tests.${category}.${subcategory} ${name})
  endif()
endfunction(einsums_add_test_and_deps_compile_test)

function(einsums_add_unit_compile_test subcategory name)
  einsums_add_test_and_deps_compile_test("unit" "${subcategory}" ${name} ${ARGN})
endfunction(einsums_add_unit_compile_test)

function(einsums_add_regression_compile_test subcategory name)
  einsums_add_test_and_deps_compile_test("regressions" "${subcategory}" ${name} ${ARGN})
endfunction(einsums_add_regression_compile_test)

function(einsums_add_headers_compile_test subcategory name)
  # Important to keep the double quotes around subcategory otherwise it doesn't consider empty
  # argument but just removes it
  einsums_add_test_and_deps_compile_test("headers" "${subcategory}" ${name} ${ARGN} OBJECT)
endfunction(einsums_add_headers_compile_test)

function(einsums_add_header_tests category)
  set(options NOLIBS)
  set(one_value_args HEADER_ROOT)
  set(multi_value_args HEADERS EXCLUDE EXCLUDE_FROM_ALL DEPENDENCIES)

  cmake_parse_arguments(${category} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(_additional_flags)
  if(${category}_NOLIBS)
    set(_additional_flags ${_additional_flags} NOLIBS)
  endif()

  set(all_headers)
  add_custom_target(tests.headers.${category})

  foreach(header ${${category}_HEADERS})

    # skip all headers in directories containing 'detail'
    string(FIND "${header}" "detail" detail_pos)
    list(FIND ${category}_EXCLUDE "${header}" exclude_pos)

    if(${detail_pos} EQUAL -1 AND ${exclude_pos} EQUAL -1)
      # extract relative path of header
      string(REGEX REPLACE "${${category}_HEADER_ROOT}/" "" relpath "${header}")

      # .hpp --> .cpp
      string(REGEX REPLACE ".hpp" ".cpp" full_test_file "${relpath}")
      # remove extension, '/' --> '_'
      string(REGEX REPLACE ".hpp" "_hpp" test_file "${relpath}")
      string(REGEX REPLACE "/" "_" test_name "${test_file}")

      # generate the test
      file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${full_test_file}
           "#include <${relpath}>\n" "#ifndef EINSUMS_MAIN_DEFINED\n" "int main() { return 0; }\n"
           "#endif\n"
      )

      set(exclude_all_pos -1)
      list(FIND ${category}_EXCLUDE_FROM_ALL "${header}" exclude_all_pos)
      if(${exclude_all_pos} EQUAL -1)
        set(all_headers ${all_headers} "#include <${relpath}>\n")
      endif()

      get_filename_component(header_dir "${relpath}" DIRECTORY)

      einsums_add_headers_compile_test(
        "${category}" ${test_name}
        SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${full_test_file}"
        SOURCE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/${header_dir}"
        FOLDER "Tests/Headers/${header_dir}"
        DEPENDENCIES ${${category}_DEPENDENCIES} einsums_private_flags einsums_public_flags
                     ${_additional_flags}
      )

    endif()
  endforeach()

  set(test_name "all_headers")
  set(all_headers_test_file "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.cpp")
  file(WRITE ${all_headers_test_file} ${all_headers} "#ifndef EINSUMS_MAIN_DEFINED\n"
                                      "int main() { return 0; }\n" "#endif\n"
  )

  einsums_add_headers_compile_test(
    "${category}" ${test_name}
    SOURCES "${all_headers_test_file}"
    SOURCE_ROOT "${CMAKE_CURRENT_BINARY_DIR}"
    FOLDER "Tests/Headers"
    DEPENDENCIES ${${category}_DEPENDENCIES} ${_additional_flags}
  )
endfunction()
