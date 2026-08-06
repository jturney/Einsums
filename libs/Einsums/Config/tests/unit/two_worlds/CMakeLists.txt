#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
#
# Two copies of one library in one process.
#
# None of these targets link Einsums, on purpose: the point is a self-contained
# reproduction of the loader behaviour that the sealed-worlds work depends on,
# small enough to build twice where Einsums is not. Only the test driver links
# Einsums, and only for Catch2.
#
# The version scripts are the mechanism under test. On ELF they give each copy's
# symbols a distinct version, so the dynamic linker binds a consumer to the copy
# it was linked against even when the other copy reached the global scope first.
# Delete them and the RTLD_GLOBAL case in TwoWorlds.cpp should go red on Linux.

set(MINIWORLD_A_VALUE 1001)
set(MINIWORLD_B_VALUE 2002)
set(MINIWORLD_STATIC_VALUE 3003)

# ELF only. Mach-O binds through a two-level namespace that records the source dylib per symbol, and
# PE binds per-DLL by name; neither has the flat-namespace race that versioning exists to settle,
# and neither linker takes the flag.
set(_miniworld_versioned FALSE)
if(CMAKE_SYSTEM_NAME MATCHES "Linux|BSD|SunOS")
  include(CheckLinkerFlag)
  check_linker_flag(
    CXX "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/miniworld_a.map"
    EINSUMS_LINKER_SUPPORTS_VERSION_SCRIPT
  )
  if(EINSUMS_LINKER_SUPPORTS_VERSION_SCRIPT)
    set(_miniworld_versioned TRUE)
  endif()
endif()

# One shared copy of the miniworld, with its own constant and version tag.
function(einsums_add_miniworld name value tag)
  add_library(${name} SHARED MiniWorld.cpp)
  target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${name} PRIVATE MINIWORLD_VALUE=${value})
  set_target_properties(
    ${name} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
  )
  if(_miniworld_versioned)
    target_link_options(
      ${name} PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/${name}.map"
    )
    # A version script is an input file, not just a flag: without this the library does not relink
    # when the script changes, and a broken script would keep passing from cache.
    set_target_properties(${name} PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.map)
  endif()
endfunction()

einsums_add_miniworld(miniworld_a ${MINIWORLD_A_VALUE} A)
einsums_add_miniworld(miniworld_b ${MINIWORLD_B_VALUE} B)

# The static copy, for the case a stage module absorbs the library instead of linking it shared.
add_library(miniworld_static STATIC MiniWorld.cpp)
target_include_directories(miniworld_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(miniworld_static PRIVATE MINIWORLD_VALUE=${MINIWORLD_STATIC_VALUE})
set_target_properties(
  miniworld_static PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden
)

# One consumer per copy, standing in for a compiled stage module. MODULE rather than SHARED: these
# are dlopen targets, never linked against.
function(einsums_add_miniworld_consumer name world)
  add_library(${name} MODULE Consumer.cpp)
  target_link_libraries(${name} PRIVATE ${world})
  set_target_properties(
    ${name} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
  )
endfunction()

einsums_add_miniworld_consumer(two_worlds_consumer_a miniworld_a)
einsums_add_miniworld_consumer(two_worlds_consumer_b miniworld_b)
einsums_add_miniworld_consumer(two_worlds_consumer_static miniworld_static)

einsums_add_executable(
  TwoWorlds_test INTERNAL_FLAGS
  SOURCES TwoWorlds.cpp
  NOINSTALL
)

# Paths and expected constants reach the test as definitions, so it never has to guess a build
# layout.
target_compile_definitions(
  TwoWorlds_test
  PRIVATE CONSUMER_A_PATH="$<TARGET_FILE:two_worlds_consumer_a>"
          CONSUMER_B_PATH="$<TARGET_FILE:two_worlds_consumer_b>"
          CONSUMER_STATIC_PATH="$<TARGET_FILE:two_worlds_consumer_static>"
          MINIWORLD_A_VALUE=${MINIWORLD_A_VALUE}
          MINIWORLD_B_VALUE=${MINIWORLD_B_VALUE}
          MINIWORLD_STATIC_VALUE=${MINIWORLD_STATIC_VALUE}
)
add_dependencies(
  TwoWorlds_test two_worlds_consumer_a two_worlds_consumer_b two_worlds_consumer_static
)

einsums_add_unit_test("Modules.Config" TwoWorlds)
