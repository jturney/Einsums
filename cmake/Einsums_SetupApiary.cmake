#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Apiary - the annotation contract plus the binding/docs codegen tool.
#
# Two independent pieces come out of this, and the distinction drives everything below:
#
# * apiary::annotations - header-only APIARY_* macros. Reached transitively from Logging.hpp, so
#   EVERY translation unit needs them, which makes this an unconditional dependency of the C++ build
#   and of any downstream find_package(Einsums) - including a pure-C++ (EINSUMS_BUILD_PYTHON=OFF)
#   install that never generates a binding.
# * apiary::apiary - the libtooling codegen executable. It drives the per-module pybind11 bindings
#   (EINSUMS_BUILD_PYTHON) and the C++ API reference (EINSUMS_WITH_DOCUMENTATION), and it pulls in
#   the Clang/LLVM dev stack, so only those two consumers need it.
#
# Apiary is on conda-forge, so the einsums-dev environment supplies both. The FIND_PACKAGE_ARGS
# branch below takes that install when it is there and falls back to building the tagged release
# from source when it is not - the same shape as the other Einsums_Setup*.cmake modules.
#
# The source fallback is where APIARY_BUILD_TOOL matters: a from-source build would otherwise drag
# Clang/LLVM into a configuration that only wants the macros. An installed package ships whatever it
# was built with (conda-forge's has the tool), so the option is simply ignored on that path. Our
# 3.25 floor means CMP0077 is NEW, so this normal variable drives Apiary's option() directly.

include(FetchContent)

if(EINSUMS_BUILD_PYTHON OR EINSUMS_WITH_DOCUMENTATION)
  set(APIARY_BUILD_TOOL ON)
else()
  set(APIARY_BUILD_TOOL OFF)
endif()

fetchcontent_declare(
  Apiary
  URL https://github.com/Einsums/Apiary/archive/refs/tags/v1.0.0.tar.gz
  URL_HASH SHA256=3dd50a38af85f9e47ade73981caffb26ca7d36e4cfc52db1decf68ef4a113199
  # Apiary's emitted C++ is tied to the Clang type printer it was built against, and its CMake
  # helper API is what einsums_finalize_pybind() calls into, so the range is bounded on both sides
  # to stay on 1.x.
  FIND_PACKAGE_ARGS 1...<2 CONFIG
)

fetchcontent_makeavailable(Apiary)

# Where <apiary/Annotations.hpp> lives, as a plain path. The codegen custom commands hand apiary its
# own include roots as bare ``-I`` strings rather than as usage requirements of a target, so a
# genex-bearing property value is no use to them.
if(Apiary_FOUND)
  get_target_property(EINSUMS_APIARY_INCLUDE_DIR apiary::annotations INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT EINSUMS_APIARY_INCLUDE_DIR)
    message(FATAL_ERROR "Apiary was found at ${Apiary_DIR} but apiary::annotations carries no include "
                        "directory, so <apiary/Annotations.hpp> is unreachable. The install is incomplete."
    )
  endif()
  # The install writes the same directory through both INCLUDES DESTINATION and the target's own
  # INSTALL_INTERFACE, so the list can repeat itself; one entry is all the flag needs.
  list(GET EINSUMS_APIARY_INCLUDE_DIR 0 EINSUMS_APIARY_INCLUDE_DIR)
else()
  set(EINSUMS_APIARY_INCLUDE_DIR "${apiary_SOURCE_DIR}/include")
endif()

# einsums_public_flags carries apiary::annotations in its link interface, and Einsums exports that
# target into the build tree with export(TARGETS ...) (see Einsums_GeneratePackage). That command
# refuses a dependency which is neither IMPORTED nor itself exported into the build tree. A
# from-source Apiary is exactly that: install(EXPORT ApiaryTargets) puts apiary_annotations in an
# INSTALL export set, which the install rules are happy with and the build-tree export is not. So
# give it a build-tree export set as well. A found Apiary needs none of this - its targets arrive
# IMPORTED and satisfy the check on their own, which is why this failure only ever appears on the
# fallback path.
if(NOT Apiary_FOUND)
  export(EXPORT ApiaryTargets NAMESPACE apiary:: FILE "${CMAKE_BINARY_DIR}/ApiaryTargets.cmake")
endif()

# Probe the build compiler for the system/stdlib include paths libtooling needs (resource-dir, conda
# isystem, C++ stdlib dirs). Results land in APIARY_SYSTEM_FLAGS / APIARY_* and are consumed by
# einsums_finalize_pybind(). Only meaningful when the codegen executable is actually available.
if(TARGET apiary::apiary)
  apiary_detect_toolchain(CXX_STANDARD ${EINSUMS_WITH_CXX_STANDARD})
endif()
