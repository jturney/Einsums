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
# APIARY_BUILD_TOOL decides both paths. An installed Apiary is two components, and conda-forge ships
# them as two packages: apiary-annotations (the macros, no Clang/LLVM) and apiary (the tool, on top).
# A configuration that generates code asks find_package for the tool component, so an install of the
# annotations alone is passed over for the source build rather than accepted and then failing in
# codegen. On the source fallback the option keeps Clang/LLVM out of a configuration that only wants
# the macros. Our 3.25 floor means CMP0077 is NEW, so this normal variable drives Apiary's option()
# directly.

include(FetchContent)

# Apiary's emitted C++ is tied to the Clang type printer it was built against, and its CMake helper
# API is what einsums_finalize_pybind() calls into, so the range is bounded on both sides to stay on
# 1.x. The floor is 1.3.1: 1.3 introduced the tool component and the annotations-only package, and
# 1.3.1 fixed both for consumers (1.3.0 exported apiary::annotations with include directories that
# do not exist, and left it defined after refusing a find_package for the tool). Since 1.2 it builds against LLVM 22 and 23, parses with the Clang
# builtin headers of its own LLVM, and its C++ docs driver takes --exclude-header, which
# docs/CMakeLists.txt passes; since 1.2.1 it gives a template kwarg its C++ default in Python, which
# syev and heev rely on.
set(_einsums_apiary_find_args 1.3.1...<2 CONFIG)
if(EINSUMS_BUILD_PYTHON OR EINSUMS_WITH_DOCUMENTATION)
  set(APIARY_BUILD_TOOL ON)
  list(APPEND _einsums_apiary_find_args COMPONENTS tool)
else()
  set(APIARY_BUILD_TOOL OFF)
endif()

fetchcontent_declare(
  Apiary
  URL https://github.com/Einsums/Apiary/archive/refs/tags/v1.3.1.tar.gz
  URL_HASH SHA256=3edcd792091825280127e886774b808dfa84d2d2ec08d56b51dbff1976fe9918
  # An installed apiary outside the range, or without the tool when it is needed, is passed over,
  # and this builds 1.3.1 instead, against whatever LLVM the environment provides.
  FIND_PACKAGE_ARGS ${_einsums_apiary_find_args}
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
