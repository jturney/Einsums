#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_BUILD_PYTHON)
  find_package(pybind11 2.11 REQUIRED)

  # Record the version we actually resolved, not the minimum we asked for.
  #
  # An out-of-tree stage module has to share TYPES with einsums._core: it
  # receives a RuntimeTensor that _core created. pybind11 keys its type registry
  # on an internals blob whose identity includes the pybind11 build, so a module
  # compiled against a different pybind11 gets its own registry and cannot see
  # _core's types at all - the failure is an unhelpful "incompatible function
  # arguments" at call time rather than anything at load time.
  #
  # EinsumsConfig.cmake re-finds this exact version so a stage module cannot
  # silently pick up a different one from the environment.
  set(EINSUMS_PYBIND11_VERSION
      "${pybind11_VERSION}"
      CACHE INTERNAL "pybind11 version this Einsums was built against"
  )
endif()
