#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# The Python-binding library einsums._core is built against: apiary's build-wide choice,
# APIARY_BACKEND (pybind11, the default, or nanobind; nanobind 3.0 or newer). With nanobind,
# APIARY_STABLE_ABI=ON builds one stable-ABI _core for every Python from 3.12 on.
#
# apiary_find_backend finds Python and the library and makes it the default for everything below
# this directory, so nothing in Einsums names either library:
#
# * apiary_add_bindings and apiary_aggregate_extension (einsums_finalize_pybind) generate and build
#   for it, and define APIARY_BACKEND_PYBIND11 or APIARY_BACKEND_NANOBIND on _core;
# * the hand-written binding code includes <apiary/Python.hpp>, which reads that define;
# * the test probes are made with apiary_add_extension_module, which defines it too;
# * Waggle, built here, calls apiary_find_backend itself and so takes the same library.
#
# APIARY_BACKEND and APIARY_BACKEND_VERSION also go into the installed package
# (EinsumsConfig.cmake): an out-of-tree stage module receives RuntimeTensors that _core created,
# and both libraries share types only between modules whose internals agree, so the package pins
# its stage modules to this library at this exact release.

if(EINSUMS_BUILD_PYTHON)
  apiary_find_backend()
endif()
