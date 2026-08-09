#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# The top-level examples are documented deliverables rather than build targets:
# they are driven by scripts, several of them need psi4, and the ones that build
# C++ (dlpno/cpp, psi4-plugin-template) are built out of tree against an
# INSTALLED Einsums on purpose, because that is the configuration a user of a
# stage module is in. So nothing here is compiled by this build.
#
# What is registered is the part that can be checked without leaving the build:
# the tests an example ships that need only einsums and its own fixtures.
if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
  add_subdirectory(dlpno)
endif()
