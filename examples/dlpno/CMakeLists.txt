#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# The psi4-free half of this example is testable and worth testing: the saved
# fixtures carry a converged reference, so the whole port runs and checks itself
# against recorded psi4 energies with nothing installed but einsums.
#
# Only that half is registered. Everything reached through psi4_source.py needs
# psi4 in the process, which is not a build dependency and must not become one.
# Both scripts locate themselves and their fixtures from __file__, so neither
# cares what directory ctest runs them in.
einsums_add_pseudo_target(Tests.Unit.Examples.DLPNO)
einsums_add_pseudo_dependencies(Tests.Examples Tests.Unit.Examples.DLPNO)

einsums_add_python_unit_test("Examples.DLPNO" ReferenceIO SCRIPT test_reference_io.py)
einsums_add_python_unit_test("Examples.DLPNO" Integrals SCRIPT test_integrals.py)
