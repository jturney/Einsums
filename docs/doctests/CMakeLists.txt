#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Tests that hold the documentation to the code it describes.
#
# These live under EINSUMS_WITH_TESTS rather than EINSUMS_WITH_DOCUMENTATION, which defaults OFF:
# the point is that an ordinary build catches doc drift, and a gate nobody enables is not a gate.
# Nothing here needs Sphinx or a rendered manual. The one check that does need apiary's generated
# reference reports itself as skipped when that is absent, and the documentation build runs the
# same script with the index so the full set still gates a docs job.

# The category pseudo-target these tests hang from. Modules get theirs from the module macro;
# this tree is not a module, so it creates its own and attaches it to Tests.Unit.
include(Einsums_AddPseudoTarget)
include(Einsums_AddPseudoDependencies)
einsums_add_pseudo_target(Tests.Unit.Docs)
einsums_add_pseudo_dependencies(Tests.Unit Tests.Unit.Docs)

# ── The documented C++ examples, compiled and run ───────────────────────────

einsums_add_executable(
  DocSnippets_test INTERNAL_FLAGS
  SOURCES DocSnippets.cpp
  NOINSTALL
)

einsums_add_unit_test("Docs" DocSnippets)

einsums_add_executable(
  TutorialSnippets_test INTERNAL_FLAGS
  SOURCES TutorialSnippets.cpp
  NOINSTALL
)

einsums_add_unit_test("Docs" TutorialSnippets)

# ── The documented Python examples, and the claims a machine can settle ─────

if(EINSUMS_BUILD_PYTHON)
  einsums_add_python_unit_test("Docs" Snippets SCRIPT test_doc_snippets_python.py)

  # check_doc_claims.py is an ordinary script rather than a pytest module, so it is registered
  # directly. EINSUMS_WITH_CXX_STANDARD is passed in because the claim being checked is about
  # this build, not about whatever the script might guess.
  add_test(
    NAME Tests.Unit.Docs.Claims
    COMMAND
      ${Python_EXECUTABLE} "${PROJECT_SOURCE_DIR}/docs/tools/check_doc_claims.py"
      --docs-dir "${PROJECT_SOURCE_DIR}/docs/sphinx"
      --repo-root "${PROJECT_SOURCE_DIR}"
      --cxx-standard "${EINSUMS_WITH_CXX_STANDARD}"
  )
  set_tests_properties(
    Tests.Unit.Docs.Claims
    PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib" LABELS "UNIT_ONLY"
  )
endif()
