..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _howto-add-a-module:

****************
Add a Module
****************

Einsums is built from modules under ``libs/Einsums/<Name>/``, each with a fixed layout and an explicit dependency list.
This page is the workflow from nothing to a module that builds, tests, and appears in the documentation.
:doc:`/contributors/create_module_skeleton` is the reference for the generator's own options.

.. contents:: On this page
    :local:
    :depth: 1

Generate the skeleton
=====================

Do not create the directories by hand.
The generator writes the layout, the CMake file, and the documentation stubs, and it knows which of those are index files that must not be edited afterwards.

.. code-block:: bash

    python3 libs/create_module_skeleton Einsums MyModule

``Einsums`` is the top-level library and ``MyModule`` is the new module.
Two options are worth knowing at this point:

``--python``
    Adds the ``PYBIND`` keyword to the module's ``einsums_add_module`` call, so apiary generates bindings from the module's annotated headers when ``EINSUMS_BUILD_PYTHON`` is on.
    The module stays an ordinary C++ module under its real library.

``--gpu``
    Sets the module up as a GPU-enabled one.

What you get
============

.. code-block:: text

    libs/Einsums/MyModule/
      CMakeLists.txt              calls einsums_add_module()
      include/Einsums/MyModule/   public headers
      src/                        implementation
      tests/unit/                 unit tests, C++ and test_*_python.py
      tests/performance/          benchmarks
      docs/index.rst              the module's documentation page

The public header path matters.
A header at ``include/Einsums/MyModule/Thing.hpp`` is included by consumers as ``<Einsums/MyModule/Thing.hpp>``, and that is the only path that is part of the module's interface.

Declare sources and dependencies
================================

``libs/Einsums/MyModule/CMakeLists.txt`` is yours to edit.
List headers and sources, then name the modules you depend on:

.. code-block:: cmake

    set(MyModuleHeaders Einsums/MyModule/Thing.hpp)
    set(MyModuleSources Thing.cpp)

    include(Einsums_AddModule)
    einsums_add_module(
      Einsums MyModule
      SOURCES ${MyModuleSources}
      HEADERS ${MyModuleHeaders}
      DEPENDENCIES
      MODULE_DEPENDENCIES Einsums_Config Einsums_Errors
      CMAKE_SUBDIRS examples tests
    )

``MODULE_DEPENDENCIES`` names other Einsums modules, prefixed with ``Einsums_``.
``DEPENDENCIES`` is for external targets.
Keep both minimal: the dependency list is what makes the build order tractable, and a module that depends on everything is a module nobody can build in isolation.

Reindex, then place the module
==============================

``libs/Einsums/CMakeLists.txt`` holds the module list, and ``libs/overview.rst`` links every module page into the site.
The generator maintains both:

.. code-block:: bash

    python3 libs/create_module_skeleton --reindex

Run this after adding a module and after removing one.
It adds the new module's page to the ``overview.rst`` toctree, where a page must be for the warnings-as-errors documentation build to pass.

The module list is in dependency order, and that order is kept by hand.
The generator preserves it, drops modules that are gone, and appends new ones at the end with a note.
When a module already in the list needs yours at configure time, move yours above it; the next reindex keeps it there.

Where shared build helpers go
=============================

A CMake helper used by more than one module belongs in the top-level ``cmake/`` directory.
That directory is on ``CMAKE_MODULE_PATH`` from the root scope, so it imposes no ordering constraint on anything.

Putting a helper in ``libs/Einsums/MyModule/cmake/`` instead looks tidier and is a trap.
The directory-scoped append has to be exported with ``set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)``, and from then on every consumer of the helper must be configured after your module.
Avoid it.

Add tests
=========

C++ tests go in ``tests/unit/`` and register with two calls:

.. code-block:: cmake

    einsums_add_executable(MyThing_test INTERNAL_FLAGS SOURCES MyThing.cpp NOINSTALL)
    einsums_add_unit_test("Modules.MyModule" MyThing)

The test appears to ctest as ``Tests.Unit.Modules.MyModule.MyThing``.

**Do not add ``target_link_libraries`` for Einsums modules.**
``einsums_add_executable`` already links the library, and linking it again produces duplicate symbols.

Python tests live beside them, per module, as ``tests/unit/test_*_python.py``:

.. code-block:: cmake

    einsums_add_python_unit_test("Modules.MyModule" Name SCRIPT test_name_python.py)

Run one by hand with:

.. code-block:: bash

    PYTHONPATH=build/lib python -m pytest libs/Einsums/MyModule/tests/unit/test_name_python.py

``pytest.ini`` points ``testpaths`` at ``libs``, so a bare ``PYTHONPATH=build/lib python -m pytest`` collects the whole per-module suite.

Prefer parametrizing over ``einsums.testing``'s ``ALL_DTYPES`` rather than testing float64 alone.
Tests written only for float64 have hidden real bugs before.

Add options and bindings
========================

If the module reads a runtime setting, declare it in the module's own ``Options.hpp``.
See :ref:`howto-add-an-option`, and note the rule about registering from a namespace-scope initializer rather than from the ``register_arguments`` hook.

If the module exposes types to Python, annotate the headers and build with ``-DEINSUMS_BUILD_PYTHON=ON``.
See :ref:`howto-expose-to-python`, particularly the part about ``EINSUMS_EXPORT``, which is the failure that shows up only when loading ``_core.so``.

Checklist
=========

#. ``python3 libs/create_module_skeleton Einsums MyModule``
#. Fill in ``CMakeLists.txt``: sources, headers, module dependencies.
#. ``python3 libs/create_module_skeleton --reindex``, then move the module earlier in the list if a module above it needs it.
#. Write the code under ``include/Einsums/MyModule/`` and ``src/``.
#. Add unit tests under ``tests/unit/``, without extra ``target_link_libraries``.
#. Configure and build; run ``ctest --test-dir build -R MyModule``.

Next
====

- :doc:`/contributors/create_module_skeleton` for every option the generator takes.
- :ref:`howto-add-an-option` for module configuration.
- :ref:`howto-expose-to-python` for bindings.
