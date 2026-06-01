..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_Einsums_Python:

######
Python
######

The ``Python`` module exposes Einsums to Python via pybind11. The bindings
are **automatically generated** at build time from the C++ headers by a
purpose-built libclang tool (``einsums-pybind``), so the Python surface
tracks the C++ surface without hand-written glue.

What gets bound
===============

The generated Python package ``einsums`` mirrors the C++ namespace structure:

- ``einsums`` — tensor types, factory functions (``create_random_tensor`` …),
  configuration entry points (``initialize``, ``finalize``, ``rc``).
- ``einsums.linalg`` — :ref:`LinearAlgebra <modules_Einsums_LinearAlgebra>`
  bindings (``gemm``, ``invert``, ``syev``, ``det`` …).
- ``einsums.graph`` — :ref:`ComputeGraph <modules_Einsums_ComputeGraph>`
  capture, executors, and the bound subset of optimization passes.
- ``einsums.io`` — slab I/O wrappers (``read_slice``, ``write_slice``).
- ``einsums.profile`` — profiling annotations
  (``annotate``, ``mem_*``, ``section`` context manager).
- ``einsums.testing`` — pytest helpers
  (``ALL_DTYPES``, ``REAL_DTYPES``, ``assert_close``).

A typical session looks like:

.. code-block:: python

    import einsums
    import einsums.linalg as la
    import einsums.graph as cg

    A = einsums.create_random_tensor("A", [16, 16])
    B = einsums.create_random_tensor("B", [16, 16])
    C = einsums.create_zero_tensor("C", [16, 16])

    g = cg.Graph("matmul")
    with cg.capture(g):
        la.gemm(1.0, A, B, 0.0, C)
    g.execute()

Build configuration
===================

The Python bindings are gated on a single CMake option::

    cmake -S . -B build -DEINSUMS_BUILD_PYTHON=ON

Turning this on builds the codegen tool, runs it over every annotated
C++ header, compiles the generated pybind translation units into
``_core.cpython-*.so``, and emits ``.pyi`` stubs for editor / IDE
integration.

The generated module
====================

The native extension lives at ``${CMAKE_BINARY_DIR}/lib/einsums/_core.*.so``.
Pure-Python wrappers in ``libs/Einsums/Python/python/einsums/`` add an
ergonomic surface on top — a configuration object (``einsums.rc``),
pretty repr, and a few convenience helpers.

einsums.rc
==========

Pre-import configuration for the runtime. Set fields *before* the first
compute call and they flow into ``einsums::initialize()``:

.. code-block:: python

    import einsums.rc
    einsums.rc.threads = 8
    einsums.rc.log_level = einsums.rc.LogLevel.INFO

    import einsums                 # initialize() fires here
    einsums.gemm(...)

Once the runtime is up the fields are read-only as far as Einsums is
concerned (changing them post-init has no effect).

The same fields can be set via environment variables — useful for test
harnesses that need to disable signal handlers and debugger prompts
without monkey-patching::

    EINSUMS_DEBUG_NO_INSTALL_SIGNAL_HANDLERS=1
    EINSUMS_DEBUG_NO_ATTACH_DEBUGGER=1

See the :ref:`API reference <modules_Einsums_Python_api>` for the full
binding surface and the codegen tool's annotation reference.
