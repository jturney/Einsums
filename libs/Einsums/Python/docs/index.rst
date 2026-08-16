..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_Einsums_Python:

######
Python
######

The ``Python`` module exposes Einsums to Python via pybind11. The bindings are
generated at build time from the C++ headers by `Apiary
<https://github.com/Einsums/Apiary>`_, a purpose-built libclang tool vendored at
``external/apiary``, so the Python surface tracks the C++ surface without
hand-written glue.

This page covers the mechanism and the package layout. For how to *use* the
bindings, see :doc:`Einsums in Python </user/python>`; for the full generated
listing, see the :ref:`Python API reference <api_python>`.

How the bindings are generated
==============================

Apiary parses each annotated header with libclang, builds a JSON IR of the
declarations marked for export, and emits both the pybind11 translation units
and the ``.pyi`` stubs from that one IR. Nothing is hand-maintained, so a
signature cannot drift between C++, Python, and the type stubs.

Headers opt in with ``APIARY_*`` annotations:

``APIARY_EXPOSE``
    Bind this declaration.
``APIARY_MODULE("graph")``
    Place it in the ``einsums.graph`` submodule rather than the package root.
``APIARY_INSTANTIATE_AS("scale", einsums::RuntimeTensor<double>)``
    Instantiate a template at a concrete type and bind it under the given name.
    Overloads that share a Python signature but differ only in element type are
    folded into **one** Python entry taking a ``dtype=`` keyword.
``APIARY_HOLDER(std::shared_ptr)`` / ``APIARY_NOCOPY`` / ``APIARY_NOMOVE``
    Holder and value-semantics control.
``APIARY_GETTER("num_merged")``
    Bind an accessor as a Python **property**, not a method. This is why the
    optimization passes' counters are ``p.num_merged``, without parentheses.
``APIARY_RELEASE_GIL`` / ``APIARY_RVP(...)`` / ``APIARY_KEEP_ALIVE(0, 1)``
    GIL release around long-running calls, return-value policy, and lifetime
    tethering for views.

``ComputeGraph/Operations.hpp`` is the reference example: to expose a new operand
type, add an ``APIARY_INSTANTIATE_AS`` line and the codegen folds it into the
existing dtype-dispatched entry and regenerates the stubs.

.. note::

   A symbol missing ``EINSUMS_EXPORT`` breaks ``_core.so`` at dlopen time even
   when the C++ tests link fine. Binding functions that print must call
   ``std::cout.flush()`` for pytest's ``capfd`` to see the output.

The package
===========

The native extension lives at ``${CMAKE_BINARY_DIR}/lib/einsums/_core.*.so``.
Thin pure-Python shells in ``libs/Einsums/Python/python/einsums/`` re-export from
it lazily, so the first attribute access is what brings the runtime up and
``einsums.rc`` still has effect at import time.

``einsums``
    Tensor types (``RuntimeTensor{F,D,C,Z}``, the matching views, and
    ``TiledRuntimeTensor{F,D,C,Z}``), the NumPy-style factories (``zeros``,
    ``ones``, ``empty``, ``full``, ``eye``, ``array``, ``asarray`` and their
    ``*_like`` forms), ``create_random_tensor`` / ``create_zero_tensor``,
    ``einsum``, ``permute``, and the configuration entry points.
``einsums.linalg``
    :ref:`LinearAlgebra <modules_Einsums_LinearAlgebra>`: ``gemm``, ``gemv``,
    ``syev``/``heev``, ``svd``, ``axpy``/``axpby``, ``direct_product``,
    ``direct_division``, ``element_transform``, the norms, and the rest.
``einsums.graph``
    :ref:`ComputeGraph <modules_Einsums_ComputeGraph>`: ``Graph``, ``Pipeline``,
    ``Workspace``, ``CaptureContext``, the executors, the bound optimization
    passes, and the Python-side ``capture`` / ``default_pass_manager`` /
    ``diis`` helpers.
``einsums.io``
    :ref:`TensorIO <modules_Einsums_TensorIO>`: ``TensorFile``, ``Slab``,
    ``read`` / ``write`` and the slab-granular ``read_slice`` / ``write_slice``.
``einsums.gpu``
    Device queries: ``device_name``, ``device_synchronize``,
    ``available_device_memory``, and the mock-device memory limit used by tests.
``einsums.profile``
    :ref:`Profile <modules_Einsums_Profile>` annotations: the ``section``
    context manager, ``annotate`` / ``annotate_dims``, ``mem_alloc`` /
    ``mem_free``, ``print_report``, ``export_json``.
``einsums.testing``
    Pytest helpers, see `einsums.testing`_ below.
``einsums.interop``
    Bridges to other codes, see `einsums.interop`_ below.

A typical session:

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
    g.apply(cg.default_pass_manager())
    g.execute()

einsums.graph
=============

The graph module carries the full capture surface, plus a few things that only
exist on the Python side.

``cg.capture(graph)``
    Context manager wrapping ``CaptureGuard``. It also maintains a Python-side
    stack so the NumPy-ergonomics operators know where to allocate the
    intermediates for expressions like ``A + B`` and ``A @ B``: inside capture
    they come from ``graph.create_zero_tensor`` rather than the process-owned
    factory, which is what keeps them alive past ``graph.execute()``.
``cg.current_graph()``
    The innermost graph being captured, or ``None``.
``cg.default_pass_manager()``
    A fresh ``PassManager`` pre-loaded with the canonical pass list, mirroring
    C++'s ``PassManager::create_default()``. The static factory cannot be bound
    directly (``PassManager`` holds non-copyable members), so this constructs an
    empty one and calls ``populate_default()`` in place.
``cg.diis(pairs, k=8)`` / ``cg.DIISAccelerator``
    Pulay DIIS extrapolation for fixed-point iterations captured as graph loops.
    It keeps a short history of (amplitude, step) snapshots and replaces the
    amplitudes with the least-squares extrapolant between replays, taking the
    update step itself as the error vector.

These optimization passes are bound as Python classes: ``ConstantFolding``,
``ScaleAbsorption``, ``CSE``, ``DeadNodeElimination``, ``ElementWiseFusion``,
``SymmetrizedAccumulation``, ``LinearCombinationContractionFolding``,
``LoopInvariantHoisting``, ``ScratchPrivatization``, ``StreamContractionFusion``,
``DistributiveFactoring``, ``Reorder``, ``SymmetryPropagation``,
``InplaceOptimization`` and ``MemoryPlanning``. Each is constructible standalone
and each exposes its counters as **properties**:

.. code-block:: python

    p  = cg.InplaceOptimization()
    pm = cg.PassManager()
    pm.add(p)
    pm.run(g)
    print(p.num_merged, p.num_candidates)   # properties, not methods

``cg.default_pass_manager()`` still runs the whole C++ pipeline, including the
passes that have no Python class of their own. Only individually-constructible
passes are listed above.

einsums.testing
===============

Helpers for the pytest suites, and for anyone writing numerical tests against
Einsums:

``ALL_DTYPES`` / ``REAL_DTYPES`` / ``COMPLEX_DTYPES``
    Parametrization lists. Prefer parametrizing over ``ALL_DTYPES``:
    float64-only graph tests have a history of hiding real bugs.
``EXACT_DTYPES`` / ``EXACT_PREFACTORS``
    The subset for which integer-valued data contracts exactly.
``tolerance_for(dtype)``
    The (rtol, atol) pair for a dtype.
``assert_close(actual, expected)``
    Tolerance-aware comparison that infers the dtype from the operands.
``integer_data(shape, dtype, rng, radius=4)`` / ``assert_exact(actual, expected)``
    Exact-arithmetic testing: integer-valued inputs small enough that the
    contraction stays inside the 2**53 exact range, then an exact comparison.

Python tests live per module under
``libs/Einsums/<Module>/tests/unit/test_*_python.py``, never in the top-level
``tests/``. Run one by hand with::

    PYTHONPATH=build/lib python -m pytest libs/Einsums/ComputeGraph/tests/unit/test_einsum_python.py

einsums.interop
===============

``einsums.interop.psi4`` assembles Einsums tensors from psi4's neutral integral
exports. Nothing in it imports psi4: every function is duck-typed against the
data it needs (``nirrep()`` / ``symmetry()`` / ``.nph`` for symmetry-blocked
matrices, or any 2D array-like for single-block data), so the two projects never
need to be compiled against each other and either side can be rebuilt freely.

``from_matrix``
    A symmetry-blocked ``psi4.core.Matrix`` (``mints.so_overlap()`` and friends).
``so_eri``
    ``mints.so_eri_blocked()``: a list of ``([hp, hq, hr, hs], Matrix)`` pairs.
``mo_bra_half_transform``
    ``mints.mo_bra_half_transform(C1, C2)``: an ``(n1*n2, nbf*nbf)`` Matrix.
``df_tensor``
    ``DFTensor.Qso()`` / ``Qov()`` / ...: a ``(naux, d2*d3)`` Matrix.
``dense``
    Any single-block 2D array-like.

Build configuration
===================

The Python bindings are gated on a single CMake option::

    cmake -S . -B build -GNinja -DEINSUMS_BUILD_PYTHON=ON

Turning this on builds the Apiary codegen tool, runs it over every annotated
C++ header, compiles the generated pybind translation units into
``_core.cpython-*.so``, emits the ``.pyi`` stubs for editor and IDE integration,
and registers the Python unit tests with ctest. It also drives the Python half
of the documentation build, which renders the same JSON IR to reStructuredText.

einsums.rc
==========

Pre-import configuration for the runtime. Set fields before the first compute
call and they are turned into ``--einsums:*`` flags and handed to
``einsums::initialize()``:

.. code-block:: python

    import einsums.rc
    einsums.rc.threads = 8
    einsums.rc.log_level = einsums.rc.LogLevel.INFO
    einsums.rc.pass_disable = "GEMMBatching,ContractionPlanning"

    import einsums                 # initialize() fires here
    einsums.einsum(...)

Once the runtime is up the fields are read-only as far as Einsums is concerned.
Changing them post-init has no effect.

There is a field per option and nothing else, because ``rc.py`` is generated
from the option descriptors: apiary parses each module's ``Options.hpp`` and
``libs/Einsums/Python/tools/generate_rc.py`` renders the file from what it
finds, with the prose kept in ``rc.py.in``. A field's name is its option's name
with the leading ``einsums:`` dropped and every ``:`` and ``-`` turned into an
underscore, so ``einsums:profile:report`` is ``profile_report``. Each field
carries the option's help text and default in a comment; :doc:`/user/arguments`
is the longer reference.

The one field that is not an option is ``threads``: Einsums has no flag for it,
so the binding routes it through ``OMP_NUM_THREADS`` instead.

Nothing maps a field to a flag by hand. ``argv_from_rc`` walks the option
registry, asks ``rc`` for the matching attribute, and refuses outright if it
finds an ``rc`` field the registry does not claim - which is what a stale
generated file looks like. Boolean fields are three-valued: ``None`` leaves the
runtime default alone, ``True`` passes the flag, and ``False`` passes its
generated negation, because a flag that already defaults to on cannot be turned
off by staying silent.

``einsums/__init__.py`` also claims the ``--einsums:`` namespace out of
``sys.argv`` at import time and forwards those flags to the runtime, so a Python
script accepts the same options a C++ program does::

    python my_script.py --einsums:pass:disable=GEMMBatching --einsums:log:level=2

The ``debug_no_*`` fields additionally read environment variables, which is what
test harnesses want so they can disable signal handlers and the debugger prompt
without touching the script::

    EINSUMS_DEBUG_NO_INSTALL_SIGNAL_HANDLERS=1
    EINSUMS_DEBUG_NO_ATTACH_DEBUGGER=1

See the :ref:`API reference <modules_Einsums_Python_api>` for the C++ side of
this module, and the :ref:`Python API reference <api_python>` for the generated
binding surface.
