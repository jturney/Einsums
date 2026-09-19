..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _howto-expose-to-python:

************************
Expose a Type to Python
************************

Einsums' Python bindings are generated from the C++ headers by apiary, a standalone libclang tool.
You annotate a declaration and the binding, the ``.pyi`` stub, and the documentation entry follow from it.
There is no hand-written glue to keep in step, which is the whole point of the arrangement.

.. contents:: On this page
    :local:
    :depth: 1

Build with bindings on
======================

.. code-block:: bash

    cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DEINSUMS_BUILD_PYTHON=ON

That one switch brings in the apiary codegen tool, the bindings, and the Python tests.
The module you are annotating also needs ``PYBIND`` on its ``einsums_add_module`` call, which the skeleton generator adds when you pass ``--python``.

Expose a function or a class
============================

The annotations are attributes on the declaration.
``APIARY_EXPOSE`` is the one that makes something appear at all, and ``APIARY_MODULE`` says which Python submodule it lands in:

.. code-block:: cpp

    #include <Einsums/Python/Annotations.hpp>

    class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_NOCOPY APIARY_NOMOVE
        EINSUMS_EXPORT Graph {
      public:
        APIARY_EXPOSE APIARY_RELEASE_GIL void execute();

        APIARY_EXPOSE APIARY_GETTER("explain")
        [[nodiscard]] std::string const &explain() const { return _last_optimize_report; }
    };

On a non-Clang compiler the macros expand to nothing, so annotations never affect the build itself.

The annotations you will reach for
==================================

.. list-table::
    :header-rows: 1
    :widths: 34 66

    * - Annotation
      - Effect
    * - ``APIARY_EXPOSE``
      - Bind this declaration. Nothing is bound without it.
    * - ``APIARY_MODULE("name")``
      - Place it in that Python submodule.
    * - ``APIARY_RENAME("name")``
      - Use a different Python name.
    * - ``APIARY_GETTER("name")`` / ``APIARY_SETTER("name")``
      - Bind as a property rather than a method.
    * - ``APIARY_RELEASE_GIL``
      - Drop the GIL for the call. Use for anything that computes.
    * - ``APIARY_HOLDER(std::shared_ptr)``
      - Set the pybind11 holder type.
    * - ``APIARY_NOCOPY`` / ``APIARY_NOMOVE``
      - Do not bind copy or move.
    * - ``APIARY_RVP(policy)``
      - Set the return value policy, for example ``reference_internal``.
    * - ``APIARY_KEEP_ALIVE(nurse, patient)``
      - Tie one object's lifetime to another's.
    * - ``APIARY_BUFFER_PROTOCOL``
      - Expose the buffer protocol, which is what makes ``np.asarray`` work.
    * - ``APIARY_EXCEPTION``
      - Translate this type as a Python exception.
    * - ``APIARY_HIDE``
      - Exclude something that would otherwise be picked up.

A getter bound with ``APIARY_GETTER`` becomes a property, so it is read without parentheses in Python.
This catches people out: ``Graph.explain`` is a property while ``PassManager.explain`` is a method, because they are annotated differently.

Expose a template over concrete types
=====================================

Templates cannot be bound directly, so you list the instantiations you want.
``APIARY_INSTANTIATE_AS`` gives them all one Python name, and the generator folds them into a single dtype-dispatched entry point:

.. code-block:: cpp

    APIARY_EXPOSE
    APIARY_INSTANTIATE_AS("scale", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>)
    APIARY_INSTANTIATE_AS("scale", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
    APIARY_INSTANTIATE_AS("scale", einsums::RuntimeTensorView<float>)
    APIARY_INSTANTIATE_AS("scale", einsums::RuntimeTensorView<double>)
    template <typename AType>
    void scale(typename AType::ValueType factor, AType *A) { /* ... */ }

To support a new operand type, add instantiation lines.
That is the whole change: the codegen folds them into the existing Python entry and regenerates the stubs.

``libs/Einsums/ComputeGraph/include/Einsums/ComputeGraph/Operations.hpp`` is the reference for this pattern and worth reading before adding your own.

The related forms are ``APIARY_INSTANTIATE_MEMBER_AS`` for member functions, ``APIARY_INSTANTIATE_TEMPLATE`` for generating one Python name per instantiation, and ``APIARY_INSTANTIATE_BOOLS`` for expanding boolean template parameters.

The traps
=========

**Missing ``EINSUMS_EXPORT`` breaks ``_core.so`` and nothing else.**
A symbol without it can link fine in every C++ test and still make importing ``einsums`` fail at dlopen.
If the C++ side builds and tests pass but the Python module will not import, this is the first thing to check.

**A brace default emits uncompilable code.**
A parameter defaulted with ``= {}`` produces a ``py::arg`` the compiler rejects.
Give it an explicit default instead.

**Flush standard output.**
A binding that prints must call ``std::cout.flush()``, or pytest's ``capfd`` sees nothing.

**Avoid Python keywords in parameter names.**
The stub generator copies parameter names verbatim, so a parameter named ``pass`` produces a ``.pyi`` that pyright cannot parse.
``PassManager::add`` names its parameter ``optimizer_pass`` for exactly this reason.

**An exposed aggregate gets no constructor.**
If you expose a plain struct and want to build one from Python, declare a constructor and annotate it.
This is why several classes in the tree carry an explicitly defaulted constructor with ``APIARY_EXPOSE`` on it.

Check your work
===============

Build, then import and look:

.. code-block:: bash

    PYTHONPATH=build/lib python -c "import einsums; print(einsums.MyThing)"

Add a Python test next to the module, as ``libs/Einsums/<Module>/tests/unit/test_*_python.py``, and register it with ``einsums_add_python_unit_test``.
Parametrize over ``einsums.testing``'s ``ALL_DTYPES`` rather than testing one dtype.

The generated ``.pyi`` stubs are worth reading once after a change.
They are what editors and type checkers see, and a binding that looks right from Python can still produce a stub that does not type check.

Next
====

- :ref:`howto-add-a-module` for creating a module with ``--python`` from the start.
- :doc:`/user/python` for the Python surface as a user sees it.
