..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _python_module:

*********************
Einsums Python Module
*********************

Reference documentation for the Einsums Python package (``import einsums``).

The bulk of the API reference is generated directly from the C++ binding
annotations, so it always tracks the bound surface. On top of that, a thin
pure-Python NumPy-parity layer adds the array conveniences documented in the
ergonomics page.

.. py:currentmodule:: einsums

Generated API reference
=======================

The :ref:`generated Python API reference <api_python>` is produced by
``einsums-pybind --emit-docs-json`` and grouped by submodule.
It documents every bound class, function,
enum, property, and the codegen-synthesized subscript/iterator protocols.

NumPy-parity ergonomics
=======================

The :ref:`tensor ergonomics page <einsums_tensor_ergonomics>` documents the
pure-Python convenience layer installed on the runtime tensor and view
classes.

.. toctree::
    :maxdepth: 3

    python/index
    einsums_tensor_ergonomics
