..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _api_reference:

*************
API Reference
*************

Reference documentation for the Einsums C++ library and the Einsums Python
package. Both references are generated directly from the C++ headers by the
apiary libclang tool, so they always track the sources.

C++ API
=======

The C++ reference is organized per module. Each page in the
:ref:`library modules overview <modules_overview>` describes what the
module does and links to the generated reference for its public headers.

Python API
==========

Reference for the Einsums Python package (``import einsums``). The bulk of
the API reference is generated from the C++ binding annotations, so it
always tracks the bound surface. On top of that, a thin pure-Python
NumPy-parity layer adds the array conveniences documented in the ergonomics
page.

The :ref:`generated Python API reference <api_python>` is grouped by
submodule. It documents every bound class, function, enum, property, and
the codegen-synthesized subscript/iterator protocols. The
:ref:`tensor ergonomics page <einsums_tensor_ergonomics>` documents the
pure-Python convenience layer installed on the runtime tensor and view
classes.

.. toctree::
    :maxdepth: 2

    ../reference/python/index
    ../reference/einsums_tensor_ergonomics
