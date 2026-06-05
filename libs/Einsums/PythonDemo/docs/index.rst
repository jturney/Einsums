..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_Einsums_PythonDemo:

##########
PythonDemo
##########

``PythonDemo`` is a small set of header-only examples that exercise the
:ref:`Python <modules_Einsums_Python>` binding generator end-to-end. The
module is built and bound whenever ``EINSUMS_BUILD_PYTHON=ON`` and serves
two purposes:

- A reference for contributors wanting to see what annotations the codegen
  expects to produce a working Python surface.
- A smoke-test for the codegen itself — if the demo headers stop binding
  cleanly, the codegen tool (``einsums-pybind``) has regressed.

Contents
========

- ``Counter.hpp`` — a trivial class with bound methods, demonstrating
  the standard codegen path for a stateful object.
- ``Coverage.hpp`` — a broader spread of overloads, default arguments,
  and template instantiation annotations, exercising the harder parts
  of the binding generator (the dtype dispatcher, the ``.pyi`` emitter,
  return-by-tuple handling).

The bound Python surface lives in ``einsums.demo``.

See the :ref:`API reference <modules_Einsums_PythonDemo_api>` of this
module for the exposed C++ surface.
