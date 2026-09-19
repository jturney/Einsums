..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _howto:

#############
How-To Guides
#############

Each page here answers a task rather than explaining a feature.
The :doc:`User Guide </user/index>` teaches the concepts in order and is the better place to start if Einsums is new to you.
Come here when you know roughly what you want and need the specific way to do it.

Every code block on these pages was compiled and run against the version of the library the documentation was built from.
Where a page states what something returns, prints, or throws, that is measured output rather than expectation.

Find your task
==============

.. list-table::
    :header-rows: 1
    :widths: 55 45

    * - I want to
      - See
    * - Contract two tensors and know which kernel ran
      - :ref:`howto-contractions`
    * - Take a diagonal, a trace, or a transpose
      - :ref:`howto-contractions`
    * - Get a single number out of a contraction
      - :ref:`howto-contractions`
    * - Work on a sub-block without copying it
      - :ref:`howto-views`
    * - Write into a tensor I am also reading
      - :ref:`howto-contractions`
    * - Capture a calculation once and replay it
      - :ref:`howto-graphs`
    * - Find out why an optimization pass did nothing
      - :ref:`howto-graphs`
    * - Build a correlated method end to end
      - :ref:`howto-ccsd`
    * - Translate code I already have in NumPy
      - :ref:`howto-from-numpy`
    * - Move data between psi4 and Einsums
      - :ref:`howto-from-psi4`
    * - Add a runtime option to a module
      - :ref:`howto-add-an-option`
    * - Create a new module
      - :ref:`howto-add-a-module`
    * - Expose a C++ type to Python
      - :ref:`howto-expose-to-python`
    * - Write my own optimization pass
      - :ref:`howto-write-a-pass`

.. toctree::
    :caption: Using Einsums
    :maxdepth: 1

    contractions
    views
    graphs

.. toctree::
    :caption: Building a method
    :maxdepth: 1

    ccsd

.. toctree::
    :caption: Porting existing code
    :maxdepth: 1

    from_numpy
    from_psi4

.. toctree::
    :caption: Extending Einsums
    :maxdepth: 1

    add_an_option
    add_a_module
    expose_to_python
    write_a_pass

Two conventions used throughout
===============================

**Specs use a semicolon.**
An Einsums contraction spec separates the two operands with ``;`` and not with the comma NumPy uses.
``"ik;kj->ij"`` and the equivalent ``"ij <- ik ; kj"`` are both accepted.
In C++ the spec is checked while your code compiles, so a malformed one is a compile error rather than a surprise at run time.

**Graph work prefers RuntimeTensor.**
:cpp:type:`einsums::RuntimeTensor` is the tensor type the :doc:`ComputeGraph </user/tutorial_compute_graph>` is built around, and it is what the Python bindings expose, so a C++ capture and its Python equivalent read the same.
The statically ranked :cpp:type:`einsums::Tensor` works in a capture as well, and is the natural choice for eager code whose ranks are known while you write it.

What does matter in a capture is saying which graph-owned tensors are results rather than scratch, because the optimizer removes computations nothing reads.
:ref:`howto-graphs-intermediate` covers it, and it is the one mistake on these pages that can pass without an error.
