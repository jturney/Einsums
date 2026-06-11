..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _contributing_documentation:

Documenting the Code
====================

Einsums does not use Doxygen or Breathe. The C++ API reference is produced
by an in-tree libclang tool, which parses the public headers,
reads the documentation comments, and emits reStructuredText in Sphinx's C++ domain. The
same comments become the docstrings of the Python
bindings. You therefore write one doc comment per declaration and it feeds
both surfaces.

This page covers two things:

#. :ref:`how to write a doc comment <writing_doc_comments>` that the tool
   understands, and
#. :ref:`how to cross-reference other items <cross_referencing>` from hand-written
   ``.rst`` pages and from within doc comments.


.. _writing_doc_comments:

Writing a doc comment
---------------------

What gets documented
^^^^^^^^^^^^^^^^^^^^

A declaration appears in the generated reference only when all of the following hold:

* it lives in a module's public ``include/Einsums/<Module>/`` header (not under
  ``detail/``, not in an anonymous or ``detail`` namespace),
* it is not marked ``@internal``, and
* it carries a doc comment.

A public symbol with no doc comment is silently skipped. If you reference such
a symbol from prose the link will dangle, so document the things you link to.

Comment syntax
^^^^^^^^^^^^^^

Use a ``///`` line block or a ``/** ... */`` block immediately above the
declaration. A trailing ``///<`` documents the preceding member:

.. code-block:: cpp

   /// Compute the trace of a square matrix.
   ///
   /// @tparam AType  Any rank-2 tensor type.
   /// @param  A      The matrix whose diagonal is summed.
   /// @return The sum of the diagonal elements.
   template <MatrixConcept AType>
   auto trace(AType const &A) -> typename AType::ValueType;

   struct Options {
       bool   verbose = false;  ///< Print a summary after each pass.
       size_t max_threads = 0;  ///< 0 means "use every core".
   };

Structured fields
^^^^^^^^^^^^^^^^^

These become the field list of the generated declaration:

``@brief``
    One-sentence summary. The first paragraph is treated as the brief even if
    ``@brief`` is omitted.
``@param[in] name``
    A parameter. The optional ``[in]`` / ``[out]`` / ``[in,out]`` direction is
    preserved.
``@tparam name``
    A template parameter.
``@return`` / ``@returns``
    The return value.
``@throws name`` (also ``@throw``, ``@exception``)
    An exception the function may raise.

Inline markup
^^^^^^^^^^^^^

Doxygen inline commands are converted to reST inline markup:

``@c x`` / ``@p x``
    Inline code. Renders ``x`` as a literal.
``@ref Thing``
    A code-styled name. Einsums renders it as a literal, not an auto-link.
``@a x`` / ``@e x``
    *Italic* emphasis.
``@b x``
    **Bold**.
``@f$ ... @f$``
    Inline math using LaTeX.
``@f[ ... @f]``
    Display math, which gives a centered ``.. math::`` block.

``@c``, ``@p`` and ``@ref`` automatically absorb a trailing ``()`` or ``[k]``
so the closing backtick is not followed by a bracket, which reST rejects.

Block callouts
^^^^^^^^^^^^^^

These become reST admonitions. Everything until the next command or a blank line
is the body:

``@note`` (also ``@remark``)
    A ``.. note::`` admonition.
``@warning``
    A ``.. warning::`` admonition.
``@attention``
    An ``.. attention::`` admonition.
``@see`` (also ``@sa``, ``@seealso``)
    A ``.. seealso::`` admonition.
``@pre``
    A "Precondition" admonition.
``@post``
    A "Postcondition" admonition.
``@par Title``
    A bold-titled paragraph.

Version history
^^^^^^^^^^^^^^^^

.. code-block:: cpp

   /// @versionadded{1.0.0}
   /// @versionchanged{2.0.0} Now handles general-rank tensors.

For a multi-paragraph note, use the ``desc`` form and close it with
``@endversion``:

.. code-block:: cpp

   /// @versionaddeddesc{1.0.0}
   ///     The initial implementation only supported rank-1 tensors. See the
   ///     migration notes for the rank-N generalisation in 2.0.0.
   /// @endversion

Lists
^^^^^

Bullet (``-``, ``+``, ``*``) and numbered (``1.``, ``2)``) lists are recognized,
and a wrapped continuation line is joined back onto its item automatically:

.. code-block:: cpp

   /// Dispatch picks the first backend that fits:
   ///
   /// 1. a vendor BLAS call,
   /// 2. the in-tree packed-GEMM backend, falling back to
   ///    a generic loop nest when neither applies.

Raw passthrough
^^^^^^^^^^^^^^^

When you need reST markup that the normalizer will not touch, like a table, a directive, or a code
block, wrap it in one of the following:

* ``@rst ... @endrst`` — emit the body as raw reStructuredText.
* ``@code ... @endcode`` / ``@verbatim ... @endverbatim`` — a literal block.

Gotchas
^^^^^^^

* Bare ``*`` or ``<`` in prose read as the start of emphasis or as a role.
  Write ``A *= b`` as ``` ``A *= b`` ``` and ``FILE *`` as ``` ``FILE *`` ```,
  or use ``@c``.
* Do not hand-write a nested ``.. note::`` inside a ``@versionaddeddesc``
  block. The indentation will be flattened. Put the text directly in the block.
* The normalizer only implements the common Doxygen commands, not all of them.
  An unknown ``@command`` is left as-is,
  so a typo shows up verbatim in the output.


.. _cross_referencing:

Cross-referencing other items
-----------------------------

C++ entities
^^^^^^^^^^^^

Reference a documented C++ symbol with the matching cpp-domain role. Use the
role that matches the kind of the target:

``:cpp:func:``
    Free functions and member functions.
``:cpp:class:`` / ``:cpp:struct:``
    ``class`` / ``struct`` declarations.
``:cpp:type:``
    Type aliases (``using`` / ``typedef``). This is also the permissive choice, as
    it resolves classes and structs too.
``:cpp:concept:``
    Concepts.
``:cpp:member:``
    Class, struct, and union members.
``:cpp:enum:`` / ``:cpp:enumerator:``
    Enums and their values.

.. code-block:: rst

   The :cpp:func:`einsums::einsum` entry point dispatches through
   :cpp:class:`einsums::GeneralTensor` and the :cpp:concept:`einsums::TensorConcept`.

Matching the namespace
^^^^^^^^^^^^^^^^^^^^^^

A reference must resolve to the symbol's fully-qualified name. From a page
that has not set a namespace, an unqualified ``:cpp:func:`einsum``` looks only in
the global scope and dangles. Two ways to fix it:

* Qualify the reference. Prefix it with ``~`` to keep the short display name
  while linking the full path:

  .. code-block:: rst

     :cpp:func:`~einsums::einsum`        renders as "einsum", links the full path
     :cpp:func:`~einsums::blas::getrf`   renders as "getrf"

* Set a page default. Put this once near the top of a page so every later
  unqualified reference is resolved in that namespace:

  .. code-block:: rst

     .. cpp:namespace:: einsums

References resolve across modules. Every declaration registers in one global
cpp-domain index so you can link any documented symbol from anywhere.

Pages and sections
^^^^^^^^^^^^^^^^^^

* ``:doc:`/user/absolute_beginners``` links another document by path.
* ``:ref:`label``` links a labelled location. Define the label immediately above
  a section title:

  .. code-block:: rst

     .. _my_section:

     My Section
     ==========

     ... referenced elsewhere as :ref:`my_section`.

Gotchas
^^^^^^^

* A letter immediately after inline markup breaks it. To pluralise a
  reference, escape the join with a backslash-space:

  .. code-block:: rst

     :cpp:class:`Tensor`\ s          (renders "Tensors", links "Tensor")
     ``vector``\ s                   (renders "vectors")

* Type aliases are not classes. ``Tensor`` and ``RuntimeTensor`` are
  ``using`` aliases, so reference them with ``:cpp:type:``, not ``:cpp:class:``.
* You cannot reference a template instantiation. Link the bare template
  name (``:cpp:class:`einsums::Tensor```, never ``Tensor<double, 2>``).
* Headings/decorations must be at least as long as their title text, or the
  page emits an "underline too short" warning.

Building and checking the docs
------------------------------

.. code-block:: bash

   # configure with the docs toolchain present (conda: add --docs to merge_yml.py)
   cmake -S . -B build -DEINSUMS_WITH_DOCUMENTATION=ON
   cmake --build build --target docs-html

   # nitpicky check — every dangling reference becomes a warning
   sphinx-build -b html -n build/docs/sphinx build/docs/html

The ``-n`` (nitpicky) flag turns every unresolved cross-reference into a
warning. Keep the build warning-free: a new warning almost always means a
reference is mis-qualified, uses the wrong role, or points at an undocumented
symbol.
