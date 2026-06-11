..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums_tensor_ergonomics:

==============================
Tensor NumPy-parity ergonomics
==============================

The generated :ref:`Python API reference <api_python>` documents the compiled
tensor surface produced by the binding generator. On top of that, a thin
pure Python convenience layer (``einsums/__init__.py``) installs the
NumPy-style attributes and operators that array users reach for reflexively.

These are added in Python, so they are invisible to
the binding-driven docs and are documented here instead. They are installed
on every runtime tensor and view class:
``RuntimeTensorF``, ``RuntimeTensorD``, ``RuntimeTensorC``, ``RuntimeTensorZ``
and the ``RuntimeTensorView{F,D,C,Z}`` variants. ``RuntimeTensor`` below
stands for any of them.

.. versionadded:: 2.0.0

.. py:currentmodule:: einsums

.. note::

   ``@`` (matmul) and the arithmetic operators do not fall through to
   NumPy. They dispatch to :py:mod:`einsums.linalg` so the work stays on
   Einsums code paths, and they are recorded into the active graph when used
   inside ``cg.capture``.

Array attributes
----------------

.. py:class:: RuntimeTensor

    .. py:attribute:: shape
        :type: tuple[int, ...]

        Axis sizes, built from ``rank()``/``dim()`` (works on tensors and
        views). Read-only.

    .. py:attribute:: ndim
        :type: int

        Number of dimensions (``rank()``). Read-only.

    .. py:attribute:: dtype

        The NumPy dtype matching the class's BLAS letter (``F`` for float32,
        ``D`` for float64, ``C`` for complex64, ``Z`` for complex128).
        Read-only.

    .. py:attribute:: T

        Zero-copy reversed-axis (transpose) view. Graph-aware: inside
        ``cg.capture`` it routes through ``cg.permute_view`` so the view is
        recorded with the correct reversed dims/strides. Read-only.

    .. py:method:: __len__() -> int

        Size of the first axis (``dim(0)``); raises on a rank-0 tensor.

    .. py:method:: __repr__() -> str

        ``ClassName(name=..., shape=..., dtype=...)``.

    .. py:method:: __array__(dtype=None, copy=None)

        NumPy array protocol. Zero-copy via the buffer protocol unless a
        ``dtype`` cast or ``copy=True`` forces a copy, so ``np.asarray(t)``
        and ``np.array(t)`` work directly.

    Views and copies
    ^^^^^^^^^^^^^^^^

    .. py:method:: transpose(*axes)

        Zero-copy permuted view (NumPy semantics): ``transpose()`` reverses
        axes, ``transpose(0, 2, 1)`` / ``transpose((0, 2, 1))`` permute them.
        Negative axes allowed. Graph-aware.

    .. py:method:: swapaxes(axis1, axis2)

        Zero-copy view with two axes exchanged.

    .. py:method:: copy()

        A fresh dense tensor holding a copy of this one (like NumPy).

    Reductions
    ^^^^^^^^^^

    .. py:method:: sum()

        Sum of all elements. Normally returns a
        Python scalar, but inside a ``cg.capture`` block, returns a graph ``[1]`` tensor.

    .. py:method:: mean()

        Arithmetic mean (``sum() / size``).

    .. py:method:: max()

        Maximum element. Only defined for totally-ordered types, such as real types.

    Operators
    ~~~~~~~~~

    .. py:method:: __matmul__(other)

        ``A @ B``. matrix-matrix (gemm) or matrix-vector (gemv) via
        :py:mod:`einsums.linalg`, recorded into the graph under capture. The
        right-hand side must be another Einsums tensor with a matching dtype.
        Two vectors (inner product) raise an exception. For these cases, use :py:func:`einsums.linalg.dot`.

    .. py:method:: __add__(other)

        Element-wise addition. ``other`` may be another Einsums tensor or a scalar.
        Dispatches to :py:mod:`einsums.linalg`. Graph-aware. Reflected and
        in-place forms are provided to match NumPy:
        ``__radd__``, ``__sub__``, ``__rsub__``, ``__mul__``, ``__rmul__``,
        ``__truediv__``, ``__neg__``, ``__pos__``, and the in-place
        ``__iadd__``, ``__isub__``, ``__imul__``, ``__itruediv__``.
