..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _howto-ccsd:

*****************************
Build a Spin-Orbital CCSD
*****************************

A correlated method end to end: integrals in, converged correlation energy out.
CCSD is a good subject because it exercises nearly everything the library offers, including views for orbital blocks, intermediates, antisymmetrizers, an amplitude iteration, and a convergence test.

The complete program is ``examples/psi4-bridge/ccsd_spinorbital_numpy_style.py``.
This page explains the decisions in it rather than reprinting all of it, so read the two together.

Running it on cc-pVDZ water reproduces psi4:

.. code-block:: text

    MP2 (spin-orbital einsums) = -0.2041547996
    converged in 28 iters
    spin-orbital einsums CCSD corr = -0.2134804971
    psi4 conv CCSD corr            = -0.2134804971
    difference                     = 1.26e-11

.. contents:: On this page
    :local:
    :depth: 1

Get the integrals in
====================

Integrals come from psi4 and cross into Einsums as buffers.
Neither library is compiled against the other, so either side can be rebuilt freely.
:ref:`howto-from-psi4` covers the bridge in general; for spin-orbital CCSD the path is an AO integral transform, a spin block, and an antisymmetrization, all of which are NumPy work done once before the method starts:

.. code-block:: python

    mo = np.einsum("mp,nr,lq,os,mnlo->pqrs", C, C, C, C, Iao, optimize=True)
    # spin-block into interleaved spin orbitals, then antisymmetrize
    gnp = G.reshape(nso, nso, nso, nso)
    gnp = gnp - gnp.transpose(0, 1, 3, 2)          # <PQ||RS>

Doing this in NumPy is deliberate.
It runs once, it is not the expensive part, and the Einsums half of the program is the iteration.
The guidance to write correlated methods in Einsums rather than NumPy is about the parts that repeat.

Slice the orbital blocks
========================

CCSD is written over occupied and virtual blocks of the antisymmetrized integrals.
Each slice is taken once and handed to Einsums as its own tensor:

.. code-block:: python

    o, vv = slice(0, no), slice(no, nso)

    def t(name, a):
        return einsums.asarray(np.ascontiguousarray(a), name=name)

    g_oovv = t("oovv", gnp[o, o, vv, vv])
    g_ooov = t("ooov", gnp[o, o, o, vv])
    g_oooo = t("oooo", gnp[o, o, o, o])
    g_vvvv = t("vvvv", gnp[vv, vv, vv, vv])
    # ... and the rest

The names are worth setting.
They are what identifies an operation in profiler output and in a graph's ``explain()`` report, and a program with eleven integral blocks is much easier to read there when they are not all called ``tensor``.

Note ``np.ascontiguousarray`` before the handoff.
:ref:`howto-from-numpy` explains why conversion into Einsums copies; passing a non-contiguous NumPy slice works but makes the copy do more work than it needs to.

Write the contraction helper
============================

An Einsums contraction writes into an output you supply rather than returning one.
For a term-by-term transcription of equations that is a little verbose, so the program defines one helper and uses it everywhere:

.. code-block:: python

    def E(spec, A, B, shape, name="x", pf=1.0):
        out = einsums.create_zero_tensor(name, list(shape), dtype="float64")
        einsums.einsum(spec, out, A, B, c_pf=0.0, ab_pf=pf)
        return out

This is the single most useful thing to copy from the example.
It lets the rest of the program read like the equations:

.. code-block:: python

    Fme = E("me <- nf ; mnef", t1, g_oovv, (no, nv), "Fme")

The explicit ``c_pf=0.0`` says the output is overwritten rather than accumulated, and ``ab_pf`` carries the coefficient from the equation, so a term with a factor of one half does not need a separate scaling step.

A second helper does the same for ``einsums.permute``, and the next section is the only place the program needs it:

.. code-block:: python

    def perm(spec, x, shape, name="p"):
        out = einsums.create_zero_tensor(name, list(shape), dtype="float64")
        einsums.permute(spec, out, x)
        return out

Write the antisymmetrizers where the equations write them
=========================================================

The :math:`P(ij)` and :math:`P(ab)` operators go in the subscript string, as a prefix on the term, exactly as the coupled-cluster literature prints them.
There is no helper to write and no permute-and-subtract to get backwards:

.. code-block:: python

    # one contraction, antisymmetrized in a and b
    E("i,j,a,b <- P(ab) i,j,a,e ; b,e", t2, be, SH2, "t2ab")

    # a tensor you already have, antisymmetrized in both pairs
    perm("i,j,a,b <- P(ij) P(ab) i,j,a,b", ring, SH2, "ringP")

Both spellings run eagerly and both record into a graph; neither is capture-only.
Commas between index letters are allowed and are worth using in a rank-4 spec, where ``i,j,b,a`` is a good deal easier to check against the paper than ``ijba``.

**An operator names letters, not axes.**
That is the whole reason to prefer it, and it removes a class of bug this example was originally written around: in the T2 residual :math:`T2[i,j,a,b]`, :math:`P(ij)` is axes (0,1) and :math:`P(ab)` is axes (2,3), but in :math:`W_{mnij}[m,n,i,j]` the same :math:`P(ij)` is axes (2,3), and in :math:`W_{abef}[a,b,e,f]` the same :math:`P(ab)` is axes (0,1).
Write ``P(ij)`` and the letters decide; write a transposition and you have to work it out per intermediate, which is where a 2e-6 ghost hid while MP2 stayed exact.

The general form partitions the output letters with slashes: ``P(i/jk)`` is three terms, ``P(ij/kl)`` is six, and ``P(ij)`` is sugar for ``P(i/j)``.
The two spellings differ in what they cost.
On ``einsum`` the contraction runs once into a temporary shaped like the output, and each term is a signed accumulation out of it; on ``permute`` there is no temporary at all, because every term reads a source the operation never writes.
So fold the operator into the ``einsum`` when the antisymmetrized quantity *is* one contraction, and use ``permute`` when it is a sum of several, as the ring term is.
Captured, neither allocates per call: ``AntisymmetrizerExpansion`` lowers the operator and the temporary becomes the graph's to plan.

Form the amplitudes
===================

The initial guess is MP2, which is the integrals divided by the denominators.
Division is an operator, and it maps onto a real elementwise kernel rather than a temporary-heavy expression:

.. code-block:: python

    Dijab = t("Dijab", eo[:, None, None, None] + eo[None, :, None, None]
                       - ev[None, None, :, None] - ev[None, None, None, :])

    t1 = einsums.zeros((no, nv), name="t1")
    t2 = t("t2", gnp[o, o, vv, vv]) / Dijab

The same operator closes each iteration, where the new amplitudes are the residual over the denominators:

.. code-block:: python

    t1n = t1n / Dia
    t2n = t2n / Dijab

Assemble the intermediates
==========================

The one-body and two-body intermediates are direct transcriptions.
Each is a sum of contractions, and accumulating them with ``+`` is clear enough at this size:

.. code-block:: python

    Fae = E("ae <- mf ; amef", t1, g_vovv, (nv, nv), "Fae") \
        - E("ae <- mnaf ; mnef", taut, g_oovv, (nv, nv), "Fae2", pf=0.5)

    wt = E("m,n,i,j <- P(ij) j,e ; m,n,i,e", t1, g_ooov, OOOO, "wt")
    Wmnij = g_oooo + wt \
        + E("mnij <- ijef ; mnef", tau, g_oovv, OOOO, "Wmnij2", pf=0.25)

Shapes are hoisted into named tuples at the top of the loop, because they are used repeatedly and a transposed pair here is a tedious bug to find:

.. code-block:: python

    OOOO, OVVO, SH2 = (no, no, no, no), (no, nv, nv, no), (no, no, nv, nv)
    VVVV = (nv, nv, nv, nv)

Compute the energy
==================

The CCSD energy is two dot products, one over the doubles and one over a singles intermediate:

.. code-block:: python

    def energy(t1, t2):
        e = 0.25 * float(la.dot(g_oovv, t2))
        X = E("ia <- ijab ; jb", g_oovv, t1, (no, nv), "Xen")
        return e + 0.5 * float(la.dot(t1, X))

``la.dot`` returns a value, which is what you want in eager code.
Inside a graph capture the returning form throws and you use the pointer-writing spelling instead; see :ref:`howto-graphs`.

Converge
========

The loop is ordinary Python.
Convergence is on the energy change:

.. code-block:: python

    for it in range(100):
        # ... form intermediates, update t1 and t2 ...
        e_new = energy(t1, t2)
        if abs(e_new - e_old) < 1e-11:
            break
        e_old = e_new

On cc-pVDZ water this converges in 28 iterations and agrees with psi4 to 1.26e-11.

Validate against a reference
============================

Do not skip this step, and do not write it afterwards.
The example exists in two forms on purpose: ``ccsd_spinorbital_oracle.py`` is a pure NumPy implementation checked against psi4, and ``ccsd_spinorbital_numpy_style.py`` is the Einsums transcription of it term for term.

Having the oracle first is what makes a wrong term findable.
Every contraction in the Einsums version corresponds to one NumPy einsum that is already known to be right, so a disagreement localizes to a single line instead of to a method.

Writing this way found two genuine library bugs, recorded in the example's own header: a rank-4 by rank-4 to rank-2 contraction aborting in the packed backend, and a transposed-output GEMM such as ``"ia <- ma ; mi"`` ignoring the requested index order.
Both are fixed, and both were found because a term disagreed with a reference rather than because a result looked wrong.

Making it faster
================

The version on this page is eager, which is the right place to start because eager is the reference semantics.
It is not where the performance is.

The iteration body is the same work every time, which is exactly the shape a :doc:`ComputeGraph </user/tutorial_compute_graph>` is for.
Capture the body once, optimize it, and replay it, as ``ccsd_rhf_graph_numpy_style.py`` does for the closed-shell case.
Fusion, memory planning, and batching all operate on the captured form, and none of them are available to the eager calls above.

Read :doc:`/user/tutorial_best_practices` before doing that conversion.
Two points from it matter most here: everything you allocate per iteration wants to come from a pool rather than a fresh allocation, and the intermediates that die on first read should never be materialized at all.

Next
====

- :ref:`howto-graphs` for converting the iteration body to a captured graph.
- :ref:`howto-from-psi4` for the integral bridge in detail.
- :doc:`/user/tutorial_best_practices` for what to fix once it is captured.
