#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""What the indices of this port's tensors actually range over.

Four index sets appear in every contraction DLPNO writes, and until now they
were distinguishable only by the letter a human chose and by comments about how
each one scales. :mod:`einsums.graph` can be told instead: an index space is
registered once, a tensor's axes are annotated with the spaces they range over,
and the cost model then reads a contraction's scaling off the graph rather than
off a comment.

That is the whole content of this module. It registers the four spaces, states
the handful of relations between them that are chemically true, and gives the
package one helper for annotating a tensor as it is built.

The scale symbols are the letters this port's own comments already use for the
extents, so a symbolic cost polynomial reads the way the hand-written scaling
arguments in :mod:`dlpno.integrals` read::

    2*Q*n^2*u          the PAO-first half transform, naux * nbf^2 * npao
    2*Q*i*n^2          the LMO-first one,            naux * nbf^2 * naocc

Annotation needs a graph to live on, because it is stored on the graph's handle
for the tensor. :func:`annotate` is therefore a no-op outside ``cg.capture``,
which is what lets the eager paths in this package carry the same annotation
calls as the captured ones: the call documents what a buffer's axes are either
way, and it becomes load-bearing the moment that code is captured.
"""

import einsums.graph as cg

__all__ = [
    "AO", "AUX", "LMO", "PAO",
    "C_LMO", "C_PAO", "F_LMO", "S_PAO", "F_PAO",
    "QMN_REVERSED", "HALF_LMO_REVERSED", "HALF_PAO_REVERSED",
    "Q_IA", "Q_IJ", "Q_AB",
    "register", "annotate",
]

#: Atomic orbitals: the basis the integrals arrive in. Extent ``nbf``.
AO = "ao"
#: The auxiliary (density fitting) basis. Extent ``naux``.
AUX = "aux"
#: Localized active occupied orbitals. Extent ``naocc``.
LMO = "lmo"
#: Projected atomic orbitals, the local virtual set. Extent ``npao``.
PAO = "pao"

# Advisory extents, taken from ethanol/cc-pVTZ, the size the scaling comments in
# dlpno.integrals are written against. They only ever break a tie between two
# cost polynomials that the declared scale order cannot rank, so being an order
# of magnitude right is all that is asked of them.
_TYPICAL = {AO: 174.0, AUX: 500.0, LMO: 13.0, PAO: 174.0}

#: Scale symbol per space. Chosen to match the index letters this port writes
#: its contractions with, so a cost polynomial reads like its comments.
_SYMBOL = {AO: "n", AUX: "Q", LMO: "i", PAO: "u"}

# ── Axis layouts of the tensors this package builds ─────────────────────────
#
# One tuple per persistent buffer, in axis order. Named here rather than spelled
# out at each call site so that the two ends of a transform cannot drift apart:
# the source that builds (Q|i u) and the test that asserts what it costs read
# the same tuple.

#: ``C (LMO)``, the localized occupied coefficients, ``(nbf, naocc)``.
C_LMO = (AO, LMO)
#: ``C (PAO)``, the projected atomic orbital coefficients, ``(nbf, npao)``.
C_PAO = (AO, PAO)
#: ``F (LMO)``, the Fock matrix in the LMO basis, ``(naocc, naocc)``.
F_LMO = (LMO, LMO)
#: ``S (PAO)``, the PAO overlap, ``(npao, npao)``.
S_PAO = (PAO, PAO)
#: ``F (PAO)``, the Fock matrix in the PAO basis, ``(npao, npao)``.
F_PAO = (PAO, PAO)

#: The AO three-index integrals as this port reads them, ``(n, m, Q)``. Reversed
#: relative to psi4's ``(Q|mn)``; see :meth:`dlpno.integrals.DenseSource.build`.
QMN_REVERSED = (AO, AO, AUX)
#: The LMO half transform, ``(n, i, Q)``.
HALF_LMO_REVERSED = (AO, LMO, AUX)
#: The PAO half transform, ``(n, u, Q)``.
HALF_PAO_REVERSED = (AO, PAO, AUX)

#: ``(Q | i u)``, ``(naux, naocc, npao)``.
Q_IA = (AUX, LMO, PAO)
#: ``(Q | i j)``, ``(naux, naocc, naocc)``.
Q_IJ = (AUX, LMO, LMO)
#: ``(Q | u v)``, ``(naux, npao, npao)``.
Q_AB = (AUX, PAO, PAO)


def register(registry=None):
    """Register the four spaces and their relations, and return the registry.

    Idempotent: registering a space that is already there with the same content
    is a no-op, and so is re-declaring a relation, so this may be called from
    anywhere that is about to annotate. ``registry`` defaults to the
    process-wide one, which is what an unconfigured :class:`einsums.graph.Graph`
    resolves its annotations against.

    Only relations that hold for every molecule and every basis are declared;
    the registry derives nothing it was not told, and a wrong declaration is
    worse than a missing one because the passes act on it.
    """
    reg = registry if registry is not None else cg.global_space_registry()

    # Registration order is the order the scale symbols appear in a rendered
    # cost polynomial, so it is chosen to make one read like the hand-written
    # scaling arguments it replaces: ``2*Q*n^2*u`` for ``naux * nbf^2 * npao``.
    ids = {
        name: reg.register_space(cg.index_space(name, _SYMBOL[name], _TYPICAL[name]))
        for name in (AUX, AO, LMO, PAO)
    }

    # naocc < nbf: the occupied orbitals are a proper part of what the basis
    # spans, for every molecule with at least one virtual orbital.
    reg.declare_less(ids[LMO], ids[AO])
    # naocc < npao: there is one PAO per atomic orbital (the projector removes
    # the occupied space from each, it does not remove any of them), so npao is
    # nbf and the previous fact carries over.
    reg.declare_less(ids[LMO], ids[PAO])
    # nbf < naux and npao < naux: a fitting basis has to span products of
    # orbital basis functions, so it is larger than the orbital basis. True of
    # every standard auxiliary set.
    reg.declare_less(ids[AO], ids[AUX])
    reg.declare_less(ids[PAO], ids[AUX])

    # An LMO index and a PAO index never name the same orbital: the PAOs are
    # built by projecting the occupied space out of the AO basis, which is
    # exactly the statement that the two sets share nothing. This is the one
    # relation of the four that a pass can use to call a contraction zero, so
    # it is also the one worth being sure of.
    reg.declare_disjoint(ids[LMO], ids[PAO])

    # Deliberately NOT declared:
    #  - anything relating ao to pao. There are equally many of each, and
    #    neither index set contains the other: a PAO is a combination of AOs,
    #    which is a statement about the spanned subspace, not about the index
    #    sets that `contained` describes.
    #  - containment of lmo or pao in ao, for the same reason.
    #  - disjointness of aux from anything. It is true and nothing reads it;
    #    an auxiliary index and an orbital index never meet under one letter.
    return reg


def annotate(tensor, axes, graph=None):
    """Annotate ``tensor``'s axes with the spaces they range over.

    ``axes`` is one space name per axis, normally one of the layout tuples in
    this module. ``graph`` defaults to the graph currently being captured.

    Outside a capture there is no graph to store the annotation on, and this
    returns the tensor unchanged. That is what lets the eager and captured
    paths in this package share one spelling: the call says what a buffer's
    axes are wherever it appears, and the compute graph picks it up wherever
    there is a graph to pick it up with.

    Returns ``tensor``, so it can wrap an allocation in one line.
    """
    g = graph if graph is not None else cg.current_graph()
    if g is None:
        return tensor
    register(g.space_registry)
    return cg.annotate(tensor, axes, graph=g)
