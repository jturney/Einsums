#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Where the three-index integrals come from.

The solver never wants ``(Q|mn)`` whole. Every consumer downstream reads
``(Q|i u)``: auxiliary functions against one LMO and a set of PAOs, and always
restricted to some pair's domains. Asking for the dense AO tensor and slicing it
is the one shape of request that cannot be screened and cannot be threaded, so
the seam belongs after the first transform rather than before it.

That is what this module is: the request, rather than the array. A source is
told what will be asked for, builds it however it likes, and answers. The
current dense implementation ignores the demand entirely and builds everything,
which is a perfectly good way to satisfy a declarative request and is what makes
it the reference the others are checked against.
"""

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import einsums

from . import tensors as ten

__all__ = ["KINDS", "Spaces", "Demand", "ThreeIndexSource", "DenseSource",
           "check_kinds"]

#: The three-index integral classes a DLPNO calculation can ask for, named for
#: the accessor that serves each. MP2 reads only ``q_ia``; coupled cluster adds
#: ``q_ij`` for the one-external integrals and ``q_ab`` for the two- and
#: three-external ones. psi4 builds them in ``compute_qia``, ``compute_qij`` and
#: ``compute_qab``.
KINDS = ("q_ia", "q_ij", "q_ab")


def check_kinds(source, demand):
    """Raise unless *source* implements every kind *demand* asks for.

    Called by a source at the top of its ``declare``, which is the earliest
    point at which the question can be answered and long before the answer
    would otherwise be needed. A source that cannot build ``(Q|a b)`` should say
    so while the calculation is still setting up, naming itself and the kind,
    rather than raise an ``AttributeError`` three phases later from inside a
    contraction.
    """
    unknown = [k for k in demand.kinds if k not in KINDS]
    if unknown:
        raise ValueError(f"unknown integral kind(s) {unknown}; expected {list(KINDS)}")
    missing = [k for k in demand.kinds if getattr(source, k, None) is None]
    if missing:
        raise NotImplementedError(
            f"{type(source).__name__} cannot build {', '.join(missing)}. "
            f"It serves {', '.join(k for k in KINDS if getattr(source, k, None))}. "
            "Use the dense source, or teach this one the missing kind."
        )


@dataclass(frozen=True)
class Spaces:
    """The orbital spaces a source transforms into.

    Both are AO-by-something coefficient matrices, which is all any producer
    needs to know about them: the LMOs it will put in the second index and the
    PAOs it will put in the third.
    """

    #: Localized active occupieds, ``(nbf, naocc)``.
    C_lmo: object
    #: Projected atomic orbitals, ``(nbf, npao)``.
    C_pao: object


@dataclass
class Demand:
    """Which blocks of ``(Q|i u)`` the solver intends to read.

    Declared before anything is built, because a producer that knows the whole
    demand can screen, batch and thread, and one answering a block at a time
    cannot. The lists are per pair domain, in the solver's own indexing.

    A source may satisfy this more coarsely than asked - the dense one satisfies
    it by ignoring it. That is deliberate: if honouring the domains exactly were
    part of the contract, every producer would have to implement domain
    screening and there would be exactly one producer.

    The demand is carried at two granularities, because two useful kinds of
    producer want different things. The distinct-domain lists say which sets of
    auxiliary functions and PAOs exist at all, which is what a producer sizing
    buffers or planning batches asks. The per-auxiliary-atom lists say which
    LMOs and PAOs will be read against each atom's auxiliary functions, which
    is the *pairing* the first pair of lists throws away - and a producer that
    builds AO integrals only where they will be transformed cannot do without
    it, since the atom is what localizes the integral loop.
    """

    #: Auxiliary function indices, one list per distinct domain.
    aux_domains: list = field(default_factory=list)
    #: PAO indices, one list per distinct domain.
    pao_domains: list = field(default_factory=list)

    #: Per auxiliary ATOM: the LMO indices that will be read against that
    #: atom's auxiliary functions.
    aux_atom_to_lmos: list = field(default_factory=list)
    #: Per auxiliary ATOM: the PAO indices that will be read against that
    #: atom's auxiliary functions.
    aux_atom_to_paos: list = field(default_factory=list)

    #: Which of :data:`KINDS` the calculation will read. MP2 declares ``q_ia``
    #: alone; coupled cluster declares all three.
    #:
    #: Part of the demand rather than implied by which accessor gets called,
    #: because the point of declaring is to be asked everything at once. A
    #: source sizing buffers needs to know it will be asked for ``(Q|a b)``
    #: before it builds ``(Q|i a)``, and a source that cannot build one at all
    #: should refuse now (:func:`check_kinds`) rather than at the call.
    kinds: tuple = ("q_ia",)

    def is_empty(self):
        return not self.aux_domains and not self.pao_domains


@runtime_checkable
class ThreeIndexSource(Protocol):
    """Half-transformed three-index integrals, delivered on request.

    Implementations differ in where the numbers come from, not in what they
    mean. :class:`DenseSource` reads a materialized ``(Q|mn)``; a psi4-backed
    one drives ``DFHelper``; a native builder would drive a screened shell
    triplet loop. All of them answer the same question.
    """

    @property
    def screening_threshold(self) -> float:
        """Zero when the result is exact, else the cutoff in force.

        Part of the interface rather than a backend detail because the port's
        validation rests on reproducing canonical DF-MP2 exactly when nothing
        is truncated. A source that screened silently would break that in a way
        that reads as a domain bug, so the untruncated fixtures assert this is
        zero.
        """

    def declare(self, spaces: Spaces, demand: Demand) -> None:
        """Say what will be asked for, before any of it is."""

    def build(self) -> None:
        """Satisfy the declared demand, however this source likes."""

    def q_ia(self):
        """``(Q | i u)`` over the full LMO and PAO spaces.

        The whole tensor, which is what today's consumers slice. A source that
        builds only the declared domains still has to materialize this until
        those consumers ask by block instead; that is the next move, not this
        one.
        """

    def q_ij(self):
        """``(Q | i j)`` over the full LMO space, ``(naux, naocc, naocc)``.

        psi4's ``compute_qij``. Read by the coupled-cluster one-external
        integrals; MP2 never asks for it. Optional: a source that does not
        define it declines the ``q_ij`` kind, and :func:`check_kinds` says so.
        """

    def q_ab(self):
        """``(Q | u v)`` over the full PAO space, ``(naux, npao, npao)``.

        psi4's ``compute_qab``. Read by the coupled-cluster two- and
        three-external integrals, and by far the largest of the three: the PAOs
        span the whole AO basis, so this is ``naux * nbf^2`` where ``q_ia`` is
        ``naux * naocc * nbf``. Optional, like :meth:`q_ij`.
        """


class DenseSource:
    """Build the whole ``(naux, naocc, npao)`` tensor from a dense ``(Q|mn)``.

    The reference implementation, and deliberately the dumbest one: it ignores
    the declared demand and transforms everything. That makes it exact, makes it
    independent of psi4 (it reads what ``reference_io`` froze into a fixture),
    and makes it the oracle a screened source is differentially tested against -
    the same role eager execution plays for the compute graph.
    """

    def __init__(self, eri_3index):
        self._eri_3index = eri_3index
        self._spaces = None
        self._kinds = ("q_ia",)
        self._blocks = {}

    @property
    def screening_threshold(self) -> float:
        return 0.0

    def declare(self, spaces: Spaces, demand: Demand) -> None:
        # Only the KINDS are consulted, and only so that a run wanting q_ia
        # alone does not pay for the two the coupled-cluster layers want. The
        # domains are ignored, which is the whole point. See the class docstring.
        check_kinds(self, demand)
        self._spaces = spaces
        self._kinds = tuple(demand.kinds)

    def build(self) -> None:
        if self._spaces is None:
            raise RuntimeError("declare() before build()")
        C_lmo, C_pao = self._spaces.C_lmo, self._spaces.C_pao
        naux, nbf = self._eri_3index.shape[0], self._eri_3index.shape[1]
        naocc = ten.shape(C_lmo)[1]
        npao = ten.shape(C_pao)[1]

        # Read the integrals in the layout psi4 handed them over in. A dense
        # (Q|mn) is the largest single buffer this phase touches - 97.7 MiB at
        # ethanol/cc-pVTZ - and it arrives C-contiguous, so copying it into a
        # column-major (Q, m, n) tensor reorders every element for nothing: 18 ms
        # of transpose against 3 ms of memcpy. Taking it reversed makes the copy
        # a memcpy and costs only that the contractions below are written with
        # their indices reversed as well.
        Qmn = ten.from_numpy_reversed("(n m Q)", self._eri_3index)

        # Occupied index first. Both orders give the same answer, but the
        # half-transform carries whichever index has already been contracted,
        # and there are naocc of those against npao of the other - 13 against
        # 174 at ethanol/cc-pVTZ, since the PAOs span the whole AO basis. Doing
        # the PAO index first makes the dominant contraction naux*nbf^2*npao
        # instead of naux*nbf^2*naocc and leaves a half-transform the size of
        # the integrals themselves: 4.79 GFLOP and 97.7 MiB against 0.67 GFLOP
        # and 7.3 MiB. The gap is npao/naocc, so it widens with basis set.
        #
        # The half-transform stays reversed too. It is small (7.3 MiB), so the
        # choice is not about its own copy cost: leaving it as (n, i, Q) keeps
        # the contracted index leading in both operands, which is the batched
        # GEMM shape, where mixing the orders puts a permute back in.
        #
        # Both LMO-first kinds share this one half-transform, which is most of
        # what either costs.
        if {"q_ia", "q_ij"} & set(self._kinds):
            half = ten.empty("(n i Q)", [nbf, naocc, naux])
            einsums.einsum("niQ <- nmQ ; mi", half, Qmn, C_lmo)
            if "q_ia" in self._kinds:
                q_ia = ten.empty("(Q|i u)", [naux, naocc, npao])
                einsums.einsum("Qiu <- niQ ; nu", q_ia, half, C_pao)
                self._blocks["q_ia"] = q_ia
            if "q_ij" in self._kinds:
                q_ij = ten.empty("(Q|i j)", [naux, naocc, naocc])
                einsums.einsum("Qij <- niQ ; nj", q_ij, half, C_lmo)
                self._blocks["q_ij"] = q_ij
            del half

        # (Q|u v) has no cheap index to lead with: both are PAOs, so its half
        # transform is nbf * npao * naux, the size of the AO integrals
        # themselves, and the result is that size again. It is the reason the
        # coupled-cluster layers are the ones that declare it - a dense
        # (Q|uv) is what makes this source the oracle rather than the
        # production path, exactly as it already is for (Q|iu).
        if "q_ab" in self._kinds:
            half_pao = ten.empty("(n u Q)", [nbf, npao, naux])
            einsums.einsum("nuQ <- nmQ ; mu", half_pao, Qmn, C_pao)
            q_ab = ten.empty("(Q|u v)", [naux, npao, npao])
            einsums.einsum("Quv <- nuQ ; nv", q_ab, half_pao, C_pao)
            self._blocks["q_ab"] = q_ab
            del half_pao

    def _block(self, kind):
        hit = self._blocks.get(kind)
        if hit is None:
            if self._spaces is None:
                raise RuntimeError(f"build() before {kind}()")
            raise RuntimeError(
                f"{kind}() was not declared; the demand asked for "
                f"{', '.join(self._kinds)}"
            )
        return hit

    def q_ia(self):
        return self._block("q_ia")

    def q_ij(self):
        return self._block("q_ij")

    def q_ab(self):
        return self._block("q_ab")

    def describe(self):
        """One line for the solver's own report."""
        parts = []
        for kind in KINDS:
            block = self._blocks.get(kind)
            if block is None:
                continue
            shape = ten.shape(block)
            label = {"q_ia": "(Q|iu)", "q_ij": "(Q|ij)", "q_ab": "(Q|uv)"}[kind]
            parts.append(f"{label} {shape[0]}x{shape[1]}x{shape[2]} "
                         f"({ten.view(block).nbytes / 2**20:.1f} MiB)")
        return ", ".join(parts) + " (dense)"
