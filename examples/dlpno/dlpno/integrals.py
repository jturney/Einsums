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

__all__ = ["Spaces", "Demand", "ThreeIndexSource", "DenseSource"]


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
    """

    #: Auxiliary function indices, one list per distinct domain.
    aux_domains: list = field(default_factory=list)
    #: PAO indices, one list per distinct domain.
    pao_domains: list = field(default_factory=list)

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
        self._q_ia = None

    @property
    def screening_threshold(self) -> float:
        return 0.0

    def declare(self, spaces: Spaces, demand: Demand) -> None:
        # The demand is not consulted. See the class docstring.
        self._spaces = spaces

    def build(self) -> None:
        if self._spaces is None:
            raise RuntimeError("declare() before build()")
        C_lmo, C_pao = self._spaces.C_lmo, self._spaces.C_pao
        naux, nbf = self._eri_3index.shape[0], self._eri_3index.shape[1]
        naocc = ten.shape(C_lmo)[1]
        npao = ten.shape(C_pao)[1]

        Qmn = ten.from_numpy("(Q|mn)", self._eri_3index)

        # Occupied index first. Both orders give the same answer, but the
        # half-transform carries whichever index has already been contracted,
        # and there are naocc of those against npao of the other - 13 against
        # 174 at ethanol/cc-pVTZ, since the PAOs span the whole AO basis. Doing
        # the PAO index first makes the dominant contraction naux*nbf^2*npao
        # instead of naux*nbf^2*naocc and leaves a half-transform the size of
        # the integrals themselves: 4.79 GFLOP and 97.7 MiB against 0.67 GFLOP
        # and 7.3 MiB. The gap is npao/naocc, so it widens with basis set.
        half = ten.zeros("(Q|i n)", [naux, naocc, nbf])
        einsums.einsum("Qin <- Qmn ; mi", half, Qmn, C_lmo)
        self._q_ia = ten.zeros("(Q|i u)", [naux, naocc, npao])
        einsums.einsum("Qiu <- Qin ; nu", self._q_ia, half, C_pao)

    def q_ia(self):
        if self._q_ia is None:
            raise RuntimeError("build() before q_ia()")
        return self._q_ia

    def describe(self):
        """One line for the solver's own report."""
        shape = ten.shape(self._q_ia)
        return (f"(Q|iu) is {shape[0]} x {shape[1]} x {shape[2]} "
                f"({ten.view(self._q_ia).nbytes / 2**20:.1f} MiB, dense)")
