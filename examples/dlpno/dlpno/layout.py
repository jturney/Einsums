#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""How per-pair blocks are laid out in contiguous stores.

psi4 keeps every pair's block in its own ``SharedMatrix``, which forces the
residual into a loop of individually-dispatched small GEMMs. Here each per-pair
quantity is a handful of contiguous ``(M_b, M_b, n_b)`` tensors, one per bucket
of PNO count, and the pair index TRAILS so that a run of pairs is the batch
index a strided-batched GEMM wants.

That layout is a *value*, not a property of the solver that happens to be
running. A DLPNO-CCSD calculation builds it twice - once for the MP2-level PNOs
the prescreening LMP2 runs in, and again for the tighter CC-level PNOs
``recompute_pnos`` produces - and the two coexist, so nothing about the layout
can live as loose fields on the calculation object.

:class:`PairLayout` is therefore the whole of it: which bucket each pair sits
in, which slot within that bucket, and the store construction and block access
that follow. Choosing the bucket boundaries is :meth:`PairLayout.choose`.
"""

from . import cost
from . import tensors as ten

__all__ = ["PairLayout"]


class PairLayout:
    """Pairs bucketed by PNO count, and the padded stores that follow.

    Blocks are padded to their bucket's dimension and the padding is inert: the
    integrals and amplitudes are zero there, the overlaps are zero there, and
    the energy denominators are one, so padded components stay zero for the life
    of the calculation and contribute nothing to any dot product.
    """

    def __init__(self, n_pno, bucket_dims):
        """Assign every pair to a bucket and a slot within it.

        Args:
            n_pno: Per pair, its PNO count. Zero means the pair is dead and gets
                no slot at all.
            bucket_dims: Ascending padded dimensions, one per bucket. Every
                nonzero count must be no larger than the last.
        """
        self.n_pno = list(n_pno)
        self.bucket_dims = list(bucket_dims)
        self.bucket_of = [-1] * len(self.n_pno)
        self.slot_of = [-1] * len(self.n_pno)
        self.bucket_members = [[] for _ in self.bucket_dims]
        for ij, n in enumerate(self.n_pno):
            if n == 0:
                continue
            b = next(i for i, e in enumerate(self.bucket_dims) if e >= n)
            self.bucket_of[ij] = b
            self.slot_of[ij] = len(self.bucket_members[b])
            self.bucket_members[b].append(ij)

    # -- construction ------------------------------------------------------

    @classmethod
    def choose(cls, n_pno, thresholds, report=None):
        """Partition pairs into groups by PNO count, and choose how many groups.

        Padding every pair to the global maximum is what makes the coupling
        GEMMs batchable, but it is paid in padded volume: on ethanol/cc-pVDZ the
        PNO counts run 5..48 against an average of 29, so a single store carries
        2.7x the elements it needs. Splitting into buckets, each padded to its
        own maximum, keeps the shapes uniform *within* a bucket (which is all the
        batching needs) while cutting most of that waste.

        More buckets is therefore cheaper per element, and it used to be much
        dearer per call: shape classes are pairs of buckets, so B buckets meant
        up to B^2 classes and the iteration issued a batched GEMM per class per
        width group, each entering its own OpenMP region at tens of microseconds
        on ten threads. Which side won was a property of the machine rather than
        of the molecule, and a fixed count could not be right - measured on
        ethanol/cc-pVTZ the best count ran 12, 8, 8, 4 at 1, 2, 4 and 10 threads.

        That tension is now mostly gone. Both halves emit
        ``grouped_batched_gemm``, which covers every shape under one region, so
        the iteration issues ``1 + B`` calls rather than ``2 * groups * B^2``:
        754 became 13 at twelve buckets on ethanol. The objective below still
        carries a call term, because the calls are not free and the residual's
        accumulation keeps one per partner bucket, but it no longer fights the
        volume term hard enough to pull the choice down to 4.

        Hence :func:`~dlpno.cost.bucket_penalty`, which turns the measured region
        cost into the only free parameter here: how many padded elements one
        extra batched call is worth. The objective is then

            padded_volume + penalty * n_calls

        Volume rather than the cubic flops this used to minimize. Fitting
        measured iteration time against both, the cubic term's coefficient is
        negligible and turns negative once threads are on: the padded GEMM flops
        are simply not what the time is made of, and an objective that minimized
        them was optimizing a term worth a few percent.

        For a fixed bucket count the call term is constant, so the boundaries
        still come from an exact segmentation DP - each PNO count contributes
        independently once its bucket's maximum is fixed - and the count itself
        is chosen by running that DP for each candidate. O(n^2 B^2) over the
        distinct counts, microseconds at these sizes.

        ``Thresholds.n_buckets`` overrides the choice when it is not None, which
        is how the sweeps that calibrated this pin it.

        Args:
            n_pno: Per pair, its PNO count.
            thresholds: Reads ``n_buckets`` and ``max_buckets``.
            report: Optional one-argument callable for the chooser's line of
                output; ``None`` prints nothing, which is what a caller running
                this twice per calculation wants for the second one.
        """
        counts = sorted({n for n in n_pno if n})
        if not counts:
            return cls(n_pno, [])

        weight = [0] * len(counts)
        index = {c: i for i, c in enumerate(counts)}
        for n in n_pno:
            if n:
                weight[index[n]] += 1
        prefix = [0] * (len(counts) + 1)
        for i, w in enumerate(weight):
            prefix[i + 1] = prefix[i] + w

        # seg(j, i): counts[j:i] in one bucket, padded to counts[i-1]. The
        # padded stores are (M, M, members), so a bucket's cost is its members
        # times the square of the dimension they are padded to.
        def seg(j, i):
            return (prefix[i] - prefix[j]) * counts[i - 1] ** 2

        def best_boundaries(n_buckets):
            """Exact DP: the boundaries minimizing padded volume for this count."""
            INF = float("inf")
            dp = [[INF] * (n_buckets + 1) for _ in range(len(counts) + 1)]
            back = [[0] * (n_buckets + 1) for _ in range(len(counts) + 1)]
            dp[0][0] = 0
            for i in range(1, len(counts) + 1):
                for b in range(1, n_buckets + 1):
                    for j in range(i):
                        if dp[j][b - 1] == INF:
                            continue
                        c = dp[j][b - 1] + seg(j, i)
                        if c < dp[i][b]:
                            dp[i][b] = c
                            back[i][b] = j
            edges, i = [], len(counts)
            for b in range(n_buckets, 0, -1):
                edges.append(counts[i - 1])
                i = back[i][b]
            return sorted(edges), dp[len(counts)][n_buckets]

        if thresholds.n_buckets is not None:
            n_buckets = max(1, min(int(thresholds.n_buckets), len(counts)))
            dims, _ = best_boundaries(n_buckets)
            return cls(n_pno, dims)

        penalty = cost.bucket_penalty()
        class_cost = cost.class_penalty()
        best = None
        for n_buckets in range(1, min(thresholds.max_buckets, len(counts)) + 1):
            dims, volume = best_boundaries(n_buckets)
            # One call for the whole first half, plus one per partner bucket for
            # the second.
            #
            # This used to be ``2 * groups * n_buckets ** 2``, one call per shape
            # class per width group per half, and that was right when a batched
            # GEMM could only cover one shape. It no longer is: both halves emit
            # ``grouped_batched_gemm``, which covers every shape under one OpenMP
            # region. The couplings all write disjoint slots and collapse to a
            # single call; the residual accumulates, so it collapses only as far
            # as disjoint pairs allow, which is one call per partner bucket.
            # See LMP2Solver._emit_couplings and _emit_residual.
            #
            # The consequence is that the call term barely bends the objective
            # any more, so this chooses close to the volume minimum - which is
            # the whole point of having removed the calls. ``groups`` no longer
            # enters: width groups subdivide a class, and subdividing a grouped
            # call costs nothing.
            n_calls = 1 + n_buckets
            # The one-time graph build, which grows with the shape classes and
            # used to be hidden behind the call term. See
            # cost.CLASS_FLOOR_ELEMENTS: with the calls collapsed, this is the
            # only thing left pulling against finer buckets, and without it the
            # objective just saturates at max_buckets.
            #
            # B**2 is an upper bound on the classes actually present, which the
            # coupling structure decides; it runs 60-95% of it here and is
            # monotone in the bucket count, which is all the comparison needs.
            # Same approximation the call term used to make.
            n_classes = n_buckets ** 2
            total = volume + penalty * n_calls + class_cost * n_classes
            if best is None or total < best[0]:
                best = (total, dims, n_buckets)
        if report is not None:
            report(f"  buckets:  {best[2]} chosen at {penalty:.0f} elements per call and "
                   f"{class_cost:.0f} per shape class ({cost.penalty_source()}), "
                   f"dims {best[1]}")
        return cls(n_pno, best[1])

    # -- stores and blocks -------------------------------------------------

    def new_stores(self, name):
        """One ``(M_b, M_b, n_b)`` tensor per bucket."""
        return [
            ten.zeros(f"{name} (bucket {b}, dim {M})", [M, M, len(members)])
            for b, (M, members) in enumerate(zip(self.bucket_dims, self.bucket_members))
        ]

    def dim(self, ij):
        """The padded block dimension pair ``ij`` is stored at."""
        return self.bucket_dims[self.bucket_of[ij]]

    def block(self, stores, ij):
        """The padded numpy view of pair ``ij``'s block within its bucket."""
        return ten.view(stores[self.bucket_of[ij]])[:, :, self.slot_of[ij]]

    def view(self, stores, ij):
        """The einsums (capture-safe) view of pair ``ij``'s PADDED block."""
        return stores[self.bucket_of[ij]][:, :, self.slot_of[ij]]

    def logical_view(self, stores, ij):
        """The einsums view of pair ``ij``'s block WITHOUT its padding.

        The padding is inert - integrals and amplitudes are zero there - so a
        term that contracts two padded blocks of the same bucket may read
        either form. A term that contracts a padded block against something
        sized to the pair, which is every overlap bridge into another basis,
        may not: the shapes simply disagree. Both callers exist, so both
        accessors do, and the name says which one a reader is looking at.
        """
        n = self.n_pno[ij]
        return stores[self.bucket_of[ij]][:n, :n, self.slot_of[ij]]
