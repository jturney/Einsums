#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""What a batched call costs, in the currency the bucket chooser spends.

The chooser trades padded elements against batched calls, so it needs the two in
one unit. This module supplies the exchange rate, and it is deliberately the only
place a measured constant lives: everything else in the port is arithmetic.
"""

from einsums import hardware as _hardware

__all__ = ["bucket_penalty", "penalty_source"]

#: Padded elements a batched call costs before any team is launched.
#:
#: A call is not free on one thread, only cheap: emitting the node, walking its
#: operands and replaying it cost something whatever the thread count. Measured
#: on ethanol/cc-pVTZ, where the serial optimum is 12 buckets rather than the 16
#: an unopposed volume objective would run to, this term lies in [28, 66]
#: elements; 45 is the middle of that.
#:
#: Its job is as much structural as numerical. Without it the serial objective
#: has nothing pulling against finer buckets and simply saturates at
#: ``Thresholds.max_buckets``, which happens to cost only 1.04x here but is an
#: artifact of where that cap sits rather than an answer.
CALL_FLOOR_ELEMENTS = 45.0

#: Padded elements one nanosecond of OpenMP region cost is worth.
#:
#: The chooser minimizes ``volume + penalty * calls``, so only the RATIO of the
#: per-call cost to the per-element cost matters, and above one thread that cost
#: is dominated by entering the parallel region - measured at 3.1 us on two
#: threads, 7.1 on four and 26.5 on ten, growing with the team and varying by an
#: order of magnitude across OpenMP runtimes, which is exactly why it is read
#: from :mod:`einsums.hardware` rather than written down.
#:
#: This constant is the part that is not a hardware fact: how expensive an
#: element of padded per-pair work is, relative to that. Fitted 2026-08-09
#: against measured DLPNO-MP2 iteration times on ethanol/cc-pVTZ and a
#: six-monomer water chain at 1, 2, 4 and 10 threads. Over those eight
#: (geometry, thread count) cases the resulting choice costs a geometric mean of
#: 1.05x the best bucket count available, worst case 1.23x, against 1.22x and
#: 1.64x for the fixed four buckets it replaces. It is never worse than the
#: fixed default on any case measured.
#:
#: Three honest caveats. The two geometries disagree at four threads - ethanol
#: wants a penalty under 245, the chain over 776, and no single value satisfies
#: both, nor does adding the repack traffic as a third term; the chain's
#: run-to-run spread is 16% against ethanol's 1%, so that is most likely noise
#: rather than a missing term, but it is not proven. No single slope hits the
#: two-thread and ten-thread bands at once either, missing each by under 10%,
#: which is why this is fitted on realized cost rather than on reproducing every
#: optimum. And this is one machine with OpenBLAS: re-derive under MKL, where
#: the vendor's own batching may change what a call costs.
ELEMENTS_PER_REGION_NS = 0.03


def penalty_source():
    """Whether the penalty rests on a calibration or on a fresh measurement.

    Worth reporting rather than assuming: an in-process measurement is taken
    under whatever the machine is doing at that moment and drifts by tens of
    percent, so a bucket count derived from one is reproducible only by luck.
    Run ``calibrate_hardware`` once and it stops being luck.
    """
    return "calibrated" if _hardware.region_cost_is_calibrated() else "measured in-process"


def bucket_penalty():
    """Padded elements one extra batched call is worth, on this machine now.

    A floor plus what the team costs, so it falls to the floor on one thread
    rather than to zero: the call still has to be emitted and replayed, it just
    has no region to enter.
    """
    return CALL_FLOOR_ELEMENTS + ELEMENTS_PER_REGION_NS * _hardware.omp_region_cost_ns()
