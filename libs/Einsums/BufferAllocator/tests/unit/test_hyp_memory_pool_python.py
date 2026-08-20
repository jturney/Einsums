# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Stateful Hypothesis fuzzer for ``einsums.MemoryPool``.

The unit tests check one sequence each. This drives arbitrary interleavings of
carve, release, epoch open/close and reset, and after EVERY step re-checks the
things a pool can only get wrong under interleaving:

  * every live tensor still holds the value written into it, which is the sharp
    test for two carves overlapping or for one being freed early;
  * live carves occupy disjoint byte ranges and distinct addresses, including
    the zero-extent ones;
  * the pool's own accounting - ``live_borrows``, ``epoch_depth``,
    ``bytes_used`` back to zero when nothing is live - matches the model.

Alignment is checked per carve. The epoch and reset rules assert the REFUSALS
too: the model knows which scope each live tensor was carved in, so it knows
whether a close must throw, and a close that succeeded when the model said it
should throw is a use-after-free the pool failed to catch.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

import einsums
from einsums.testing import ALL_DTYPES

MIB = 1024 * 1024

#: Shapes small enough that thousands of steps stay inside one arena, but not
#: all the same size class: mimalloc segregates by size, and a bug that only
#: shows up when two size classes share a page needs both to be in play.
_SHAPES = st.sampled_from([[], [1], [7], [64], [3, 5], [16, 16], [2, 3, 4], [128], [1024], [33, 17]])
_DTYPES = st.sampled_from(ALL_DTYPES)

#: Deep enough to nest, shallow enough that the machine still closes what it opens.
_MAX_DEPTH = 3


def _address(tensor) -> int:
    return np.asarray(tensor).__array_interface__["data"][0]


def _nbytes(tensor) -> int:
    return int(np.asarray(tensor).nbytes)


class PoolMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.pool = einsums.MemoryPool(16 * MIB, "fuzz")
        #: key -> (tensor, fill value, scope depth it was carved in)
        self.live: dict[int, tuple] = {}
        self.epochs: list = []
        self._next_key = 0
        self._next_fill = 1.0
        self._grew = False

    # -- rules ---------------------------------------------------------------

    @rule(shape=_SHAPES, dtype=_DTYPES, zeroed=st.booleans())
    def carve(self, shape, dtype, zeroed):
        fill = self._next_fill
        self._next_fill += 1.0
        name = f"fuzz{self._next_key}"

        if zeroed:
            t = self.pool.zeros(shape, dtype=dtype, name=name)
            if t.size:
                # A zeroed carve is checked as zeros first, then repainted, so
                # the zeroing itself is under test and not just the storage.
                assert not np.asarray(t).any(), f"pool.zeros returned nonzero for {shape}"
        else:
            t = self.pool.empty(shape, dtype=dtype, name=name)

        if t.size:
            np.asarray(t)[...] = fill
        assert _address(t) % 64 == 0, "carve is not 64-byte aligned"

        self.live[self._next_key] = (t, fill, len(self.epochs))
        self._next_key += 1

    @precondition(lambda self: bool(self.live))
    @rule(pick=st.integers(min_value=0, max_value=1 << 30))
    def release(self, pick):
        keys = sorted(self.live)
        del self.live[keys[pick % len(keys)]]

    @precondition(lambda self: len(self.epochs) < _MAX_DEPTH)
    @rule()
    def open_epoch(self):
        self.epochs.append(self.pool.epoch())

    @precondition(lambda self: bool(self.epochs))
    @rule()
    def close_epoch(self):
        depth = len(self.epochs)
        held = any(scope >= depth for (_, _, scope) in self.live.values())
        scope = self.epochs[-1]
        if held:
            # Closing over a live carve is the use-after-free the epoch exists
            # to refuse; the scope must survive the refusal, still open.
            try:
                scope.close()
            except RuntimeError:
                assert scope.open
                return
            raise AssertionError("epoch closed while a carve made inside it was still live")
        scope.close()
        assert not scope.open
        self.epochs.pop()

    @rule()
    def reset(self):
        if self.live or self.epochs:
            try:
                self.pool.reset()
            except RuntimeError:
                return
            raise AssertionError("reset ran with live carves or an open epoch")
        self.pool.reset()
        assert self.pool.bytes_used == 0

    @rule(extra=st.integers(min_value=0, max_value=8))
    def reserve(self, extra):
        # At most one real growth per machine, on purpose. Every growth takes
        # another arena, mimalloc caps how many a process may hold and never
        # reclaims one, so a fuzzer that grew freely would exhaust them and
        # report the pool's own resource limit as a bug. The no-op path (a
        # reserve at or below current capacity) stays unbounded.
        target = self.pool.bytes_reserved + extra * MIB
        if extra and self._grew:
            target = self.pool.bytes_reserved
        before = self.pool.bytes_reserved
        self.pool.reserve(target)
        assert self.pool.bytes_reserved >= max(target, before)
        if target > before:
            self._grew = True

    # -- invariants ----------------------------------------------------------

    @invariant()
    def contents_intact(self):
        for tensor, fill, _ in self.live.values():
            if tensor.size:
                arr = np.asarray(tensor)
                assert np.all(arr == fill), f"live carve was overwritten (expected {fill})"

    @invariant()
    def carves_are_disjoint(self):
        spans = sorted((_address(t), _nbytes(t)) for (t, _, _) in self.live.values())
        for (lo, n), (nxt, _) in zip(spans, spans[1:]):
            # Zero-extent tensors still get real bytes, so even they may not
            # share an address with anything else.
            assert nxt >= lo + max(n, 1), f"live carves overlap at {lo:#x}"

    @invariant()
    def accounting_matches(self):
        assert self.pool.live_borrows == len(self.live)
        assert self.pool.epoch_depth == len(self.epochs)
        if not self.live:
            assert self.pool.bytes_used == 0

    def teardown(self):
        # Epochs are closed innermost first; their carves have to go first too,
        # or the close refuses and the next machine inherits a warning.
        self.live.clear()
        while self.epochs:
            self.epochs.pop().close()


TestPoolMachine = PoolMachine.TestCase
TestPoolMachine.settings = settings(
    max_examples=50,
    stateful_step_count=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
