#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""The session: a sequence of graph segments, and the timing table over them.

A session holds a :class:`einsums.graph.Pipeline`, which is the library's own
sequence-of-graphs abstraction, and adds one Pipeline stage per *segment*. A
segment holds every framework stage captured between two eager splits, so
several framework stages share one graph and the passes still see whole runs of
them. Mapping one framework stage to one Pipeline stage would put a graph
boundary at every stage boundary, and then no pass would ever see the method.

The session drives execution segment by segment rather than through
``Pipeline::execute()``. It has to: an eager stage needs its inputs
materialized, so the pending segment must run in the middle of the capture
block, and a later whole-pipeline execute would then run it a second time.
Passes are therefore applied per segment as each one closes, which is also why
they are supplied when the session is built rather than when it runs.
"""

import contextlib as _contextlib
import json as _json
import threading as _threading
import time as _time

import einsums.graph as _cg

__all__ = ["Session", "session", "active_session", "StageTiming"]

_active = _threading.local()


def active_session():
    """The session capturing on this thread, or None."""
    return getattr(_active, "session", None)


class StageTiming:
    """One row of the timing table."""

    __slots__ = ("name", "backend", "eager", "promotable", "capture_ms", "execute_ms", "split", "_ids")

    def __init__(self, name, backend, *, eager, promotable):
        self.name = name
        self.backend = backend
        self.eager = eager
        self.promotable = promotable
        self.capture_ms = 0.0
        self.execute_ms = 0.0
        self.split = False
        self._ids = None  # (segment index, first node id, last+1) for captured stages

    @property
    def notes(self) -> str:
        parts = []
        if self.eager:
            parts.append("eager")
        if self.split:
            parts.append("SPLIT")
        if not self.promotable:
            parts.append("python-only")
        return ", ".join(parts)


class Session:
    """A sequence of graph segments plus the per-stage timing that spans them.

    Built directly rather than through a context manager, because the block
    delimits capture, not the session's lifetime::

        s = stages.session("dlpno", passes=cg.default_pass_manager())
        with s.capture():
            ...
        s.run()
        print(s.report())
    """

    def __init__(self, name: str = "stages", *, passes=None):
        self.name = name
        self._pipeline = _cg.Pipeline(name)
        self._passes = passes
        self._segments = []  # graphs, in order
        self._executed = []  # parallel to _segments
        self._timings = []
        self._capturing = False
        self._capture_cm = None
        self._had_node_timings = False

    # -- segments -------------------------------------------------------
    @property
    def pipeline(self):
        """The underlying :class:`einsums.graph.Pipeline`."""
        return self._pipeline

    @property
    def segments(self) -> list:
        """The graph of each segment, in order."""
        return list(self._segments)

    def _open_segment(self):
        g = self._pipeline.add_stage(f"{self.name}.segment{len(self._segments)}")
        self._segments.append(g)
        self._executed.append(False)
        return g

    def _current(self):
        return self._segments[-1] if self._segments else None

    def _begin_capture(self, graph):
        self._capture_cm = _cg.capture(graph)
        self._capture_cm.__enter__()
        self._capturing = True

    def _end_capture(self):
        if self._capture_cm is not None:
            self._capture_cm.__exit__(None, None, None)
            self._capture_cm = None
        self._capturing = False

    def _execute_segment(self, i: int):
        """Apply passes to segment *i* and execute it, once."""
        if self._executed[i]:
            return
        g = self._segments[i]
        if self._passes is not None and g.num_nodes() > 0:
            g.apply(self._passes)
        g.execute()
        self._executed[i] = True

    # -- capture --------------------------------------------------------
    @_contextlib.contextmanager
    def capture(self):
        """Capture stage calls into this session's segments.

        Mirrors :func:`einsums.graph.capture`: the block delimits capture, and
        the session outlives it so ``run()`` and ``report()`` still work.
        """
        if active_session() is not None:
            raise RuntimeError("a session is already capturing on this thread")
        self._open_segment()
        _active.session = self
        self._begin_capture(self._current())
        try:
            yield self
        finally:
            self._end_capture()
            _active.session = None

    # -- dispatch -------------------------------------------------------
    def run_stage(self, st, fn, args, kwargs):
        """Called by the dispatching wrapper for every stage inside a session."""
        row = StageTiming(st.name, st.selected, eager=st.eager, promotable=st.promotable)

        if st.eager:
            # Materialize everything the eager stage might read, then run it
            # outside capture, then open the next segment.
            row.split = True
            self._end_capture()
            self._execute_segment(len(self._segments) - 1)
            t0 = _time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            finally:
                row.execute_ms = (_time.perf_counter() - t0) * 1e3
                self._timings.append(row)
                self._begin_capture(self._open_segment())
            return result

        seg = len(self._segments) - 1
        graph = self._current()
        first = graph.num_nodes()
        t0 = _time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        finally:
            row.capture_ms = (_time.perf_counter() - t0) * 1e3
            row._ids = (seg, first, graph.num_nodes())
            self._timings.append(row)
        return result

    # -- execution ------------------------------------------------------
    def run(self):
        """Apply passes to and execute every segment that has not run yet."""
        if self._capturing:
            raise RuntimeError("run() called while still capturing; leave the capture block first")
        for i in range(len(self._segments)):
            self._execute_segment(i)
        self._attribute_execute_time()

    def execute(self):
        """Alias for :meth:`run`, for readers coming from ``Graph.execute``."""
        self.run()

    # -- timing ---------------------------------------------------------
    def _attribute_execute_time(self):
        """Bucket per-node execute times into the stage that captured them.

        Node ids are handed out from a monotonic counter and never reused, so
        the ``[first, last)`` range recorded during capture still names the
        stage's nodes after passes have fused or deleted some of them.
        """
        per_segment = []
        for g in self._segments:
            try:
                doc = _json.loads(g.to_json())
            except Exception:
                per_segment.append({})
                continue
            per_segment.append(
                {n["id"]: n.get("timing_ms", 0.0) or 0.0 for n in doc.get("nodes", [])}
            )

        for row in self._timings:
            if row._ids is None:
                continue
            seg, first, last = row._ids
            if seg >= len(per_segment):
                continue
            timings = per_segment[seg]
            row.execute_ms = sum(t for nid, t in timings.items() if first <= nid < last)
            if row.execute_ms:
                self._had_node_timings = True

    @property
    def timings(self) -> list:
        """The per-stage rows behind :meth:`report`."""
        return list(self._timings)

    def _timing_note(self) -> str | None:
        """Why the execute column is empty, when it is.

        Asked empirically rather than by probing the profiler: the runtime gate
        (``Profiler::enabled()``) is not bound to Python, and "did any node
        report a time" is the question that actually matters anyway.
        """
        if self._had_node_timings or not any(self._executed):
            return None
        try:
            import einsums.profile as _p

            available = bool(_p.available())
        except Exception:
            available = False
        if not available:
            return (
                "Execute times are blank: this build has the profiler compiled out, and "
                "per-node timing comes from it."
            )
        return (
            "Execute times are blank: per-node timing needs the profiler enabled at runtime "
            "(it is off under --einsums:profile-disable). Numbers it reports include its own "
            "overhead, which is not the configuration you ship."
        )

    def report(self) -> str:
        """The per-stage table: backend, capture time, execute time, and splits.

        This is the table the whole framework exists to print. A developer sees
        their own measured number before writing any C++, instead of guessing
        it, and a guess here can be off by more than an order of magnitude.
        """
        rows = self._timings
        name_w = max([len("stage")] + [len(r.name) for r in rows])
        back_w = max([len("backend")] + [len(r.backend) for r in rows])
        num_w = 11  # "%8.1f ms"

        def cell(ms: float) -> str:
            return f"{ms:8.1f} ms" if ms else " " * num_w

        out = [
            f"{'stage':<{name_w}}  {'backend':<{back_w}}  "
            f"{'capture':>{num_w}}  {'execute':>{num_w}}  notes"
        ]
        for r in rows:
            out.append(
                f"{r.name:<{name_w}}  {r.backend:<{back_w}}  "
                f"{cell(r.capture_ms)}  {cell(r.execute_ms)}  {r.notes}".rstrip()
            )

        total_cap = sum(r.capture_ms for r in rows)
        total_exe = sum(r.execute_ms for r in rows)
        out.append(
            f"{'total':<{name_w}}  {'':<{back_w}}  "
            f"{cell(total_cap)}  {cell(total_exe)}"
        )

        splits = sum(1 for r in rows if r.split)
        if splits:
            out.append(f"\n{splits} eager split(s); each forces a graph boundary no pass can cross.")
        note = self._timing_note()
        if note:
            out.append(f"\n{note}")
        return "\n".join(out)

    def __repr__(self) -> str:
        return (
            f"<Session {self.name!r} segments={len(self._segments)} "
            f"stages={len(self._timings)}>"
        )


def session(name: str = "stages", *, passes=None) -> Session:
    """Build a :class:`Session`.

    Args:
        name: Used to name the pipeline and its segments.
        passes: A ``PassManager`` applied to each segment as it closes. Supplied
            here rather than at ``run()`` because an eager stage closes a
            segment mid-block, before any argument to ``run()`` exists.
    """
    return Session(name, passes=passes)
