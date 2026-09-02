# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Which pass moved the answer, asked from Python.

A capability that is not reachable from Python is not shipped. These prove a
person can construct the driver, hand it their own pipeline, choose a mode,
widen a bound, and read the verdict.

What is NOT here, deliberately: the injected-wrong-pass cases. A pass cannot be
AUTHORED in Python - ``OptimizerPass`` is an abstract base and the binding
generator emits no trampoline - so the test that proves the driver names a
culprit lives in the C++ file, where a wrong pass can be written. These cover
the surface; that one covers the finding.

One thing worth knowing about the builder: it is handed the graph as a
non-owning view, valid for the duration of the call. Stash it and you are
holding a dangling pointer once the trial ends.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg

I, K, X, J, Y = 6, 3, 7, 5, 4


def _chain():
    """A builder plus the pool that keeps each trial's tensors alive.

    Fresh buffers per call, held by the pool rather than by the graph, which is
    the discipline the driver asks for: two trials sharing an output buffer
    would each start from the previous trial's answer.
    """
    rng = np.random.default_rng(3)
    a = rng.standard_normal((I, K))
    b = rng.standard_normal((K, X, J))
    d = rng.standard_normal((J, Y))
    pool = []

    def keep(name, shape, values=None):
        t = einsums.create_zero_tensor(name, list(shape), dtype="float64")
        if values is not None:
            np.asarray(t)[...] = values
        pool.append(t)
        return t

    def build(graph):
        A = keep("A", (I, K), a)
        B = keep("B", (K, X, J), b)
        D = keep("D", (J, Y), d)
        R = keep("R", (I, X, Y))
        W = graph.declare_tensor("W", [I, J, X], intermediate=True, dtype="float64")
        with cg.capture(graph):
            einsums.einsum("i,j,x <- i,k ; k,x,j", W, A, B)
            einsums.einsum("i,x,y <- i,j,x ; j,y", R, W, D)

    return build, pool


def test_the_default_pipeline_comes_back_clean():
    build, _pool = _chain()
    report = cg.BisectDriver(build).run()

    assert report.baseline_ok
    assert report.clean, report.text
    assert report.trials

    # Something has to have run, or "clean" is a statement about nothing.
    fired = [t for t in report.trials if t.modified]
    assert fired, report.text
    for trial in fired:
        assert trial.norm_relative <= trial.bound


def test_a_caller_can_bisect_their_own_pipeline():
    build, _pool = _chain()

    # The primary use: the passes the caller is actually running, in their order.
    driver = cg.BisectDriver(build)
    driver.set_passes([cg.CSE(), cg.LayoutAssignment()])
    report = driver.run()

    assert [t.name for t in report.trials] == ["CSE", "LayoutAssignment"]
    assert report.clean, report.text


def test_the_bound_follows_the_tier():
    build, _pool = _chain()
    driver = cg.BisectDriver(build)
    driver.set_passes([cg.CSE(), cg.LayoutAssignment()])
    report = driver.run()

    by_name = {t.name: t for t in report.trials}
    # A pass promising the same arithmetic is held to bit equality; one that re-associates is not.
    assert by_name["CSE"].tier == "bitwise-exact"
    assert by_name["CSE"].bound == 0.0
    assert by_name["LayoutAssignment"].tier == "re-associating"
    assert by_name["LayoutAssignment"].bound > 0.0


def test_cumulative_mode_is_reachable():
    build, _pool = _chain()
    driver = cg.BisectDriver(build)
    driver.set_mode(cg.BisectMode.Cumulative)
    driver.set_passes([cg.CSE(), cg.LayoutAssignment()])
    report = driver.run()

    assert report.clean, report.text
    assert len(report.trials) == 2


def test_the_bound_scale_is_checked():
    build, _pool = _chain()
    driver = cg.BisectDriver(build)

    driver.set_bound_scale(10.0)
    assert driver.bound_scale == 10.0

    # Zero would report every re-associating pass as wrong, so it is refused rather than accepted
    # as a stricter setting.
    with pytest.raises(ValueError):
        driver.set_bound_scale(0.0)


def test_a_program_with_nothing_to_read_says_so():
    def build(graph):
        T = graph.declare_tensor("T", [4, 4], intermediate=True, dtype="float64")
        with cg.capture(graph):
            einsums.linalg.scale(2.0, T)

    report = cg.BisectDriver(build).run()

    # Reporting this as clean would be a driver that passes every pipeline it cannot see.
    assert not report.baseline_ok
    assert not report.clean
    assert "nothing to compare" in report.text


def test_the_report_reads_as_a_table():
    build, _pool = _chain()
    driver = cg.BisectDriver(build)
    driver.set_passes([cg.LayoutAssignment()])
    text = driver.run().text

    assert "LayoutAssignment" in text
    assert "re-associating" in text
    assert "bound" in text
