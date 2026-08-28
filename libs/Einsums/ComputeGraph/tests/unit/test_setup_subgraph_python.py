# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Setup subgraphs, from Python.

A capability that is not reachable from Python is not shipped, and this module's history
is five separate misses of exactly that: C++ that worked perfectly with no spelling
anyone writing an example could use. The C++ cases prove the semantics; these prove a
person can get at them, which the passing C++ never does.

The counting trick is the same one the C++ uses: the setup body adds one to every element
of its output, so the value in a slot IS the number of times the body ran.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg as la


def _tensor(name, array):
    t = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(t)[...] = array
    return t


def _build(name):
    """A graph whose `fit` is produced by a setup body and read by the replay body."""
    ones = _tensor("ones", np.ones((2, 2)))
    fit = _tensor("fit", np.zeros((2, 2)))
    out = _tensor("out", np.zeros((2, 2)))

    g = cg.Graph(name)
    body = g.add_setup("fit")
    with cg.capture(body):
        la.axpy(1.0, ones, fit)
    with cg.capture(g):
        einsums.permute("i,j <- i,j", out, fit, c_pf=0.0, a_pf=1.0)
    return g, ones, fit, out


def test_the_body_runs_once_and_the_replays_skip_it():
    g, _ones, fit, out = _build("setup_once")
    assert g.has_setup

    g.execute()
    assert np.asarray(fit)[0, 0] == pytest.approx(1.0)
    assert np.asarray(out)[0, 0] == pytest.approx(1.0)

    g.execute()
    g.execute()
    assert np.asarray(fit)[0, 0] == pytest.approx(1.0)


def test_run_setup_pulls_the_body_forward_and_force_repeats_it():
    g, _ones, fit, _out = _build("setup_run")

    g.run_setup()
    assert np.asarray(fit)[0, 0] == pytest.approx(1.0)

    # Idempotent: the same guards apply here as inside a replay.
    g.run_setup()
    assert np.asarray(fit)[0, 0] == pytest.approx(1.0)

    # The escape hatch for a caller who changed a bound tensor's CONTENTS, which nothing
    # can observe.
    g.run_setup(True)
    assert np.asarray(fit)[0, 0] == pytest.approx(2.0)


def test_invalidate_setup_puts_the_body_back_to_work():
    g, _ones, fit, _out = _build("setup_invalidate")

    g.execute()
    assert np.asarray(fit)[0, 0] == pytest.approx(1.0)

    g.invalidate_setup()
    g.execute()
    assert np.asarray(fit)[0, 0] == pytest.approx(2.0)


def test_a_bind_refits_and_a_matching_key_does_not():
    g, _ones, _fit, _out = _build("setup_key")
    g.set_setup_key("problem-a")
    assert g.setup_key == "problem-a"

    g.execute()

    ones2 = _tensor("ones", np.ones((2, 2)))
    fit2 = _tensor("fit", np.ones((2, 2)))  # the caller's own copy of the previous fit
    out2 = _tensor("out", np.zeros((2, 2)))

    # Same key: the caller is saying this is the same problem, so the factors on hand stand.
    cg.bind(g, {"ones": ones2, "fit": fit2, "out": out2})
    g.execute()
    assert np.asarray(fit2)[0, 0] == pytest.approx(1.0)
    assert np.asarray(out2)[0, 0] == pytest.approx(1.0)

    # A different key is a different problem, and the body runs again.
    g.set_setup_key("problem-b")
    cg.bind(g, {"ones": ones2, "fit": fit2, "out": out2})
    g.execute()
    assert np.asarray(fit2)[0, 0] == pytest.approx(2.0)


def test_no_key_means_every_bind_refits():
    g, _ones, _fit, _out = _build("setup_nokey")
    g.execute()

    ones2 = _tensor("ones", np.ones((2, 2)))
    fit2 = _tensor("fit", np.zeros((2, 2)))
    out2 = _tensor("out", np.zeros((2, 2)))

    # The default is no cache: refitting is always correct, and a graph whose caller has
    # said nothing about problem identity gets the behavior that cannot be wrong.
    cg.bind(g, {"ones": ones2, "fit": fit2, "out": out2})
    g.execute()
    assert np.asarray(fit2)[0, 0] == pytest.approx(1.0)


def test_a_saved_graph_reloads_having_fitted_nothing(tmp_path):
    g, _ones, _fit, _out = _build("setup_saved")
    g.execute()

    path = str(tmp_path / "setup.eig")
    cg.save_graph(g, path)

    loaded = cg.load_graph(path)
    ones2 = _tensor("ones", np.ones((2, 2)))
    fit2 = _tensor("fit", np.zeros((2, 2)))
    out2 = _tensor("out", np.zeros((2, 2)))

    cg.bind(loaded, {"ones": ones2, "fit": fit2, "out": out2})
    loaded.execute()
    assert np.asarray(fit2)[0, 0] == pytest.approx(1.0)
    assert np.asarray(out2)[0, 0] == pytest.approx(1.0)

    loaded.execute()
    assert np.asarray(fit2)[0, 0] == pytest.approx(1.0)


def test_the_lambda_form_captures_the_body():
    ones = _tensor("ones", np.ones((2, 2)))
    fit = _tensor("fit", np.zeros((2, 2)))
    out = _tensor("out", np.zeros((2, 2)))

    g = cg.Graph("setup_lambda")
    g.add_setup("fit", lambda: la.axpy(1.0, ones, fit))
    with cg.capture(g):
        einsums.permute("i,j <- i,j", out, fit, c_pf=0.0, a_pf=1.0)

    g.execute()
    g.execute()
    assert np.asarray(fit)[0, 0] == pytest.approx(1.0)
    assert np.asarray(out)[0, 0] == pytest.approx(1.0)


def test_the_default_pass_manager_does_not_wipe_what_a_setup_produced():
    ones = _tensor("ones", np.ones((2, 2)))
    out = _tensor("out", np.zeros((2, 2)))

    g = cg.Graph("setup_passes")
    # A graph-owned DEFERRED intermediate, which is what a factorization pass will create
    # for its factors: written once per bound problem, read every replay. Its Materialize
    # and Initialize belong on the setup body's schedule, not the replay's, or the second
    # replay re-zeroes a fitting the skipped body will not recompute.
    fit = g.declare_zero_tensor("fit", [2, 2], True)
    body = g.add_setup("fit")
    with cg.capture(body):
        la.axpy(1.0, ones, fit)
    with cg.capture(g):
        einsums.permute("i,j <- i,j", out, fit, c_pf=0.0, a_pf=1.0)

    g.apply(cg.default_pass_manager())

    g.execute()
    assert np.asarray(out)[0, 0] == pytest.approx(1.0)

    np.asarray(out)[...] = 0.0
    g.execute()
    assert np.asarray(out)[0, 0] == pytest.approx(1.0)
