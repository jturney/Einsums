# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""``python -m einsums``'s commands, and what Einsums adds to Waggle's viewer: the TaskPool panel and
the compute-graph screen. The viewer itself is tested with Waggle (``libs/Waggle/tests/python``).

One test starts a real profiler server in a child interpreter and checks the viewer draws what it
sends, so a change to the wire format in Waggle's ``src/Server.cpp`` that the viewer does not follow
fails here rather than in someone's terminal.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from waggle import analysis
from waggle.testing import META, fake_server, server_export, snapshot_msg, wait_for

from einsums.cli import build_parser


# ── command line ──────────────────────────────────────────────────────────────


def test_parser_has_every_command():
    from waggle.cli import viewer_parser

    parser = build_parser()
    args = viewer_parser("einsums profiler").parse_args(["box:3", "--port", "1", "2", "--no-mdns"])
    assert args.port == [1, 2] and args.endpoints == ["box:3"] and args.no_mdns
    assert parser.parse_args(["bench", "list-runs"]).bench_command == "list-runs"
    # profiler forwards everything, so its server arguments and report/diff both reach it.
    assert parser.parse_args(["profiler", "box:3", "--no-mdns"]).args == ["box:3", "--no-mdns"]
    for command in ("info", "options", "completion", "stages"):
        assert command in parser.format_help()


def _launcher():
    """build/bin/einsums, from the package in build/lib/einsums (None when installed elsewhere)."""
    import einsums

    path = Path(einsums.__file__).resolve().parents[2] / "bin" / "einsums"
    return path if path.is_file() else None


@pytest.mark.skipif(_launcher() is None, reason="no build-tree einsums launcher")
def test_einsums_launcher_runs_without_pythonpath(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cmd = [str(_launcher()), "bench", "--db", str(tmp_path / "b.db"), "list-runs"]
    if sys.platform == "win32":
        cmd = [sys.executable, *cmd]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=tmp_path, check=False)
    assert result.returncode == 0, result.stderr
    assert "No runs found" in result.stdout


def test_stages_command_forwards_to_the_stages_parser(capsys):
    # Every argument, options included, reaches einsums.stages' own parser.
    from einsums.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["stages", "promote", "--help"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("usage: einsums stages promote")
    with pytest.raises(SystemExit) as exit_info:
        main(["stages", "nonsense"])
    assert exit_info.value.code == 2 and "invalid choice: 'nonsense'" in capsys.readouterr().err


def test_module_entry_point_names_itself():
    result = subprocess.run([sys.executable, "-m", "einsums", "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0 and result.stdout.startswith("usage: python -m einsums")


# ── the app, headless ─────────────────────────────────────────────────────────


needs_textual = pytest.mark.skipif(importlib.util.find_spec("textual") is None, reason="the app needs Textual")


def app_module():
    from waggle import app

    return app


def einsums_plugins():
    from einsums.cli.profiler.einsums_plugin import einsums_viewer_plugin

    return [einsums_viewer_plugin()]


@needs_textual
def test_the_einsums_plugin_binds_its_keys():
    app = app_module().ProfilerApp(load=["unused"], mdns=False, plugins=einsums_plugins())
    # A plugin's keys are bound per app, to the two plugin actions.
    plugin_keys = {b.key: b.action for bindings in app._bindings.key_to_bindings.values() for b in bindings}
    assert plugin_keys["W"] == "plugin_panel('einsums', 'taskpool')"
    assert plugin_keys["K"] == "plugin_action('einsums', 'compute_graphs')"


def test_einsums_profiler_report_runs_waggles(tmp_path, capsys):
    from einsums.cli import main

    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"sessions": [server_export("first")]}))
    assert main(["profiler", "report", str(path), "--format", "json"]) == 0
    assert [r["name"] for r in json.loads(capsys.readouterr().out)] == ["inner", "outer", "other"]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


SERVER_SCRIPT = textwrap.dedent(
    """
    import sys, time
    import einsums, einsums.profile as prof
    einsums._ensure_initialized()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with prof.section("outer"):
            prof.annotate("flops", 2_000_000)
            with prof.section("inner"):
                time.sleep(0.002)
    """
)


@needs_textual
@pytest.mark.skipif(sys.platform == "win32", reason="the profile server has no Windows socket implementation")
def test_viewer_follows_the_real_server():
    import einsums.profile as prof

    ProfilerApp = app_module().ProfilerApp

    if not prof.available():
        pytest.skip("built without the profiler")
    port = _free_port()
    child = subprocess.Popen(
        [sys.executable, "-c", SERVER_SCRIPT, "--einsums:profile:server", f"--einsums:profile:port={port}",
         "--einsums:profile:wait-for-viewer", "--einsums:profile:report=false", "--einsums:debug:no-attach-debugger"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )  # fmt: skip

    # Found by path, not position: a build with EINSUMS_WITH_PROFILER_INTERNAL records the runtime's
    # start-up zones first, on the same thread, and roots keep their insertion order.
    async def main():
        app = ProfilerApp([("127.0.0.1", port)], mdns=False)

        def paths():
            return {r.path for r in app.active_view._rows} if app.active_view is not None else set()

        async with app.run_test(size=(120, 40)) as pilot:
            await wait_for(pilot, lambda: {"outer", "outer/inner"} <= paths(), timeout=30)
            session = app.active_session
            await pilot.press("q")
        return session

    try:
        session = asyncio.run(main())
    finally:
        child.kill()
        _, err = child.communicate()
    assert session.meta.pid == child.pid, err
    assert "einsums" in [c.get("name") for c in session.meta.clients], session.meta.clients
    outer = next(node for node in session.snapshot.all_roots() if node.name == "outer")
    assert outer.call_count > 0 and analysis.numeric_annotation(outer.annotations, "flops") == 2_000_000


# ── compute graphs and TaskPool metrics ───────────────────────────────────────


GRAPHS = {
    "graphs": [
        {
            "name": "scf",
            "stage_name": "fock",
            "tensors": [{"id": 1, "name": "D", "rank": 2, "dims": [4, 4], "dtype": "f64"},
                        {"id": 2, "name": "F", "rank": 2, "dims": [4, 4], "dtype": "f64"}],
            "nodes": [{"id": 0, "kind": "einsum", "label": "J", "target": "cpu", "stream_id": 0, "inputs": [1],
                       "outputs": [2], "timing_ms": 3.0, "c_indices": "ij", "a_indices": "ijkl", "b_indices": "kl"},
                      {"id": 1, "kind": "scale", "label": "scale", "target": "cpu", "stream_id": 0, "inputs": [2],
                       "outputs": [2], "timing_ms": 1.0}],
            "edges": [{"from": 0, "to": 1, "tensor_id": 2}],
        }
    ]
}  # fmt: skip


def test_graph_report_parses_and_summarizes():
    from einsums.cli.profiler.graphs import graph_summary, node_summary, parse_graphs

    [graph] = parse_graphs(GRAPHS)
    assert graph.context == "fock" and graph.total_ms == 4.0
    assert graph.nodes[0].spec == "ij <- ijkl,kl" and graph.predecessors(1) == [0]
    assert "#0 einsum J" in graph_summary(graph) and "75.0%" in graph_summary(graph)
    text = node_summary(graph, graph.nodes[1])
    assert "D [4x4]" not in text and "F [4x4]" in text and "after:[/bold] #0" in text
    assert parse_graphs(GRAPHS["graphs"])[0].name == "scf"  # the saved-session form, a bare list


def test_taskpool_metrics_and_their_absence():
    from einsums.cli.profiler.graphs import parse_taskpool, taskpool_text

    metrics = parse_taskpool({"total_submitted": 10, "total_completed": 7, "total_steals": 2, "active_workers": 1,
                              "num_workers": 3, "per_worker_executed": [5, 2, 0], "per_worker_stolen": [0, 2, 0]})  # fmt: skip
    text = taskpool_text(metrics)
    assert "pending 3" in text and "worker   1" in text
    assert "worker   2" not in text and "(1 worker(s) have run nothing yet)" in text
    assert parse_taskpool({"error": "unknown method"}) is None
    assert "no TaskPool" in taskpool_text(None)


@needs_textual
def test_graph_screen_from_a_saved_session(tmp_path):
    ProfilerApp = app_module().ProfilerApp
    path = tmp_path / "s.json"
    path.write_text(json.dumps(server_export() | {"extensions": {"einsums.compute_graphs": GRAPHS["graphs"]}}))

    async def main():
        app = ProfilerApp(load=[str(path)], mdns=False, plugins=einsums_plugins())
        async with app.run_test(size=(140, 50)) as pilot:
            await wait_for(pilot, lambda: app.active_view is not None and app.active_view._rows)
            await pilot.press("K")
            await wait_for(pilot, lambda: type(app.screen).__name__ == "GraphScreen")
            from textual.widgets import Static, Tree

            tree = app.screen.query_one(Tree)
            assert len(tree.root.children) == 1 and len(tree.root.children[0].children) == 2
            detail = app.screen.query_one("#graph-detail-text", Static)
            assert "scf" in str(detail.render())
            # TaskPool metrics come only from a live program, so a loaded session does not open the panel.
            await pilot.press("escape", "W")
            await pilot.pause()
            assert not app.query_one("#einsums-taskpool").has_class("visible")
            await pilot.press("q")

    asyncio.run(main())


@needs_textual
@pytest.mark.parametrize("advertised", [True, False])
def test_a_plugin_panel_opens_only_for_a_program_that_answers_it(advertised):
    ProfilerApp = app_module().ProfilerApp
    metrics = {"total_submitted": 4, "total_completed": 1, "total_steals": 0, "active_workers": 1,
               "num_workers": 2, "per_worker_executed": [1, 0], "per_worker_stolen": [0, 0]}  # fmt: skip
    meta = META | {"handlers": ["get_taskpool_metrics"] if advertised else []}

    async def main():
        server, port = await fake_server(
            [(json.dumps(m) + "\n").encode() for m in (meta, snapshot_msg(1))],
            on_request=lambda method: metrics if method == "get_taskpool_metrics" else {"error": "unknown method"},
        )
        app = ProfilerApp([("127.0.0.1", port)], mdns=False, plugins=einsums_plugins())
        async with app.run_test(size=(140, 50)) as pilot:
            await wait_for(pilot, lambda: app.active_view is not None and app.active_view._rows)
            await pilot.press("W")
            panel = app.query_one("#einsums-taskpool")
            if advertised:
                await wait_for(pilot, lambda: "pending 3" in panel._text)
            else:
                await pilot.pause()
                assert not panel.has_class("visible")
            await pilot.press("q")
        server.close()

    asyncio.run(main())
