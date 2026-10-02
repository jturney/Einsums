# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""``python -m einsums profiler``: its data model, files, client, and the app driven headless.

The last test starts a real profiler server in a child interpreter and checks the viewer
draws what it sends, so a change to the wire format in ``Profile/src/Server.cpp`` that the
viewer does not follow fails here rather than in someone's terminal.
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
import time
from pathlib import Path

import pytest

from einsums.cli import build_parser
from einsums.cli.profiler import analysis, format as fmt
from einsums.cli.profiler.client import ProfileClient, StreamState, parse_endpoint, read_recording, Recorder
from einsums.cli.profiler.disasm import match_symbol, strip_listing
from einsums.cli.profiler.model import ProfileNode, node_to_dict, parse_node, parse_snapshot
from einsums.cli.profiler.session import (
    Session,
    export_snapshot,
    read_session_file,
    session_from_dict,
    write_session_file,
)


def node(name, excl=1.0, incl=None, calls=1, children=(), **kw):
    kids = list(children)
    return ProfileNode(
        name=name,
        exclusive_ms=excl,
        inclusive_ms=incl if incl is not None else excl + sum(c.inclusive_ms for c in kids),
        call_count=calls,
        children=kids,
        **kw,
    )


def wire_node(name, excl=1.0, children=()):
    kids = list(children)
    return {
        "name": name,
        "call_count": 1,
        "exclusive_ms": excl,
        "inclusive_ms": excl + sum(k["inclusive_ms"] for k in kids),
        "file": "f.cpp",
        "line": 3,
        "function": name,
        "children": kids,
    }


META = {"type": "meta", "pid": 42, "executable": "prog", "executable_path": "/nonexistent/prog", "start_time": "T0"}


def snapshot_msg(seq=1):
    tree = wire_node("outer", 2.0, [wire_node("inner", 5.0), wire_node("other", 1.0)])
    return {"type": "snapshot", "seq": seq, "dropped": 0, "threads": {"7": {"name": "main", "children": [tree]}}}


# ── model ─────────────────────────────────────────────────────────────────────


def test_node_round_trips_through_its_wire_form():
    original = node(
        "zone",
        2.5,
        calls=3,
        children=[node("child", 0.5)],
        annotations={"flops": {"avg": 10, "min": 10, "max": 10}},
        mem_alloc_bytes=1024,
        mem_peak_bytes=512,
        histogram={"1us": 2},
    )
    assert parse_node(json.loads(json.dumps(node_to_dict(original)))) == original


def test_parse_snapshot_keys_threads_by_string_id():
    snap = parse_snapshot(snapshot_msg())
    assert list(snap.threads) == ["7"]
    assert snap.threads["7"].label == "main"
    assert [n.name for n in snap.all_roots()] == ["outer"]


def test_stream_state_folds_each_message_type():
    state = StreamState()
    assert state.apply(META) == "meta" and state.meta.executable == "prog"
    state.apply(snapshot_msg())
    assert state.snapshot.threads["7"].children[0].name == "outer"
    state.apply({"type": "timeline", "events": [{"tid": 7, "name": "z", "start_ms": 0.0, "end_ms": 1.0}]})
    assert state.timeline[0].thread_id == "7"
    state.apply({"type": "output", "timestamp": "t", "message": "hello"})
    state.apply({"type": "log", "level": 3, "message": "careful"})
    assert [e.message for e in state.log_entries] == ["hello", "careful"] and state.log_total == 2


# ── analysis ──────────────────────────────────────────────────────────────────


def tree():
    return [node("a", 1.0, children=[node("b", 4.0, children=[node("c", 2.0)]), node("c", 3.0)])]


def test_flatten_keeps_ancestors_of_filter_matches_and_honours_collapse():
    assert [r.node.name for r in analysis.flatten_tree(tree())] == ["a", "b", "c", "c"]
    assert [r.path for r in analysis.flatten_tree(tree(), "^b$")] == ["a", "a/b"]
    collapsed = analysis.flatten_tree(tree(), collapsed={"a/b"})
    assert [r.path for r in collapsed] == ["a", "a/b", "a/c"] and collapsed[1].collapsed


def test_invalid_regex_filter_falls_back_to_substring():
    assert analysis.name_matches("pack[A", "pack[")


def test_aggregate_flat_sums_each_name_across_call_sites():
    flat = {n.name: n for n in analysis.aggregate_flat(tree())}
    assert flat["c"].exclusive_ms == 5.0 and flat["c"].call_count == 2


def test_bottom_up_lists_callers_under_each_zone():
    up = {n.name: n for n in analysis.bottom_up(tree())}
    assert sorted(c.name for c in up["c"].children) == ["a", "b"]
    assert up["a"].children == []


def test_hot_path_follows_the_most_expensive_child():
    assert analysis.hot_path(tree()) == {"a", "a/b", "a/b/c"}


def test_sort_tree_does_not_touch_the_snapshot():
    roots = tree()
    ordered = analysis.sort_tree(roots, "exclusive")
    assert [n.name for n in ordered[0].children] == ["b", "c"]
    assert [n.name for n in analysis.sort_tree(roots, "name")[0].children] == ["b", "c"]
    assert [n.name for n in roots[0].children] == ["b", "c"]


def test_anomaly_flags_an_outlying_call():
    steady = node("z", 10.0, calls=10, exclusive_min_ms=0.9, exclusive_max_ms=1.1, stddev_ms=0.1)
    spiky = node("z", 10.0, calls=10, exclusive_min_ms=0.5, exclusive_max_ms=5.0, stddev_ms=0.5)
    assert not analysis.is_anomaly(steady) and analysis.is_anomaly(spiky)


def test_roofline_needs_flops_and_bytes():
    roots = [node("k", 1000.0, annotations={"flops": 2e9, "bytes_read": {"avg": 1e9}}), node("plain")]
    [point] = analysis.roofline_points(roots)
    assert point.name == "k" and point.intensity == pytest.approx(2.0) and point.gflops == pytest.approx(2.0)


def test_compare_orders_by_largest_change():
    a = parse_snapshot(snapshot_msg())
    b = parse_snapshot(snapshot_msg())
    b.threads["7"].children[0].children[1].exclusive_ms = 11.0  # "other": 1 -> 11
    rows = analysis.compare_snapshots(a, b)
    assert rows[0].name == "other" and rows[0].delta_ms == pytest.approx(10.0)


def test_format_helpers():
    assert len(fmt.make_bar(37.0)) == 10 and fmt.make_bar(100.0) == "█" * 10
    assert fmt.format_bytes(0) == "" and fmt.format_bytes(1536) == "1.5K" and fmt.format_bytes(-(3 << 20)) == "-3.0M"
    # Stable across processes, unlike hash(): the same zone keeps its color between runs.
    assert fmt.flame_color("pack_A") == fmt.flame_color("pack_A")
    assert fmt.sparkline([1, 2, 3]) == "▁▄█"


# ── session files ─────────────────────────────────────────────────────────────


def server_export(label="prog"):
    """The shape Server::export_session writes: the snapshot fields beside meta, at top level."""
    snap = snapshot_msg()
    return {"label": label, "type": "snapshot", "meta": dict(META), "seq": 3, "dropped": 0, "threads": snap["threads"]}


def test_loads_the_servers_export_format(tmp_path):
    # Regression: the old viewer read only data["snapshot"], so every file written by
    # --einsums:profile:save opened as an empty tab.
    path = tmp_path / "s.json"
    path.write_text(json.dumps(server_export()))
    [record] = read_session_file(path)
    session = session_from_dict(record, "s1")
    assert session.snapshot is not None and session.snapshot.seq == 3
    assert session.label == "prog (T0)"  # the bare executable name is shared by every run


def test_loads_the_servers_appended_multi_session_format(tmp_path):
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"sessions": [server_export(), server_export("second run")]}))
    sessions = [session_from_dict(r, f"s{i}") for i, r in enumerate(read_session_file(path))]
    assert [s.label for s in sessions] == ["prog (T0)", "second run"]
    assert all(s.snapshot for s in sessions)


def test_saved_sessions_load_back(tmp_path):
    session = Session("s1", "mine", snapshot=parse_snapshot(snapshot_msg()), bookmarks={"inner"})
    session.record_snapshot(session.snapshot)
    for count in (1, 2):
        path = tmp_path / f"{count}.json"
        write_session_file(path, [session] * count)
        loaded = [session_from_dict(r, "x") for r in read_session_file(path)]
        assert len(loaded) == count
        assert loaded[0].snapshot == session.snapshot and loaded[0].bookmarks == {"inner"}
        assert list(loaded[0].history["inner"]) == [5.0]


def test_export_writes_json_and_one_csv_row_per_node(tmp_path):
    export_snapshot(parse_snapshot(snapshot_msg()), tmp_path / "s.json", tmp_path / "s.csv")
    assert json.loads((tmp_path / "s.json").read_text())["threads"]["7"]["name"] == "main"
    assert len((tmp_path / "s.csv").read_text().splitlines()) == 1 + 3


# ── client ────────────────────────────────────────────────────────────────────


def test_parse_endpoint():
    assert parse_endpoint("19216") == ("127.0.0.1", 19216)
    assert parse_endpoint("box:1") == ("box", 1)
    with pytest.raises(ValueError):
        parse_endpoint("box:port")


async def fake_server(chunks, on_request=None):
    """A server that writes *chunks* raw, then answers requests with on_request(method)."""

    async def handle(reader, writer):
        for chunk in chunks:
            writer.write(chunk)
            await writer.drain()
            await asyncio.sleep(0.01)
        while line := await reader.readline():
            req = json.loads(line)
            data = on_request(req["method"]) if on_request else {}
            writer.write((json.dumps({"type": "response", "id": req["id"], "data": data}) + "\n").encode())
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


def test_client_reassembles_long_lines_split_mid_character():
    # A snapshot line far longer than asyncio's 64 KiB readline limit, cut inside a
    # multi-byte character: the client must split lines from bytes, not decoded chunks.
    big = snapshot_msg()
    big["threads"]["7"]["children"][0]["annotations"] = {"note": "é" * 50_000}
    payload = (json.dumps(META) + "\n" + json.dumps(big, ensure_ascii=False) + "\n").encode()
    cut = payload.index("é".encode()) + 1
    chunks = [payload[:cut], payload[cut:]]

    async def main():
        server, port = await fake_server(chunks, on_request=lambda method: {"echo": method})
        client = ProfileClient("127.0.0.1", port)
        assert await client.connect()
        seen = []

        async def consume():
            async for msg in client.messages():
                seen.append(client.state.apply(msg))

        # Requests come from another task: the reply is read by the messages() loop.
        reader = asyncio.create_task(consume())
        while len(seen) < 2:
            await asyncio.sleep(0.01)
        assert await client.request("get_compute_graphs") == {"echo": "get_compute_graphs"}
        await client.close()
        await reader
        server.close()
        return seen, client.state

    seen, state = asyncio.run(main())
    assert seen == ["meta", "snapshot"]
    assert state.snapshot.threads["7"].children[0].annotations["note"] == "é" * 50_000


def test_request_without_connection_reports_it():
    assert asyncio.run(ProfileClient("127.0.0.1", 1).request("x")) == {"error": "not connected"}


def test_recording_round_trips(tmp_path):
    rec = Recorder(tmp_path / "r.jsonl")
    rec.write(META)
    rec.write(snapshot_msg())
    rec.close()
    assert [msg["type"] for _, msg in read_recording(tmp_path / "r.jsonl")] == ["meta", "snapshot"]


# ── disassembly lookup ────────────────────────────────────────────────────────


def test_symbol_lookup_matches_the_bare_function_name():
    mangled = "0000 b _ZGVZ4workvE5guard\n0010 T _ZN7einsums4blas5dgemmEv\n0020 T _Z10dgemm_like\n"
    pretty = "0000 b guard variable for work()::guard\n0010 T einsums::blas::dgemm()\n0020 T dgemm_like\n"
    assert match_symbol(mangled, pretty, "dgemm") == "_ZN7einsums4blas5dgemmEv"
    assert match_symbol(mangled, pretty, "work") is None  # only text symbols count


def test_strip_listing_drops_headers():
    listing = "\nfile:     file format elf64\n\nDisassembly of section .text:\n\n0010 <f()>:\n  10: ret\n\n"
    assert strip_listing(listing) == "0010 <f()>:\n  10: ret"


# ── command line ──────────────────────────────────────────────────────────────


def test_parser_has_every_command():
    from einsums.cli.profiler import _viewer_parser

    parser = build_parser()
    args = _viewer_parser("einsums profiler").parse_args(["box:3", "--port", "1", "2", "--no-mdns"])
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
    from einsums.cli.profiler import app

    return app


@needs_textual
def test_every_key_has_an_action():
    KEYMAP, ProfilerApp = app_module().KEYMAP, app_module().ProfilerApp
    app = ProfilerApp(load=["unused"], mdns=False)
    for _, keys in KEYMAP:
        for key, action, _, _ in keys:
            name = action.split("(")[0]
            assert key == "space" or hasattr(app, f"action_{name}"), action


async def wait_for(pilot, condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await pilot.pause(0.05)


@needs_textual
def test_live_session_draws_and_every_key_runs():
    PANELS, ProfilerApp = app_module().PANELS, app_module().ProfilerApp

    async def main():
        server, port = await fake_server(
            [(json.dumps(m) + "\n").encode() for m in (META, snapshot_msg(1), snapshot_msg(2))]
        )
        app = ProfilerApp([("127.0.0.1", port)], mdns=False)
        async with app.run_test(size=(140, 60)) as pilot:
            await wait_for(pilot, lambda: app.active_view is not None and app.active_view._rows)
            assert [r.node.name for r in app.active_view._rows] == ["outer", "inner", "other"]
            assert app.active_session.label == "prog (T0)"
            for panel in PANELS:
                await app.run_action(f"toggle_panel('{panel}')")
                await pilot.pause()
            assert "outer" in app.query_one("#hotspots")._text
            await pilot.press("down", "space", "space", "e", "w", "e", "s", "a", "a", "u", "u", "h", "T", "b", "B", "d", "y")
            await pilot.pause()
            assert len(app.active_session.bookmarks) == 1
            await pilot.press("slash", *"other", "enter")
            await wait_for(pilot, lambda: [r.node.name for r in app.active_view._rows] == ["outer", "other"])
            await pilot.press("escape")
            await wait_for(pilot, lambda: len(app.active_view._rows) == 3)
            await pilot.press("question_mark")
            assert type(app.screen).__name__ == "HelpScreen"
            await pilot.press("escape")
            await pilot.press("q")
        server.close()

    asyncio.run(main())


@needs_textual
def test_loaded_server_export_shows_its_tree(tmp_path):
    ProfilerApp = app_module().ProfilerApp
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"sessions": [server_export(), server_export("b")]}))

    async def main():
        app = ProfilerApp(load=[str(path)], mdns=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await wait_for(pilot, lambda: app.active_view is not None and app.active_view._rows)
            assert len(app._ui) == 2
            await pilot.press("C")
            assert type(app.screen).__name__ == "CompareScreen"
            await pilot.press("escape", "q")

    asyncio.run(main())


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
    outer = next(node for node in session.snapshot.all_roots() if node.name == "outer")
    assert outer.call_count > 0 and analysis.numeric_annotation(outer.annotations, "flops") == 2_000_000


# ── report and diff, without the viewer ───────────────────────────────────────


def _run_cli(*argv):
    from einsums.cli import main

    return main(list(argv))


def test_report_prints_hotspots_and_the_tree(tmp_path, capsys):
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"sessions": [server_export("first"), server_export("second")]}))
    assert _run_cli("profiler", "report", str(path), "--format", "json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["name"] for r in rows] == ["inner", "outer", "other"]  # by exclusive time
    assert sum(r["pct"] for r in rows) == pytest.approx(100.0)
    assert _run_cli("profiler", "report", str(path), "--session", "first", "--tree", "--format", "csv") == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "thread,name,calls,exclusive_ms,inclusive_ms,mean_ms" and len(lines) == 4
    assert _run_cli("profiler", "report", str(path), "--top", "1") == 0
    out = capsys.readouterr().out
    assert out.startswith("second") and "inner" in out and "other" not in out


def test_diff_compares_sessions_and_can_fail_a_ci_job(tmp_path, capsys):
    slow = server_export("slow")
    slow["threads"]["7"]["children"][0]["children"][1]["exclusive_ms"] = 3.0  # "other": 1 ms -> 3 ms
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(server_export("base")))
    b.write_text(json.dumps(slow))
    assert _run_cli("profiler", "diff", str(a), str(b), "--format", "json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["name"] == "other" and rows[0]["delta_pct"] == pytest.approx(200.0)
    assert _run_cli("profiler", "diff", str(a), str(b), "--fail-above", "50") == 1
    assert "other (+200.0%)" in capsys.readouterr().err
    assert _run_cli("profiler", "diff", str(a), str(b), "--fail-above", "50", "--min-ms", "2") == 0


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
    path.write_text(json.dumps(server_export() | {"compute_graphs": GRAPHS["graphs"]}))

    async def main():
        app = ProfilerApp(load=[str(path)], mdns=False)
        async with app.run_test(size=(140, 50)) as pilot:
            await wait_for(pilot, lambda: app.active_view is not None and app.active_view._rows)
            await pilot.press("K")
            await wait_for(pilot, lambda: type(app.screen).__name__ == "GraphScreen")
            from textual.widgets import Static, Tree

            tree = app.screen.query_one(Tree)
            assert len(tree.root.children) == 1 and len(tree.root.children[0].children) == 2
            detail = app.screen.query_one("#graph-detail-text", Static)
            assert "scf" in str(detail.render())
            await pilot.press("escape", "W")
            await wait_for(pilot, lambda: "live connection" in app.query_one("#taskpool")._text)
            await pilot.press("q")

    asyncio.run(main())
