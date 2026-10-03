# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""The viewer: its data model, session files, client, ``report`` and ``diff``, and the app driven
headless. What a library adds through a plugin is tested with that library."""

from __future__ import annotations

import asyncio
import importlib.util
import json

import pytest

from waggle import analysis, format as fmt
from waggle.client import ProfileClient, StreamState, parse_endpoint, read_recording, Recorder
from waggle.disasm import match_symbol, strip_listing
from waggle.model import ProfileNode, meta_to_dict, node_to_dict, parse_meta, parse_node, parse_snapshot
from waggle.session import (
    Session,
    export_snapshot,
    read_session_file,
    session_from_dict,
    write_session_file,
)
from waggle.testing import META, fake_server, server_export, snapshot_msg, wait_for


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


def test_loads_library_data_from_either_session_layout(tmp_path):
    graphs = [{"name": "scf", "nodes": [], "tensors": [], "edges": []}]
    current = server_export() | {"format": "waggle-session", "version": 1, "extensions": {"einsums.compute_graphs": graphs}}
    # Files written before the profiler became Waggle kept Einsums' graphs at the top level.
    legacy = server_export() | {"compute_graphs": graphs}
    for record in (current, legacy):
        path = tmp_path / "s.json"
        path.write_text(json.dumps(record))
        [loaded] = read_session_file(path)
        assert session_from_dict(loaded, "s1").extensions == {"einsums.compute_graphs": graphs}


def test_meta_carries_the_programs_handlers_and_clients():
    meta = parse_meta(dict(META) | {"handlers": ["get_taskpool_metrics"], "clients": [{"name": "einsums", "version": "2.0.0"}]})
    assert meta.handlers == ["get_taskpool_metrics"]
    assert [c["name"] for c in meta.clients] == ["einsums"]
    assert parse_meta(meta_to_dict(meta)) == meta
    # A server from before advertised nothing, which is not the same as advertising no handlers.
    assert parse_meta(dict(META)).handlers is None
    assert parse_meta(dict(META) | {"handlers": []}).handlers == []


def test_a_handler_registered_after_connecting_reaches_the_viewer():
    # ComputeGraph registers get_compute_graphs on its first graph, often after the viewer
    # connected; the meta message is sent once, so snapshots repeat the list.
    state = StreamState()
    state.apply({"type": "meta"} | dict(META) | {"handlers": []})
    state.apply(snapshot_msg() | {"handlers": ["get_compute_graphs"]})
    assert state.meta.handlers == ["get_compute_graphs"]


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


def test_saved_sessions_say_what_they_are(tmp_path):
    session = Session("s1", "mine", snapshot=parse_snapshot(snapshot_msg()))
    session.extensions["einsums.compute_graphs"] = [{"name": "g"}]
    path = tmp_path / "s.json"
    write_session_file(path, [session])
    data = json.loads(path.read_text())
    assert (data["format"], data["version"]) == ("waggle-session", 1)
    assert data["extensions"] == {"einsums.compute_graphs": [{"name": "g"}]}
    assert "compute_graphs" not in data


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


# ── the app, headless ─────────────────────────────────────────────────────────


needs_textual = pytest.mark.skipif(importlib.util.find_spec("textual") is None, reason="the app needs Textual")


def app_module():
    from waggle import app

    return app


@needs_textual
def test_every_key_has_an_action():
    KEYMAP, ProfilerApp = app_module().KEYMAP, app_module().ProfilerApp
    app = ProfilerApp(load=["unused"], mdns=False)
    for _, keys in KEYMAP:
        for key, action, _, _ in keys:
            name = action.split("(")[0]
            assert key == "space" or hasattr(app, f"action_{name}"), action


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


# ── report and diff, without the viewer ───────────────────────────────────────


def _run_cli(*argv):
    from waggle.cli import main

    return main(list(argv))


def test_report_prints_hotspots_and_the_tree(tmp_path, capsys):
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"sessions": [server_export("first"), server_export("second")]}))
    assert _run_cli("report", str(path), "--format", "json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["name"] for r in rows] == ["inner", "outer", "other"]  # by exclusive time
    assert sum(r["pct"] for r in rows) == pytest.approx(100.0)
    assert _run_cli("report", str(path), "--session", "first", "--tree", "--format", "csv") == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "thread,name,calls,exclusive_ms,inclusive_ms,mean_ms" and len(lines) == 4
    assert _run_cli("report", str(path), "--top", "1") == 0
    out = capsys.readouterr().out
    assert out.startswith("second") and "inner" in out and "other" not in out


def test_diff_compares_sessions_and_can_fail_a_ci_job(tmp_path, capsys):
    slow = server_export("slow")
    slow["threads"]["7"]["children"][0]["children"][1]["exclusive_ms"] = 3.0  # "other": 1 ms -> 3 ms
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(server_export("base")))
    b.write_text(json.dumps(slow))
    assert _run_cli("diff", str(a), str(b), "--format", "json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["name"] == "other" and rows[0]["delta_pct"] == pytest.approx(200.0)
    assert _run_cli("diff", str(a), str(b), "--fail-above", "50") == 1
    assert "other (+200.0%)" in capsys.readouterr().err
    assert _run_cli("diff", str(a), str(b), "--fail-above", "50", "--min-ms", "2") == 0


def test_a_server_from_before_advertising_handlers_keeps_every_panel():
    from waggle.plugin import Requirement
    from waggle.session import Session

    requirement = Requirement(handler="get_taskpool_metrics")
    old = Session("s1", "old", meta=parse_meta(dict(META)))  # no handler list at all
    assert requirement.met_by(old, live=True)
    told_none = Session("s2", "new", meta=parse_meta(dict(META) | {"handlers": []}))
    assert not requirement.met_by(told_none, live=True)
    assert not requirement.met_by(old, live=False)  # and never without a live program
