# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""``einsums profiler report`` and ``einsums profiler diff``: saved sessions without the viewer.

For scripts and CI: the same hotspot table and session comparison the viewer shows, as
text, CSV or JSON. Reads every session-file shape the viewer loads (``--einsums:profile:save``
and the viewer's own saves, single or multi-session). Needs no Textual.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys

from .analysis import aggregate_flat, compare_snapshots, name_matches, walk
from .session import Session, read_session_file, session_from_dict


def add_parsers(sub) -> None:
    rep = sub.add_parser("report", help="Hotspots or the call tree of a saved session")
    rep.add_argument("file", help="session file (--einsums:profile:save, or the viewer's Save)")
    rep.add_argument("--session", default="-1", help="which session in a multi-session file: index or label (default: the last)")
    rep.add_argument("--thread", help="only threads whose name matches this regex")
    rep.add_argument("--filter", default="", help="only zones whose name matches this regex")
    rep.add_argument("--tree", action="store_true", help="the call tree instead of the flat hotspot table")
    rep.add_argument("--top", type=int, default=20, help="rows of the hotspot table (default: 20; 0 for all)")
    rep.add_argument("--format", choices=("text", "csv", "json"), default="text")
    rep.set_defaults(action=report)

    diff = sub.add_parser(
        "diff",
        help="Compare two saved sessions zone by zone",
        description="Compare two sessions: A and B are files, or one file holding several sessions "
        "(then the first and last are compared unless --sessions picks two).",
    )
    diff.add_argument("files", nargs="+", metavar="FILE", help="one or two session files")
    diff.add_argument("--sessions", nargs=2, metavar=("A", "B"), help="index or label of each session to compare")
    diff.add_argument("--filter", default="", help="only zones whose name matches this regex")
    diff.add_argument("--top", type=int, default=0, help="only the N largest changes (default: all)")
    diff.add_argument("--format", choices=("text", "csv", "json"), default="text")
    diff.add_argument(
        "--fail-above",
        type=float,
        metavar="PCT",
        help="exit 1 if any zone's exclusive time grew by more than PCT percent (zones under --min-ms ignored)",
    )
    diff.add_argument("--min-ms", type=float, default=1.0, help="ignore zones below this in A for --fail-above (default: 1)")
    diff.set_defaults(action=diff_sessions)


def _load(path: str) -> list[Session]:
    sessions = [session_from_dict(record, f"s{i}", path) for i, record in enumerate(read_session_file(path))]
    if not sessions:
        raise SystemExit(f"{path}: no sessions")
    return sessions


def _pick(sessions: list[Session], which: str) -> Session:
    try:
        return sessions[int(which)]
    except ValueError:
        pass
    except IndexError:
        raise SystemExit(f"no session {which}; the file holds {len(sessions)}") from None
    for session in sessions:
        if session.label == which:
            return session
    raise SystemExit(f"no session labelled {which!r}; labels: {', '.join(s.label for s in sessions)}")


def _emit(rows: list[dict], columns: list[str], fmt: str, title: str = "") -> None:
    if fmt == "json":
        print(json.dumps(rows, indent=2))
        return
    if fmt == "csv":
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        sys.stdout.write(out.getvalue())
        return
    if title:
        print(title)
    if not rows:
        print("(no zones)")
        return
    cells = [[_cell(row[c]) for c in columns] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]
    numeric = [all(isinstance(row[c], (int, float)) or row[c] is None for row in rows) for c in columns]
    print("  ".join(c.rjust(w) if n else c.ljust(w) for c, w, n in zip(columns, widths, numeric)).rstrip())
    for r in cells:
        print("  ".join(v.rjust(w) if n else v.ljust(w) for v, w, n in zip(r, widths, numeric)).rstrip())


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def report(args: argparse.Namespace) -> int:
    session = _pick(_load(args.file), args.session)
    if session.snapshot is None:
        raise SystemExit(f"session {session.label!r} has no profile data")
    threads = [t for t in session.snapshot.threads.values() if not args.thread or name_matches(t.label, args.thread)]
    title = f"{session.label}  ({len(threads)} thread(s))"
    if args.tree:
        rows = [
            {"thread": t.label, "depth": depth, "name": "  " * depth + node.name, "calls": node.call_count,
             "exclusive_ms": node.exclusive_ms, "inclusive_ms": node.inclusive_ms, "mean_ms": node.mean_ms}
            for t in threads
            for node, depth in walk(t.children)
            if name_matches(node.name, args.filter)
        ]  # fmt: skip
        if args.format != "text":
            for row in rows:
                row["name"] = row["name"].strip()
        _emit(rows, ["thread", "name", "calls", "exclusive_ms", "inclusive_ms", "mean_ms"], args.format, title)
        return 0
    flat = sorted(aggregate_flat([n for t in threads for n in t.children]), key=lambda n: n.exclusive_ms, reverse=True)
    total = sum(n.exclusive_ms for n in flat) or 1.0
    flat = [n for n in flat if name_matches(n.name, args.filter)]
    rows = [
        {"name": n.name, "pct": n.exclusive_ms / total * 100.0, "exclusive_ms": n.exclusive_ms,
         "calls": n.call_count, "mean_ms": n.mean_ms, "inclusive_ms": n.inclusive_ms}
        for n in (flat[: args.top] if args.top else flat)
    ]  # fmt: skip
    _emit(rows, ["pct", "exclusive_ms", "calls", "mean_ms", "name"], args.format, title)
    return 0


def diff_sessions(args: argparse.Namespace) -> int:
    if len(args.files) > 2:
        raise SystemExit("diff takes one or two files")
    if len(args.files) == 2:
        a_all, b_all = _load(args.files[0]), _load(args.files[1])
        picks = args.sessions or ("-1", "-1")
        a, b = _pick(a_all, picks[0]), _pick(b_all, picks[1])
    else:
        sessions = _load(args.files[0])
        if len(sessions) < 2 and not args.sessions:
            raise SystemExit(f"{args.files[0]} holds one session; give a second file")
        a, b = (_pick(sessions, w) for w in (args.sessions or ("0", "-1")))
    rows = [
        {"name": r.name, "a_ms": r.exclusive_a, "b_ms": r.exclusive_b, "delta_ms": r.delta_ms,
         "delta_pct": r.delta_pct if r.exclusive_a > 0 else None, "a_calls": r.calls_a, "b_calls": r.calls_b}
        for r in compare_snapshots(a.snapshot, b.snapshot)
        if name_matches(r.name, args.filter)
    ]  # fmt: skip
    shown = rows[: args.top] if args.top else rows
    _emit(shown, ["delta_ms", "delta_pct", "a_ms", "b_ms", "a_calls", "b_calls", "name"], args.format, f"A: {a.label}\nB: {b.label}")
    if args.fail_above is not None:
        grew = [r for r in rows if r["a_ms"] >= args.min_ms and r["delta_pct"] is not None and r["delta_pct"] > args.fail_above]
        if grew:
            names = ", ".join(f"{r['name']} (+{r['delta_pct']:.1f}%)" for r in grew[:10])
            print(f"{len(grew)} zone(s) grew by more than {args.fail_above:g}%: {names}", file=sys.stderr)
            return 1
    return 0
