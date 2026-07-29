#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Regenerate the pydata-sphinx-theme version switcher for the docs site.

The switcher must list EVERY published version, so it is rebuilt on every
deploy: the version directories already on gh-pages (``git ls-tree`` of the
fetched branch) are unioned with the version being deployed right now.
``latest`` (main) is listed first; the highest tag is marked preferred and
labeled stable.

Usage::

    generate_switcher.py --deploying v1.2.0 --out site-root/switcher.json \
        [--base-url https://einsums.github.io/Einsums] [--ref origin/gh-pages]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

VERSION_RE = re.compile(r"^v\d+(\.\d+)*([.-].*)?$")


def published_dirs(ref: str) -> set[str]:
    """Top-level directories on the deployed branch, or empty if it does not
    exist yet (first deploy)."""
    res = subprocess.run(["git", "ls-tree", "--name-only", ref],
                         capture_output=True, text=True)
    if res.returncode != 0:
        return set()
    return {ln.strip() for ln in res.stdout.splitlines() if ln.strip()}


def semver_key(tag: str) -> tuple:
    nums = re.findall(r"\d+", tag)
    return tuple(int(n) for n in nums)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deploying", required=True,
                    help="version directory being deployed now (latest or vX.Y.Z)")
    ap.add_argument("--out", required=True, help="output switcher.json path")
    ap.add_argument("--base-url", default="https://einsums.github.io/Einsums")
    ap.add_argument("--ref", default="origin/gh-pages",
                    help="git ref of the deployed site to enumerate")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    dirs = published_dirs(args.ref)
    dirs.add(args.deploying)
    tags = sorted((d for d in dirs if VERSION_RE.match(d)), key=semver_key, reverse=True)
    have_latest = "latest" in dirs

    entries = []
    if have_latest:
        entries.append({
            "name": "latest (main)",
            "version": "latest",
            "url": f"{base}/latest/",
        })
    for i, tag in enumerate(tags):
        entry = {
            "name": f"{tag} (stable)" if i == 0 else tag,
            "version": tag,
            "url": f"{base}/{tag}/",
        }
        if i == 0:
            entry["preferred"] = True
        entries.append(entry)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"switcher: {len(entries)} entries -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
