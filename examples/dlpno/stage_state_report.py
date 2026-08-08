#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""How many ``self`` fields each DLPNO phase reads and writes.

The diagnostic behind "cut where the interface is narrow". Promoting a method
means turning it into a free function, so the number of ``self`` fields it
touches is the width of the contract it would need, and the design's rule of
thumb is that roughly forty means the stage is cut in the wrong place.

    python examples/dlpno/stage_state_report.py

Run before choosing what to promote. It is a static AST pass over the source, so
it needs no einsums build and no fixtures.

Two limits, both of which make the numbers a LOWER bound:

* A phase that mutates an existing structure in place shows no write at all.
  ``precompute_fits`` reports zero writes and is not writing nothing.
* Attributes reached through anything but a bare ``self.x`` are missed.

That is the right direction to be wrong in for this purpose: a phase this
reports as wide certainly is.

This is the measurement half of the method-promotion work deferred from M4 to
M6. It is useful as a diagnostic well before it is useful as a generator, which
is most of the argument for having deferred the generator.
"""

import argparse
import ast
import os
import pathlib
import sys

DEFAULT_SOURCES = ["dlpno/base.py", "dlpno/mp2.py"]
DEFAULT_PHASES = [
    "setup_orbitals", "compute_doi", "prep_sparsity", "compute_metric", "compute_qia",
    "precompute_fits", "pno_transform", "plan_pno_couplings", "lmp2_iterations",
]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--sources", nargs="*", default=DEFAULT_SOURCES)
parser.add_argument("--phases", nargs="*", default=DEFAULT_PHASES)
parser.add_argument("--wide", type=int, default=40,
                    help="reads+writes above which a stage is flagged as cut wrong")
args = parser.parse_args()

here = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))

methods = {}
for source in args.sources:
    tree = ast.parse((here / source).read_text())
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for fn in cls.body:
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods[fn.name] = fn


def self_fields(fn, seen=None, depth=0):
    """``self.x`` reads and writes reachable from *fn*, following ``self.helper()``."""
    seen = set() if seen is None else seen
    reads, writes, calls = set(), set(), set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "self":
            (writes if isinstance(node.ctx, ast.Store) else reads).add(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            calls.add(node.func.attr)

    helpers = calls & set(methods)      # a call target is a method, not state
    reads -= helpers
    writes -= helpers
    if depth < 3:
        for helper in sorted(helpers - seen):
            seen.add(helper)
            r, w = self_fields(methods[helper], seen, depth + 1)
            reads |= r
            writes |= w
    return reads, writes


print(f"{'phase':<24} {'reads':>6} {'writes':>7} {'total':>6}  produces")
missing = []
for phase in args.phases:
    if phase not in methods:
        missing.append(phase)
        continue
    reads, writes = self_fields(methods[phase])
    reads -= writes                     # written before use is not an input
    total = len(reads) + len(writes)
    flag = "  <-- cut wrong?" if total >= args.wide else ""
    produced = ", ".join(sorted(writes)[:4]) + (" ..." if len(writes) > 4 else "")
    print(f"{phase:<24} {len(reads):>6} {len(writes):>7} {total:>6}  {produced}{flag}")

if missing:
    print(f"\nnot found (renamed or promoted already): {', '.join(missing)}")
print("\nCounts are lower bounds: in-place mutation shows no write. See the module docstring.")
