#!/usr/bin/env python3
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Render one head-to-head result set as a box table of percent-of-GEMM.

    table96.py <resultdir> [> table.txt]

Takes a single run's directory so every row shares one reference-GEMM
calibration; %GEMM is NOT comparable between runs (the reference itself drifts
1.5% single / 4.6% double), so never stitch directories together here.
"""
import csv, sys, os

GROUP_ORDER = ("ccsd", "ccsd_t", "intensli", "ao2mo")

def main(d):
    path = os.path.join(d, "results-1thread-reversed.csv")
    pct, grp = {}, {}
    for r in csv.DictReader(l for l in open(path) if not l.startswith("#")):
        pct[(r["case"], r["precision"], r["library"])] = float(r["pct_of_gemm"])
        grp[(r["case"], r["precision"])] = r["group"]
    keys = sorted({(c, p) for (c, p, l) in pct if l == "einsums"},
                  key=lambda k: (GROUP_ORDER.index(grp[k]) if grp[k] in GROUP_ORDER else 99, k[0], k[1]))
    rows = [(grp[k], k[0], k[1], pct[(k[0], k[1], "einsums")],
             pct[(k[0], k[1], "tblis")], pct[(k[0], k[1], "tcl")]) for k in keys]
    f = lambda v: f"{v:.1f}%"
    wg = max(len("category"), max(len(r[0]) for r in rows))
    wn = max(len("case"), max(len(r[1]) for r in rows))
    we = max(len("einsums"), max(len(f(r[3])) for r in rows))
    wt = max(len("TBLIS"), max(len(f(r[4])) for r in rows))
    wc = max(len("TCL"), max(len(f(r[5])) for r in rows))
    bar = lambda a, b, c: a + "─"*(wg+2) + b + "─"*(wn+2) + b + "─"*5 + b + "─"*(we+2) + b + "─"*(wt+2) + b + "─"*(wc+2) + c
    print(f"# {os.path.basename(os.path.normpath(d))}: percent of an equally-sized GEMM, one run, one calibration")
    print(bar("┌", "┬", "┐"))
    print(f"│ {'category':^{wg}} │ {'case':^{wn}} │ {'p':^3} │ {'einsums':^{we}} │ {'TBLIS':^{wt}} │ {'TCL':^{wc}} │")
    print(bar("├", "┼", "┤"))
    for i, (g, c, p, e, t, x) in enumerate(rows):
        print(f"│ {g:<{wg}} │ {c:<{wn}} │ {p:<3} │ {f(e):<{we}} │ {f(t):<{wt}} │ {f(x):<{wc}} │")
        print(bar("├", "┼", "┤") if i < len(rows)-1 else bar("└", "┴", "┘"))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
