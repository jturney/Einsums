# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import re, sys, statistics
R = "/home/jturney/git/EinsumsPaper/benchmarks/head-to-head/results/"
def parse(path):
    out = {}
    cur = None
    for line in open(path):
        m = re.match(r"\[\s*\d+/\d+\]\s+(\S+)\s+([sd])\s+m=", line)
        if m: cur = (m.group(1), m.group(2)); out[cur] = {}; continue
        m = re.match(r"\s+(einsums|tblis|tcl)\s+([\d.]+) GF\s+([\d.]+)% of GEMM", line)
        if m and cur: out[cur][m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out
group = sys.argv[1]
modes = sys.argv[2:] if len(sys.argv) > 2 else ["off", "on", "thp"]
data = {m: parse(f"{R}2026-09-13-hp-{m}-{group}.log") for m in modes}
cases = list(data[modes[0]].keys())
hdr = f"{'case':18s} p | {'off: eins':>9s} {'tblis':>6s} {'tcl':>6s} | {'on: eins':>8s} {'d%':>6s} | {'thp: eins':>9s} {'tblis':>6s} {'tcl':>6s} {'d%eins':>6s} {'d%tblis':>7s}"
print(hdr)
deltas = {"on": [], "thp": [], "thp_tblis": [], "thp_tcl": []}
wins = {m: {"tblis": 0, "tcl": 0} for m in modes}
for c in cases:
    o = data["off"][c]; n = data.get("on", {}).get(c, {}); t = data.get("thp", {}).get(c, {})
    e0 = o["einsums"][0]
    don = (n["einsums"][0]/e0-1)*100 if n else float("nan")
    dth = (t["einsums"][0]/e0-1)*100 if t else float("nan")
    dtb = (t["tblis"][0]/o["tblis"][0]-1)*100 if t else float("nan")
    dtc = (t["tcl"][0]/o["tcl"][0]-1)*100 if t else float("nan")
    if n: deltas["on"].append(don)
    if t: deltas["thp"].append(dth); deltas["thp_tblis"].append(dtb); deltas["thp_tcl"].append(dtc)
    for m in modes:
        d = data[m].get(c)
        if d:
            wins[m]["tblis"] += d["einsums"][0] > d["tblis"][0]
            wins[m]["tcl"]   += d["einsums"][0] > d["tcl"][0]
    print(f"{c[0]:18s} {c[1]} | {o['einsums'][1]:8.1f}% {o['tblis'][1]:5.1f}% {o['tcl'][1]:5.1f}% | "
          f"{n['einsums'][1] if n else float('nan'):7.1f}% {don:+6.1f} | "
          f"{t['einsums'][1] if t else float('nan'):8.1f}% {t['tblis'][1] if t else float('nan'):5.1f}% {t['tcl'][1] if t else float('nan'):5.1f}% {dth:+6.1f} {dtb:+7.1f}")
print()
for k, v in deltas.items():
    if v: print(f"{k:10s} GF/s change vs off: mean {statistics.mean(v):+.1f}%  median {statistics.median(v):+.1f}%  min {min(v):+.1f}%  max {max(v):+.1f}%")
for m in modes:
    print(f"{m:4s}: einsums beats tblis on {wins[m]['tblis']}/{len(cases)}, tcl on {wins[m]['tcl']}/{len(cases)}")
