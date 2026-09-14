#!/usr/bin/env bash
# Live view of a head-to-head run: each completed case against BOTH references,
# and against the recorded baseline. Reads the in-progress .log files, one per
# --only group, so it is safe to run while the harness is going.
#
#   watch_h2h.sh                 one pass over whatever has landed so far
#   watch_h2h.sh -f              refresh every 20 s until the runs finish
#
# BASE / NEW name the two result sets by their log prefix.
set -uo pipefail
H2H=/home/jturney/git/EinsumsPaper/benchmarks/head-to-head
BASE="${BASE:-$H2H/results/2026-09-14-shapeblocking}"
NEW="${NEW:-$H2H/results/2026-09-14-core8}"

render() {
python3 - "$BASE" "$NEW" <<'PY'
import glob, os, re, sys, statistics as st
base_pfx, new_pfx = sys.argv[1], sys.argv[2]
case_re = re.compile(r'^\[\s*\d+/\s*\d+\]\s+(\S+)\s+([sd])\s')
lib_re  = re.compile(r'^\s+(einsums|tblis|tcl)\s+[\d.]+ GF\s+([\d.]+)% of GEMM')

def parse(path):
    out, cur = {}, None
    try: fh = open(path, errors='replace')
    except OSError: return out
    with fh:
        for ln in fh:
            m = case_re.match(ln)
            if m: cur = (m.group(1), m.group(2)); continue
            m = lib_re.match(ln)
            if m and cur: out.setdefault(cur, {})[m.group(1)] = float(m.group(2))
    return out

groups = sorted({os.path.basename(p)[len(os.path.basename(new_pfx)) + 1:-4]
                 for p in glob.glob(new_pfx + '-*.log')})
if not groups:
    print('no run logs yet'); raise SystemExit

def mad(xs):
    if len(xs) < 3: return 0.0
    m = st.median(xs)
    return st.median([abs(x - m) for x in xs])

tot = dict(n=0, w=0, t=0, l=0, bt=0)
for g in groups:
    new  = parse(f'{new_pfx}-{g}.log')
    base = parse(f'{base_pfx}-{g}.log')
    done = [k for k, v in new.items() if {'einsums', 'tcl', 'tblis'} <= v.keys()]
    if not done: continue

    deltas = [new[k]['einsums'] - base[k]['einsums'] for k in done if k in base]
    # Flag a drop only when it clears this group's own spread. einsums' loop-based
    # kernels scatter far more than TCL's single-GEMM route (ccsd_t: sd 2.0 vs
    # 0.25), so one fixed threshold cries wolf on one group and misses the other.
    band = 1.0 + 3.0 * mad(deltas)

    rows = []
    for k in done:
        n = new[k]; b = base.get(k, {})
        e, tcl, tb = n['einsums'], n['tcl'], n['tblis']
        d = e - b['einsums'] if 'einsums' in b else None
        v = 'WIN ' if e > tcl + 0.05 else ('loss' if e < tcl - 0.05 else 'tie ')
        rows.append((v, k[0], k[1], e, tcl, tb, d, d is not None and d < -band))
    rows.sort(key=lambda r: (r[0] != 'loss', r[6] if r[6] is not None else 0))

    w = sum(r[0] == 'WIN ' for r in rows); l = sum(r[0] == 'loss' for r in rows)
    t = len(rows) - w - l; bt = sum(r[5] > r[3] + 0.05 for r in rows)
    tot['n'] += len(rows); tot['w'] += w; tot['t'] += t; tot['l'] += l; tot['bt'] += bt

    print(f'=== {g}  ({len(rows)}/{len(base) or "?"} cases)   drop flagged below {-band:.1f} pts ===')
    print(f'{"":4}  {"case":<20} {"":1}  {"einsums":>8} {"tcl":>7} {"tblis":>7} {"vs base":>8}')
    for v, case, prec, e, tcl, tb, d, flag in rows:
        ds = f'{d:+7.1f}' if d is not None else '      -'
        star = '*' if tb > e + 0.05 else ' '   # * = TBLIS ahead of us
        print(f'{v}  {case:<20} {prec}  {e:7.1f}% {tcl:6.1f}% {tb:6.1f}%{star} {ds}'
              + ('  <-- REGRESSED' if flag else ''))
    if deltas:
        ctl = [new[k]['tcl'] - base[k]['tcl'] for k in done if k in base]
        print(f'  einsums vs base: mean {st.mean(deltas):+5.2f} sd {st.pstdev(deltas):4.2f}'
              f'   | tcl CONTROL (code unchanged): mean {st.mean(ctl):+5.2f} sd {st.pstdev(ctl):4.2f}')
    print(f'  {w} win, {t} tie, {l} loss vs TCL   ({bt} also behind TBLIS)\n')

print(f'TOTAL {tot["n"]} cases: {tot["w"]} win, {tot["t"]} tie, {tot["l"]} loss vs TCL'
      f'   ({tot["bt"]} behind TBLIS)')
PY
}

if [[ "${1:-}" == "-f" ]]; then
    while true; do
        clear; date '+%H:%M:%S'; render
        pgrep -f 'head_to_head' >/dev/null || { echo; echo '(harness not running)'; break; }
        sleep 20
    done
else
    render
fi
