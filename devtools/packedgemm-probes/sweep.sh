#!/bin/bash
# Sweep one constant over the groups it can affect, appending a progress line per point.
# usage: sweep.sh <ENVVAR> <label> <values...> -- groups come from GROUPS=
H=/home/jturney/git/EinsumsPaper/benchmarks/head-to-head
LOG=$H/results/SWEEP-PROGRESS.txt
var=$1; label=$2; shift 2
groups=${SWEEP_GROUPS:-"intensli ccsd_t"}
for v in "$@"; do
  for g in $groups; do
    out=$H/results/sweep-$label-$v-$g
    env $var=$v $H/run_head_to_head.sh --einsums-root /home/jturney/git/Einsums/build \
        --numa-node 2 --pin 8 --only $g --outdir $out > /dev/null 2>&1
    python3 - "$out" "$label" "$v" "$g" >> $LOG <<'PY'
import csv,statistics,sys
d,label,v,g=sys.argv[1:5]
R={}
for r in csv.DictReader(l for l in open(f"{d}/results-1thread-reversed.csv") if not l.startswith('#')):
    R[(r['case'],r['precision'],r['library'])]=float(r['gflops_min'])
keys=sorted({(c,p) for (c,p,l) in R if l=='einsums'})
out=[]
for pr in ('s','d'):
    rr=[R[(c,p,'einsums')]/R[(c,p,'tblis')] for c,p in keys if p==pr]
    out.append(f"{pr}={statistics.median(rr):.3f}x(w{sum(1 for x in rr if x>1)}/{len(rr)},min {min(rr):.2f})")
print(f"{label:<10} = {v:<4} {g:<9} " + "  ".join(out), flush=True)
PY
  done
done
