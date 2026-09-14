#!/bin/bash
# usage: runset.sh <probe-binary> [casefile]
probe=${1:-./ttgt_probe_base}
cf=${2:-cases_work.txt}
while IFS='|' read -r p name ext reps; do
  [ -z "$p" ] && continue
  PROBE_PACKED_ONLY=1 numactl --cpunodebind=2 --membind=2 --physcpubind=8 \
     "$probe" "$p" "$name" "$ext" "$reps" 2>&1 | grep -E '^case|packed'
done < "$cf"
