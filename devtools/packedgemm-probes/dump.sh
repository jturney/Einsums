#!/bin/bash
while IFS='|' read -r p name ext reps; do
  [ -z "$p" ] && continue
  line=$(EINSUMS_DEBUG_BLOCK=1 PROBE_PACKED_ONLY=1 numactl --cpunodebind=2 --membind=2 --physcpubind=8 ./ttgt_probe_dbg "$p" "$name" "$ext" 1 2>&1 | grep -m1 '^\[blk\]')
  [ -z "$line" ] && continue
  echo "$name $p | $line"
done
