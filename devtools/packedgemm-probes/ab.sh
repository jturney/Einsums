#!/bin/bash
# usage: ab.sh <baseBin> <newBin> [casefile] [rounds]
A=${1:-./ttgt_probe_base}; B=${2:-./ttgt_probe_new}; cf=${3:-cases_work.txt}; R=${4:-3}
declare -A sa sb
for r in $(seq 1 $R); do
  for bin in A B; do
    [ $bin = A ] && p=$A || p=$B
    while IFS='|' read -r pr name ext reps; do
      [ -z "$pr" ] && continue
      g=$(PROBE_PACKED_ONLY=1 numactl --cpunodebind=2 --membind=2 --physcpubind=8 "$p" "$pr" "$name" "$ext" "$reps" 2>/dev/null | awk '/packed /{print $4}')
      key="$name $pr"
      if [ $bin = A ]; then sa[$key]="${sa[$key]} $g"; else sb[$key]="${sb[$key]} $g"; fi
    done < "$cf"
  done
done
printf "%-26s %-3s %10s %10s %8s\n" case p base new "delta%"
for k in "${!sa[@]}"; do
  ba=$(echo ${sa[$k]} | tr ' ' '\n' | sort -g | tail -1)
  bb=$(echo ${sb[$k]} | tr ' ' '\n' | sort -g | tail -1)
  d=$(python3 -c "print(f'{100*($bb/$ba-1):+.1f}')")
  printf "%-30s %10.2f %10.2f %8s   base:[%s] new:[%s]\n" "$k" $ba $bb $d "${sa[$k]}" "${sb[$k]}"
done | sort
