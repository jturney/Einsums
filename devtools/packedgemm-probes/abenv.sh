#!/bin/bash
# usage: abenv.sh <bin> <casefile> <rounds> -- A = with EINSUMS_NO_MN_SWAP=1, B = without
P=${1:-./ttgt_probe_new}; cf=${2:-cases_work.txt}; R=${3:-3}
declare -A sa sb
for r in $(seq 1 $R); do
  for arm in A B; do
    while IFS='|' read -r pr name ext reps; do
      [ -z "$pr" ] && continue
      if [ $arm = A ]; then EV="EINSUMS_NO_MN_SWAP=1"; else EV="EINSUMS_NO_MN_SWAP_UNUSED=1"; fi
      g=$(env $EV PROBE_PACKED_ONLY=1 numactl --cpunodebind=2 --membind=2 --physcpubind=8 "$P" "$pr" "$name" "$ext" "$reps" 2>/dev/null | awk '/packed /{print $4}')
      key="$name $pr"
      if [ $arm = A ]; then sa[$key]="${sa[$key]} $g"; else sb[$key]="${sb[$key]} $g"; fi
    done < "$cf"
  done
done
printf "%-30s %10s %10s %8s\n" case base new "delta%"
for k in "${!sa[@]}"; do
  ba=$(echo ${sa[$k]} | tr ' ' '\n' | sort -g | tail -1)
  bb=$(echo ${sb[$k]} | tr ' ' '\n' | sort -g | tail -1)
  d=$(python3 -c "print(f'{100*($bb/$ba-1):+.1f}')")
  printf "%-30s %10.2f %10.2f %8s   base:[%s] new:[%s]\n" "$k" $ba $bb $d "${sa[$k]}" "${sb[$k]}"
done | sort
