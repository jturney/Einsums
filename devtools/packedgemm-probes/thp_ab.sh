#!/bin/bash
# A/B the glibc THP malloc tunable on one TCB case, packed route only, pinned core 8 / node 2.
# usage: hp_ab.sh <s|d> <case> "<extents>" [reps] [rounds]
# Prints one line per run: label, peak AnonHugePages of the probe process, and the probe's packed line.
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close PROBE_PACKED_ONLY=1
prec=$1; name=$2; ext=$3; reps=${4:-3}; rounds=${5:-2}
tmp=$(mktemp)
run_one() {
  local label=$1 tun=$2 peak=0 hp
  if [ -n "$tun" ]; then
    GLIBC_TUNABLES=$tun numactl --membind=2 -- taskset -c 8 ./ttgt_probe "$prec" "$name" "$ext" "$reps" >"$tmp" 2>&1 &
  else
    numactl --membind=2 -- taskset -c 8 ./ttgt_probe "$prec" "$name" "$ext" "$reps" >"$tmp" 2>&1 &
  fi
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    hp=$(awk '/AnonHugePages/{print $2}' /proc/$pid/smaps_rollup 2>/dev/null)
    [ -n "$hp" ] && [ "$hp" -gt "$peak" ] && peak=$hp
    sleep 0.3
  done
  wait "$pid"
  printf '%-5s AnonHuge_peak=%9s kB  %s\n' "$label" "$peak" "$(grep -E 'packed ' "$tmp" | head -1)"
  grep -q 'packed ' "$tmp" || { echo "  !! no packed line; output was:"; cat "$tmp"; }
}
echo "### $prec $name $ext reps=$reps rounds=$rounds  ($(date -u +%H:%M:%S))"
for r in $(seq "$rounds"); do
  run_one base ""
  run_one thp glibc.malloc.hugetlb=1
done
rm -f "$tmp"
