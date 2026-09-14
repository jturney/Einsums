#!/bin/bash
# MC sweep on one intensli case, packed path only. Ascending, then descending, then MC_FROM_KC.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close PROBE_PACKED_ONLY=1
run() { numactl --membind=2 -- taskset -c 8 ./ttgt_probe "$@" 3 2>&1 | grep -E "packed|blocking" | head -3; }
case_name=$1; ext=$2; prec=${3:-s}
echo "### $prec $case_name  (report blocking)"
EINSUMS_EXPERIMENT_REPORT_BLOCKING=1 run $prec $case_name "$ext" | sort -u
for order in "32 128 512 2048" "2048 512 128 32"; do
  echo "### sweep order: $order"
  for mc in $order; do
    printf "MC=%-5s " $mc; EINSUMS_EXPERIMENT_MC=$mc run $prec $case_name "$ext"
  done
done
echo "### MC_FROM_KC=1"
EINSUMS_EXPERIMENT_MC_FROM_KC=1 EINSUMS_EXPERIMENT_REPORT_BLOCKING=1 run $prec $case_name "$ext" | sort -u
