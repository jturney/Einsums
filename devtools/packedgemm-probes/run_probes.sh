#!/bin/bash
# $1 = cpu to pin, rest = list of "prec name extents" lines on stdin
cpu=$1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close
while read -r prec name ext; do
  [ -z "$prec" ] && continue
  numactl --membind=2 -- taskset -c $cpu ./ttgt_probe $prec $name "$ext" 2 2>&1 | grep -v "^\s*$"
done
