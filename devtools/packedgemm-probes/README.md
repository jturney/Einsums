# PackedGemm probes (uncommitted scratch tools from the 2026-09-12 session)

Standalone drivers linked against `build/lib/libEinsums-abi1.so`. They compile the
packed templates themselves (like the head-to-head harness does), so a header
change is picked up by recompiling the probe, without rebuilding the library.

    CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
    E=$CONDA_PREFIX/lib
    $CXX $(cat compile_flags.txt) -o ttgt_probe ttgt_probe.cpp $(cat link_libs.txt) \
        -Wl,-rpath,/home/jturney/git/Einsums/build/lib -Wl,-rpath,$E
    # add -DEINSUMS_HAVE_PROFILER_INTERNAL for per-phase zones (pack_A, tile scatter, ...)
    # kernel_bench.cpp wants the harness flags with -march=x86-64-v3 substituted for nocona

* `ttgt_probe s|d <case> "<extents>" [reps]` runs one TCB case three ways: the
  packed route as the harness calls it (reversed row-major layout), a hand TTGT
  (HPTT permutes + one vendor GEMM), and the reference GEMM. `PROBE_PACKED_ONLY=1`
  skips the last two. Pass `--einsums:profile:filename=F` with the profiler build.
* `kernel_bench` times the resolved tile kernel, a prototype kernel, and OpenBLAS on
  one packed cache block.
* `transpose_bench <ghz>` times the pack's transpose four ways (scalar gather,
  scalar tiled, LxL in registers, memcpy) hot and cold, in cycles/element. NOTE
  it does NOT interleave the micro-kernel, so it OVERSTATES what a pack change
  is worth in the engine - it said 2.47 cycles/element where the engine
  delivered 6.5. Use it to bound the permutation cost, never to predict a win.
* `watch_h2h.sh [-f]` live per-case einsums/TCL/TBLIS view of a running
  harness, with a per-group noise band. `BASE=` / `NEW=` select the result sets.
* `const_probe` prints cpu_config, compute_blocking, and micro_kernel_shape.
* `run_probes.sh <cpu> < cases.txt`, `mc_sweep.sh` drive them pinned.

`compile_flags.txt` / `link_libs.txt` were extracted from the harness's
compile_commands.json and build.ninja; regenerate them if the toolchain moves.

## Added 2026-09-13 (huge-page session)

* `thp_ab.sh s|d <case> "<extents>" [reps] [rounds]` A/Bs transparent huge pages on one
  case with ZERO code: the `thp` arm sets `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`, which
  makes malloc madvise every large chunk, and the script samples the probe's
  `AnonHugePages` so a null result cannot hide behind pages that never became huge.
  Pinned to core 8 / node 2. To A/B the shipped code path instead, set
  `EINSUMS_TENSOR_HUGE_PAGES=1` in the environment (the probe calls `einsums::initialize`).
* `h2h_table.py <group> [modes...]` tabulates `results/2026-09-13-hp-<mode>-<group>.log`
  harness logs side by side (einsums / tblis / tcl %GEMM, GF/s deltas, win counts).
* `run_probes.sh` and `mc_sweep.sh` now pin to core 8 / node 2 (see the handoff's section 3;
  core 2 / node 0 was the old, invalid convention).

## Added 2026-09-14 (C-traffic session)

* `runset.sh <probe> [casefile]` runs a probe over a case list, pinned to core 8
  / node 2. A case list is one `precision|case|extents|reps` per line; generate
  one from the benchmark's own `cases.csv` rather than keeping a snapshot:

      awk -F, '!/^#/ && NF>=5 {print $4"|"$2"|"$5"|3"}' \
          /home/jturney/git/EinsumsPaper/benchmarks/head-to-head/cases.csv > all_cases.txt

  and note that `cases.csv` has FOUR groups - ccsd, ccsd_t, intensli and
  **ao2mo** - not the three that most of these notes name. Check with
  `awk -F, '!/^#/{print $1}' cases.csv | sort -u`.
* `ab.sh <baseBin> <newBin> [casefile] [rounds]` interleaves two probe binaries
  over a case list and prints best-of per case with the raw samples, so a
  suspicious delta can be read against its own spread.
* `abenv.sh <bin> [casefile] [rounds]` does the same A/B with ONE binary and an
  environment switch, which is the better shape when the change can be gated at
  runtime: it removes the binary as a variable.
* `dump.sh` reads a case list and prints one line per case from whatever
  `EINSUMS_DEBUG_*` instrumentation is currently compiled in. Nothing is
  committed instrumented; add the getenv block, build, dump, remove. Doing this
  first is worth more than it looks - two sessions of wrong guesses about the
  C write-back collapsed in one build once `compose`, `Fm`, `Fn`, `MC` and
  `n_inner` were printed per case.

**The probes link the prebuilt `build/lib/libEinsums-abi1.so` while compiling
the PackedGemm templates themselves.** That is what makes header-only iteration
a 30 s rebuild instead of two minutes, and it breaks SILENTLY if a type that
crosses the library boundary changes layout: adding one `bool` to `PackingPlan`
made a probe read the field as false while the plan's M and N came through
swapped, with wrong results, no crash and no warning. After any such change,
`cmake --build build --target Einsums` first, then rebuild EVERY probe binary.
