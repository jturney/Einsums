..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums-command:

The einsums Command
===================

The Python package carries Einsums' developer tools behind one command, ``einsums``. A build with
``EINSUMS_BUILD_PYTHON=ON`` writes it to ``build/bin``, and ``cmake --install`` puts it in
``<prefix>/bin`` beside the installed package (``<prefix>/${EINSUMS_INSTALL_PYMODDIR}/einsums``).
Put that directory on ``PATH``. The command runs the interpreter the build used, so ``_core``
loads with the Python it was compiled for, and finds the package relative to itself.
``python -m einsums`` is the same thing wherever the package is importable.

.. code-block:: bash

    export PATH=$PWD/build/bin:$PATH
    einsums info                       # what this Einsums is, for a bug report
    einsums options profile            # runtime options matching "profile"
    einsums profiler                   # attach to a program run with --einsums:profile:server
    einsums bench run                  # build and run the performance tests, store the results
    einsums stages promote mymethod/stages.py --out cpp/

``info``
    The version, commit and build type, which ``EINSUMS_WITH_*`` features are compiled in, the
    package and ``libEinsums`` locations, the BLAS library ``libEinsums`` links, the Python and
    its optional packages, and the CPU with the highest x86-64 level it supports (the ceiling for
    the SIMD rung; ``EINSUMS_SIMD_ARCH`` and ``STRIPES_ARCH`` overrides are listed when set).
    ``--json`` for scripts. The runtime is not started.

``options [PATTERN]``
    Every runtime option this build accepts, read from the library's own option registry: its
    ``--einsums:`` flag, the ``EINSUMS_*`` environment variable, the ``einsums.rc`` field, the
    default and the help text. ``PATTERN`` is a case-insensitive regular expression over the
    name, category and help. ``--json`` for scripts.

``profiler``
    The terminal profile viewer, described in :ref:`tutorial-performance`; ``profiler report`` and
    ``profiler diff`` print saved sessions without it.

``bench``
    Runs the performance tests and keeps their results in a SQLite database, so a run can be
    compared with earlier ones and a regression bisected; see below.

``stages``
    The promote and extract tools of ``einsums.stages``, also reachable as
    ``python -m einsums.stages``.

``completion bash|zsh|fish``
    A completion script generated from the commands' own parsers, for example
    ``einsums completion bash > ~/.local/share/bash-completion/completions/einsums``.

A command that starts the runtime (``stages promote`` imports the stages module) writes no
``profile.txt``: the tool turns the profiler's text report off unless ``einsums.rc`` asks for it.

Tracking performance
--------------------

.. code-block:: bash

    einsums bench run          # build Tests.Performance, run the contraction suite, store results
    einsums bench compare      # the latest run against recent runs on main

The performance tests are built with ``EINSUMS_WITH_TESTS_BENCHMARKS=ON`` and are disabled in
ctest, so they run only through ``bench run`` or ``ctest -L PERFORMANCE_ONLY``.

Suites
^^^^^^

Every performance test belongs to one suite, given as ``SUITE`` to
``einsums_add_performance_test`` and carried as the ctest label ``SUITE_CONTRACTION`` or
``SUITE_INFRASTRUCTURE``:

``contraction``
    Contractions and the operations they are built from: einsum dispatch and its routes,
    PackedGemm, sort-GEMM, permutes and transposes, BLAS and LAPACK, and the Tensor Contraction
    Benchmark (below). ``bench run`` runs this suite unless told otherwise.

``infrastructure``
    The machinery around them: allocators and pooled tensors, the profiler's own cost, the task
    pool and graph executors, graph overhead, bandwidth and peak-FLOP probes.

``bench run --suite infrastructure`` or ``--suite all`` runs the others; ``--targets`` names
binaries whatever their suite. ``bench list-tests`` shows each test's suite.

The Tensor Contraction Benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``BenchmarkTCB`` runs the shapes of the Tensor Contraction Benchmark
(`TCCG <https://github.com/HPAC/tccg>`_) through the string-spec einsum, the way the paper's
head-to-head harness calls Einsums: its four groups (ccsd, ccsd_t, intensli, ao2mo), each case in
single and double precision. The case table is ``tcb_cases.csv`` beside it, copied from that
harness. Each case is first checked at small extents against ``reference_einsum``; then it is
timed with the harness's protocol (trash the cache, call once, keep the minimum of three) and
stored as ``t_einsum``, with the vendor GEMM of the equivalent M, N, K as ``t_gemm``, under the
label ``tcb <group> <case> <s|d>``. GFLOP/s, the percentage of GEMM, the dispatch route and
M, N, K are annotations.

It runs on one thread by default, as the harness reports its baseline. Environment variables
adjust it, since ``bench run`` passes no arguments: ``EINSUMS_TCB_ONLY`` (a group, or part of a
case name), ``EINSUMS_TCB_PRECISION`` (``s``, ``d`` or ``sd``), ``EINSUMS_TCB_REPS``,
``EINSUMS_TCB_TRASH_MB`` and ``EINSUMS_TCB_THREADS`` (a count other than 1 is added to each label,
so serial and threaded results are never compared). Pin it to a memory-attached core, which on a
Threadripper WX is not every core:

.. code-block:: bash

    taskset -c 8 einsums bench run --targets BenchmarkTCB
    EINSUMS_TCB_ONLY=intensli einsums bench run --targets BenchmarkTCB

How results are collected
^^^^^^^^^^^^^^^^^^^^^^^^^

Each test reports through ``performance::publish_benchmark_result``, which sends a structured
``benchmark_result`` event to the profiler server and prints one line,
``[label N=n] Time: ... metric: name``. ``bench run`` starts each binary with its own server on a
free port, so a viewer on the default port is undisturbed, and stores the events. In a build
without the profiler there is no server, and the printed lines are read instead; both paths give
the same label and metric. The ``source`` column records which path each result came from
(``profiler`` or ``stdout``; rows from before the column existed are ``stdout``).

Results printed before the line carried its metric are stored as ``t_generic``, which is also
what every stdout-parsed row in an older database is called. Comparisons are keyed by label and
metric, so the first runs collected through the server do not line up with that history.

The database
^^^^^^^^^^^^

Results go to a per-user database, ``benchmarks.db`` in ``~/.local/share/einsums/`` on Linux
(``$XDG_DATA_HOME/einsums/`` when that is set), ``~/Library/Application Support/einsums/`` on
macOS and ``%LOCALAPPDATA%\einsums\`` on Windows. Every checkout and worktree shares it, and each
run records the commit and branch it measured, so the history survives removing a worktree.
``EINSUMS_BENCH_DB`` names another file, and ``--db PATH`` before the command
(``bench --db PATH run``) overrides both; a CI job usually sets one of them.

Einsums Studio's benchmark bundle reads ``<checkout>/.einsums-studio/benchmarks.db``; to keep
feeding it, point ``EINSUMS_BENCH_DB`` there.

The schema has three tables: ``runs`` (one row per execution, with the commit, branch, host,
compiler, BLAS vendor, CPU and its frequency and power source), ``results`` (one row per
benchmark and metric, in microseconds, with its collection ``source``) and ``baselines`` (named
runs). Migrations are numbered SQL files in ``einsums/cli/bench/migrations/``; to change the
schema, add the next file and bump ``SCHEMA_VERSION`` in ``einsums/cli/bench/models.py``. A
database is migrated when it is opened.

Commands
^^^^^^^^

Every command that prints a table also takes ``--json``. Where a command takes ``--metric`` and
it is left out, the benchmark's only recorded metric is used; if it has several, the command
lists them and asks.

``run``
    Runs the ``--suite`` (default ``contraction``). Builds ``Tests.Performance`` (skip with
    ``--no-build``) with ``--jobs`` (default:
    ``$CMAKE_BUILD_PARALLEL_LEVEL``, else half the cores and at most 16, rather than Ninja's every
    core), runs the suite's performance test binaries (``--targets`` names binaries instead), and
    stores the results. Each binary is cut off at its ctest ``TIMEOUT``, 600 s unless it sets one. ``--reps N`` repeats the run, building once; ``--notes`` is stored with it.
    ``--build-dir`` defaults to ``build`` and ``--source-dir`` to the checkout.

``compare``
    Classifies each benchmark of a run (default: the latest) as a regression, improvement,
    unchanged, or noisy, by a robust z-score against the median and MAD of the last
    ``--baseline-count`` runs on ``--baseline`` (default ``main``), or against one run with
    ``--baseline-run-id``. ``--z-threshold`` (3) and ``--min-pct`` (5) set what is flagged; it
    exits 1 when anything regressed.

``show``, ``diff``
    One run's results (``--filter`` on labels, ``--metric`` to pick columns), or two runs side by
    side with the speedup (``--before``, ``--after``, ``--prefix``; every metric unless
    ``--metric`` names one).

``trend``, ``scaling``
    One benchmark over successive runs, with a sparkline (``--benchmark "blas-gemm N=1024"``), or
    one run's timings against N for every label starting with ``--benchmark``.

``list-runs``, ``list-tests``, ``tag-baseline``
    Browse the history, list the performance binaries in the build tree, and name a run.

``export``
    A run as JSON on stdout, or with ``--html FILE`` as a self-contained HTML report that
    includes its comparison against ``--baseline``.

``bisect``
    One step of ``git bisect run``: build (``--jobs`` as for ``run``), run the binaries until one
    reports ``--benchmark``, and exit 0 if its metric stays under ``--threshold-us``, 1 if not
    (a bad result is confirmed by the median of ``--reps`` runs), and 125 (skip) if the build
    fails or the benchmark or metric is missing.

    .. code-block:: bash

        git bisect start HEAD v2.0
        git bisect run einsums bench bisect --benchmark "blas-gemm N=1024" --threshold-us 500

    The step runs the tool from the build tree, and a reconfigure at a bisected commit replaces it
    with that commit's version. For a range that changes the tool itself, copy ``build/lib``
    before starting and run the step as ``env PYTHONPATH=<copy> python -m einsums bench bisect``.

``delete-db``
    Remove the database file (asks first unless ``--yes``).
