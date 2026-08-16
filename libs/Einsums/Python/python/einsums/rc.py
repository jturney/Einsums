#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Pre-import configuration for einsums.

Set fields here before the first compute call to influence
``einsums::initialize()``. Once the runtime is up these fields are read-only
as far as einsums is concerned, so changing them after init has no effect.

Usage::

    import einsums.rc
    einsums.rc.threads = 8
    einsums.rc.log_level = einsums.rc.LogLevel.INFO

    import einsums
    einsums.gemm(...)   # initialize() fires here using the rc settings
"""

from enum import Enum

# Every option below also has an environment variable, read by the C++ runtime
# itself rather than by this module: einsums:log:level is EINSUMS_LOG_LEVEL,
# einsums:debug:no-attach-debugger is EINSUMS_DEBUG_NO_ATTACH_DEBUGGER, and so
# on. Precedence is default < environment < the fields below, since these are
# handed to einsums::initialize() as command line arguments.
#
# So a launcher that wants to flip a setting without touching this module - the
# pytest harness in cmake/Einsums_AddTest.cmake, which disables the runtime's
# "waiting for debugger" prompt so a failing test cannot hang - just exports the
# variable. Leaving a field as ``None`` here is what lets that through.
#
# Flags are declared in the positive and the runtime generates the ``no-``
# spelling, so ``--einsums:profile:report`` turns off as
# ``--einsums:profile:no-report``. Several fields below still carry the older
# negated names (``profile_no_report``, ``debug_no_attach_debugger``); they are
# unchanged and still work, because the negated spellings all still parse.
# Setting one to ``True`` requests the negation, which is what the name says.

# Buffer Allocator:
#   --einsums:buffer-size <value>                        Total size of buffers allocated for tensor contractions
#   --einsums:work-buffer-size <value>                   The largest buffer size to use for buffered contractions. Should be much smaller than the max buffer size. The maximum should be the value of --einsums:buffer-size divided by three times the number of threads. In reality, the program will need more space for other buffers, so the size should be much smaller than that. Setting to zero will let the program decide.

# ComputeGraph Passes:
#   --einsums:pass:disable <PASSES>                      Comma-separated list of optimization pass names to skip (e.g. CSE,Reorder)
#   --einsums:pass:analyze                               Run all passes in analysis-only mode (report findings but do not modify the graph)
#   --einsums:pass:verbose                               Log node count and timing before and after each optimization pass
#   --einsums:graph:profile-groups                       Break a grouped batched GEMM into one profiler zone per shape class
#   --einsums:graph:verify-levels                        Check that no execution level holds two nodes touching overlapping storage

# Debug:
#   --einsums:debug:install-signal-handlers              Install signal handlers that report a fatal signal before aborting (default: true)
#   --einsums:debug:attach-debugger                      Provide a mechanism to attach a debugger on detected errors (default: true)
#   --einsums:debug:diagnostics-on-terminate             Print additional diagnostic information on termination (default: true)

# GPU:
#   --einsums:gpu:disable                                Keep every node on the host (GPUPlacement becomes a no-op)

# HPTT:
#   --einsums:hptt:selection-method <METHOD>             HPTT plan selection method (estimate, measure, patient, crazy)

# Help:
#   --help                                         -h    Show this help message and exit
#   --version                                            Show version and exit

# Logging:
#   --einsums:log:level <LogLevel>                       Log level
#   --einsums:log:destination <value>                    Log destination
#   --einsums:log:format <value>                         Log format

# Profile:
#   --einsums:profile:disable                            Do not record profiling zones or annotations (large speedup for small operations)
#   --einsums:profile:report                             Generate a profile report on exit (default: true)
#   --einsums:profile:filename <filename>                Profile report file name
#   --einsums:profile:append                             Append to the profile file instead of truncating it (default: true)
#   --einsums:profile:detailed                           Print a detailed profile report
#   --einsums:profile:save <filename>                    Save the profile session as JSON for the imgui viewer
#   --einsums:profile:port <PORT>                        Profile server port
#   --einsums:profile:wait-for-viewer                    Wait for the profiler viewer to connect before running

# Tensor Options:
#   --einsums:row-major                                  Construct tensors in row-major order rather than column-major
#   --einsums:scratch-dir <value>                        The scratch directory for Einsums tensor files.
#   --einsums:hdf5-file-name <value>                     The name of the HDF5 file for Einsums. Defaults to einsums.[pid].h5, where [pid] is the PID of the current process.
#   --einsums:delete-hdf5-files                          Clean up the HDF5 scratch file on exit. (default: true)

# Number of OpenMP / task-pool worker threads. ``None`` means einsums picks
# based on the host's logical core count. einsums has no CLI flag for this;
# the binding routes a non-``None`` value through the ``OMP_NUM_THREADS``
# environment variable before ``einsums::initialize`` runs.
threads: int | None = None

# Buffer Allocator
#   Sizes accept the same shorthand strings the runtime understands,
#   such as ``"1G"`` or ``"512M"``. ``None`` means einsums default.
buffer_size: str | None = None
work_buffer_size: str | None = None

# ComputeGraph Passes
pass_disable: str | None = None  # comma-separated pass names to skip
pass_analyze: bool | None = None
pass_verbose: bool | None = None

# Debug
debug_no_install_signal_handlers: bool | None = None
debug_no_attach_debugger: bool | None = None
debug_no_diagnostics_on_terminate: bool | None = None

# HPTT
hptt_selection_method: str | None = None  # "estimate" / "measure" / "patient" / "crazy"


class LogLevel(Enum):
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4


# Logging
# Log level. One of ``LogLevel.TRACE``..``LogLevel.ERROR``; ``None`` means einsums default.
log_level: LogLevel | None = None
# Log destination. ``None`` means the einsums default of ``cerr``.
log_destination: str | None = None
# Log format. ``None`` means einsums default.
log_format: str | None = None

# Profile
# Don't generate profile report. ``None`` means the einsums default of ``False``.
profile_no_report: bool | None = None
# Generate profile filename. ``None`` means the einsums default of ``"profile.txt"``.
profile_filename: str | None = None
# Don't append to profile file.
profile_no_append: bool | None = None
# Print detailed profile report.
profile_detailed: bool | None = None
# Save profile session JSON for imgui viewer.
profile_save: str | None = None
# Profile server port.
profile_port: int | None = None
# Wait for profiler viewer to connect before running.
profile_wait_for_viewer: bool | None = None

# Tensor Options
# The scratch directory for Einsums tensor files. ``None`` means einsums default.
scratch_dir: str | None = None
# The name of the HDF5 file for Einsums. ``None`` means the einsums default
# of ``einsums.[pid].h5``, where ``[pid]`` is the PID of the current process.
hdf5_file_name: str | None = None
# Tell einsums not to clean up HDF5 files on exit.
no_delete_hdf5_files: bool | None = None
