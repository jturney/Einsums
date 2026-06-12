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

import os as _os
from enum import Enum


def _bool_env(name: str) -> bool | None:
    """Read a boolean override from the environment.

    Returns ``True`` or ``False`` for recognized truthy or falsy strings,
    and ``None`` when the variable is unset. This lets pytest harnesses and
    any other launcher flip the corresponding rc field without monkey-patching
    this module before ``import einsums``. In particular the test launcher
    in cmake/Einsums_AddTest.cmake sets EINSUMS_DEBUG_NO_INSTALL_SIGNAL_HANDLERS
    and EINSUMS_DEBUG_NO_ATTACH_DEBUGGER so a failing pytest doesn't hang on
    the runtime's "waiting for debugger" prompt.
    """
    v = _os.environ.get(name)
    if v is None:
        return None
    return v.strip().lower() in ("1", "true", "yes", "on")

# Buffer Allocator:
#   --einsums:buffer-size <value>                        Total size of buffers allocated for tensor contractions
#   --einsums:work-buffer-size <value>                   The largest buffer size to use for buffered contractions. Should be much smaller than the max buffer size. The maximum should be the value of --einsums:buffer-size divided by three times the number of threads. In reality, the program will need more space for other buffers, so the size should be much smaller than that. Setting to zero will let the program decide.

# ComputeGraph Passes:
#   --einsums:pass:disable <PASSES>                      Comma-separated list of optimization pass names to skip (e.g. CSE,Reorder)
#   --einsums:pass:analyze                               Run all passes in analysis-only mode (report findings but don't modify the graph)
#   --einsums:pass:verbose                               Log node count and timing before/after each optimization pass

# Debug:
#   --einsums:debug:no-install-signal-handlers           Do not install signal handlers
#   --einsums:debug:no-attach-debugger                   Do not provide a mechanism to attach a debugger on detected errors
#   --einsums:debug:no-diagnostics-on-terminate          Print additional diagnostic information on termination

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
#   --einsums:profile:no-report                          Don't generate profile report
#   --einsums:profile:filename <filename>                Generate profile filename
#   --einsums:profile:no-append <N>                      Don't append to profile file
#   --einsums:profile:detailed                           Print detailed profile report
#   --einsums:profile:save <filename>                    Save profile session JSON for imgui viewer
#   --einsums:profile:port <PORT>                        Profile server port
#   --einsums:profile:wait-for-viewer                    Wait for profiler viewer to connect before running

# Tensor Options:
#   --einsums:scratch-dir <value>                        The scratch directory for Einsums tensor files.
#   --einsums:hdf5-file-name <value>                     The name of the HDF5 file for Einsums. Defaults to einsums.[pid].h5, where [pid] is the PID of the current process.
#   --einsums:no-delete-hdf5-files                       Tells Einsums not to clean up HDF5 files on exit.

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
debug_no_install_signal_handlers: bool | None = _bool_env("EINSUMS_DEBUG_NO_INSTALL_SIGNAL_HANDLERS")
debug_no_attach_debugger: bool | None = _bool_env("EINSUMS_DEBUG_NO_ATTACH_DEBUGGER")
debug_no_diagnostics_on_terminate: bool | None = _bool_env("EINSUMS_DEBUG_NO_DIAGNOSTICS_ON_TERMINATE")

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
