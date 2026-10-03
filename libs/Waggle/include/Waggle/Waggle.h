/*----------------------------------------------------------------------------------------------
 * Copyright (c) The Einsums Developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for license information.
 *----------------------------------------------------------------------------------------------*/

/*
 * Waggle's C interface: everything a library or program calls the profiler through.
 *
 * Plain exported functions, so a library built against one minor version of Waggle runs against any
 * later one: adding functions raises the minor version and changes nothing that exists. An
 * incompatible change raises the major version, which is in the shared library's file name.
 *
 * Strings are passed with their lengths and need not be terminated, except where a parameter says
 * otherwise. Nothing allocated on one side of this interface is freed on the other: a handler
 * answers by handing its bytes to waggle_reply_set, which copies them.
 *
 * C++ code uses <Waggle/Waggle.hpp>, which wraps this.
 */

#ifndef WAGGLE_WAGGLE_H
#define WAGGLE_WAGGLE_H

#include <stddef.h> /* NOLINT(modernize-deprecated-headers): a C header */
#include <stdint.h> /* NOLINT(modernize-deprecated-headers) */

#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(WAGGLE_EXPORTS)
#        define WAGGLE_C_EXPORT __declspec(dllexport)
#    else
#        define WAGGLE_C_EXPORT __declspec(dllimport)
#    endif
#else
#    define WAGGLE_C_EXPORT __attribute__((visibility("default")))
#endif

/* The version of this interface the header describes. */
#define WAGGLE_ABI_MAJOR 0
#define WAGGLE_ABI_MINOR 1

#ifdef __cplusplus
extern "C" {
#endif

/* The version of the interface the loaded library provides. */
WAGGLE_C_EXPORT void waggle_abi_version(uint32_t *major, uint32_t *minor);

/* ---- Names ------------------------------------------------------------------------------------
 * Every id is the collector's, stable for the life of the process. Id 0 is the empty string, no
 * site, and the unnamed domain. */

/* The id of a string; the same characters always give the same id. */
WAGGLE_C_EXPORT uint32_t waggle_intern(char const *s, size_t length);

/* The id of the library named `name`, registered if new. */
WAGGLE_C_EXPORT uint32_t waggle_register_domain(char const *name, size_t length);

/* The id of a call site; the same description always gives the same id. `file` and `func` are
 * terminated. */
WAGGLE_C_EXPORT uint32_t waggle_register_site(char const *name, size_t name_length, char const *file, int line, char const *func,
                                              uint32_t domain);

/* ---- Recording ---------------------------------------------------------------------------- */

/* Whether zones and annotations are recorded now. */
WAGGLE_C_EXPORT int  waggle_enabled(void);
WAGGLE_C_EXPORT void waggle_set_enabled(int on);

/* The recording switch itself, 1 or 0, so a caller can skip a call into the collector while
 * recording is off. Read it only with a relaxed atomic load (C11 atomic_load_explicit, C++
 * std::atomic_ref); never write it. The address is fixed for the life of the process. */
WAGGLE_C_EXPORT int32_t const *waggle_enabled_flag(void);

/* Open a zone at `site`, named `name_id`, or by its site when `name_id` is 0. Returns 1 when it
 * opened one and 0 when recording is off; call waggle_zone_end exactly when it returned 1, so a
 * switch of recording while the zone is open cannot unbalance the pairs. */
WAGGLE_C_EXPORT int waggle_zone_begin(uint32_t site, uint32_t name_id);
/* Close the zone the calling thread opened last. */
WAGGLE_C_EXPORT void waggle_zone_end(void);

/* Annotate the calling thread's innermost zone. */
WAGGLE_C_EXPORT void waggle_annotate_str(uint32_t key_id, uint32_t value_id);
WAGGLE_C_EXPORT void waggle_annotate_i64(uint32_t key_id, int64_t value);
WAGGLE_C_EXPORT void waggle_annotate_f64(uint32_t key_id, double value);

/* Record an allocation or a free in the innermost zone; `address` may be null. A free gives its
 * size: freed on another thread, it may be processed before its allocation. */
WAGGLE_C_EXPORT void waggle_mem_alloc(void const *address, int64_t bytes);
WAGGLE_C_EXPORT void waggle_mem_free(void const *address, int64_t bytes);

/* Name the calling thread in reports and viewers. */
WAGGLE_C_EXPORT void waggle_set_thread_name(char const *name, size_t length);
/* The calling thread's id, as reports and viewers show it. */
WAGGLE_C_EXPORT uint32_t waggle_current_thread_id(void);

/* ---- Settings and lifecycle -------------------------------------------------------------- */

/* Set `count` settings by name, each from its value written as on a command line ("true", "19216",
 * a path), all as one change: the server starts on the port set beside it. Returns how many were
 * refused because another library already set them to something else (each refusal is also
 * reported as a diagnostic), or -1, changing nothing, for an unknown name or a bad value. Every
 * key and value is terminated. */
WAGGLE_C_EXPORT int waggle_config_set(char const *const *keys, char const *const *values, size_t count);

/* As waggle_config_set, whoever set the settings before. For tests and tools that put a value
 * back; returns 0, or -1 as waggle_config_set does. */
WAGGLE_C_EXPORT int waggle_config_override(char const *const *keys, char const *const *values, size_t count);

/* Write the setting's current value, terminated, into `buffer` when it fits, and return its
 * length without the terminator; -1 for an unknown name. */
WAGGLE_C_EXPORT int64_t waggle_config_get(char const *key, char *buffer, size_t size);

/* Count a library as using the profiler until its waggle_finalize. Every argument is
 * terminated; any may be empty. */
WAGGLE_C_EXPORT void waggle_init(char const *name, char const *version, char const *git_commit, char const *git_branch, int git_dirty,
                                 char const *build_type);

/* Release the waggle_init of the library named `name` (terminated). The last release writes the
 * session file and report the settings ask for and stops recording for the process. */
WAGGLE_C_EXPORT void waggle_finalize(char const *name);

/* Drain every thread's recorded events into the aggregated trees. */
WAGGLE_C_EXPORT void waggle_flush(void);

/* Hold the calling thread until a viewer connects, if the settings ask for that. */
WAGGLE_C_EXPORT void waggle_wait_for_viewer(void);

/* ---- The live server --------------------------------------------------------------------- */

/* Start the server on `port`, or the first free port after it, unless one runs. */
WAGGLE_C_EXPORT void     waggle_server_start(uint16_t port);
WAGGLE_C_EXPORT int      waggle_server_running(void);
WAGGLE_C_EXPORT uint16_t waggle_server_port(void);
WAGGLE_C_EXPORT int      waggle_viewer_connected(void);

/* What a handler writes its answer into. */
typedef struct waggle_reply waggle_reply; /* NOLINT(modernize-use-using): a C header */

/* Answer with `data` (JSON), copied. */
WAGGLE_C_EXPORT void waggle_reply_set(waggle_reply *reply, char const *data, size_t length);

/* A request handler: receives the request's params (a JSON object) and answers through `reply`.
 * Called from the collector's thread. */
typedef void (*waggle_handler_fn)(void *user, char const *params, size_t length, waggle_reply *reply); /* NOLINT(modernize-use-using) */

/* Called once on `user` when the collector no longer needs it; may be null. */
typedef void (*waggle_release_fn)(void *user); /* NOLINT(modernize-use-using) */

/* Register a handler for requests named `method` (terminated), replacing any earlier one. */
WAGGLE_C_EXPORT void waggle_register_handler(char const *method, waggle_handler_fn handler, void *user, waggle_release_fn release);

/* Remove the handler for `method`, once any call of it in progress returns. */
WAGGLE_C_EXPORT void waggle_unregister_handler(char const *method);

/* Embed what `section` answers (with empty params) under `key` in every session file. */
WAGGLE_C_EXPORT void waggle_register_session_section(char const *key, waggle_handler_fn section, void *user, waggle_release_fn release);

/* Send `json` (an object) to every connected viewer as a message of type `type` (terminated). */
WAGGLE_C_EXPORT void waggle_publish(char const *type, char const *json, size_t length);

/* Stream a log message (level 0 trace to 5 critical) to connected viewers. `unix_ns` is when it
 * was logged; `file` and `func` are terminated and may be empty. */
WAGGLE_C_EXPORT void waggle_log(int level, int64_t unix_ns, char const *file, int line, char const *func, char const *message,
                                size_t length);

/* Stream a line the program printed to connected viewers. */
WAGGLE_C_EXPORT void waggle_output(char const *message, size_t length);

/* ---- Diagnostics ------------------------------------------------------------------------- */

/* Receives the profiler's own messages: level 0 debug, 1 info, 2 warning, 3 error. */
typedef void (*waggle_diagnostic_fn)(void *user, int level, char const *message, size_t length); /* NOLINT(modernize-use-using) */

/* Send the profiler's messages to `handler` instead of stderr; null restores stderr. */
WAGGLE_C_EXPORT void waggle_set_diagnostic_handler(waggle_diagnostic_fn handler, void *user, waggle_release_fn release);

/* ---- Reports ----------------------------------------------------------------------------- */

/* Print the text report to standard output. */
WAGGLE_C_EXPORT void waggle_print_report(int detailed);

/* Write the aggregated trees as JSON to `path` (terminated); 0 on success. */
WAGGLE_C_EXPORT int waggle_export_json(char const *path);

/* Zones opened and closed so far, on every thread, whether or not their events were dropped. */
WAGGLE_C_EXPORT uint64_t waggle_total_push_count(void);
WAGGLE_C_EXPORT uint64_t waggle_total_pop_count(void);

/* What one recorded push or pop costs, in nanoseconds, measured once. */
WAGGLE_C_EXPORT double waggle_push_overhead_ns(void);
WAGGLE_C_EXPORT double waggle_pop_overhead_ns(void);

/* Receives one key and value. */
typedef void (*waggle_pair_fn)(void *user, char const *key, size_t key_length, char const *value, size_t value_length); /* NOLINT(modernize-use-using) */

/* Call `fn` with each string annotation on the calling thread's open zones, outermost first,
 * after a flush. */
WAGGLE_C_EXPORT void waggle_open_zone_annotations(waggle_pair_fn fn, void *user);

/* ---- Snapshots and reset ----------------------------------------------------------------- */

/* A copy of the aggregated trees, owned by the caller until waggle_snapshot_release. Recording
 * continues while it is read. */
typedef struct waggle_snapshot waggle_snapshot; /* NOLINT(modernize-use-using) */

/* One zone in a snapshot, valid until its snapshot is released. */
typedef struct waggle_node waggle_node; /* NOLINT(modernize-use-using) */

/* Merge every thread's tree into one, zones matched by their name path as within a thread. */
#define WAGGLE_SNAPSHOT_MERGE_THREADS 1u

/* Flush, then copy the trees. `flags` is 0 or WAGGLE_SNAPSHOT_MERGE_THREADS. */
WAGGLE_C_EXPORT waggle_snapshot *waggle_snapshot_take(uint32_t flags);
WAGGLE_C_EXPORT void             waggle_snapshot_release(waggle_snapshot *snapshot);

/* The snapshot's threads, by index: one, with id 0 and no name, in a merged snapshot. */
WAGGLE_C_EXPORT size_t   waggle_snapshot_thread_count(waggle_snapshot const *snapshot);
WAGGLE_C_EXPORT uint32_t waggle_snapshot_thread_id(waggle_snapshot const *snapshot, size_t thread);
WAGGLE_C_EXPORT char const *waggle_snapshot_thread_name(waggle_snapshot const *snapshot, size_t thread, size_t *length);

/* The unnamed node above a thread's outermost zones; null for an index out of range. */
WAGGLE_C_EXPORT waggle_node const *waggle_snapshot_root(waggle_snapshot const *snapshot, size_t thread);

/* The zone at `path` below a thread's root: names joined by '/', so "solve/iterate/gemm". Null
 * when there is none. A name that itself contains '/' cannot be found this way; walk the
 * children instead. */
WAGGLE_C_EXPORT waggle_node const *waggle_snapshot_find(waggle_snapshot const *snapshot, size_t thread, char const *path, size_t length);

/* A node's children, in the order they were first entered. */
WAGGLE_C_EXPORT size_t             waggle_node_child_count(waggle_node const *node);
WAGGLE_C_EXPORT waggle_node const *waggle_node_child(waggle_node const *node, size_t index);

/* Where a node was entered: its name, and the file, line and function of its call site. The
 * strings are terminated and live as long as the snapshot. */
WAGGLE_C_EXPORT char const *waggle_node_name(waggle_node const *node, size_t *length);
WAGGLE_C_EXPORT char const *waggle_node_file(waggle_node const *node, size_t *length);
WAGGLE_C_EXPORT int         waggle_node_line(waggle_node const *node);
WAGGLE_C_EXPORT char const *waggle_node_function(waggle_node const *node, size_t *length);

/* The library the node's site belongs to (its domain); empty for the unnamed domain. */
WAGGLE_C_EXPORT char const *waggle_node_domain(waggle_node const *node, size_t *length);

/* Number of log2 histogram buckets: bucket i counts calls whose exclusive time was in
 * [2^i, 2^(i+1)) microseconds, the first also holding everything shorter. */
#define WAGGLE_HISTOGRAM_BUCKETS 21

/* A node's statistics. Times are nanoseconds, exclusive of the node's children unless named
 * inclusive. Fields are only ever appended: set `size` to sizeof(waggle_node_stats) and the
 * collector fills no more than that. */
typedef struct waggle_node_stats { /* NOLINT(modernize-use-using,readability-identifier-naming): a C type */
    uint32_t size;
    uint64_t call_count;
    int64_t  exclusive_ns;
    int64_t  inclusive_ns;
    int64_t  exclusive_min_ns; /* 0 when never called */
    int64_t  exclusive_max_ns;
    double   exclusive_mean_ns;
    double   exclusive_m2; /* sum of squared deviations from the mean: variance = m2 / count */
    uint64_t mem_alloc_count;
    uint64_t mem_free_count;
    int64_t  mem_alloc_bytes;
    int64_t  mem_free_bytes;
    int64_t  mem_peak_bytes;
    uint64_t histogram[WAGGLE_HISTOGRAM_BUCKETS];
} waggle_node_stats;

/* Fill `stats` up to stats->size bytes. */
WAGGLE_C_EXPORT void waggle_node_stats_get(waggle_node const *node, waggle_node_stats *stats);

/* Call `fn` with each of a node's annotations as text, in the order they were first made: each
 * key's latest value, numeric ones included. */
WAGGLE_C_EXPORT void waggle_node_annotations(waggle_node const *node, waggle_pair_fn fn, void *user);

/* Receives one numeric annotation: its key, and the total, smallest, largest and count of the
 * values given it. */
typedef void (*waggle_numeric_fn)(void *user, char const *key, size_t key_length, double total, double min, double max, /* NOLINT(modernize-use-using) */
                                  uint64_t count);

/* Call `fn` with each of a node's numeric annotations, in the order they were first made. */
WAGGLE_C_EXPORT void waggle_node_numeric_annotations(waggle_node const *node, waggle_numeric_fn fn, void *user);

/* Flush, then clear every statistic, annotation and timeline entry and drop every zone not open
 * on some thread. A zone open across the reset keeps its node and is timed from the reset. Ids
 * are kept. Process-wide: it clears every library's data. */
WAGGLE_C_EXPORT void waggle_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* WAGGLE_WAGGLE_H */
