//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/defines.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
// On Windows, make sure winsock.h is not included even if windows.h is
// included before winsock2.h
#    define _WINSOCKAPI_
#    include <winsock2.h>
#endif

#include <einsums/config/Attributes.hpp>
#include <einsums/config/BranchHints.hpp>
#include <einsums/config/CompilerSpecific.hpp>
#include <einsums/config/Constexpr.hpp>
#include <einsums/config/Debug.hpp>
#include <einsums/config/EmulateDeleted.hpp>
#include <einsums/config/ForceInline.hpp>
#include <einsums/config/Forward.hpp>
#include <einsums/config/ManualProfiling.hpp>
#include <einsums/config/ModulesEnabled.hpp>
#include <einsums/config/Move.hpp>
#include <einsums/config/ThreadsStack.hpp>
#include <einsums/config/compiler_fence.hpp>
#include <einsums/config/export_definitions.hpp>
#include <einsums/config/version.hpp>
#include <einsums/preprocessor/cat.hpp>
#include <einsums/preprocessor/stringize.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
// On Windows, make sure winsock.h is not included even if windows.h is
// included before winsock2.h
#    define _WINSOCKAPI_
#endif

///////////////////////////////////////////////////////////////////////////////
/// By default, enable minimal thread deadlock detection in debug builds only.
#if !defined(EINSUMS_SPINLOCK_DEADLOCK_DETECTION_LIMIT)
#    define EINSUMS_SPINLOCK_DEADLOCK_DETECTION_LIMIT 10000000
#endif

/// Print a warning about potential deadlocks after this many iterations.
#if !defined(EINSUMS_SPINLOCK_DEADLOCK_WARNING_LIMIT)
#    define EINSUMS_SPINLOCK_DEADLOCK_WARNING_LIMIT 1000000
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the default number of coroutine heaps.
#if !defined(EINSUMS_COROUTINE_NUM_HEAPS)
#    define EINSUMS_COROUTINE_NUM_HEAPS 7
#endif

///////////////////////////////////////////////////////////////////////////////
/// By default we do not maintain stack back-traces on suspension. This is a
/// pure debugging aid to be able to see in the debugger where a suspended
/// thread got stuck.
#if defined(EINSUMS_HAVE_THREAD_BACKTRACE_ON_SUSPENSION) && !defined(EINSUMS_HAVE_STACKTRACES)
#    error EINSUMS_HAVE_THREAD_BACKTRACE_ON_SUSPENSION requires EINSUMS_HAVE_STACKTRACES to be defined!
#endif

/// By default we capture only 20 levels of stack back trace on suspension
#if !defined(EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH)
#    define EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH 20
#endif

///////////////////////////////////////////////////////////////////////////////
//  Characters used
//    - to delimit several einsums ini paths
//    - used as file extensions for shared libraries
//    - used as path delimiters
#ifdef EINSUMS_WINDOWS // windows
#    define EINSUMS_INI_PATH_DELIMITER   ";"
#    define EINSUMS_SHARED_LIB_EXTENSION ".dll"
#    define EINSUMS_EXECUTABLE_EXTENSION ".exe"
#    define EINSUMS_PATH_DELIMITERS      "\\/"
#else // unix like
#    define EINSUMS_INI_PATH_DELIMITER ":"
#    define EINSUMS_PATH_DELIMITERS    "/"
#    ifdef __APPLE__ // apple
#        define EINSUMS_SHARED_LIB_EXTENSION ".dylib"
#    elif defined(EINSUMS_HAVE_STATIC_LINKING)
#        define EINSUMS_SHARED_LIB_EXTENSION ".a"
#    else // linux & co
#        define EINSUMS_SHARED_LIB_EXTENSION ".so"
#    endif
#    define EINSUMS_EXECUTABLE_EXTENSION ""
#endif

///////////////////////////////////////////////////////////////////////////////
// Count number of empty (no einsums thread available) thread manager loop executions
#if !defined(EINSUMS_IDLE_LOOP_COUNT_MAX)
#    define EINSUMS_IDLE_LOOP_COUNT_MAX 200000
#endif

///////////////////////////////////////////////////////////////////////////////
// Count number of busy thread manager loop executions before forcefully
// cleaning up terminated thread objects
#if !defined(EINSUMS_BUSY_LOOP_COUNT_MAX)
#    define EINSUMS_BUSY_LOOP_COUNT_MAX 2000
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum number of threads to create in the thread queue, except when there is
// no work to do, in which case the count will be increased in steps of
// EINSUMS_THREAD_QUEUE_MIN_ADD_NEW_COUNT.
#if !defined(EINSUMS_THREAD_QUEUE_MAX_THREAD_COUNT)
#    define EINSUMS_THREAD_QUEUE_MAX_THREAD_COUNT 1000
#endif

///////////////////////////////////////////////////////////////////////////////
// Minimum number of pending tasks required to steal tasks.
#if !defined(EINSUMS_THREAD_QUEUE_MIN_TASKS_TO_STEAL_PENDING)
#    define EINSUMS_THREAD_QUEUE_MIN_TASKS_TO_STEAL_PENDING 0
#endif

///////////////////////////////////////////////////////////////////////////////
// Minimum number of staged tasks required to steal tasks.
#if !defined(EINSUMS_THREAD_QUEUE_MIN_TASKS_TO_STEAL_STAGED)
#    define EINSUMS_THREAD_QUEUE_MIN_TASKS_TO_STEAL_STAGED 0
#endif

///////////////////////////////////////////////////////////////////////////////
// Minimum number of staged tasks to add to work items queue.
#if !defined(EINSUMS_THREAD_QUEUE_MIN_ADD_NEW_COUNT)
#    define EINSUMS_THREAD_QUEUE_MIN_ADD_NEW_COUNT 10
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum number of staged tasks to add to work items queue.
#if !defined(EINSUMS_THREAD_QUEUE_MAX_ADD_NEW_COUNT)
#    define EINSUMS_THREAD_QUEUE_MAX_ADD_NEW_COUNT 10
#endif

///////////////////////////////////////////////////////////////////////////////
// Minimum number of terminated threads to delete in one go.
#if !defined(EINSUMS_THREAD_QUEUE_MIN_DELETE_COUNT)
#    define EINSUMS_THREAD_QUEUE_MIN_DELETE_COUNT 10
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum number of terminated threads to delete in one go.
#if !defined(EINSUMS_THREAD_QUEUE_MAX_DELETE_COUNT)
#    define EINSUMS_THREAD_QUEUE_MAX_DELETE_COUNT 1000
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum number of terminated threads to keep before cleaning them up.
#if !defined(EINSUMS_THREAD_QUEUE_MAX_TERMINATED_THREADS)
#    define EINSUMS_THREAD_QUEUE_MAX_TERMINATED_THREADS 100
#endif

///////////////////////////////////////////////////////////////////////////////
// Number of threads (of the default stack size) to pre-allocate when
// initializing a thread queue.
#if !defined(EINSUMS_THREAD_QUEUE_INIT_THREADS_COUNT)
#    define EINSUMS_THREAD_QUEUE_INIT_THREADS_COUNT 10
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum sleep time for idle backoff in milliseconds (used only if
// EINSUMS_HAVE_THREAD_MANAGER_IDLE_BACKOFF is defined).
#if !defined(EINSUMS_IDLE_BACKOFF_TIME_MAX)
#    define EINSUMS_IDLE_BACKOFF_TIME_MAX 1000
#endif

///////////////////////////////////////////////////////////////////////////////
// This limits how deep the internal recursion of future continuations will go
// before a new operation is re-spawned.
#if !defined(EINSUMS_CONTINUATION_MAX_RECURSION_DEPTH)
#    if defined(__has_feature)
#        if __has_feature(address_sanitizer)
// if we build under AddressSanitizer we set the max recursion depth to 1 to not
// run into stack overflows.
#            define EINSUMS_CONTINUATION_MAX_RECURSION_DEPTH 1
#        endif
#    endif
#endif

#if !defined(EINSUMS_CONTINUATION_MAX_RECURSION_DEPTH)
#    if defined(EINSUMS_DEBUG)
#        define EINSUMS_CONTINUATION_MAX_RECURSION_DEPTH 14
#    else
#        define EINSUMS_CONTINUATION_MAX_RECURSION_DEPTH 20
#    endif
#endif

///////////////////////////////////////////////////////////////////////////////
// Make sure we have support for more than 64 threads for Xeon Phi
#if defined(__MIC__) && !defined(EINSUMS_HAVE_MORE_THAN_64_THREADS)
#    define EINSUMS_HAVE_MORE_THAN_64_THREADS
#endif
#if defined(__MIC__) && !defined(EINSUMS_HAVE_MAX_CPU_COUNT)
#    define EINSUMS_HAVE_MAX_CPU_COUNT 256
#endif
