//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstddef>

namespace einsums::task_pool {

/// @brief Per-worker thread context.
///
/// Passed to tasks that need to know which worker they're running on.
/// When MPI support lands, the rank field will be populated from MPI_Comm_rank.
struct WorkerContext {
    int   worker_id{-1};      ///< Index of this worker (0..N-1). -1 for the submitting thread.
    void *user_data{nullptr}; ///< Opaque user data slot for thread-local scratch buffers.
    int   rank{0};            ///< MPI rank (0 for now; populated when MPI support lands).
};

} // namespace einsums::task_pool
