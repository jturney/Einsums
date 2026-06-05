//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/Logging.hpp>

#include <atomic>

#if defined(EINSUMS_HAVE_MPI)
#    include <mpi.h>
#endif

namespace einsums::comm {

namespace {
std::atomic<bool> g_initialized{false};
} // namespace

void initialize(int *argc, char ***argv) {
    if (g_initialized.exchange(true))
        return; // Already initialized

#if defined(EINSUMS_HAVE_MPI)
    int provided = 0;
    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        EINSUMS_LOG_WARN("MPI_Init_thread: requested THREAD_FUNNELED but got {}", provided);
    }
    EINSUMS_LOG_INFO("Comm: MPI initialized (rank {}/{}, thread support={})", world_rank(), world_size(), provided);
#else
    (void)argc;
    (void)argv;
    EINSUMS_LOG_INFO("Comm: mock backend (single process, no MPI)");
#endif
}

void finalize() {
    if (!g_initialized.exchange(false))
        return;

#if defined(EINSUMS_HAVE_MPI)
    MPI_Finalize();
    EINSUMS_LOG_INFO("Comm: MPI finalized");
#endif
}

bool is_initialized() {
    return g_initialized.load();
}

int world_rank() {
#if defined(EINSUMS_HAVE_MPI)
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
#else
    return 0;
#endif
}

int world_size() {
#if defined(EINSUMS_HAVE_MPI)
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
#else
    return 1;
#endif
}

bool is_root() {
    return world_rank() == 0;
}

} // namespace einsums::comm
