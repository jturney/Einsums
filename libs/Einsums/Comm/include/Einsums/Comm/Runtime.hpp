//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Comm/InitModule.hpp>
#include <Einsums/Comm/Platform.hpp>
#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN(comm)

/// Initialize the communication layer: MPI_Init_thread(MPI_THREAD_FUNNELED), or nothing on the
/// mock backend. Called by the module's startup hook.
EINSUMS_EXPORT void initialize(int *argc = nullptr, char ***argv = nullptr);

/// Finalize the communication layer (MPI_Finalize). Called by the module's shutdown hook.
EINSUMS_EXPORT void finalize();

/// Check if the communication layer has been initialized.
[[nodiscard]] EINSUMS_EXPORT bool is_initialized();

/// Rank of this process in the world communicator.
/// Mock: always returns 0.
[[nodiscard]] EINSUMS_EXPORT int world_rank();

/// Total number of processes in the world communicator.
/// Mock: always returns 1.
[[nodiscard]] EINSUMS_EXPORT int world_size();

/// True if this is rank 0 in the world communicator.
[[nodiscard]] EINSUMS_EXPORT bool is_root();

EINSUMS_NAMESPACE_END(comm)
