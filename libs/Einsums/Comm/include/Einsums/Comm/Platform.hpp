//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

namespace einsums::comm {

/// True when MPI support is compiled in.
inline constexpr bool has_mpi =
#if defined(EINSUMS_HAVE_MPI)
    true;
#else
    false;
#endif

/// True when NCCL (NVIDIA GPU-direct communication) is available.
inline constexpr bool has_nccl =
#if defined(EINSUMS_HAVE_NCCL)
    true;
#else
    false;
#endif

/// True when RCCL (AMD GPU-direct communication) is available.
inline constexpr bool has_rccl =
#if defined(EINSUMS_HAVE_RCCL)
    true;
#else
    false;
#endif

/// True when running without MPI (serial mock backend).
inline constexpr bool is_mock = !has_mpi;

} // namespace einsums::comm
