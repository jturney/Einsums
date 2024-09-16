//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

// clang-format off
#include <einsums/coroutines/detail/context_impl.hpp>
// clang-format on

#include <einsums/assert.hpp>
#include <einsums/coroutines/detail/swap_context.hpp>
#include <einsums/coroutines/detail/tss.hpp>
#include <einsums/coroutines/thread_id_type.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <utility>

#define EINSUMS_COROUTINE_NUM_ALL_HEAPS                                                               \
(EINSUMS_COROUTINE_NUM_HEAPS + EINSUMS_COROUTINE_NUM_HEAPS / 2 + EINSUMS_COROUTINE_NUM_HEAPS / 4 +      \
EINSUMS_COROUTINE_NUM_HEAPS / 4) /**/


