//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/runtime/StartupFunction.hpp>
#include <einsums/runtime/State.hpp>

#include <atomic>
#include <list>

// clang-format off
#include <einsums/config/warnings_prefix.hpp>
// clang-format on

namespace einsums {
namespace detail {
extern std::list<StartupFunctionType> global_pre_startup_functions;
extern std::list<StartupFunctionType> global_startup_functions;
} // namespace detail

struct EINSUMS_EXPORT runtime {

    runtime();

    [[nodiscard]] auto get_state() const -> RuntimeState;
    void               set_state(RuntimeState s);

  protected:
    void init();

    std::atomic<RuntimeState> _state;
};

} // namespace einsums

#include <einsums/config/warnings_suffix.hpp>
