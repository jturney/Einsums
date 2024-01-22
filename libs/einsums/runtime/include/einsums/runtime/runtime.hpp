//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/runtime/startup_function.hpp>
#include <einsums/runtime/state.hpp>

#include <atomic>
#include <list>

// clang-format off
#include <einsums/config/warnings_prefix.hpp>
// clang-format on

namespace einsums {
namespace detail {
extern std::list<startup_function_type> global_pre_startup_functions;
extern std::list<startup_function_type> global_startup_functions;
} // namespace detail

struct EINSUMS_EXPORT runtime {

    runtime();

    [[nodiscard]] auto get_state() const -> runtime_state;
    void               set_state(runtime_state s);

  protected:
    void init();

    std::atomic<runtime_state> _state;
};

} // namespace einsums

#include <einsums/config/warnings_suffix.hpp>
