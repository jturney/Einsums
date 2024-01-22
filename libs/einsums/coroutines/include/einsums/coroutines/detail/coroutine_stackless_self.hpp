//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/coroutines/detail/coroutine_self.hpp>
#include <einsums/coroutines/thread_enums.hpp>
#include <einsums/coroutines/thread_id_type.hpp>
#include <einsums/functional/function.hpp>

#include <cstddef>
#include <exception>
#include <limits>
#include <utility>

namespace einsums::threads::coroutines::detail {
class stackless_coroutine;

class coroutine_stackless_self : public coroutine_self {
  public:
    explicit coroutine_stackless_self(stackless_coroutine *pimpl) : coroutine_self(nullptr), _pimpl(pimpl) {}

    auto yield_impl(result_type) -> arg_type override {
        // stackless coroutines don't support suspension
        EINSUMS_ASSERT(false);
        return threads::detail::thread_restart_state::abort;
    }

    auto get_thread_id() const -> thread_id_type override {
        EINSUMS_ASSERT(_pimpl);
        return _pimpl->get_thread_id();
    }

    auto get_thread_phase() const -> std::size_t override {
#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
        EINSUMS_ASSERT(_pimpl);
        return _pimpl->get_thread_phase();
#else
        return 0;
#endif
    }

    auto get_available_stack_space() -> std::ptrdiff_t override { return (std::numeric_limits<std::ptrdiff_t>::max)(); }

    auto get_thread_data() const -> std::size_t override {
        EINSUMS_ASSERT(_pimpl);
        return _pimpl->get_thread_data();
    }
    auto set_thread_data(std::size_t data) -> std::size_t override {
        EINSUMS_ASSERT(_pimpl);
        return _pimpl->set_thread_data(data);
    }

    auto get_thread_tss_data() -> tss_storage * override {
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
        EINSUMS_ASSERT(_pimpl);
        return _pimpl->get_thread_tss_data(false);
#else
        return nullptr;
#endif
    }

    auto get_or_create_thread_tss_data() -> tss_storage * override {
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
        EINSUMS_ASSERT(_pimpl);
        return _pimpl->get_thread_tss_data(true);
#else
        return nullptr;
#endif
    }

    auto get_continuation_recursion_count() -> std::size_t & override {
        EINSUMS_ASSERT(_pimpl);
        return _pimpl->get_continuation_recursion_count();
    }

  private:
    stackless_coroutine *_pimpl;
};
} // namespace einsums::threads::coroutines::detail
