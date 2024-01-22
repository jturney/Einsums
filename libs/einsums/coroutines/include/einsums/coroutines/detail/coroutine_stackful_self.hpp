//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/coroutines/detail/coroutine_impl.hpp>
#include <einsums/coroutines/detail/coroutine_self.hpp>
#include <einsums/coroutines/thread_enums.hpp>
#include <einsums/coroutines/thread_id_type.hpp>

#include <cstddef>
#include <exception>
#include <limits>
#include <utility>

namespace einsums::threads::coroutines::detail {

class coroutine_stackful_self : public coroutine_self {
  public:
    explicit coroutine_stackful_self(impl_type *pimpl, coroutine_self *next_self = nullptr)
        : coroutine_self(next_self), _pimpl(pimpl) {}

    auto yield_impl(result_type arg) -> arg_type override {
        EINSUMS_ASSERT(_pimpl);

        this->_pimpl->bind_result(arg);

        {
            reset_self_on_exit on_exit(this);
            this->_pimpl->yield();
        }

        return *_pimpl->args();
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

    auto get_available_stack_space() -> std::ptrdiff_t override {
#if defined(EINSUMS_HAVE_THREADS_GET_STACK_POINTER)
        return _pimpl->get_available_stack_space();
#else
        return (std::numeric_limits<std::ptrdiff_t>::max)();
#endif
    }

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
    auto            get_impl() -> coroutine_impl            *override { return _pimpl; }
    coroutine_impl *_pimpl;
};
} // namespace einsums::threads::coroutines::detail
