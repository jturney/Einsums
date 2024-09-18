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
    explicit coroutine_stackful_self(impl_type *pimpl, coroutine_self *next_self = nullptr) : coroutine_self(next_self), _pimpl(pimpl) {}

    arg_type yield_impl(result_type arg) override {
        EINSUMS_ASSERT(_pimpl);

        this->_pimpl->bind_result(arg);

        {
            reset_self_on_exit on_exit(this);
            this->_pimpl->yield();
        }

        return *_pimpl->args();
    }

    thread_id_type get_thread_id() const override {
        EINSUMS_ASSERT(_pimpl);
        return _pimpl->get_thread_id();
    }

    std::size_t get_thread_phase() const override {
#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
        EINSUMS_ASSERT(_pimpl);
        return _pimpl->get_thread_phase();
#else
        return 0;
#endif
    }

    std::ptrdiff_t get_available_stack_space() override {
#if defined(EINSUMS_HAVE_THREADS_GET_STACK_POINTER)
        return _pimpl->get_available_stack_space();
#else
        return (std::numeric_limits<std::ptrdiff_t>::max)();
#endif
    }

    std::size_t get_thread_data() const override {
        EINSUMS_ASSERT(_pimpl);
        return _pimpl->get_thread_data();
    }
    std::size_t set_thread_data(std::size_t data) override {
        EINSUMS_ASSERT(_pimpl);
        return _pimpl->set_thread_data(data);
    }

    tss_storage *get_thread_tss_data() override {
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
        EINSUMS_ASSERT(_pimpl);
        return _pimpl->get_thread_tss_data(false);
#else
        return nullptr;
#endif
    }

    tss_storage *get_or_create_thread_tss_data() override {
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
        EINSUMS_ASSERT(_pimpl);
        return _pimpl->get_thread_tss_data(true);
#else
        return nullptr;
#endif
    }

    std::size_t &get_continuation_recursion_count() override {
        EINSUMS_ASSERT(_pimpl);
        return _pimpl->get_continuation_recursion_count();
    }

  private:
    coroutine_impl *get_impl() override { return _pimpl; }
    coroutine_impl *_pimpl;
};
} // namespace einsums::threads::coroutines::detail
