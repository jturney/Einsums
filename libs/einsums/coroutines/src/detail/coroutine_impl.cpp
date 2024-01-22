//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/coroutines/coroutine.hpp>
#include <einsums/coroutines/detail/coroutine_impl.hpp>
#include <einsums/coroutines/detail/coroutine_stackful_self.hpp>

#include <cstddef>
#include <exception>
#include <utility>

namespace einsums::threads::coroutines::detail {
#if defined(EINSUMS_DEBUG)
coroutine_impl::~coroutine_impl() {
    EINSUMS_ASSERT(!_fun); // functor should have been reset by now
}
#endif

void coroutine_impl::operator()() noexcept {
    using context_exit_status  = super_type::context_exit_status;
    context_exit_status status = super_type::ctx_not_exited;

    // yield value once the thread function has finished executing
    result_type result_last(threads::detail::thread_schedule_state::unknown, threads::detail::invalid_thread_id);

    // loop as long this coroutine has been rebound
    do {
#if defined(EINSUMS_HAVE_ADDRESS_SANITIZER)
        finish_switch_fiber(nullptr, m_caller);
#endif
        std::exception_ptr tinfo;
        {
            coroutine_self         *old_self = coroutine_self::get_self();
            coroutine_stackful_self self(this, old_self);
            reset_self_on_exit      on_exit(&self, old_self);
            try {
                result_last = _fun(*this->args());
                EINSUMS_ASSERT(result_last.first == threads::detail::thread_schedule_state::terminated);
                status = super_type::ctx_exited_return;
            } catch (...) {
                status = super_type::ctx_exited_abnormally;
                tinfo  = std::current_exception();
            }

            // Reset early as the destructors may still yield.
            this->reset_tss();
            this->reset();

            // return value to other side of the fence
            this->bind_result(result_last);
        }

        this->do_return(status, EINSUMS_MOVE(tinfo));
    } while (this->_state == super_type::ctx_running);

    // should not get here, never
    EINSUMS_ASSERT(this->_state == super_type::ctx_running);
}
} // namespace einsums::threads::coroutines::detail
