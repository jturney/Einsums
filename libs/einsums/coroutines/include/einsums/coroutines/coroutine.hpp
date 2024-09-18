//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/coroutines/coroutine_fwd.hpp>
#include <einsums/coroutines/detail/coroutine_accessor.hpp>
#include <einsums/coroutines/detail/coroutine_impl.hpp>
#include <einsums/coroutines/detail/coroutine_self.hpp>
#include <einsums/coroutines/thread_enums.hpp>
#include <einsums/coroutines/thread_id_type.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace einsums::threads::coroutines::detail {
/////////////////////////////////////////////////////////////////////////////
class coroutine {
  public:
    friend struct coroutine_accessor;

    using impl_type      = coroutine_impl;
    using thread_id_type = impl_type::thread_id_type;

    using result_type = impl_type::result_type;
    using arg_type    = impl_type::arg_type;

    using functor_type = util::detail::unique_function<result_type(arg_type)>;

    coroutine(functor_type &&f, thread_id_type id, std::ptrdiff_t stack_size = default_stack_size) : _impl(std::move(f), id, stack_size) {
        EINSUMS_ASSERT(_impl.is_ready());
    }

    coroutine(coroutine const &src)            = delete;
    coroutine &operator=(coroutine const &src) = delete;
    coroutine(coroutine &&src)                 = delete;
    coroutine &operator=(coroutine &&src)      = delete;

    thread_id_type get_thread_id() const { return _impl.get_thread_id(); }

#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
    std::size_t get_thread_phase() const { return _impl.get_thread_phase(); }
#endif

    std::size_t get_thread_data() const { return _impl.get_thread_data(); }

    std::size_t set_thread_data(std::size_t data) { return _impl.set_thread_data(data); }

    void init() { _impl.init(); }

    void rebind(functor_type &&f, thread_id_type id) { _impl.rebind(std::move(f), id); }

    EINSUMS_FORCEINLINE result_type operator()(arg_type arg = arg_type()) {
        EINSUMS_ASSERT(_impl.is_ready());

        _impl.bind_args(&arg);

        _impl.invoke();

        return _impl.result();
    }

    bool is_ready() const { return _impl.is_ready(); }

    std::ptrdiff_t get_available_stack_space() {
#if defined(EINSUMS_HAVE_THREADS_GET_STACK_POINTER)
        return _impl.get_available_stack_space();
#else
        return (std::numeric_limits<std::ptrdiff_t>::max)();
#endif
    }

    impl_type *impl() { return &_impl; }

  private:
    impl_type _impl;
};
} // namespace einsums::threads::coroutines::detail
