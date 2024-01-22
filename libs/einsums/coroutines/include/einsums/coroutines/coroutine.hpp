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

    coroutine(functor_type &&f, thread_id_type id, std::ptrdiff_t stack_size = default_stack_size)
        : _impl(EINSUMS_MOVE(f), id, stack_size) {
        EINSUMS_ASSERT(impl_.is_ready());
    }

    coroutine(coroutine const &src)                     = delete;
    auto operator=(coroutine const &src) -> coroutine & = delete;
    coroutine(coroutine &&src)                          = delete;
    auto operator=(coroutine &&src) -> coroutine      & = delete;

    auto get_thread_id() const -> thread_id_type { return _impl.get_thread_id(); }

#if defined(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
    std::size_t get_thread_phase() const { return impl_.get_thread_phase(); }
#endif

    auto get_thread_data() const -> std::size_t { return _impl.get_thread_data(); }

    auto set_thread_data(std::size_t data) -> std::size_t { return _impl.set_thread_data(data); }

    void init() { _impl.init(); }

    void rebind(functor_type &&f, thread_id_type id) { _impl.rebind(EINSUMS_MOVE(f), id); }

    EINSUMS_FORCEINLINE result_type operator()(arg_type arg = arg_type()) {
        EINSUMS_ASSERT(impl_.is_ready());

        _impl.bind_args(&arg);

        _impl.invoke();

        return _impl.result();
    }

    auto is_ready() const -> bool { return _impl.is_ready(); }

    auto get_available_stack_space() -> std::ptrdiff_t {
#if defined(EINSUMS_HAVE_THREADS_GET_STACK_POINTER)
        return _impl.get_available_stack_space();
#else
        return (std::numeric_limits<std::ptrdiff_t>::max)();
#endif
    }

    auto impl() -> impl_type * { return &_impl; }

  private:
    impl_type _impl;
};
} // namespace einsums::threads::coroutines::detail
