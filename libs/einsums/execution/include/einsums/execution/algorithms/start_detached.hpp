//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_STDEXEC)
#    include <einsums/execution_base/stdexec_forward.hpp>
#endif

#include <einsums/allocator_support/allocator_deleter.hpp>
#include <einsums/allocator_support/internal_allocator.hpp>
#include <einsums/allocator_support/traits/is_allocator.hpp>
#include <einsums/assert.hpp>
#include <einsums/execution_base/operation_state.hpp>
#include <einsums/execution_base/receiver.hpp>
#include <einsums/execution_base/sender.hpp>
#include <einsums/functional/detail/tag_fallback_invoke.hpp>
#include <einsums/type_support/unused.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace einsums::start_detached_detail {
template <typename Sender, typename Allocator>
struct operation_state_holder {
    struct start_detached_receiver {
        EINSUMS_STDEXEC_RECEIVER_CONCEPT

        operation_state_holder &op_state;

        template <typename Error>
#if !defined(__NVCC__)
        [[noreturn]]
#endif
        friend void
        tag_invoke(einsums::execution::experimental::set_error_t, start_detached_receiver &&r, Error error) noexcept {
            r.op_state.release();

            if constexpr (std::is_same_v<std::decay_t<Error>, std::exception_ptr>) {
                std::rethrow_exception(std::move(error));
            }

            EINSUMS_ASSERT_MSG(false, "set_error was called on the receiver of start_detached, terminating. If you "
                                      "want to allow errors from the predecessor sender, handle them first with e.g. "
                                      "let_error.");
            std::terminate();
        }

        friend void tag_invoke(einsums::execution::experimental::set_stopped_t, start_detached_receiver &&r) noexcept {
            r.op_state.release();
        };

        template <typename... Ts>
        friend void tag_invoke(einsums::execution::experimental::set_value_t, start_detached_receiver &&r, Ts &&...) noexcept {
            r.op_state.release();
        }
    };

  private:
    using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<operation_state_holder>;
    EINSUMS_NO_UNIQUE_ADDRESS allocator_type alloc;

    using operation_state_type = einsums::execution::experimental::connect_result_t<Sender, start_detached_receiver>;
    std::decay_t<operation_state_type> op_state;

  public:
    template <typename Sender_, typename = std::enable_if_t<!std::is_same<std::decay_t<Sender_>, operation_state_holder>::value>>
    explicit operation_state_holder(Sender_ &&sender, allocator_type const &alloc)
        : alloc(alloc), op_state(einsums::execution::experimental::connect(std::forward<Sender_>(sender), start_detached_receiver{*this})) {
        einsums::execution::experimental::start(op_state);
    }

    void release() noexcept {
        allocator_type other_alloc(alloc);
        std::allocator_traits<allocator_type>::destroy(other_alloc, this);
        std::allocator_traits<allocator_type>::deallocate(other_alloc, this, 1);
    }
};
} // namespace einsums::start_detached_detail

namespace einsums::execution::experimental {
inline constexpr struct start_detached_t final : einsums::functional::detail::tag_fallback<start_detached_t> {
  private:
    template <typename Sender, typename Allocator = einsums::detail::internal_allocator<>>
        requires(is_sender_v<Sender> && einsums::detail::is_allocator_v<Allocator>)
    friend constexpr EINSUMS_FORCEINLINE void tag_fallback_invoke(start_detached_t, Sender &&sender,
                                                                  Allocator const &allocator = Allocator{}) {
        using allocator_type       = Allocator;
        using operation_state_type = start_detached_detail::operation_state_holder<Sender, Allocator>;
        using other_allocator      = typename std::allocator_traits<allocator_type>::template rebind_alloc<operation_state_type>;
        using allocator_traits     = std::allocator_traits<other_allocator>;
        using unique_ptr           = std::unique_ptr<operation_state_type, einsums::detail::allocator_deleter<other_allocator>>;

        other_allocator alloc(allocator);
        unique_ptr      p(allocator_traits::allocate(alloc, 1), einsums::detail::allocator_deleter<other_allocator>{alloc});

        new (p.get()) operation_state_type{std::forward<Sender>(sender), alloc};
        EINSUMS_UNUSED(p.release());
    }
} start_detached{};
} // namespace einsums::execution::experimental
