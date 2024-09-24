//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_STDEXEC)
#    include <einsums/execution_base/stdexec_forward.hpp>
#endif

#include <einsums/concurrency/spinlock.hpp>
#include <einsums/datastructures/variant.hpp>
#include <einsums/execution/algorithms/detail/helpers.hpp>
#include <einsums/execution_base/operation_state.hpp>
#include <einsums/execution_base/receiver.hpp>
#include <einsums/execution_base/sender.hpp>
#include <einsums/functional/detail/tag_fallback_invoke.hpp>
#include <einsums/synchronization/counting_semaphore.hpp>
#include <einsums/type_support/pack.hpp>
#include <einsums/type_support/unused.hpp>

#include <atomic>
#include <exception>
#include <type_traits>
#include <utility>

namespace einsums::sync_wait_detail {
struct sync_wait_error_visitor {
    void EINSUMS_STATIC_CALL_OPERATOR(std::exception_ptr ep) { std::rethrow_exception(ep); }

    template <typename Error>
    void EINSUMS_STATIC_CALL_OPERATOR(Error &error) {
        throw error;
    }
};

template <typename Sender>
struct sync_wait_receiver_impl {
    struct sync_wait_receiver_type;
};

template <typename Sender>
using sync_wait_receiver = typename sync_wait_receiver_impl<Sender>::sync_wait_receiver_type;

template <typename Sender>
struct sync_wait_receiver_impl<Sender>::sync_wait_receiver_type {
    EINSUMS_STDEXEC_RECEIVER_CONCEPT

#if defined(EINSUMS_HAVE_STDEXEC)
    // value and error_types of the predecessor sender
    template <template <typename...> class Tuple, template <typename...> class Variant>
    using predecessor_value_types = einsums::util::detail::unique_t<
        einsums::execution::experimental::value_types_of_t<Sender, einsums::execution::experimental::empty_env, Tuple, Variant>>;

    template <template <typename...> class Variant>
    using predecessor_error_types =
        einsums::execution::experimental::error_types_of_t<Sender, einsums::execution::experimental::empty_env, Variant>;

    // The type of the single void or non-void result that we store. If
    // there are multiple variants or multiple values sync_wait will
    // fail to compile.
    using result_type = std::decay_t<einsums::execution::experimental::detail::single_result_t<
        predecessor_value_types<einsums::util::detail::decayed_pack, einsums::util::detail::pack>>>;

    // Constant to indicate if the type of the result from the
    // predecessor sender is void or not
    static constexpr bool is_void_result = std::is_void_v<result_type>;

    // Dummy type to indicate that set_value with void has been called
    struct void_value_type {};

    // The type of the value to store in the variant, void_value_type if
    // result_type is void, or result_type if it is not
    using value_type = std::conditional_t<is_void_result, void_value_type, result_type>;

    // The type of errors to store in the variant. This in itself is a
    // variant.
    using error_type = einsums::util::detail::unique_t<einsums::util::detail::prepend_t<
        einsums::util::detail::transform_t<predecessor_error_types<einsums::detail::variant>, std::decay>, std::exception_ptr>>;
#else
    // value and error_types of the predecessor sender
    template <template <typename...> class Tuple, template <typename...> class Variant>
    using predecessor_value_types = typename einsums::execution::experimental::sender_traits<Sender>::template value_types<Tuple, Variant>;

    template <template <typename...> class Variant>
    using predecessor_error_types = typename einsums::execution::experimental::sender_traits<Sender>::template error_types<Variant>;

    // The type of the single void or non-void result that we store. If
    // there are multiple variants or multiple values sync_wait will
    // fail to compile.
    using result_type = std::decay_t<einsums::execution::experimental::detail::single_result_t<
        predecessor_value_types<einsums::util::detail::pack, einsums::util::detail::pack>>>;

    // Constant to indicate if the type of the result from the
    // predecessor sender is void or not
    static constexpr bool is_void_result = std::is_void_v<result_type>;

    // Dummy type to indicate that set_value with void has been called
    struct void_value_type {};

    // The type of the value to store in the variant, void_value_type if
    // result_type is void, or result_type if it is not
    using value_type = std::conditional_t<is_void_result, void_value_type, result_type>;

    // The type of errors to store in the variant. This in itself is a
    // variant.
    using error_type = einsums::util::detail::unique_t<einsums::util::detail::prepend_t<
        einsums::util::detail::transform_t<predecessor_error_types<einsums::detail::variant>, std::decay>, std::exception_ptr>>;
#endif

    // We use a spinlock here to allow taking the lock on non-einsums threads.
    using mutex_type = einsums::concurrency::detail::spinlock;

    struct shared_state {
        einsums::binary_semaphore<>                                                  sem{0};
        einsums::detail::variant<einsums::detail::monostate, error_type, value_type> value;

        void wait() { sem.acquire(); }

        auto get_value() {
            if (einsums::detail::holds_alternative<value_type>(value)) {
                if constexpr (is_void_result) {
                    return;
                } else {
                    return std::move(einsums::detail::get<value_type>(value));
                }
            } else if (einsums::detail::holds_alternative<error_type>(value)) {
                einsums::detail::visit(sync_wait_error_visitor{}, einsums::detail::get<error_type>(value));
            }

            // If the variant holds a einsums::detail::monostate something has gone
            // wrong and we terminate
            EINSUMS_UNREACHABLE;
        }
    };

    shared_state &state;

    void signal_set_called() noexcept { state.sem.release(); }

    template <typename Error>
    friend void tag_invoke(einsums::execution::experimental::set_error_t, sync_wait_receiver_type &&r, Error &&error) noexcept {
        r.state.value.template emplace<error_type>(std::forward<Error>(error));
        r.signal_set_called();
    }

    friend void tag_invoke(einsums::execution::experimental::set_stopped_t, sync_wait_receiver_type &&r) noexcept { r.signal_set_called(); }

    template <typename... Us,
              typename = std::enable_if_t<(is_void_result && sizeof...(Us) == 0) || (!is_void_result && sizeof...(Us) == 1)>>
    friend void tag_invoke(einsums::execution::experimental::set_value_t, sync_wait_receiver_type &&r, Us &&...us) noexcept {
        r.state.value.template emplace<value_type>(std::forward<Us>(us)...);
        r.signal_set_called();
    }

    friend constexpr einsums::execution::experimental::empty_env tag_invoke(einsums::execution::experimental::get_env_t,
                                                                            sync_wait_receiver_type const &) noexcept {
        return {};
    }
};
} // namespace einsums::sync_wait_detail

namespace einsums::this_thread::experimental {
inline constexpr struct sync_wait_t final : einsums::functional::detail::tag_fallback<sync_wait_t> {
  private:
    template <typename Sender>
        requires(einsums::execution::experimental::is_sender_v<Sender>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(sync_wait_t, Sender &&sender) {
        using receiver_type = sync_wait_detail::sync_wait_receiver<Sender>;
        using state_type    = typename receiver_type::shared_state;

        state_type state{};
        auto       op_state = einsums::execution::experimental::connect(std::forward<Sender>(sender), receiver_type{state});
        einsums::execution::experimental::start(op_state);

        state.wait();
        return state.get_value();
    }
} sync_wait{};
} // namespace einsums::this_thread::experimental
