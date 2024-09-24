//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_STDEXEC)
#    include <einsums/execution_base/stdexec_forward.hpp>
#endif

#include <einsums/assert.hpp>
#include <einsums/datastructures/variant.hpp>
#include <einsums/execution/algorithms/detail/helpers.hpp>
#include <einsums/execution_base/operation_state.hpp>
#include <einsums/execution_base/receiver.hpp>
#include <einsums/execution_base/sender.hpp>
#include <einsums/functional/detail/tag_fallback_invoke.hpp>
#include <einsums/type_support/detail/with_result_of.hpp>
#include <einsums/type_support/pack.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsums::when_all_vector_detail {
template <typename Sender>
struct when_all_vector_sender_impl {
    struct when_all_vector_sender_type;
};

template <typename Sender>
using when_all_vector_sender = typename when_all_vector_sender_impl<Sender>::when_all_vector_sender_type;

template <typename Sender>
struct when_all_vector_sender_impl<Sender>::when_all_vector_sender_type {
    EINSUMS_STDEXEC_SENDER_CONCEPT

    using senders_type = std::vector<Sender>;
    senders_type senders;

    explicit constexpr when_all_vector_sender_type(senders_type &&senders) : senders(std::move(senders)) {}

    explicit constexpr when_all_vector_sender_type(senders_type const &senders) : senders(senders) {}

#if defined(EINSUMS_HAVE_STDEXEC)
    // We expect a single value type or nothing from the predecessor
    // sender type
    using element_value_type =
        std::decay_t<einsums::execution::experimental::detail::single_result_t<einsums::execution::experimental::value_types_of_t<
            Sender, einsums::execution::experimental::empty_env, einsums::util::detail::pack, einsums::util::detail::pack>>>;

    static constexpr bool is_void_value_type = std::is_void_v<element_value_type>;

    // This is a helper empty type for the case that nothing is sent
    // from the predecessors
    struct void_value_type {};

    // This sender sends a single vector of the type sent by the
    // predecessor senders or nothing if the predecessor senders send
    // nothing
    template <typename...>
    using set_value_helper = einsums::execution::experimental::completion_signatures<
        std::conditional_t<is_void_value_type, einsums::execution::experimental::set_value_t(),
                           einsums::execution::experimental::set_value_t(std::vector<element_value_type>)>>;

    // This sender sends any error types sent by the predecessor senders
    // or std::exception_ptr
    template <template <typename...> class Variant>
    using error_types = einsums::util::detail::unique_concat_t<
        einsums::util::detail::transform_t<
            einsums::execution::experimental::error_types_of_t<Sender, einsums::execution::experimental::empty_env, Variant>, std::decay>,
        Variant<std::exception_ptr>>;

    static constexpr bool sends_done = false;

    using completion_signatures = einsums::execution::experimental::transform_completion_signatures_of<
        Sender, einsums::execution::experimental::empty_env,
        einsums::execution::experimental::completion_signatures<einsums::execution::experimental::set_error_t(std::exception_ptr)>,
        set_value_helper>;
#else
    // We expect a single value type or nothing from the predecessor
    // sender type
    using element_value_type =
        std::decay_t<einsums::execution::experimental::detail::single_result_t<typename einsums::execution::experimental::sender_traits<
            Sender>::template value_types<einsums::util::detail::pack, einsums::util::detail::pack>>>;

    static constexpr bool is_void_value_type = std::is_void_v<element_value_type>;

    // This is a helper empty type for the case that nothing is sent
    // from the predecessors
    struct void_value_type {};

    // This sender sends a single vector of the type sent by the
    // predecessor senders or nothing if the predecessor senders send
    // nothing
    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = Variant<std::conditional_t<is_void_value_type, Tuple<>, Tuple<std::vector<element_value_type>>>>;

    // This sender sends any error types sent by the predecessor senders
    // or std::exception_ptr
    template <template <typename...> class Variant>
    using error_types = einsums::util::detail::unique_concat_t<
        einsums::util::detail::transform_t<typename einsums::execution::experimental::sender_traits<Sender>::template error_types<Variant>,
                                           std::decay>,
        Variant<std::exception_ptr>>;

    static constexpr bool sends_done = false;
#endif

    template <typename Receiver>
    struct operation_state {
        struct when_all_vector_receiver {
            EINSUMS_STDEXEC_RECEIVER_CONCEPT

            operation_state  &op_state;
            std::size_t const i;

            template <typename Error>
            friend void tag_invoke(einsums::execution::experimental::set_error_t, when_all_vector_receiver &&r, Error &&error) noexcept {
                if (!r.op_state.set_stopped_error_called.exchange(true)) {
                    try {
                        r.op_state.error = std::forward<Error>(error);
                    } catch (...) {
                        // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
                        r.op_state.error = std::current_exception();
                    }
                }

                r.op_state.finish();
            }

            friend void tag_invoke(einsums::execution::experimental::set_stopped_t, when_all_vector_receiver &&r) noexcept {
                r.op_state.set_stopped_error_called = true;
                r.op_state.finish();
            };

            template <typename... Ts>
            friend void tag_invoke(einsums::execution::experimental::set_value_t, when_all_vector_receiver &&r, Ts &&...ts) noexcept {
                if (!r.op_state.set_stopped_error_called) {
                    try {
                        // We only have something to store if the
                        // predecessor sends the single value that it
                        // should send. We have nothing to store for
                        // predecessor senders that send nothing.
                        if constexpr (sizeof...(Ts) == 1) {
                            r.op_state.ts[r.i].emplace(std::forward<Ts>(ts)...);
                        }
                    } catch (...) {
                        if (!r.op_state.set_stopped_error_called.exchange(true)) {
                            // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
                            r.op_state.error = std::current_exception();
                        }
                    }
                }

                r.op_state.finish();
            }

            friend constexpr einsums::execution::experimental::empty_env tag_invoke(einsums::execution::experimental::get_env_t,
                                                                                    when_all_vector_receiver const &) noexcept {
                return {};
            }
        };

        std::size_t const      num_predecessors;
        std::decay_t<Receiver> receiver;

        // Number of predecessor senders that have not yet called any of
        // the set signals.
        std::atomic<std::size_t> predecessors_remaining{num_predecessors};

        // The values sent by the predecessor senders are stored in a
        // vector of optional or the dummy type void_value_type if the
        // predecessor senders send nothing
        using value_types_storage_type =
            std::conditional_t<is_void_value_type, void_value_type, std::vector<std::optional<element_value_type>>>;
        value_types_storage_type ts;

        // The first error sent by any predecessor sender is stored in a
        // optional of a variant of the error_types
        using error_types_storage_type = std::optional<error_types<einsums::detail::variant>>;
        error_types_storage_type error;

        // Set to true when set_stopped or set_error has been called
        std::atomic<bool> set_stopped_error_called{false};

        // The operation states are stored in an array of optionals of
        // the operation states to handle the non-movability and
        // non-copyability of them
        using operation_state_type              = einsums::execution::experimental::connect_result_t<Sender, when_all_vector_receiver>;
        using operation_states_storage_type     = std::unique_ptr<std::optional<operation_state_type>[]>;
        operation_states_storage_type op_states = nullptr;

        template <typename Receiver_>
        operation_state(Receiver_ &&receiver, std::vector<Sender> senders)
            : num_predecessors(senders.size()), receiver(std::forward<Receiver_>(receiver)) {
            op_states     = std::make_unique<std::optional<operation_state_type>[]>(num_predecessors);
            std::size_t i = 0;
            for (auto &sender : senders) {
                op_states[i].emplace(einsums::detail::with_result_of([&]() {
                    return einsums::execution::experimental::connect(
#if defined(__NVCC__) && defined(EINSUMS_CUDA_VERSION) && (EINSUMS_CUDA_VERSION >= 1204)
                        std::move(sender)
#else
                        std::move(sender)
#endif
                            ,
                        when_all_vector_receiver{*this, i});
                }));
                ++i;
            }

            if constexpr (!is_void_value_type) {
                ts.resize(num_predecessors);
            }
        }

                         operation_state(operation_state &&)      = delete;
        operation_state &operator=(operation_state &&)            = delete;
                         operation_state(operation_state const &) = delete;
        operation_state &operator=(operation_state const &)       = delete;

        void finish() noexcept {
            if (--predecessors_remaining == 0) {
                if (!set_stopped_error_called) {
                    if constexpr (is_void_value_type) {
                        einsums::execution::experimental::set_value(std::move(receiver));
                    } else {
                        std::vector<element_value_type> values;
                        values.reserve(num_predecessors);
                        for (auto &&t : ts) {
                            EINSUMS_ASSERT(t.has_value());
                            values.push_back(
#if defined(__NVCC__) && defined(EINSUMS_CUDA_VERSION) && (EINSUMS_CUDA_VERSION >= 1204)
                                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                                std::move(*t)
#else
                                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                                std::move(*t)
#endif
                            );
                        }
                        einsums::execution::experimental::set_value(std::move(receiver), std::move(values));
                    }
                } else if (error) {
                    einsums::detail::visit(
                        [this](auto &&error) {
                            einsums::execution::experimental::set_error(std::move(receiver), std::forward<decltype(error)>(error));
                        },
                        std::move(*error));
                } else {
#if defined(EINSUMS_HAVE_STDEXEC)
                    if constexpr (einsums::execution::experimental::sends_stopped<Sender>)
#else
                    if constexpr (einsums::execution::experimental::sender_traits<Sender>::sends_done)
#endif
                    {
                        einsums::execution::experimental::set_stopped(std::move(receiver));
                    } else {
                        EINSUMS_UNREACHABLE;
                    }
                }
            }
        }

        friend void tag_invoke(einsums::execution::experimental::start_t, operation_state &os) noexcept {
            // If there are no predecessors we can signal the
            // continuation as soon as start is called.
            if (os.num_predecessors == 0) {
                // If the predecessor sender type sends nothing, we also
                // send nothing to the continuation.
                if constexpr (is_void_value_type) {
                    einsums::execution::experimental::set_value(std::move(os.receiver));
                }
                // If the predecessor sender type sends something we
                // send an empty vector of that type to the continuation.
                else {
                    einsums::execution::experimental::set_value(std::move(os.receiver), std::vector<element_value_type>{});
                }
            }
            // Otherwise we start all the operation states and wait for
            // the predecessors to signal completion.
            else {
                // After the call to start on the last child operation state the current
                // when_all_vector operation state may already have been released. We read the
                // number of predecessors from the operation state into a stack-local variable
                // so that the loop can end without reading freed memory.
                auto const num_predecessors = os.num_predecessors;
                for (std::size_t i = 0; i < num_predecessors; ++i) {
                    EINSUMS_ASSERT(os.op_states[i].has_value());
                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                    einsums::execution::experimental::start(*(os.op_states.get()[i]));
                }
            }
        }
    };

    template <typename Receiver>
    friend auto tag_invoke(einsums::execution::experimental::connect_t, when_all_vector_sender_type &&s, Receiver &&receiver) {
        return operation_state<Receiver>(std::forward<Receiver>(receiver), std::move(s.senders));
    }

    template <typename Receiver>
    friend auto tag_invoke(einsums::execution::experimental::connect_t, when_all_vector_sender_type const &s, Receiver &&receiver) {
        return operation_state<Receiver>(std::forward<Receiver>(receiver), s.senders);
    }
};
} // namespace einsums::when_all_vector_detail

namespace einsums::execution::experimental {
struct when_all_vector_t final : einsums::functional::detail::tag_fallback<when_all_vector_t> {
  private:
    template <typename Sender>
        requires(is_sender_v<Sender>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(when_all_vector_t, std::vector<Sender> &&senders) {
        return when_all_vector_detail::when_all_vector_sender<Sender>{std::move(senders)};
    }

    template <typename Sender>
        requires(is_sender_v<Sender>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(when_all_vector_t, std::vector<Sender> const &senders) {
        return when_all_vector_detail::when_all_vector_sender<Sender>{senders};
    }
};

/// \brief Returns a sender that completes when all senders in the input vector have completed.
///
/// Sender adaptor that takes a vector of senders and returns a sender that sends a vector of
/// the values sent by the input senders. The vector sent has the same size as the input vector.
/// An empty vector of senders completes immediately on start. When the input vector of senders
/// contains senders that send no value the output sender sends no value instead of a vector.
/// The senders in the input vector must send at most a single type.
///
/// Added in 0.2.0.
inline constexpr when_all_vector_t when_all_vector{};
} // namespace einsums::execution::experimental
