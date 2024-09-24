//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_STDEXEC)
#    include <einsums/execution_base/stdexec_forward.hpp>
#else
#    include <einsums/assert.hpp>
#    include <einsums/datastructures/variant.hpp>
#    include <einsums/errors/try_catch_exception_ptr.hpp>
#    include <einsums/execution/algorithms/detail/partial_algorithm.hpp>
#    include <einsums/execution_base/receiver.hpp>
#    include <einsums/execution_base/sender.hpp>
#    include <einsums/functional/detail/tag_fallback_invoke.hpp>
#    include <einsums/type_support/detail/with_result_of.hpp>
#    include <einsums/type_support/pack.hpp>

#    include <exception>
#    include <type_traits>
#    include <utility>

namespace einsums::let_error_detail {
template <typename PredecessorSender, typename F>
struct let_error_sender_impl {
    struct let_error_sender_type;
};

template <typename PredecessorSender, typename F>
using let_error_sender = typename let_error_sender_impl<PredecessorSender, F>::let_error_sender_type;

template <typename PredecessorSender, typename F>
struct let_error_sender_impl<PredecessorSender, F>::let_error_sender_type {
    EINSUMS_NO_UNIQUE_ADDRESS typename std::decay_t<PredecessorSender> predecessor_sender;
    EINSUMS_NO_UNIQUE_ADDRESS typename std::decay_t<F>                 f;

    // Type of the potential values returned from the predecessor sender
    template <template <typename...> class Tuple, template <typename...> class Variant>
    using predecessor_value_types =
        typename einsums::execution::experimental::sender_traits<std::decay_t<PredecessorSender>>::template value_types<Tuple, Variant>;

    // Type of the potential errors returned from the predecessor sender
    template <template <typename...> class Variant>
    using predecessor_error_types = einsums::util::detail::transform_t<
        typename einsums::execution::experimental::sender_traits<PredecessorSender>::template error_types<Variant>, std::decay>;
    static_assert(!std::is_same<predecessor_error_types<einsums::util::detail::pack>, einsums::util::detail::pack<>>::value,
                  "let_error used with a predecessor that has an empty error_types. Is let_error "
                  "misplaced?");

    template <typename Error>
    struct successor_sender_types_helper {
        using type = std::invoke_result_t<F, std::add_lvalue_reference_t<Error>>;
        static_assert(einsums::execution::experimental::is_sender<std::decay_t<type>>::value,
                      "let_error expects the invocable sender factory to return a sender");
    };

    // Type of the potential senders returned from the sender factory F
    template <template <typename...> class Variant>
    using successor_sender_types = einsums::util::detail::unique_t<
        einsums::util::detail::transform_t<predecessor_error_types<Variant>, successor_sender_types_helper>>;

    // The workaround for clang is due to a parsing bug in clang < 11
    // in CUDA mode (where >>> also has a different meaning in kernel
    // launches).
    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = einsums::util::detail::unique_concat_t<
        predecessor_value_types<Tuple, Variant>,
        einsums::util::detail::concat_pack_of_packs_t<einsums::util::detail::transform_t<
            successor_sender_types<Variant>, einsums::execution::experimental::detail::value_types<Tuple, Variant>::template apply
#    if defined(EINSUMS_CLANG_VERSION) && EINSUMS_CLANG_VERSION < 110000
            >
                                                      //
                                                      >>;
#    else
            >>>;
#    endif

    template <template <typename...> class Variant>
    using error_types = einsums::util::detail::unique_t<einsums::util::detail::prepend_t<
        einsums::util::detail::concat_pack_of_packs_t<einsums::util::detail::transform_t<
            successor_sender_types<Variant>, einsums::execution::experimental::detail::error_types<Variant>::template apply>>,
        std::exception_ptr>>;

    static constexpr bool sends_done = false;

    template <typename Receiver>
    struct operation_state {
        struct let_error_predecessor_receiver {
            EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<F> f;
            operation_state                          &op_state;

            template <typename Receiver_, typename F_>
            let_error_predecessor_receiver(Receiver_ &&receiver, F_ &&f, operation_state &op_state)
                : receiver(std::forward<Receiver_>(receiver)), f(std::forward<F_>(f)), op_state(op_state) {}

            struct start_visitor {
                [[noreturn]] void EINSUMS_STATIC_CALL_OPERATOR(einsums::detail::monostate) { EINSUMS_UNREACHABLE; }

                template <typename OperationState_,
                          typename = std::enable_if_t<!std::is_same_v<std::decay_t<OperationState_>, einsums::detail::monostate>>>
                void EINSUMS_STATIC_CALL_OPERATOR(OperationState_ &op_state) {
                    einsums::execution::experimental::start(op_state);
                }
            };

            struct set_error_visitor {
                EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<F> f;
                operation_state                          &op_state;

                [[noreturn]] void operator()(einsums::detail::monostate) const { EINSUMS_UNREACHABLE; }

                template <typename Error, typename = std::enable_if_t<!std::is_same_v<std::decay_t<Error>, einsums::detail::monostate>>>
                void operator()(Error &error) {
                    using operation_state_type =
                        decltype(einsums::execution::experimental::connect(EINSUMS_INVOKE(std::move(f), error), std::declval<Receiver>()));

#    if defined(EINSUMS_HAVE_CXX17_COPY_ELISION)
                    // with_result_of is used to emplace the operation
                    // state returned from connect without any
                    // intermediate copy construction (the operation
                    // state is not required to be copyable nor movable).
                    op_state.successor_op_state.template emplace<operation_state_type>(einsums::detail::with_result_of([&]() {
                        return einsums::execution::experimental::connect(EINSUMS_INVOKE(std::move(f), error), std::move(receiver));
                    }));
#    else
                    // MSVC doesn't get copy elision quite right, the operation
                    // state must be constructed explicitly directly in place
                    op_state.successor_op_state.template emplace_f<operation_state_type>(
                        einsums::execution::experimental::connect, EINSUMS_INVOKE(std::move(f), error), std::move(receiver));
#    endif
                    einsums::detail::visit(start_visitor{}, op_state.successor_op_state);
                }
            };

            template <typename Error>
            friend void tag_invoke(einsums::execution::experimental::set_error_t, let_error_predecessor_receiver &&r,
                                   Error &&error) noexcept {
                einsums::detail::try_catch_exception_ptr(
                    [&]() {
                        // TODO: receiver is moved before the visit, but
                        // the invoke inside the visit may throw.
                        r.op_state.predecessor_error.template emplace<std::decay_t<Error>>(std::forward<Error>(error));
                        einsums::detail::visit(set_error_visitor{std::move(r.receiver), std::move(r.f), r.op_state},
                                               r.op_state.predecessor_error);
                    },
                    [&](std::exception_ptr ep) { einsums::execution::experimental::set_error(std::move(r.receiver), std::move(ep)); });
            }

            friend void tag_invoke(einsums::execution::experimental::set_stopped_t, let_error_predecessor_receiver &&r) noexcept {
                einsums::execution::experimental::set_stopped(std::move(r.receiver));
            };

            template <typename... Ts,
                      typename = std::enable_if_t<std::is_invocable_v<einsums::execution::experimental::set_value_t, Receiver &&, Ts...>>>
            friend void tag_invoke(einsums::execution::experimental::set_value_t, let_error_predecessor_receiver &&r, Ts &&...ts) noexcept {
                einsums::execution::experimental::set_value(std::move(r.receiver), std::forward<Ts>(ts)...);
            }
        };

        // Type of the operation state returned when connecting the
        // predecessor sender to the let_error_predecessor_receiver
        using predecessor_operation_state_type =
            std::decay_t<einsums::execution::experimental::connect_result_t<PredecessorSender &&, let_error_predecessor_receiver>>;

        // Type of the potential operation states returned when
        // connecting a sender in successor_sender_types to the receiver
        // connected to the let_error_sender
        template <typename Sender>
        struct successor_operation_state_types_helper {
            using type = einsums::execution::experimental::connect_result_t<Sender, Receiver>;
        };
        template <template <typename...> class Variant>
        using successor_operation_state_types =
            einsums::util::detail::transform_t<successor_sender_types<Variant>, successor_operation_state_types_helper>;

        // Operation state from connecting predecessor sender to
        // let_error_predecessor_receiver
        predecessor_operation_state_type predecessor_operation_state;

        // Potential errors returned from the predecessor sender
        einsums::util::detail::prepend_t<predecessor_error_types<einsums::detail::variant>, einsums::detail::monostate> predecessor_error;

        // Potential operation states returned when connecting a sender
        // in successor_sender_types to the receiver connected to the
        // let_error_sender
        einsums::util::detail::prepend_t<successor_operation_state_types<einsums::detail::variant>, einsums::detail::monostate>
            successor_op_state;

        template <typename PredecessorSender_, typename Receiver_, typename F_>
        operation_state(PredecessorSender_ &&predecessor_sender, Receiver_ &&receiver, F_ &&f)
            : predecessor_operation_state{einsums::execution::experimental::connect(
                  std::forward<PredecessorSender_>(predecessor_sender),
                  let_error_predecessor_receiver(std::forward<Receiver_>(receiver), std::forward<F_>(f), *this))} {}

                         operation_state(operation_state &&)      = delete;
        operation_state &operator=(operation_state &&)            = delete;
                         operation_state(operation_state const &) = delete;
        operation_state &operator=(operation_state const &)       = delete;

        friend void tag_invoke(einsums::execution::experimental::start_t, operation_state &os) noexcept {
            einsums::execution::experimental::start(os.predecessor_operation_state);
        }
    };

    template <typename Receiver>
    friend auto tag_invoke(einsums::execution::experimental::connect_t, let_error_sender_type &&s, Receiver &&receiver) {
        return operation_state<Receiver>(std::move(s.predecessor_sender), std::forward<Receiver>(receiver), std::move(s.f));
    }

    template <typename Receiver>
    friend auto tag_invoke(einsums::execution::experimental::connect_t, let_error_sender_type const &, Receiver &&) {
        static_assert(sizeof(Receiver) == 0, "Are you missing a std::move? The let_error sender is not copyable and thus not "
                                             "l-value connectable. Make sure you are passing a non-const r-value reference of "
                                             "the sender.");
    }
};
} // namespace einsums::let_error_detail

namespace einsums::execution::experimental {
inline constexpr struct let_error_t final : einsums::functional::detail::tag_fallback<let_error_t> {
  private:
    template <typename PredecessorSender, typename F>
        requires(is_sender_v<PredecessorSender>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(let_error_t, PredecessorSender &&predecessor_sender, F &&f) {
        return let_error_detail::let_error_sender<PredecessorSender, F>{std::forward<PredecessorSender>(predecessor_sender),
                                                                        std::forward<F>(f)};
    }

    template <typename F>
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(let_error_t, F &&f) {
        return detail::partial_algorithm<let_error_t, F>{std::forward<F>(f)};
    }
} let_error{};
} // namespace einsums::execution::experimental
#endif
