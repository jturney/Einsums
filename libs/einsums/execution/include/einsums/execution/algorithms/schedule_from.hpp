//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(einsums_HAVE_STDEXEC)
#    include <einsums/execution_base/stdexec_forward.hpp>
#else
#    include <einsums/datastructures/variant.hpp>
#    include <einsums/execution_base/completion_scheduler.hpp>
#    include <einsums/execution_base/receiver.hpp>
#    include <einsums/execution_base/sender.hpp>
#    include <einsums/functional/bind_front.hpp>
#    include <einsums/functional/detail/invoke_result_plain_function.hpp>
#    include <einsums/functional/detail/tag_fallback_invoke.hpp>
#    include <einsums/type_support/detail/with_result_of.hpp>
#    include <einsums/type_support/pack.hpp>

#    include <atomic>
#    include <cstddef>
#    include <exception>
#    include <optional>
#    include <tuple>
#    include <type_traits>
#    include <utility>

namespace einsums::schedule_from_detail {
template <typename Sender, typename Scheduler>
struct schedule_from_sender_impl {
    struct schedule_from_sender_type;
};

template <typename Sender, typename Scheduler>
using schedule_from_sender = typename schedule_from_sender_impl<Sender, Scheduler>::schedule_from_sender_type;

template <typename Sender, typename Scheduler>
struct schedule_from_sender_impl<Sender, Scheduler>::schedule_from_sender_type {
    EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Sender> predecessor_sender;
    EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;

    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = typename einsums::execution::experimental::sender_traits<Sender>::template value_types<Tuple, Variant>;

    template <template <typename...> class Variant>
    using predecessor_sender_error_types = typename einsums::execution::experimental::sender_traits<Sender>::template error_types<Variant>;

    using scheduler_sender_type = einsums::detail::invoke_result_plain_function_t<einsums::execution::experimental::schedule_t, Scheduler>;
    template <template <typename...> class Variant>
    using scheduler_sender_error_types =
        typename einsums::execution::experimental::sender_traits<scheduler_sender_type>::template error_types<Variant>;

    template <template <typename...> class Variant>
    using error_types =
        einsums::util::detail::unique_concat_t<predecessor_sender_error_types<Variant>, scheduler_sender_error_types<Variant>>;

    static constexpr bool sends_done = false;

    template <typename Env>
    struct env {
        EINSUMS_NO_UNIQUE_ADDRESS Env e;
        EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;

        friend std::decay_t<Scheduler>
        tag_invoke(einsums::execution::experimental::get_completion_scheduler_t<einsums::execution::experimental::set_value_t>,
                   env const &e) noexcept {
            return e.scheduler;
        }

        template <typename Tag>
            requires(std::is_invocable_v<Tag, env const &>)
        friend decltype(auto) tag_invoke(Tag, env const &e) {
            return Tag{}(e.e);
        }
    };

    friend auto tag_invoke(einsums::execution::experimental::get_env_t, schedule_from_sender_type const &s) noexcept {
        auto e = einsums::execution::experimental::get_env(s.predecessor_sender);
        return env<std::decay_t<decltype(e)>>{std::move(e), s.scheduler};
    }

    template <typename Receiver>
    struct operation_state {
        EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;
        EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

        struct predecessor_sender_receiver;
        struct scheduler_sender_receiver;

        template <typename Tuple>
        struct value_types_helper {
            using type = einsums::util::detail::transform_t<Tuple, std::decay>;
        };

        using value_type = einsums::util::detail::prepend_t<
            einsums::util::detail::transform_t<typename einsums::execution::experimental::sender_traits<Sender>::template value_types<
                                                   std::tuple, einsums::detail::variant>,
                                               value_types_helper>,
            einsums::detail::monostate>;
        value_type ts;

        using sender_operation_state_type = einsums::execution::experimental::connect_result_t<Sender, predecessor_sender_receiver>;
        sender_operation_state_type sender_os;

        using scheduler_operation_state_type = einsums::execution::experimental::connect_result_t<
            einsums::detail::invoke_result_plain_function_t<einsums::execution::experimental::schedule_t, Scheduler>,
            scheduler_sender_receiver>;
        std::optional<scheduler_operation_state_type> scheduler_op_state;

        template <typename Sender_, typename Scheduler_, typename Receiver_>
        operation_state(Sender_ &&predecessor_sender, Scheduler_ &&scheduler, Receiver_ &&receiver)
            : scheduler(std::forward<Scheduler_>(scheduler)), receiver(std::forward<Receiver_>(receiver)),
              sender_os(einsums::execution::experimental::connect(std::forward<Sender_>(predecessor_sender),
                                                                  predecessor_sender_receiver{*this})) {}

                         operation_state(operation_state &&)      = delete;
                         operation_state(operation_state const &) = delete;
        operation_state &operator=(operation_state &&)            = delete;
        operation_state &operator=(operation_state const &)       = delete;

        struct predecessor_sender_receiver {
            operation_state &op_state;

            template <typename Error>
            friend void tag_invoke(einsums::execution::experimental::set_error_t, predecessor_sender_receiver &&r, Error &&error) noexcept {
                r.op_state.set_error_predecessor_sender(std::forward<Error>(error));
            }

            friend void tag_invoke(einsums::execution::experimental::set_stopped_t, predecessor_sender_receiver &&r) noexcept {
                r.op_state.set_stopped_predecessor_sender();
            }

            // These typedefs are duplicated from the parent struct. The
            // parent typedefs are not instantiated early enough for use
            // here.
            template <typename Tuple>
            struct value_types_helper {
                using type = einsums::util::detail::transform_t<Tuple, std::decay>;
            };

            using value_type = einsums::util::detail::prepend_t<
                einsums::util::detail::transform_t<typename einsums::execution::experimental::sender_traits<Sender>::template value_types<
                                                       std::tuple, einsums::detail::variant>,
                                                   value_types_helper>,
                einsums::detail::monostate>;

            template <typename... Ts>
            friend auto tag_invoke(einsums::execution::experimental::set_value_t, predecessor_sender_receiver &&r, Ts &&...ts) noexcept
                -> decltype(std::declval<value_type>().template emplace<std::tuple<std::decay_t<Ts>...>>(std::forward<Ts>(ts)...), void()) {
                // nvcc fails to compile this with std::forward<Ts>(ts)...
                // or static_cast<Ts&&>(ts)... so we explicitly use
                // static_cast<decltype(ts)>(ts)... as a workaround.
#    if defined(EINSUMS_HAVE_CUDA)
                r.op_state.set_value_predecessor_sender(static_cast<decltype(ts) &&>(ts)...);
#    else
                r.op_state.set_value_predecessor_sender(std::forward<Ts>(ts)...);
#    endif
            }
        };

        template <typename Error>
        void set_error_predecessor_sender(Error &&error) noexcept {
            einsums::execution::experimental::set_error(std::move(receiver), std::forward<Error>(error));
        }

        void set_stopped_predecessor_sender() noexcept { einsums::execution::experimental::set_stopped(std::move(receiver)); }

        template <typename... Us>
        void set_value_predecessor_sender(Us &&...us) noexcept {
            ts.template emplace<std::tuple<std::decay_t<Us>...>>(std::forward<Us>(us)...);
#    if defined(einsums_HAVE_CXX17_COPY_ELISION)
            // with_result_of is used to emplace the operation
            // state returned from connect without any
            // intermediate copy construction (the operation
            // state is not required to be copyable nor movable).
            scheduler_op_state.emplace(einsums::detail::with_result_of([&]() {
                return einsums::execution::experimental::connect(einsums::execution::experimental::schedule(std::move(scheduler)),
                                                                 scheduler_sender_receiver{*this});
            }));
#    else
            // MSVC doesn't get copy elision quite right, the operation
            // state must be constructed explicitly directly in place
            scheduler_op_state.emplace_f(einsums::execution::experimental::connect,
                                         einsums::execution::experimental::schedule(std::move(scheduler)),
                                         scheduler_sender_receiver{*this});
#    endif
            einsums::execution::experimental::start(*scheduler_op_state);
        }

        struct scheduler_sender_receiver {
            operation_state &op_state;

            template <typename Error>
            friend void tag_invoke(einsums::execution::experimental::set_error_t, scheduler_sender_receiver &&r, Error &&error) noexcept {
                r.op_state.set_error_scheduler_sender(std::forward<Error>(error));
            }

            friend void tag_invoke(einsums::execution::experimental::set_stopped_t, scheduler_sender_receiver &&r) noexcept {
                r.op_state.set_stopped_scheduler_sender();
            }

            friend void tag_invoke(einsums::execution::experimental::set_value_t, scheduler_sender_receiver &&r) noexcept {
                r.op_state.set_value_scheduler_sender();
            }
        };

        struct scheduler_sender_value_visitor {
            EINSUMS_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            [[noreturn]] void operator()(einsums::detail::monostate) const { EINSUMS_UNREACHABLE; }

            template <typename Ts, typename = std::enable_if_t<!std::is_same_v<std::decay_t<Ts>, einsums::detail::monostate>>>
            void operator()(Ts &&ts) {
                std::apply(einsums::util::detail::bind_front(einsums::execution::experimental::set_value, std::move(receiver)),
                           std::forward<Ts>(ts));
            }
        };

        template <typename Error>
        void set_error_scheduler_sender(Error &&error) noexcept {
            scheduler_op_state.reset();
            einsums::execution::experimental::set_error(std::move(receiver), std::forward<Error>(error));
        }

        void set_stopped_scheduler_sender() noexcept {
            scheduler_op_state.reset();
            einsums::execution::experimental::set_stopped(std::move(receiver));
        }

        void set_value_scheduler_sender() noexcept {
            scheduler_op_state.reset();
            einsums::detail::visit(scheduler_sender_value_visitor{std::move(receiver)}, std::move(ts));
        }

        friend void tag_invoke(einsums::execution::experimental::start_t, operation_state &os) noexcept {
            einsums::execution::experimental::start(os.sender_os);
        }
    };

    template <typename Receiver>
    friend operation_state<Receiver> tag_invoke(einsums::execution::experimental::connect_t, schedule_from_sender_type &&s,
                                                Receiver &&receiver) {
        return {std::move(s.predecessor_sender), std::move(s.scheduler), std::forward<Receiver>(receiver)};
    }

    template <typename Receiver>
    friend operation_state<Receiver> tag_invoke(einsums::execution::experimental::connect_t, schedule_from_sender_type const &s,
                                                Receiver &&receiver) {
        return {s.predecessor_sender, s.scheduler, std::forward<Receiver>(receiver)};
    }
};
} // namespace einsums::schedule_from_detail

namespace einsums::execution::experimental {
inline constexpr struct schedule_from_t final : einsums::functional::detail::tag_fallback<schedule_from_t> {
  private:
    template <typename Scheduler, typename Sender>
        requires(is_sender_v<Sender>)
    friend constexpr EINSUMS_FORCEINLINE auto tag_fallback_invoke(schedule_from_t, Scheduler &&scheduler, Sender &&predecessor_sender) {
        return schedule_from_detail::schedule_from_sender<Sender, Scheduler>{std::forward<Sender>(predecessor_sender),
                                                                             std::forward<Scheduler>(scheduler)};
    }
} schedule_from{};
} // namespace einsums::execution::experimental
#endif
