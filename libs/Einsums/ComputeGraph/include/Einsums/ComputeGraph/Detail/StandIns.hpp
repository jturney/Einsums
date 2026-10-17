//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file StandIns.hpp
/// @brief Which caller's tensor each stand-in stands in for, and which tensors a graph owns.
///
/// @ref Graph::adopt_operand gives a graph a stand-in for an operand it does not own, so the
/// caller may destroy its own tensor before the graph runs. Once it has, the caller's address is
/// free for the next allocation and no longer says which tensor a handle means, so an adopted
/// operand is identified by its stand-in (see @ref TensorHandle::tensor_ptr). Every graph has to
/// agree on that identity: a loop body names the tensor its parent names, and a pass that rewrites
/// a graph captures over the stand-in into the graph and into the bodies it builds. Hence one
/// stand-in per caller tensor, shared by every graph that adopts it while it lives, and a record
/// that lets a capture over a stand-in find the operand it stands in for.
///
/// The records are process-wide rather than per graph because a graph knows nothing of the graphs
/// around it. Each lasts exactly as long as the object it describes.

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <memory>
#include <optional>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// @brief What a stand-in stands in for.
struct StandIn {
    void                 *identity{nullptr}; ///< What a handle for the operand records as @ref TensorHandle::tensor_ptr.
    void                 *caller{nullptr};   ///< The caller's tensor, a handle's @ref TensorHandle::caller_ptr.
    std::weak_ptr<void>   caller_token;      ///< The caller's liveness token, a handle's @ref TensorHandle::caller_token.
    std::shared_ptr<void> owner;             ///< The stand-in itself, kept alive by whoever registers it next.
};

/// @brief Record what the object at @p stand_in stands in for, until @ref forget_stand_in.
/// @param[in] stand_in The stand-in object.
/// @param[in] identity What a handle for the operand records as its identity.
/// @param[in] caller The caller's tensor.
/// @param[in] caller_token The caller's liveness token.
/// @param[in] owner The owning reference the graph holds to the stand-in, observed and never extended.
EINSUMS_EXPORT void note_stand_in(void const *stand_in, void *identity, void *caller, std::weak_ptr<void> caller_token,
                                  std::weak_ptr<void> owner);

/// @brief Stop recording @p stand_in; called when the stand-in is destroyed.
EINSUMS_EXPORT void forget_stand_in(void const *stand_in) noexcept;

/// @brief What the object at @p object stands in for, or nothing when it is not a live stand-in.
[[nodiscard]] EINSUMS_EXPORT std::optional<StandIn> stand_in_of(void const *object);

/// @brief The live stand-in for the caller's tensor at @p caller, or nothing when it has none.
/// @param[in] caller The caller's tensor.
/// @param[in] caller_token Its liveness token. An untracked token never matches: an address alone
///            cannot tell the tensor from one allocated where a dead one was.
[[nodiscard]] EINSUMS_EXPORT std::shared_ptr<void> stand_in_for(void const *caller, std::weak_ptr<void> const &caller_token);

/// @brief Record that a graph owns the tensor at @p tensor, until @ref forget_graph_owned.
EINSUMS_EXPORT void note_graph_owned(void const *tensor);

/// @brief Stop recording @p tensor; called just before its graph destroys it.
EINSUMS_EXPORT void forget_graph_owned(void const *tensor) noexcept;

/// @brief Whether a graph owns the tensor at @p tensor.
///
/// A tensor a graph owns lives as long as that graph, so its address names it for as long as any
/// loop body of the graph can name it, and the address is its identity even where another graph
/// adopts it.
[[nodiscard]] EINSUMS_EXPORT bool is_graph_owned(void const *tensor);

/// @brief Records one stand-in for exactly the stand-in's lifetime.
///
/// A member of the block @ref Graph::adopt_operand allocates, declared after the stand-in so it is
/// destroyed first: the record goes before the object does, so an address a later object reuses is
/// never mistaken for a stand-in.
class StandInMark {
  public:
    StandInMark()                               = default;
    StandInMark(StandInMark const &)            = delete;
    StandInMark &operator=(StandInMark const &) = delete;
    StandInMark(StandInMark &&)                 = delete;
    StandInMark &operator=(StandInMark &&)      = delete;
    ~StandInMark() {
        if (_stand_in != nullptr) {
            forget_stand_in(_stand_in);
        }
    }

    /// @brief Record @p stand_in; once per stand-in. The parameters are those of @ref note_stand_in.
    void mark(void const *stand_in, void *identity, void *caller, std::weak_ptr<void> caller_token, std::weak_ptr<void> owner) {
        _stand_in = stand_in;
        note_stand_in(stand_in, identity, caller, std::move(caller_token), std::move(owner));
    }

  private:
    void const *_stand_in{nullptr};
};

/// @brief A stand-in and its record, allocated together.
/// @tparam Tensor The operand's type.
template <typename Tensor>
struct AdoptedOperand {
    template <typename... Args>
    explicit AdoptedOperand(Args &&...args) : tensor(std::forward<Args>(args)...) {}

    Tensor      tensor;
    StandInMark mark;
};

EINSUMS_NAMESPACE_END(compute_graph::detail)
