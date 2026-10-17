//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file StandIns.hpp
/// @brief Which caller's tensor each of a graph's stand-ins stands in for.
///
/// @ref Graph::adopt_operand gives a graph a stand-in for an operand it does not own, so the
/// caller may destroy its own tensor before the graph runs. A pass that rewrites the graph after
/// that must capture over the stand-in, since it is the only one of the two still alive, and
/// capture identifies a tensor by the CALLER's address. Without this record a capture over a
/// stand-in would register it as a tensor nobody had seen, under a new id, with no order against
/// the nodes that read or write the original.
///
/// The record is process-wide rather than per graph because a pass captures into the loop bodies
/// it builds as well as into the graph it rewrites, and a body has to name the tensor its parent
/// names. Each record lasts exactly as long as its stand-in.

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <memory>
#include <optional>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// @brief What a stand-in stands in for.
struct StandIn {
    void                 *identity{nullptr}; ///< The caller's tensor, which is the handle's @ref TensorHandle::tensor_ptr.
    std::weak_ptr<void>   caller_token;      ///< The caller's liveness token, which is the handle's @ref TensorHandle::caller_token.
    std::shared_ptr<void> owner;             ///< The stand-in itself, kept alive by whoever registers it next.
};

/// @brief Record that the object at @p stand_in stands in for @p identity until @ref forget_stand_in.
/// @param[in] stand_in The stand-in object.
/// @param[in] identity The caller's tensor.
/// @param[in] caller_token The caller's liveness token.
/// @param[in] owner The owning reference the graph holds to the stand-in, observed and never extended.
EINSUMS_EXPORT void note_stand_in(void const *stand_in, void *identity, std::weak_ptr<void> caller_token, std::weak_ptr<void> owner);

/// @brief Stop recording @p stand_in; called when the stand-in is destroyed.
EINSUMS_EXPORT void forget_stand_in(void const *stand_in) noexcept;

/// @brief What the object at @p object stands in for, or nothing when it is not a live stand-in.
[[nodiscard]] EINSUMS_EXPORT std::optional<StandIn> stand_in_of(void const *object);

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

    /// @brief Record @p stand_in as standing in for @p identity; once per stand-in.
    void mark(void const *stand_in, void *identity, std::weak_ptr<void> caller_token, std::weak_ptr<void> owner) {
        _stand_in = stand_in;
        note_stand_in(stand_in, identity, std::move(caller_token), std::move(owner));
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
