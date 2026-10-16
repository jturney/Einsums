//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file MaintainedViews.hpp
/// @brief Which view objects a graph's View node re-seats on every replay.
///
/// A View node keeps its view in a holder the graph owns and re-emplaces it on each replay, so
/// the object's address is stable while its data pointer, extents and strides are not. A copy of
/// one taken at capture holds whatever its parent had then, which for a tensor the graph declares
/// is only a placeholder. @ref Graph::adopt_operand therefore never makes a stand-in for one,
/// whichever graph is capturing: a loop body that names a view its parent made binds the live
/// object, as the parent itself does.

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <memory>
#include <optional>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// @brief Record @p view as maintained by a View node until @ref forget_maintained_view.
/// @param[in] view The holder's view object.
/// @param[in] life The holder's liveness token, which outlives every instance the holder emplaces.
EINSUMS_EXPORT void note_maintained_view(void const *view, std::weak_ptr<void> life);

/// @brief Stop recording @p view; called when the holder that maintains it is destroyed.
EINSUMS_EXPORT void forget_maintained_view(void const *view) noexcept;

/// @brief Whether a View node currently maintains the object at @p view.
[[nodiscard]] EINSUMS_EXPORT bool is_maintained_view(void const *view);

/// @brief The liveness token of the holder that maintains @p view, or nothing when none does.
///
/// A View node re-emplaces its view on every replay, and each instance publishes a token of its
/// own that expires with it. The holder's token is the one that says whether the view still
/// exists, which is what a handle's liveness check and a by-address lookup need to ask.
[[nodiscard]] EINSUMS_EXPORT std::optional<std::weak_ptr<void>> maintained_view_life(void const *view);

/// @brief Records one holder's view for exactly the holder's lifetime.
///
/// A member of each view holder: the holder marks its view once it first holds one, and the
/// record goes when the holder does, so an address a later object reuses is never mistaken for a
/// maintained view.
class MaintainedViewMark {
  public:
    MaintainedViewMark()                                      = default;
    MaintainedViewMark(MaintainedViewMark const &)            = delete;
    MaintainedViewMark &operator=(MaintainedViewMark const &) = delete;
    MaintainedViewMark(MaintainedViewMark &&)                 = delete;
    MaintainedViewMark &operator=(MaintainedViewMark &&)      = delete;
    ~MaintainedViewMark() {
        if (_view != nullptr) {
            forget_maintained_view(_view);
        }
    }

    /// @brief Record @p view, the holder's view object, under the holder's token @p life; once per holder.
    void mark(void const *view, std::weak_ptr<void> life) {
        _view = view;
        note_maintained_view(view, std::move(life));
    }

  private:
    void const *_view{nullptr};
};

EINSUMS_NAMESPACE_END(compute_graph::detail)
