//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file
/// The member loop the grouped element-wise nodes run under.
///
/// Shared between the capture entry points in `Operations.hpp` and the
/// executor builder, for the reason every other shared executor piece is
/// shared: a node a region rewrite emitted has to run the members the way the
/// captured node ran them, and two copies of the loop would drift on the one
/// property that matters here, which is whether it threads.

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <exception>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// Run @p member over every index as one OpenMP team, carrying the first
/// exception out by hand: one may not cross a region boundary.
///
/// @param[in] count  How many members to run.
/// @param[in] member The body, taking the member index.
template <typename F>
void run_grouped_members(std::size_t count, F &&member) {
    // A run of one IS the call the grouped form replaces, and forking a team
    // for it costs more than the member does. Worth the branch because a gated
    // capture is full of them: a conditional over one entity still wants the
    // grouped spelling, so that the ungated capture beside it can be the same
    // emitter with a longer list.
    if (count == 1) {
        member(std::size_t{0});
        return;
    }
    std::exception_ptr first;
    EINSUMS_OMP_PRAGMA(parallel for schedule(dynamic))
    for (std::size_t i = 0; i < count; i++) {
        try {
            member(i);
        } catch (...) {
            EINSUMS_OMP_PRAGMA(critical(grouped_elementwise_failure))
            if (!first) {
                first = std::current_exception();
            }
        }
    }
    if (first) {
        std::rethrow_exception(first);
    }
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
