//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <memory>

namespace einsums::detail {
///////////////////////////////////////////////////////////////////////////
template <typename Allocator>
struct allocator_deleter {
    template <typename SharedState>
    void operator()(SharedState *state) {
        using traits = std::allocator_traits<Allocator>;
        traits::deallocate(alloc_, state, 1);
    }

    Allocator alloc_;
};
} // namespace einsums::detail
