//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/execution_base/resource_base.hpp>

namespace einsums::execution::detail {

struct context_base {
    virtual ~context_base() = default;

    virtual resource_base const &resource() const = 0;
};

} // namespace einsums::execution::detail