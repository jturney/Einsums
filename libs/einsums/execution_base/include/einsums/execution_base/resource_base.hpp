//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

namespace einsums::execution::detail {

struct resource_base {
    virtual ~resource_base() = default;
};

} // namespace einsums::execution::detail