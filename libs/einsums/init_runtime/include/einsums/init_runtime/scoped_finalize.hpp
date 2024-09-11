//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

namespace einsums {

struct [[nodiscard]] scoped_finalize {
    scoped_finalize() = default;
    EINSUMS_EXPORT ~scoped_finalize();
};

} // namespace einsums