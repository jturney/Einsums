//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

namespace einsums {
    struct no_mutex
    {
        void lock() {}

        bool try_lock() { return true; };

        void unlock() {}
    };
}    // namespace einsums
