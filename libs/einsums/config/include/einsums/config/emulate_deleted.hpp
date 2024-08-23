//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// Marks a class as non-copyable and non-movable.
#define EINSUMS_NON_COPYABLE(cls)                                                                                                          \
    cls(cls const &)            = delete;                                                                                                  \
    cls(cls &&)                 = delete;                                                                                                  \
    cls &operator=(cls const &) = delete;                                                                                                  \
    cls &operator=(cls &&)      = delete /**/
