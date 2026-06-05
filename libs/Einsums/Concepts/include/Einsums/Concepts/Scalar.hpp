//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

namespace einsums {

template <typename T>

concept Scalar = requires(T a) {
    T{0};
    T{1};
    a + a;
    a *a;
};

} // namespace einsums