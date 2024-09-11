//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/init_runtime/init_runtime.hpp>
#include <einsums/init_runtime/scoped_finalize.hpp>

namespace einsums {
scoped_finalize::~scoped_finalize() {
    einsums::finalize();
}
} // namespace einsums