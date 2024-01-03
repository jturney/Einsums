//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

namespace einsums::backend::linear_algebra::netlib {
auto xerbla(const char *srname, int *info) -> int;
auto lsame(const char *ca, const char *cb) -> long int;
} // namespace einsums::backend::linear_algebra::netlib