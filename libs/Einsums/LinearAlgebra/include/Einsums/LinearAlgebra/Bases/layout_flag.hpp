//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN(linear_algebra::detail)

/// The transpose flag column-major BLAS needs for a matrix stored in the other order.
///
/// Such a matrix's memory holds its transpose as BLAS sees it, so 'n' and 't' trade places. A 'c'
/// would need the conjugate without the transpose, which BLAS has no flag for: that returns 0, and
/// the caller takes its loop kernel. A real matrix's 'c' is a 't'. The flag must be lower case.
template <typename T>
char flipped_layout_flag(char flag) {
    if (flag == 'n') {
        return 't';
    }
    if (flag == 'c' && IsComplexV<T>) {
        return 0;
    }
    return 'n';
}

EINSUMS_NAMESPACE_END(linear_algebra::detail)
