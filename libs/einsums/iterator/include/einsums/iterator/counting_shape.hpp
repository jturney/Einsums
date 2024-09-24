//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/iterator/counting_iterator.hpp>
#include <einsums/iterator/iterator_range.hpp>
#include <einsums/iterator/range.hpp>
#include <einsums/iterator/traits/is_range.hpp>

namespace einsums::util::detail {

///////////////////////////////////////////////////////////////////////////
template <typename Incrementable>
using counting_shape_type = einsums::util::iterator_range<einsums::util::counting_iterator<Incrementable>>;

EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
template <typename Incrementable>
EINSUMS_HOST_DEVICE inline counting_shape_type<Incrementable> make_counting_shape(Incrementable n) {
    return einsums::util::make_iterator_range(einsums::util::make_counting_iterator(Incrementable(0)),
                                              einsums::util::make_counting_iterator(n));
}
} // namespace einsums::util::detail