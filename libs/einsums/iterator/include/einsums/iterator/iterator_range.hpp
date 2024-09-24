//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>
#include <einsums/iterator/range.hpp>
#include <einsums/iterator/traits/is_iterator.hpp>
#include <einsums/iterator/traits/is_range.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace einsums::util {
    template <typename Iterator, typename Sentinel = Iterator>
    class iterator_range
    {
    public:
        iterator_range() = default;

        EINSUMS_HOST_DEVICE iterator_range(Iterator iterator, Sentinel sentinel)
          : _iterator(EINSUMS_MOVE(iterator))
          , _sentinel(EINSUMS_MOVE(sentinel))
        {
        }

        EINSUMS_HOST_DEVICE Iterator begin() const { return _iterator; }

        EINSUMS_HOST_DEVICE Iterator end() const { return _sentinel; }

        EINSUMS_HOST_DEVICE std::ptrdiff_t size() const { return std::distance(_iterator, _sentinel); }

        EINSUMS_HOST_DEVICE bool empty() const { return _iterator == _sentinel; }

    private:
        Iterator _iterator;
        Sentinel _sentinel;
    };

    template <typename Range, typename Iterator = typename traits::range_iterator<Range>::type,
        typename Sentinel = typename traits::range_iterator<Range>::type>
    typename std::enable_if<traits::is_range<Range>::value,
        iterator_range<Iterator, Sentinel>>::type
    make_iterator_range(Range& r)
    {
        return iterator_range<Iterator, Sentinel>(util::begin(r), util::end(r));
    }

    template <typename Range,
        typename Iterator = typename traits::range_iterator<Range const>::type,
        typename Sentinel = typename traits::range_iterator<Range const>::type>
    typename std::enable_if<traits::is_range<Range>::value,
        iterator_range<Iterator, Sentinel>>::type
    make_iterator_range(Range const& r)
    {
        return iterator_range<Iterator, Sentinel>(util::begin(r), util::end(r));
    }

    template <typename Iterator, typename Sentinel = Iterator>
    typename std::enable_if<traits::is_iterator<Iterator>::value,
        iterator_range<Iterator, Sentinel>>::type
    make_iterator_range(Iterator iterator, Sentinel sentinel)
    {
        return iterator_range<Iterator, Sentinel>(iterator, sentinel);
    }
}    // namespace einsums::util

namespace einsums::ranges {
    template <typename I, typename S = I>
    using subrange_t = einsums::util::iterator_range<I, S>;
}    // namespace einsums::ranges
