// Copyright Vladimir Prus 2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <einsums/assert.hpp>
#include <einsums/program_options/config.hpp>
#include <einsums/program_options/positional_options.hpp>

#include <cstddef>
#include <limits>
#include <string>

namespace einsums::program_options {

positional_options_description::positional_options_description() {
}

positional_options_description &positional_options_description::add(const char *name, int max_count) {
    EINSUMS_ASSERT(max_count != -1 || m_trailing.empty());

    if (max_count == -1)
        m_trailing = name;
    else {
        m_names.resize(m_names.size() + static_cast<std::size_t>(max_count), name);
    }
    return *this;
}

unsigned positional_options_description::max_total_count() const {
    return m_trailing.empty() ? static_cast<unsigned>(m_names.size()) : (std::numeric_limits<unsigned>::max)();
}

const std::string &positional_options_description::name_for_position(unsigned position) const {
    EINSUMS_ASSERT(position < max_total_count());

    if (static_cast<std::size_t>(position) < m_names.size())
        return m_names[static_cast<std::size_t>(position)];

    return m_trailing;
}

} // namespace einsums::program_options
