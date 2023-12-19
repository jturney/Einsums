//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/Assert.hpp>
#include <einsums/program_options/Config.hpp>
#include <einsums/program_options/PositionalOptions.hpp>

#include <cstddef>
#include <limits>
#include <string>

namespace einsums::program_options {

PositionalOptionsDescriptor::PositionalOptionsDescriptor() = default;

auto PositionalOptionsDescriptor::add(char const *name, int max_count) -> PositionalOptionsDescriptor & {
    EINSUMS_ASSERT(max_count != -1 || _trailing.empty());

    if (max_count == -1)
        _trailing = name;
    else {
        _names.resize(_names.size() + static_cast<std::size_t>(max_count), name);
    }
    return *this;
}

auto PositionalOptionsDescriptor::max_total_count() const -> unsigned {
    return _trailing.empty() ? static_cast<unsigned>(_names.size()) : (std::numeric_limits<unsigned>::max)();
}

auto PositionalOptionsDescriptor::name_for_position(unsigned position) const -> std::string const & {
    EINSUMS_ASSERT(position < max_total_count());

    if (static_cast<std::size_t>(position) < _names.size())
        return _names[static_cast<std::size_t>(position)];

    return _trailing;
}

} // namespace einsums::program_options