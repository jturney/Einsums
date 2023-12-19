//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/program_options/Config.hpp>

#include <string>
#include <vector>

// clang-format off
#include <einsums/config/WarningsPrefix.hpp>
// clang-format on

namespace einsums::program_options {

/** Describes positional options.

    The class allows to guess option names for positional options, which
    are specified on the command line and are identified by the position.
    The class uses the information provided by the user to associate a name
    with every positional option, or tell that no name is known.

    The primary assumption is that only the relative order of the
    positional options themselves matters, and that any interleaving
    ordinary options don't affect interpretation of positional options.

    The user initializes the class by specifying that first N positional
    options should be given the name X1, following M options should be given
    the name X2 and so on.
*/
class EINSUMS_EXPORT PositionalOptionsDescriptor {

  public:
    PositionalOptionsDescriptor();

    /** Species that up to 'max_count' next positional options
        should be given the 'name'. The value of '-1' means 'unlimited'.
        No calls to 'add' can be made after call with 'max_value' equal to
        '-1'.
    */
    PositionalOptionsDescriptor &add(char const *name, int max_count);

    /** Returns the maximum number of positional options that can
        be present. Can return (numeric_limits<unsigned>::max)() to
        indicate unlimited number. */
    [[nodiscard]] auto max_total_count() const -> unsigned;

    /** Returns the name that should be associated with positional
        options at 'position'.
        Precondition: position < max_total_count()
    */
    [[nodiscard]] auto name_for_position(unsigned position) const -> std::string const &;

  private:
    // List of names corresponding to the positions. If the number of
    // positions is unlimited, then the last name is stored in
    // m_trailing;
    std::vector<std::string> _names;
    std::string              _trailing;
};

} // namespace einsums::program_options

#include <einsums/config/WarningsSuffix.hpp>
