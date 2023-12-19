//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/program_options/CmdLine.hpp>
#include <einsums/program_options/Config.hpp>
#include <einsums/program_options/Errors.hpp>
#include <einsums/program_options/Option.hpp>
#include <einsums/program_options/OptionsDescription.hpp>
#include <einsums/program_options/PositionalOptions.hpp>

#include <functional>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include <einsums/config/WarningsPrefix.hpp>
// clang-format on

namespace einsums::program_options::detail {

/** Command line parser class. Main requirements were:
    - Powerful enough to support all common uses.
    - Simple and easy to learn/use.
    - Minimal code size and external dependencies.
    - Extensible for custom syntaxes.

    First all options are registered. After that, elements of command line
    are extracted using operator++.

    For each element, user can find
    - if it's an option or an argument
    - name of the option
    - index of the option
    - option value(s), if any

    Sometimes the registered option name is not equal to the encountered
    one, for example, because name abbreviation is supported.  Therefore
    two option names can be obtained:
    - the registered one
    - the one found at the command line

    There are lot of style options, which can be used to tune the command
    line parsing. In addition, it's possible to install additional parser
    which will process custom option styles.

    @todo minimal match length for guessing?
*/
class EINSUMS_EXPORT CmdLine {
  public:
    using Style = ::einsums::program_options::CommandLineStyle;

    using AdditionalParser = std::function<std::pair<std::string, std::string>(std::string const &)>;

    using StyleParser = std::function<std::vector<Option>(std::vector<std::string> &)>;

    /** Constructs a command line parser for (argc, argv) pair. Uses
     * style options passed in 'style', which should be binary or'ed values
     * of style_t enum. It can also be zero, in which case a "default"
     * style will be used. If 'allow_unregistered' is true, then allows
     * unregistered options. They will be assigned index 1 and are
     * assumed to have optional parameter.
     */
    CmdLine(std::vector<std::string> const &args);
};

} // namespace einsums::program_options::detail

#include <einsums/config/WarningsSuffix.hpp>
