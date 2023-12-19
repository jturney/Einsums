//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/program_options/Config.hpp>

namespace einsums::program_options {

/** Various possible styles of options.
 *
 * There are "long" options, which start with "--" and "short",
 * which start with either "-" or "/". Both kinds can be allowed or
 * disallowed, see allow_long and allow_short. The allowed character
 * for short options is also configurable.
 *
 * Option's value can be specified in the same token as name
 * ("--foo=bar"), or in the next token.
 *
 * It's possible to introduce long options by the same character as
 * short options, see allow_long_disguise.
 *
 * Finally, guessing (specifying only prefix of option) and case
 * insensitive processing are supported.
 */

enum class CommandLineStyle {
    /// Allow "--long_name" style
    AllowLong = 1,
    /// Allow "-<single character" style
    AllowShort = AllowLong << 1,
    /// Allow "-" in short options
    AllowDashForShort = AllowShort << 1,
    /// Allow "/" in short options
    AllowSlashForShort = AllowDashForShort << 1,
    /** Allow option parameter in the same token for long option, like in
     * @verbatim
     * --foo=10
     * @endverbatim
     */
    LongAllowAdjacent = AllowSlashForShort << 1,
    /// Alow option parameter in the next token for long options.
    LongAllowNext = LongAllowAdjacent << 1,
    /// Allow option parameter in the same token for short options.
    ShortAllowAdjacent = LongAllowNext << 1,
    /// Allow option parameter in the next token for short options.
    ShortAllowNext = ShortAllowAdjacent << 1,
    /** Allow to merge several short options together so that "-s -k" becomes "-sk". All of the options but last should accept no parameter.
       For example, if "-s" accepts a paramter, then "k" wil be taken as a parameter, not another short option. Dos-style short options
       cannot be stick. */
    AllowSticky = ShortAllowNext << 1,
    /** Alow abbreviated spellings for long options, if they unambiguously identify long option. No long option name should be prefix or
       other long option name if guessing is in effect. */
    AllowGuessing = AllowSticky << 1,
    /// Ignore the difference in case for long options.
    LongCaseInsensitive = AllowGuessing << 1,
    /// Ignore the difference in case for short options.
    ShortCaseInsensitive = LongCaseInsensitive << 1,
    /// Ignore the difference in case for all options
    CaseInsensitive = (LongCaseInsensitive | ShortCaseInsensitive),
    /// Allow long options with single option starting character, e.g. <tt>-foo=10</tt>
    AllowLongDisguise = ShortCaseInsensitive << 1,
    /// The more-or-less traditional unix style.
    UnixStyle = (AllowShort | ShortAllowAdjacent | ShortAllowNext | AllowLong | LongAllowAdjacent | LongAllowNext | AllowSticky |
                 AllowGuessing | AllowDashForShort),
    /// The default style
    DefaultStyle = UnixStyle
};

} // namespace einsums::program_options

#include <einsums/program_options/detail/CmdLine.hpp>