//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <ios>

namespace einsums::detail {

// this is taken from the Boost.Io library
class ios_flags_saver {
  public:
    using state_type  = ::std::ios_base;
    using aspect_type = ::std::ios_base::fmtflags;

    explicit ios_flags_saver(state_type &s) : _s_save(s), _a_save(s.flags()) {}
    ios_flags_saver(state_type &s, aspect_type const &a) : _s_save(s), _a_save(s.flags(a)) {}

    ~ios_flags_saver() { restore(); }

    ios_flags_saver(ios_flags_saver const &)            = delete;
    ios_flags_saver &operator=(ios_flags_saver const &) = delete;

    void restore() { _s_save.flags(_a_save); }

  private:
    state_type       &_s_save;
    aspect_type const _a_save;
};
} // namespace einsums::detail