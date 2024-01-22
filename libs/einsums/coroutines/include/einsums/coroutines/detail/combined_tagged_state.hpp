//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>

#include <cstddef>
#include <cstdint>

namespace einsums::threads::detail {

template <typename T1, typename T2>
struct combined_tagged_state {
  private:
    using tagged_state_type    = std::int64_t;
    using thread_state_type    = std::int8_t;
    using thread_state_ex_type = std::int8_t;
    using tag_type             = std::int64_t;

    static std::size_t const state_shift    = 56; // 8th byte
    static std::size_t const state_ex_shift = 48; // 7th byte

    static tagged_state_type const state_mask    = 0xffull;
    static tagged_state_type const state_ex_mask = 0xffull;

    // (1L << 48L) - 1;
    static tagged_state_type const tag_mask = 0x0000ffffffffffffull;

    static auto extract_tag(tagged_state_type const &i) -> tag_type { return i & tag_mask; }

    static auto extract_state(tagged_state_type const &i) -> thread_state_type {
        return static_cast<thread_state_type>((i >> state_shift) & state_mask);
    }

    static auto extract_state_ex(tagged_state_type const &i) -> thread_state_ex_type {
        return static_cast<thread_state_ex_type>((i >> state_ex_shift) & state_ex_mask);
    }

    static auto pack_state(T1 state_, T2 state_ex_, tag_type tag) -> tagged_state_type {
        auto state    = static_cast<tagged_state_type>(state_);
        auto state_ex = static_cast<tagged_state_type>(state_ex_);

        EINSUMS_ASSERT(!(state & ~state_mask));
        EINSUMS_ASSERT(!(state_ex & ~state_ex_mask));
        EINSUMS_ASSERT(!(state & ~tag_mask));

        return (state << state_shift) | (state_ex << state_ex_shift) | tag;
    }

  public:
    ///////////////////////////////////////////////////////////////////////
    combined_tagged_state() noexcept : _state(0) {}

    combined_tagged_state(T1 state, T2 state_ex, tag_type t = 0) : _state(pack_state(state, state_ex, t)) {}

    combined_tagged_state(combined_tagged_state state, tag_type t)
        : _state(pack_state(state.state(), state.state_ex(), t)) {}

    ///////////////////////////////////////////////////////////////////////
    void set(T1 state, T2 state_ex, tag_type t) { _state = pack_state(state, state_ex, t); }

    ///////////////////////////////////////////////////////////////////////
    auto operator==(combined_tagged_state const &p) const -> bool { return _state == p._state; }

    auto operator!=(combined_tagged_state const &p) const -> bool { return !operator==(p); }

    ///////////////////////////////////////////////////////////////////////
    // state access
    auto state() const -> T1 { return static_cast<T1>(extract_state(_state)); }

    void set_state(T1 state) { _state = pack_state(state, state_ex(), tag()); }

    auto state_ex() const -> T2 { return static_cast<T2>(extract_state_ex(_state)); }

    void set_state_ex(T2 state_ex) { _state = pack_state(state(), state_ex, tag()); }

    ///////////////////////////////////////////////////////////////////////
    // tag access
    [[nodiscard]] tag_type tag() const { return extract_tag(_state); }

    void set_tag(tag_type t) { _state = pack_state(state(), state_ex(), t); }

  protected:
    tagged_state_type _state;
};

} // namespace einsums::threads::detail