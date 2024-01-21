//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include "einsums/config/defines.hpp"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace einsums::debug::detail {

namespace stack_trace {

EINSUMS_EXPORT auto trace(void **addresses, std::size_t size) -> std::size_t;
EINSUMS_EXPORT void write_symbols(void *const *addresses, std::size_t size, std::ostream &);
EINSUMS_EXPORT auto get_symbol(void *address) -> std::string;
EINSUMS_EXPORT auto get_symbols(void *const *address, std::size_t size) -> std::string;

} // namespace stack_trace

struct backtrace {
  private:
    std::vector<void *> _frames;

  public:
    explicit backtrace(std::size_t frames_no = EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH) {
        if (frames_no == 0)
            return;
        frames_no += 2; // we omit two frames from printing
        _frames.resize(frames_no, nullptr);
        std::size_t size = stack_trace::trace(&_frames.front(), frames_no);
        if (size != 0)
            _frames.resize(size);
    }

    virtual ~backtrace() noexcept = default;

    [[nodiscard]] auto stack_size() const -> std::size_t { return _frames.size(); }

    [[nodiscard]] auto return_address(std::size_t frame_no) const -> void * {
        if (frame_no < stack_size())
            return _frames[frame_no];
        return nullptr;
    }

    void trace_line(std::size_t frame_no, std::ostream &out) const {
        if (frame_no < _frames.size())
            stack_trace::write_symbols(&_frames[frame_no], 1, out);
    }

    [[nodiscard]] auto trace_line(std::size_t frame_no) const -> std::string {
        if (frame_no < _frames.size())
            return stack_trace::get_symbol(_frames[frame_no]);
        return {};
    }

    [[nodiscard]] auto trace() const -> std::string {
        if (_frames.empty())
            return {};
        return stack_trace::get_symbols(&_frames.front(), _frames.size());
    }

    void trace(std::ostream &out) const {
        if (_frames.empty())
            return;
        stack_trace::write_symbols(&_frames.front(), _frames.size(), out);
    }
};

struct trace_manip {
  private:
    backtrace const *_tr;

  public:
    trace_manip(backtrace const *tr) : _tr(tr) {}

    auto write(std::ostream &out) const -> std::ostream & {
        if (_tr)
            _tr->trace(out);
        return out;
    }
};

inline auto operator<<(std::ostream &out, trace_manip const &t) -> std::ostream & {
    return t.write(out);
}

template <typename E>
inline auto trace(E const &e) -> trace_manip {
    auto const *tr = dynamic_cast<backtrace const *>(&e);
    return {tr};
}

inline auto trace(std::size_t frames_no = EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH) {
    return backtrace(frames_no).trace();
}

} // namespace einsums::debug::detail