//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Print.hpp>

#include <cstdio>
#include <iomanip>
#include <mutex>
#include <omp.h>
#include <sstream>
#include <string>
#include <thread>

#if defined(_WIN32) || defined(_WIN64)
#    include <io.h>
#else
#    include <unistd.h>
#endif

namespace einsums {
namespace print {
namespace {
std::mutex      lock;
int             indent_level{0};
std::string     indent_string{};
bool            print_master_thread_id{false};
std::thread::id main_thread_id = std::this_thread::get_id();
bool            suppress{false};

OutputSinkCallback output_sink_{};

/// Strip ANSI escape sequences from a string.
/// Handles CSI sequences (ESC[...X), OSC sequences (ESC]...ST), and other
/// two-character escape sequences (ESC + single char) used by 256-color and truecolor modes.
std::string strip_ansi(std::string const &input) {
    std::string result;
    result.reserve(input.size());
    size_t i = 0;
    while (i < input.size()) {
        if (input[i] == '\033' && i + 1 < input.size()) {
            char next = input[i + 1];
            if (next == '[') {
                // CSI sequence: ESC [ ... <letter>
                // Covers SGR (colors), 256-color (38;5;N), truecolor (38;2;R;G;B), cursor, etc.
                i += 2;
                while (i < input.size() && !std::isalpha(static_cast<unsigned char>(input[i]))) {
                    ++i;
                }
                if (i < input.size())
                    ++i; // skip the terminating letter
            } else if (next == ']') {
                // OSC sequence: ESC ] ... (ST or BEL)
                // ST is ESC backslash (\033\\) or BEL (\007)
                i += 2;
                while (i < input.size()) {
                    if (input[i] == '\007') {
                        ++i;
                        break;
                    }
                    if (input[i] == '\033' && i + 1 < input.size() && input[i + 1] == '\\') {
                        i += 2;
                        break;
                    }
                    ++i;
                }
            } else {
                // Two-character escape sequence (ESC + single character)
                i += 2;
            }
        } else {
            result.push_back(input[i]);
            ++i;
        }
    }
    return result;
}

} // namespace

void set_output_sink(OutputSinkCallback cb) {
    std::lock_guard guard(lock);
    output_sink_ = std::move(cb);
}

void clear_output_sink() {
    std::lock_guard guard(lock);
    output_sink_ = {};
}

void update_indent_string() {
    indent_string.clear();
    if (indent_level > 0)
        indent_string.insert(0, indent_level, ' ');
}

void indent() {
    if (omp_get_thread_num() == 0) {
        indent_level += 4;
        update_indent_string();
    }
}

void deindent() {
    if (omp_get_thread_num() == 0) {
        indent_level -= 4;
        if (indent_level < 0)
            indent_level = 0;
        update_indent_string();
    }
}

auto current_indent_level() -> int {
    return indent_level;
}

void always_print_thread_id(bool onoff) {
    print_master_thread_id = onoff;
}

void suppress_output(bool onoff) {
    suppress = onoff;
}

} // namespace print

namespace {

void print_line(std::string const &line) {
    std::string line_header;

    if (omp_in_parallel()) {
        if (omp_get_thread_num() == 0) {
            std::ostringstream oss;
            oss << "[ main #" << std::setw(6) << 0 << " ] ";
            line_header = oss.str();
        } else {
            std::ostringstream oss;
            oss << "[ tid  #" << std::setw(6) << omp_get_thread_num() << " ] ";
            line_header = oss.str();
        }
    }
    line_header.append(print::indent_string);

    std::lock_guard guard(print::lock);
    std::printf("%s", line_header.c_str());
    std::printf("%s\n", line.c_str());
    if (print::output_sink_) {
        print::output_sink_(print::strip_ansi(line_header + line));
    }
}

void fprint_line(std::FILE *fp, std::string const &line) {
    std::string line_header;

    if (omp_in_parallel()) {
        if (omp_get_thread_num() == 0) {
            std::ostringstream oss;
            oss << "[ main #" << std::setw(6) << 0 << " ] ";
            line_header = oss.str();
        } else {
            std::ostringstream oss;
            oss << "[ tid  #" << std::setw(6) << omp_get_thread_num() << " ] ";
            line_header = oss.str();
        }
    }
    line_header.append(print::indent_string);

    std::lock_guard guard(print::lock);
    std::fprintf(fp, "%s", line_header.c_str());
    std::fprintf(fp, "%s\n", line.c_str());
    if ((fp == stdout || fp == stderr) && print::output_sink_) {
        print::output_sink_(print::strip_ansi(line_header + line));
    }
}

void fprint_line(std::ostream &os, std::string const &line) {
    std::string line_header;

    if (omp_in_parallel()) {
        if (omp_get_thread_num() == 0) {
            std::ostringstream oss;
            oss << "[ main #" << std::setw(6) << 0 << " ] ";
            line_header = oss.str();
        } else {
            std::ostringstream oss;
            oss << "[ tid  #" << std::setw(6) << omp_get_thread_num() << " ] ";
            line_header = oss.str();
        }
    }
    line_header.append(print::indent_string);

    std::lock_guard guard(print::lock);
    os << line_header << line << '\n';
    if ((&os == &std::cout || &os == &std::cerr) && print::output_sink_) {
        print::output_sink_(print::strip_ansi(line_header + line));
    }
}

} // namespace

namespace detail {

#if defined(EINSUMS_WINDOWS)
inline bool is_terminal(std::ostream const &os) {
    if (&os == &std::cout)
        return _isatty(_fileno(stdout));
    if (&os == &std::cerr)
        return _isatty(_fileno(stderr));
    return false;
}

bool is_terminal(FILE *file) {
    return __isatty(_fileno(file));
}
#else
bool is_terminal(std::ostream const &os) {
    if (&os == &std::cout)
        return isatty(fileno(stdout));
    if (&os == &std::cerr)
        return isatty(fileno(stderr));
    return false;
}

bool is_terminal(FILE *file) {
    return isatty(fileno(file));
}
#endif

void println(std::string const &str) {
    if (!print::suppress) {
        std::istringstream iss(str);

        for (std::string line; std::getline(iss, line);) {
            print_line(line);
        }
    }
}

void fprintln(std::FILE *fp, std::string const &str) {
    if (!print::suppress) {
        std::istringstream iss(str);

        for (std::string line; std::getline(iss, line);) {
            fprint_line(fp, line);
        }
    }
}

void fprintln(std::ostream &os, std::string const &str) {
    if (!print::suppress) {
        std::istringstream iss(str);

        for (std::string line; std::getline(iss, line);) {
            fprint_line(os, line);
        }
    }
}
} // namespace detail
} // namespace einsums