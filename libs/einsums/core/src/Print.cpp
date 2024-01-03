//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/Print.hpp"

#include "einsums/OpenMP.hpp"
#include "einsums/Timer.hpp"
#if defined(EINSUMS_IN_PARALLEL)
#    include "einsums/parallel/MPI.hpp"
#endif

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>

#if defined(_WIN32) || defined(_WIN64)
#    include <io.h>
#else
#    include <unistd.h>
#endif

namespace print {

namespace {

std::mutex      lock;
int             indent_level{0};
std::string     indent_string{};
bool            print_master_thread_id{false};
std::thread::id main_thread_id = std::this_thread::get_id();
bool            suppress{false};

auto printing_to_screen() -> bool {
#if defined(_WIN32) || defined(_WIN64)
    return _isatty(_fileno(stdout));
#else
    return isatty(fileno(stdout));
#endif
}

} // namespace

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
#if defined(EINSUMS_IN_PARALLEL)
    einsums::mpi::ErrorOr<int> size = einsums::mpi::rank();
    int                        rank = size.value();
#else
    constexpr int rank = 0;
#endif

    if (rank == 0) {
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

        std::lock_guard<std::mutex> guard(print::lock);
        std::printf("%s", line_header.c_str());
        std::printf("%s", line.c_str());
        std::printf("\n");
    }
}

} // namespace

namespace detail {
void println(std::string const &str) {
    if (print::suppress == false) {
        std::istringstream iss(str);

        for (std::string line; std::getline(iss, line);) {
            print_line(line);
        }
    }
}
} // namespace detail
