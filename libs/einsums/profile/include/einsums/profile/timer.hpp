// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <fmt/format.h>

#include <chrono>
#include <string>

namespace einsums::profile::timer {

using clock      = std::chrono::high_resolution_clock;
using time_point = std::chrono::time_point<clock>;
using duration   = std::chrono::high_resolution_clock::duration;

void EINSUMS_EXPORT initialize();
void EINSUMS_EXPORT finalize();

void EINSUMS_EXPORT report();
void EINSUMS_EXPORT report(const std::string &filename);
void EINSUMS_EXPORT report(std::FILE *fp);
void EINSUMS_EXPORT report(std::ostream &os);

namespace detail {
void EINSUMS_EXPORT push(std::string name);
}

template <typename... Ts>
void push(const std::string_view &f, const Ts... ts) {
    std::string s = fmt::format(fmt::runtime(f), ts...);
    detail::push(s);
}

void EINSUMS_EXPORT pop();
void EINSUMS_EXPORT pop(duration elapsed);

struct timer {
  private:
    time_point start;

  public:
    explicit timer(const std::string &name) {
        start = clock::now();
        push(name);
    }

    ~timer() {
        auto difference = clock::now() - start;
        pop(difference);
    }
};

} // namespace einsums::profile::timer