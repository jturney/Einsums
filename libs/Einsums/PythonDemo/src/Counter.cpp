//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/PythonDemo/Counter.hpp>

#include <utility>

namespace einsums::pythondemo {

Counter::Counter() = default;

Counter::Counter(long start, std::string label) : _value(start), _label(std::move(label)) {
}

long Counter::bump() {
    ++_value;
    return _value;
}

long Counter::bump_by(long delta) {
    _value += delta;
    return _value;
}

void Counter::reset() {
    _value = 0;
}

long Counter::get_value() const {
    return _value;
}

std::string const &Counter::get_label() const {
    return _label;
}

long sum_of_squares(long n) {
    long total = 0;
    for (long i = 1; i <= n; ++i) {
        total += i * i;
    }
    return total;
}

} // namespace einsums::pythondemo
