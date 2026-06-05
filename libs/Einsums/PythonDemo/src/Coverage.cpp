//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/PythonDemo/Coverage.hpp>

#include <utility>

namespace einsums::pythondemo {

Vec3::Vec3() = default;
Vec3::Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {
}

Vec3 Vec3::operator+(Vec3 const &other) const {
    return {x + other.x, y + other.y, z + other.z};
}

Vec3 Vec3::operator*(double scalar) const {
    return {x * scalar, y * scalar, z * scalar};
}

bool Vec3::operator==(Vec3 const &other) const {
    return x == other.x && y == other.y && z == other.z;
}

Resource::Slot::Slot(int v) : _value(v) {
}
int Resource::Slot::value() const {
    return _value;
}

Resource::Resource(std::string label, int slots) : _label(std::move(label)) {
    _slots.reserve(slots);
    for (int i = 0; i < slots; ++i) {
        _slots.emplace_back(i);
    }
}

Resource::Slot &Resource::slot(int i) {
    return _slots.at(i);
}

std::string const &Resource::label() const {
    return _label;
}

void raise_counter_error(std::string const &msg) {
    throw CounterError(msg);
}

Worker::Worker() = default;

long Worker::crunch(long n) {
    long acc = 0;
    for (long i = 1; i <= n; ++i) {
        acc += i;
    }
    return acc;
}

} // namespace einsums::pythondemo
