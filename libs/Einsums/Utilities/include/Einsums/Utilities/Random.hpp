//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <numbers>
#include <random>

namespace einsums {

namespace detail {

/**
 * @struct circle_distribution
 *
 * @brief A uniformly random distribution on a circle with a center and radius.
 *
 * For real numbers, this will give a uniform distribution on an interval. For complex numbers,
 * the distribution will be such that the probability of a point being within a subregion
 * is proportional to the area of that subregion.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
struct CircleDistribution {};

template <>
struct CircleDistribution<float> {
  public:
    CircleDistribution(float center, float radius) : _mag_dist(center - radius, center + radius) {}

    ~CircleDistribution() = default;

    template <typename Generator>
    float operator()(Generator &generator) {
        return _mag_dist(generator);
    }

  private:
    std::uniform_real_distribution<float> _mag_dist;
};

template <>
struct CircleDistribution<double> {
  public:
    CircleDistribution(double center, double radius) : _mag_dist(center - radius, center + radius) {}

    ~CircleDistribution() = default;

    template <typename Generator>
    double operator()(Generator &generator) {
        return _mag_dist(generator);
    }

  private:
    std::uniform_real_distribution<double> _mag_dist;
};

// For this case, we can just use the normal uniform distribution. The boundary of the region will not be included.
template <typename T>
struct CircleDistribution<std::complex<T>> {
  public:
    CircleDistribution(std::complex<T> center, T radius)
        : _center{center}, _mag_dist(0, radius), _angle_dist(0, 2 * std::numbers::pi_v<T>) {}

    ~CircleDistribution() = default;

    template <typename Generator>
    std::complex<T> operator()(Generator &generator) {
        T mag = _mag_dist(generator), angle = _angle_dist(generator);

        return std::complex<T>{mag * std::cos(angle), mag * std::sin(angle)} + _center;
    }

  private:
    std::complex<T>                   _center;
    std::uniform_real_distribution<T> _mag_dist, _angle_dist;
};

/**
 * @struct unit_circle_distribution
 *
 * @brief A uniformly random distribution on the unit circle.
 *
 * For real numbers, this will give a uniform distribution on (-1, 1). The endpoints are not included.
 * For complex numbers, the distribution will be such that the probability of a point being within a subregion
 * is proportional to the area of that subregion. The region will be the unit disc without its boundary.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
struct UnitCircleDistribution {};

template <>
struct UnitCircleDistribution<float> {
  public:
    UnitCircleDistribution() : _mag_dist(-1.0f, 1.0f) {}

    ~UnitCircleDistribution() = default;

    template <typename Generator>
    float operator()(Generator &generator) {
        return _mag_dist(generator);
    }

  private:
    std::uniform_real_distribution<float> _mag_dist;
};

template <>
struct UnitCircleDistribution<double> {
  public:
    UnitCircleDistribution() : _mag_dist(-1.0, 1.0) {}

    ~UnitCircleDistribution() = default;

    template <typename Generator>
    double operator()(Generator &generator) {
        return _mag_dist(generator);
    }

  private:
    std::uniform_real_distribution<double> _mag_dist;
};

// For this case, we can just use the normal uniform distribution. The boundary of the region will not be included.
template <typename T>
struct UnitCircleDistribution<std::complex<T>> {
  public:
    UnitCircleDistribution() : _mag_dist(0, 1), _angle_dist(0, 2 * std::numbers::pi_v<T>) {}

    ~UnitCircleDistribution() = default;

    template <typename Generator>
    std::complex<T> operator()(Generator &generator) {
        T mag = _mag_dist(generator), angle = _angle_dist(generator);

        return std::complex<T>{std::cos(angle), std::sin(angle)};
    }

  private:
    std::uniform_real_distribution<T> _mag_dist, _angle_dist;
};

} // namespace detail

/**
 * @property random_engine
 *
 * @brief The global random engine for random number generation.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT extern std::default_random_engine random_engine;

/**
 * @brief Set the seed of the random number generator.
 *
 * @param seed The new seed for the random number generator.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void seed_random(std::default_random_engine::result_type seed);

} // namespace einsums
