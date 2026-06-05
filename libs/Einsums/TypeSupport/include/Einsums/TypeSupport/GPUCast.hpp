//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace einsums {

/**
 * @struct HipCast<To, From>
 *
 * @brief Allows casting between host types and device types.
 *
 * To cast between types, use <tt>HipCast<To, From>::cast(param)</tt>.
 *
 * Valid types are @c float , @c double , @c complex<float> , @c complex<double> , @c hipFloatComplex , and @c hipDoubleComplex .
 *
 * @tparam To The type to cast to.
 * @tparam From The type to cast from.
 *
 * @versionadded{1.0.0}
 */
template <typename To, typename From>
struct HipCast {
    /**
     * Perform the cast.
     *
     * @param[in] from The value to cast.
     *
     * @return The cast value.
     *
     * @versionadded{1.0.0}
     * @versionchangeddesc{2.0.0}
     *      Added the cast function to the base cast struct for documentation purposes.
     * @endversion
     */
    __host__ __device__ static inline To cast(From from) { return To(from); }
};

template <>
struct HipCast<float, float> {
    /**
     * Cast between types.
     */
    __host__ __device__ static inline float cast(float from) { return from; }
};

template <>
struct HipCast<double, float> {
    __host__ __device__ static inline double cast(float from) { return (double)from; }
};

template <>
struct HipCast<std::complex<float>, float> {
    __host__ static inline std::complex<float> cast(float from) { return std::complex<float>(from); }
};

template <>
struct HipCast<std::complex<double>, float> {
    __host__ static inline std::complex<double> cast(float from) { return std::complex<double>((double)from); }
};

template <>
struct HipCast<hipFloatComplex, float> {
    __host__ __device__ static inline hipFloatComplex cast(float from) { return make_hipFloatComplex(from, 0.0f); }
};

template <>
struct HipCast<hipDoubleComplex, float> {
    __host__ __device__ static inline hipDoubleComplex cast(float from) { return make_hipDoubleComplex((double)from, 0.0); }
};

template <>
struct HipCast<float, double> {
    __host__ __device__ static inline float cast(double from) { return (float)from; }
};

template <>
struct HipCast<double, double> {
    __host__ __device__ static inline double cast(double from) { return from; }
};

template <>
struct HipCast<std::complex<float>, double> {
    __host__ static inline std::complex<float> cast(double from) { return std::complex<float>((float)from); }
};

template <>
struct HipCast<std::complex<double>, double> {
    __host__ static inline std::complex<double> cast(double from) { return std::complex<double>(from); }
};

template <>
struct HipCast<hipFloatComplex, double> {
    __host__ __device__ static inline hipFloatComplex cast(double from) { return make_hipFloatComplex((float)from, 0.0f); }
};

template <>
struct HipCast<hipDoubleComplex, double> {
    __host__ __device__ static inline hipDoubleComplex cast(double from) { return make_hipDoubleComplex(from, 0.0); }
};

template <>
struct HipCast<float, std::complex<float>> {
    __host__ static inline float cast(std::complex<float> from) { return from.real(); }
};

template <>
struct HipCast<double, std::complex<float>> {
    __host__ static inline double cast(std::complex<float> from) { return (double)from.real(); }
};

template <>
struct HipCast<std::complex<float>, std::complex<float>> {
    __host__ static inline std::complex<float> cast(std::complex<float> from) { return from; }
};

template <>
struct HipCast<std::complex<double>, std::complex<float>> {
    __host__ static inline std::complex<double> cast(std::complex<float> from) {
        return std::complex<double>((double)from.real(), (double)from.imag());
    }
};

template <>
struct HipCast<hipFloatComplex, std::complex<float>> {
    __host__ static inline hipFloatComplex cast(std::complex<float> from) { return *reinterpret_cast<hipFloatComplex *>(&from); }
};

template <>
struct HipCast<hipDoubleComplex, std::complex<float>> {
    __host__ static inline hipDoubleComplex cast(std::complex<float> from) {
        return make_hipDoubleComplex((double)from.real(), (double)from.imag());
    }
};

template <>
struct HipCast<float, std::complex<double>> {
    __host__ static inline float cast(std::complex<double> from) { return (float)from.real(); }
};

template <>
struct HipCast<double, std::complex<double>> {
    __host__ static inline double cast(std::complex<double> from) { return from.real(); }
};

template <>
struct HipCast<std::complex<float>, std::complex<double>> {
    __host__ static inline std::complex<float> cast(std::complex<double> from) {
        return std::complex<float>((float)from.real(), (float)from.imag());
    }
};

template <>
struct HipCast<std::complex<double>, std::complex<double>> {
    __host__ static inline std::complex<double> cast(std::complex<double> from) { return from; }
};

template <>
struct HipCast<hipFloatComplex, std::complex<double>> {
    __host__ static inline hipFloatComplex cast(std::complex<double> from) {
        return make_hipFloatComplex((float)from.real(), (float)from.imag());
    }
};

template <>
struct HipCast<hipDoubleComplex, std::complex<double>> {
    __host__ static inline hipDoubleComplex cast(std::complex<double> from) { return *reinterpret_cast<hipDoubleComplex *>(&from); }
};

template <>
struct HipCast<float, hipFloatComplex> {
    __host__ __device__ static inline float cast(hipFloatComplex from) { return from.x; }
};

template <>
struct HipCast<double, hipFloatComplex> {
    __host__ __device__ static inline double cast(hipFloatComplex from) { return (double)from.x; }
};

template <>
struct HipCast<std::complex<float>, hipFloatComplex> {
    __host__ static inline std::complex<float> cast(hipFloatComplex from) { return *reinterpret_cast<std::complex<float> *>(&from); }
};

template <>
struct HipCast<std::complex<double>, hipFloatComplex> {
    __host__ static inline std::complex<double> cast(hipFloatComplex from) { return std::complex<double>((double)from.x, (double)from.y); }
};

template <>
struct HipCast<hipFloatComplex, hipFloatComplex> {
    __host__ __device__ static inline hipFloatComplex cast(hipFloatComplex from) { return from; }
};

template <>
struct HipCast<hipDoubleComplex, hipFloatComplex> {
    __host__ __device__ static inline hipDoubleComplex cast(hipFloatComplex from) {
        return make_hipDoubleComplex((double)from.x, (double)from.x);
    }
};

template <>
struct HipCast<float, hipDoubleComplex> {
    __host__ __device__ static inline float cast(hipDoubleComplex from) { return (float)from.x; }
};

template <>
struct HipCast<double, hipDoubleComplex> {
    __host__ __device__ static inline double cast(hipDoubleComplex from) { return from.x; }
};

template <>
struct HipCast<std::complex<float>, hipDoubleComplex> {
    __host__ static inline std::complex<float> cast(hipDoubleComplex from) { return std::complex<float>((float)from.x, (float)from.y); }
};

template <>
struct HipCast<std::complex<double>, hipDoubleComplex> {
    __host__ static inline std::complex<double> cast(hipDoubleComplex from) { return *reinterpret_cast<std::complex<double> *>(&from); }
};

template <>
struct HipCast<hipFloatComplex, hipDoubleComplex> {
    __host__ __device__ static inline hipFloatComplex cast(hipDoubleComplex from) {
        return make_hipFloatComplex((float)from.x, (float)from.y);
    }
};

template <>
struct HipCast<hipDoubleComplex, hipDoubleComplex> {
    __host__ __device__ static inline hipDoubleComplex cast(hipDoubleComplex from) { return from; }
};

} // namespace einsums