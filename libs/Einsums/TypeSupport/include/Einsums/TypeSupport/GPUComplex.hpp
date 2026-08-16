//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <type_traits>

EINSUMS_NAMESPACE_BEGIN(gpu_ops)

__device__ inline float conj(float x) {
    return x;
}
__device__ inline double conj(double x) {
    return x;
}
__device__ inline hipFloatComplex conj(hipFloatComplex x) {
    return make_hipFloatComplex(x.x, -x.y);
}
__device__ inline hipDoubleComplex conj(hipDoubleComplex x) {
    return make_hipDoubleComplex(x.x, -x.y);
}

__device__ inline float fma(float x, float y, float z) {
    return ::fmaf(x, y, z);
}

__device__ inline double fma(double x, float y, double z) {
    return ::fma(x, y, z);
}

__device__ inline double fma(float x, double y, double z) {
    return ::fma(x, y, z);
}

__device__ inline double fma(double x, double y, double z) {
    return ::fma(x, y, z);
}

__device__ inline hipFloatComplex fma(hipFloatComplex x, float y, hipFloatComplex z) {
    return make_hipFloatComplex(::fmaf(x.x, y, z.x), x.y + z.y);
}

__device__ inline hipFloatComplex fma(hipFloatComplex x, double y, hipFloatComplex z) {
    return make_hipFloatComplex(::fmaf(x.x, y, z.x), x.y + z.y);
}

__device__ inline hipFloatComplex fma(float x, hipFloatComplex y, hipFloatComplex z) {
    return make_hipFloatComplex(::fmaf(x, y.x, z.x), y.y + z.y);
}

__device__ inline hipFloatComplex fma(double x, hipFloatComplex y, hipFloatComplex z) {
    return make_hipFloatComplex(::fmaf(x, y.x, z.x), y.y + z.y);
}

__device__ inline hipFloatComplex fma(hipFloatComplex x, hipFloatComplex y, hipFloatComplex z) {
    return make_hipFloatComplex(::fmaf(-x.y, y.y, ::fmaf(x.x, y.x, z.x)), ::fmaf(x.y, y.x, ::fmaf(x.x, y.y, z.y)));
}

__device__ inline hipDoubleComplex fma(hipDoubleComplex x, float y, hipDoubleComplex z) {
    return make_hipDoubleComplex(::fma(x.x, y, z.x), x.y + z.y);
}

__device__ inline hipDoubleComplex fma(hipDoubleComplex x, double y, hipDoubleComplex z) {
    return make_hipDoubleComplex(::fma(x.x, y, z.x), x.y + z.y);
}

__device__ inline hipDoubleComplex fma(hipDoubleComplex x, hipFloatComplex y, hipDoubleComplex z) {
    return make_hipDoubleComplex(::fma(-x.y, y.y, ::fma(x.x, y.x, z.x)), ::fma(x.y, y.x, ::fma(x.x, y.y, z.y)));
}

__device__ inline hipDoubleComplex fma(float x, hipDoubleComplex y, hipDoubleComplex z) {
    return make_hipDoubleComplex(::fma(x, y.x, z.x), y.y + z.y);
}

__device__ inline hipDoubleComplex fma(double x, hipDoubleComplex y, hipDoubleComplex z) {
    return make_hipDoubleComplex(::fma(x, y.x, z.x), y.y + z.y);
}

__device__ inline hipDoubleComplex fma(hipFloatComplex x, hipDoubleComplex y, hipDoubleComplex z) {
    return make_hipDoubleComplex(::fma(-x.y, y.y, ::fma(x.x, y.x, z.x)), ::fma(x.y, y.x, ::fma(x.x, y.y, z.y)));
}

__device__ inline hipDoubleComplex fma(hipDoubleComplex x, hipDoubleComplex y, hipDoubleComplex z) {
    return make_hipDoubleComplex(::fma(-x.y, y.y, ::fma(x.x, y.x, z.x)), ::fma(x.y, y.x, ::fma(x.x, y.y, z.y)));
}

__device__ inline hipFloatComplex mult(hipFloatComplex x, hipFloatComplex y) {
    return make_hipFloatComplex(::fmaf(-x.y, y.y, x.x * y.x), ::fmaf(x.y, y.x, x.x * y.y));
}

__device__ inline hipDoubleComplex mult(hipDoubleComplex x, hipFloatComplex y) {
    return make_hipDoubleComplex(::fma(-x.y, y.y, x.x * y.x), ::fma(x.y, y.x, x.x * y.y));
}

__device__ inline hipDoubleComplex mult(hipFloatComplex x, hipDoubleComplex y) {
    return make_hipDoubleComplex(::fma(-x.y, y.y, x.x * y.x), ::fma(x.y, y.x, x.x * y.y));
}

__device__ inline hipDoubleComplex mult(hipDoubleComplex x, hipDoubleComplex y) {
    return make_hipDoubleComplex(::fma(-x.y, y.y, x.x * y.x), ::fma(x.y, y.x, x.x * y.y));
}

template <typename T1, typename T2>
__device__ inline auto mult(T1 x, T2 y) -> std::conditional_t<(sizeof(T1) >= sizeof(T2)), T1, T2> {
    return x * y;
}

__device__ inline hipFloatComplex div(hipFloatComplex x, hipFloatComplex y) {
    float const denom = y.x * y.x + y.y * y.y;
    return make_hipFloatComplex(::fmaf(x.y, y.y, x.x * y.x) / denom, ::fmaf(x.y, y.x, -x.x * y.y) / denom);
}

__device__ inline hipDoubleComplex div(hipDoubleComplex x, hipFloatComplex y) {
    double const denom = y.x * y.x + y.y * y.y;
    return make_hipDoubleComplex(::fmaf(x.y, y.y, x.x * y.x) / denom, ::fmaf(x.y, y.x, -x.x * y.y) / denom);
}

__device__ inline hipDoubleComplex div(hipFloatComplex x, hipDoubleComplex y) {
    double const denom = y.x * y.x + y.y * y.y;
    return make_hipDoubleComplex(::fmaf(x.y, y.y, x.x * y.x) / denom, ::fmaf(x.y, y.x, -x.x * y.y) / denom);
}

__device__ inline hipDoubleComplex div(hipDoubleComplex x, hipDoubleComplex y) {
    double const denom = y.x * y.x + y.y * y.y;
    return make_hipDoubleComplex(::fmaf(x.y, y.y, x.x * y.x) / denom, ::fmaf(x.y, y.x, -x.x * y.y) / denom);
}

__device__ inline hipFloatComplex div(float x, hipFloatComplex y) {
    float const denom = y.x * y.x + y.y * y.y;
    return make_hipFloatComplex((x * y.x) / denom, (-x * y.y) / denom);
}

__device__ inline hipFloatComplex div(double x, hipFloatComplex y) {
    double const denom = y.x * y.x + y.y * y.y;
    return make_hipFloatComplex((x * y.x) / denom, (-x * y.y) / denom);
}

__device__ inline hipDoubleComplex div(float x, hipDoubleComplex y) {
    double const denom = y.x * y.x + y.y * y.y;
    return make_hipDoubleComplex((x * y.x) / denom, (-x * y.y) / denom);
}

__device__ inline hipDoubleComplex div(double x, hipDoubleComplex y) {
    double const denom = y.x * y.x + y.y * y.y;
    return make_hipDoubleComplex((x * y.x) / denom, (-x * y.y) / denom);
}

template <typename T1, typename T2>
__device__ inline auto div(T1 x, T2 y) -> std::conditional_t<(sizeof(T1) >= sizeof(T2)), T1, T2> {
    return x / y;
}

EINSUMS_NAMESPACE_END(gpu_ops)