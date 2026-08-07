//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>

#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#if __has_include(<stdfloat>)
#    include <stdfloat>
#endif

EINSUMS_NAMESPACE_BEGIN(gpu)
template <typename T>
constexpr hipDataType get_hip_datatype();

template <>
constexpr hipDataType get_hip_datatype<float>() {
    return HIP_R_32F;
}

template <>
constexpr hipDataType get_hip_datatype<double>() {
    return HIP_R_64F;
}

#ifdef __STDCPP_FLOAT16_T__
template <>
constexpr hipDataType get_hip_datatype<float16_t>() {
    return HIP_R_16F;
}
#endif

template <>
constexpr hipDataType get_hip_datatype<int8_t>() {
    return HIP_R_8I;
}

template <>
constexpr hipDataType get_hip_datatype<std::complex<float>>() {
    return HIP_C_32F;
}

template <>
constexpr hipDataType get_hip_datatype<hipComplex>() {
    return HIP_C_32F;
}

template <>
constexpr hipDataType get_hip_datatype<std::complex<double>>() {
    return HIP_C_64F;
}

template <>
constexpr hipDataType get_hip_datatype<hipDoubleComplex>() {
    return HIP_C_64F;
}

#ifdef __STDCPP_FLOAT16_T__
template <>
constexpr hipDataType get_hip_datatype<std::complex<float16_t>>() {
    return HIP_C_16F;
}
#endif

template <>
constexpr hipDataType get_hip_datatype<std::complex<int8_t>>() {
    return HIP_C_8I;
}

template <>
constexpr hipDataType get_hip_datatype<uint8_t>() {
    return HIP_R_8U;
}

template <>
constexpr hipDataType get_hip_datatype<std::complex<uint8_t>>() {
    return HIP_C_8U;
}

template <>
constexpr hipDataType get_hip_datatype<int32_t>() {
    return HIP_R_32I;
}

template <>
constexpr hipDataType get_hip_datatype<std::complex<int32_t>>() {
    return HIP_C_32I;
}

template <>
constexpr hipDataType get_hip_datatype<uint32_t>() {
    return HIP_R_32U;
}

template <>
constexpr hipDataType get_hip_datatype<std::complex<uint32_t>>() {
    return HIP_C_32U;
}

#ifdef __STDCPP_BFLOAT16_T__
template <>
constexpr hipDataType get_hip_datatype<bfloat16_t>() {
    return HIP_R_16BF;
}

template <>
constexpr hipDataType get_hip_datatype<std::complex<bfloat16_t>>() {
    return HIP_C_16BF;
}
#endif

template <>
constexpr hipDataType get_hip_datatype<int16_t>() {
    return HIP_R_16I;
}

template <>
constexpr hipDataType get_hip_datatype<std::complex<int16_t>>() {
    return HIP_C_16I;
}

template <>
constexpr hipDataType get_hip_datatype<uint16_t>() {
    return HIP_R_16U;
}

template <>
constexpr hipDataType get_hip_datatype<std::complex<uint16_t>>() {
    return HIP_C_16U;
}

template <>
constexpr hipDataType get_hip_datatype<int64_t>() {
    return HIP_R_64I;
}

template <>
constexpr hipDataType get_hip_datatype<std::complex<int64_t>>() {
    return HIP_C_64I;
}

template <>
constexpr hipDataType get_hip_datatype<uint64_t>() {
    return HIP_R_64U;
}

template <>
constexpr hipDataType get_hip_datatype<std::complex<uint64_t>>() {
    return HIP_C_64U;
}

EINSUMS_NAMESPACE_END(gpu)