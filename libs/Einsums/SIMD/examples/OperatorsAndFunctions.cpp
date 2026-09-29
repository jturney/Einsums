//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file OperatorsAndFunctions.cpp
/// @brief Writing Vec arithmetic with operators or with named functions, and what fmadd changes.
///
/// Vec<T> has + - * / and unary -, which call add, sub, mul, div and neg; the two spellings compile
/// to the same instructions and give the same bits. A codebase that wants every vector operation
/// spelled out can define EINSUMS_SIMD_NO_OPERATORS before including the headers, and the
/// operators disappear.
///
/// One thing operators cannot express is a fused multiply-add. a * b + c rounds the product and
/// then the sum; fmadd(a, b, c) rounds once. The difference is easiest to see in the rounding error
/// of a product itself: with p = a * b rounded, a * b - p is zero by construction, while
/// fmadd(a, b, -p) computes the exact product before subtracting, so it returns the error p made.
/// A backend without FMA hardware computes fmadd as a multiply and an add, and gets zero too.
///
/// Compilers can fuse on their own: GCC turns a * b - p into a fused instruction by default, which
/// is why CMakeLists.txt builds this file with -ffp-contract=off under GCC. Code whose results
/// must not depend on the compiler's choice should call fmadd where it wants fusion and build with
/// contraction off.

#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/Operations.hpp>

#include <cmath>
#include <iostream>

using namespace einsums::simd;

#if defined(EINSUMS_SIMD_HAVE_FMA) || defined(__aarch64__) || defined(_M_ARM64)
constexpr bool fused = true;
#else
constexpr bool fused = false;
#endif

namespace {

/// sqrt(x^2 + y^2) / 2 - x with operators...
Vec<double> with_operators(Vec<double> x, Vec<double> y) {
    return sqrt(x * x + y * y) / broadcast(2.0) - x;
}

/// ...and with the named functions: the same instructions, so the same bits.
Vec<double> with_functions(Vec<double> x, Vec<double> y) {
    return sub(div(sqrt(add(mul(x, x), mul(y, y))), broadcast(2.0)), x);
}

} // namespace

int einsums_main() {
    constexpr int L = lanes<double>;
    double        a[L], b[L], ops[L], fns[L], naive[L], err[L];
    for (int i = 0; i < L; ++i) {
        a[i] = 1.0 / (3.0 + i); // not exact in binary, so products carry a rounding error
        b[i] = 7.0 / (11.0 + i);
    }
    Vec<double> const va = loadu(a), vb = loadu(b);

    storeu(ops, with_operators(va, vb));
    storeu(fns, with_functions(va, vb));

    Vec<double> const p = va * vb;      // the rounded product
    storeu(naive, va * vb - p);         // always zero
    storeu(err, fmadd(va, vb, neg(p))); // the rounding error p made, when fused

    int failures = 0;
    for (int i = 0; i < L; ++i) {
        double const exact_err = fused ? std::fma(a[i], b[i], -(a[i] * b[i])) : 0.0;
        std::cout << "a * b = " << a[i] * b[i] << ": a*b - p = " << naive[i] << ", fmadd(a, b, -p) = " << err[i] << "\n";
        failures += ops[i] != fns[i];
        failures += naive[i] != 0.0;
        failures += err[i] != exact_err;
    }
    std::cout << (fused ? "fmadd is fused on this build" : "fmadd is a multiply and an add on this build") << "\n";
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
