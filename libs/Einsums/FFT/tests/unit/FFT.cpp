//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/FFT.hpp>

#include <Einsums/Testing.hpp>

// Some elements of these tests come from Intel MKL
// examples/c/dft/sources/basic_dp_real_dft_1d.c

namespace {
template <typename T>
auto moda(int K, int L, int M) -> T {
    return static_cast<T>(((static_cast<long long>(K) * L) % M));
}

template <typename T>
void init_data(einsums::Tensor<T, 1> &data, int N, int H) {
    T   TWOPI = 6.2831853071795864769, phase, factor;
    int n;

    factor = (2 * (N - H) % N == 0) ? 1.0 : 2.0;
    for (n = 0; n < N; n++) {
        phase   = moda<T>(n, H, N) / N;
        data(n) = factor * cos(TWOPI * phase) / N;
    }
}

// Initialize array data to produce unit peak at y(H)
template <typename T>
void init_data(einsums::Tensor<std::complex<T>, 1> &data, int N, int H) {
    double const TWOPI = 6.2831853071795864769;
    double       phase;
    int          n;

    for (n = 0; n < N / 2 + 1; n++) {
        phase = static_cast<double>(moda<T>(n, H, N) / N);
        data(n).real(cos(TWOPI * phase) / N);
        data(n).imag(-sin(TWOPI * phase) / N);
    }
}

template <einsums::NotComplex T, int N = 6, int H = -1>
auto verify_r(einsums::Tensor<T, 1> &x) -> void {
    T   err, errthr, maxerr;
    int n;

    errthr = T{2.5} * log(static_cast<T>(N)) / std::numbers::ln2_v<T> * std::numeric_limits<T>::epsilon();
    maxerr = 0.0;
    for (n = 0; n < N; n++) {
        T re_exp = 0.0, re_got;

        if ((n - H) % N == 0)
            re_exp = 1.0;

        re_got = x(n);
        err    = abs(re_got - re_exp);
        if (err > maxerr)
            maxerr = err;
        // println("re_exp {}, re_got {}, err {}, errthr {}", re_exp, re_got, err, errthr);
        REQUIRE((err < errthr));
    }
}

template <typename Source, einsums::NotComplex Result, int N = 6, int H = -1>
auto ifft1d_1() -> void {
    using namespace einsums;

    auto x_data   = create_tensor<Source>("sample data", einsums::IsComplexV<Source> ? N / 2 + 1 : N);
    auto x_result = create_tensor<Result>("FFT result", N);

    init_data(x_data, N, H);

    // println(x_data);

    fft::ifft(x_data, &x_result);

    // println(x_result);

    verify_r(x_result);
}

template <einsums::Complex Source, einsums::Complex Result, int N = 7, int H = -N / 2>
auto ifft1d_1() -> void {
    using namespace einsums;
    using SourceBase = einsums::RemoveComplexT<Source>;
    using ResultBase = einsums::RemoveComplexT<Result>;

    auto x_data   = create_tensor<Source>("Sample data", N);
    auto x_result = create_tensor<Result>("FFT result", N);

    // Initialize data
    {
        double const TWOPI = 6.2831853071795864769;
        double       phase;
        int          n;

        for (n = 0; n < N; n++) {
            phase = static_cast<double>(moda<SourceBase>(n, -H, N) / N);
            x_data(n).real(cos(TWOPI * phase) / N);
            x_data(n).imag(sin(TWOPI * phase) / N);
        }
    }

    fft::ifft(x_data, &x_result);

    // Verify that x has unit peak at H
    {
        ResultBase err, errthr, maxerr;
        int        n;

        errthr = 5.0 * log((ResultBase)N) / log(2.0) * std::numeric_limits<ResultBase>::epsilon();
        maxerr = 0.0;

        for (n = 0; n < N; n++) {
            ResultBase re_exp = 0.0, im_exp = 0.0, re_got, im_got;

            if ((n - H) % N == 0)
                re_exp = 1.0;

            re_got = x_result(n).real();
            im_got = x_result(n).imag();
            err    = abs(re_got - re_exp) + abs(im_got - im_exp);
            if (err > maxerr)
                maxerr = err;

            // println("n {} {} -- {} {}, {} {}", n, (n - H) % N, re_got, re_exp, im_got, im_exp);
            REQUIRE((err < errthr));
        }
    }
}

template <einsums::NotComplex Source, typename Result, int N = 6, int H = -1>
auto fft1d_1() -> void {
    using namespace einsums;

    auto x_data   = create_tensor<Source>("sample data", N);
    auto x_result = create_tensor<Result>("FFT result", einsums::IsComplexV<Source> ? N : N / 2 + 1);

    init_data(x_data, N, H);

    // println(x_data);

    fft::fft(x_data, &x_result);

    // println(x_result);

    // The result of each forward transformation is a tensor with
    // exactly one 1.0 and everything else is 0.0.
    for (int n = 0; n < x_result.dim(0); n++) {
        einsums::RemoveComplexT<Result> re_exp = 0.0, im_exp = 0.0;

        // The position of the expected 1.0 is dependent on the Source data type.
        if constexpr (!einsums::IsComplexV<Source>) {
            if ((n - H) % N == 0 || (-n - H) % N == 0)
                re_exp = 1.0;
        } else {
            if (n == x_result.dim(0) - 1)
                re_exp = 1.0;
        }

        REQUIRE_THAT(x_result(n).real(), Catch::Matchers::WithinAbsMatcher(einsums::RemoveComplexT<Result>(re_exp), 0.0001));
        REQUIRE_THAT(x_result(n).imag(), Catch::Matchers::WithinAbsMatcher(einsums::RemoveComplexT<Result>(im_exp), 0.0001));
    }
}

template <einsums::Complex Source, einsums::Complex Result, int N = 7, int H = -N / 2>
auto fft1d_1() -> void {
    using namespace einsums;
    using SourceBase = einsums::RemoveComplexT<Source>;
    using ResultBase = einsums::RemoveComplexT<Result>;

    auto x_data   = create_tensor<Source>("Sample Data", N);
    auto x_result = create_tensor<Result>("Result", N);

    // Initialize data
    {
        SourceBase TWOPI = 6.2831853071795864769, phase;
        int        n;

        for (n = 0; n < N; n++) {
            phase = moda<SourceBase>(n, H, N) / N;
            x_data(n).real(cos(TWOPI * phase) / N);
            x_data(n).imag(sin(TWOPI * phase) / N);
        }
    }

    // Call the underlying fft routine
    fft::fft(x_data, &x_result);

    // println(x_data);
    // println(x_result);

    // Verify the results
    {
        ResultBase err, errthr, maxerr;
        int        n;

        errthr = 5.0 * log((ResultBase)N) / log(2.0) * std::numeric_limits<ResultBase>::epsilon();

        maxerr = 0.0;
        for (n = 0; n < N; n++) {
            ResultBase re_exp = 0.0, im_exp = 0.0, re_got, im_got;

            if ((n - H) % N == 0)
                re_exp = 1.0;

            re_got = x_result(n).real();
            im_got = x_result(n).imag();

            err = abs(re_got - re_exp) + abs(im_got - im_exp);
            if (err > maxerr)
                maxerr = err;

            REQUIRE((err < errthr));
        }
    }
}

} // namespace
TEST_CASE("fft1") {
    fft1d_1<float, std::complex<float>>();
}
TEST_CASE("fft2") {
    fft1d_1<double, std::complex<double>>();
}
TEST_CASE("fft3") {
    fft1d_1<double, std::complex<double>, 10>();
}
TEST_CASE("fft4") {
    fft1d_1<std::complex<float>, std::complex<float>>();
}
TEST_CASE("fft5") {
    fft1d_1<std::complex<double>, std::complex<double>>();
}

TEST_CASE("ifft1") {
    ifft1d_1<std::complex<float>, float>();
}
TEST_CASE("ifft2") {
    ifft1d_1<std::complex<double>, double>();
}
TEST_CASE("ifft3") {
    ifft1d_1<std::complex<float>, std::complex<float>>();
}
TEST_CASE("ifft4") {
    ifft1d_1<std::complex<double>, std::complex<double>>();
}