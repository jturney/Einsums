//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Transposes whose fastest index has a non-unit stride, on A, on B, or both: the inner stride.
//
// The traversal names each loop's role (A's stride-1 index, B's, or neither) when it builds the
// plan. The scalar remainder once recovered the role from the loop's stride instead, taking a
// unit lda or ldb to mean the stride-1 index. With an inner stride that stride is the inner
// stride, so the remainder ran B's stride-1 loop as an outer loop at the full block width: past
// the end of B, and over elements other tiles had already written. The extents here leave a
// scalar remainder at each vector width, and every element of B's buffer outside the transposed
// region must keep its sentinel.

#include <Einsums/HPTT/HPTT.hpp>

#include <complex>
#include <cstddef>
#include <tuple>
#include <vector>

#include <Einsums/Testing.hpp>

namespace {

/// B = alpha * permute(A), column-major; B's axis d is A's axis perm[d]. Each operand's axes are
/// laid out as outer sizes times its inner stride, which is how HPTT describes them.
template <typename T>
void check(std::vector<size_t> const &size, std::vector<int> const &perm, size_t pad_b, size_t inner_a, size_t inner_b, int threads) {
    int const    rank     = static_cast<int>(size.size());
    T const      alpha    = T(2);
    T const      sentinel = T(-777);
    size_t const guard    = 64;

    std::vector<size_t> outer_a = size, outer_b(static_cast<size_t>(rank)), zero(static_cast<size_t>(rank), 0);
    for (int d = 0; d < rank; ++d)
        outer_b[d] = size[static_cast<size_t>(perm[d])];
    outer_b[0] += pad_b;

    std::vector<size_t> sa(static_cast<size_t>(rank)), sb(static_cast<size_t>(rank));
    sa[0] = inner_a;
    sb[0] = inner_b;
    for (int d = 1; d < rank; ++d) {
        sa[d] = sa[d - 1] * outer_a[d - 1];
        sb[d] = sb[d - 1] * outer_b[d - 1];
    }
    std::vector<T> A(sa[rank - 1] * outer_a[rank - 1]);
    std::vector<T> B(sb[rank - 1] * outer_b[rank - 1] + guard, sentinel);
    for (size_t e = 0; e < A.size(); ++e)
        A[e] = T(static_cast<double>(e % 89) + 1.0);

    auto plan = einsums::hptt::create_plan<T>(perm.data(), rank, alpha, A.data(), size.data(), outer_a.data(), zero.data(), inner_a, T(0),
                                              B.data(), outer_b.data(), zero.data(), inner_b, einsums::hptt::ESTIMATE, threads);
    plan->execute();

    // Every index of A, and where it lands in B.
    std::vector<char>   written(B.size(), 0);
    std::vector<size_t> idx(static_cast<size_t>(rank), 0);
    size_t              total = 1;
    for (auto s : size)
        total *= s;
    for (size_t n = 0; n < total; ++n) {
        size_t rest = n, a_off = 0, b_off = 0;
        for (int d = 0; d < rank; ++d) {
            idx[d] = rest % size[d];
            rest /= size[d];
            a_off += idx[d] * sa[d];
        }
        for (int d = 0; d < rank; ++d)
            b_off += idx[static_cast<size_t>(perm[d])] * sb[d];
        written[b_off] = 1;
        REQUIRE(B[b_off] == alpha * A[a_off]);
    }
    for (size_t o = 0; o < B.size(); ++o) {
        if (!written[o]) {
            CAPTURE(o);
            REQUIRE(B[o] == sentinel);
        }
    }
}

} // namespace

TEMPLATE_TEST_CASE("HPTT - inner strides on either operand", "[HPTT]", float, double, std::complex<float>, std::complex<double>) {
    // Extents that leave a scalar remainder at every vector width, a pad on B's leading
    // dimension, and the 2 x 3 x 4 cyclic permutation whose fused form first showed the bug.
    for (auto const &[size, perm] : {std::tuple{std::vector<size_t>{2, 3, 4}, std::vector<int>{2, 0, 1}},
                                     std::tuple{std::vector<size_t>{19, 5, 21}, std::vector<int>{2, 0, 1}},
                                     std::tuple{std::vector<size_t>{37, 23}, std::vector<int>{1, 0}}}) {
        for (size_t inner_a : {size_t{1}, size_t{2}}) {
            for (size_t inner_b : {size_t{1}, size_t{2}, size_t{3}}) {
                for (size_t pad_b : {size_t{0}, size_t{1}}) {
                    for (int threads : {1, 4}) {
                        CAPTURE(size, perm, inner_a, inner_b, pad_b, threads);
                        check<TestType>(size, perm, pad_b, inner_a, inner_b, threads);
                    }
                }
            }
        }
    }
}
