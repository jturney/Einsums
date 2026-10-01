//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Transposes that keep the fastest index stream B when beta is zero: the one-dimensional and
// two-dimensional axpy paths and the constant-stride-1 path each write contiguous runs through
// stripes::stream_store_span, whose misaligned head, streamed middle and partial tail are
// three separate code paths. These cases reach each transpose path directly, at every start
// offset within a vector and at lengths that leave every tail, and check that every element of
// B's buffer outside the transposed region keeps its sentinel.

#include <Einsums/HPTT/HPTT.hpp>

#include <cstddef>
#include <vector>

#include <Einsums/Testing.hpp>

namespace {

/// One transpose, described the way HPTT takes it: column-major, the extents of A, the
/// permutation (B's dimension d is A's dimension perm[d]), padded outer sizes and offsets.
struct Case {
    std::vector<int>    perm;
    std::vector<size_t> size;
    std::vector<size_t> outer_a;
    std::vector<size_t> offset_a;
    std::vector<size_t> outer_b;
    std::vector<size_t> offset_b;
};

std::vector<size_t> strides_of(std::vector<size_t> const &outer) {
    std::vector<size_t> s(outer.size(), 1);
    for (size_t d = 1; d < outer.size(); ++d)
        s[d] = s[d - 1] * outer[d - 1];
    return s;
}

size_t volume(std::vector<size_t> const &outer) {
    size_t v = 1;
    for (auto x : outer)
        v *= x;
    return v;
}

/// Run @p c with B's buffer starting @p shift elements past a fresh allocation and compare the
/// whole buffer against a brute-force transpose that leaves the sentinel everywhere else.
template <typename T>
void check(Case const &c, size_t shift, int threads) {
    int const  rank     = static_cast<int>(c.perm.size());
    T const    alpha    = T(2.5);
    T const    sentinel = T(-777);
    auto const sa       = strides_of(c.outer_a);
    auto const sb       = strides_of(c.outer_b);

    std::vector<T> A(volume(c.outer_a));
    for (size_t e = 0; e < A.size(); ++e)
        A[e] = T(static_cast<double>(e % 97) * 0.25 + 1.0);

    std::vector<T> storage(volume(c.outer_b) + shift, sentinel);
    T             *B = storage.data() + shift;

    std::vector<T>      expected(storage);
    std::vector<size_t> idx(static_cast<size_t>(rank), 0);
    for (bool more = true; more;) {
        size_t oa = 0, ob = 0;
        for (int d = 0; d < rank; ++d)
            oa += (c.offset_a[d] + idx[d]) * sa[d];
        for (int d = 0; d < rank; ++d) {
            auto const from = static_cast<size_t>(c.perm[d]);
            ob += (c.offset_b[d] + idx[from]) * sb[d];
        }
        expected[shift + ob] = alpha * A[oa];
        more                 = false;
        for (int d = 0; d < rank; ++d) {
            if (++idx[d] < c.size[d]) {
                more = true;
                break;
            }
            idx[d] = 0;
        }
    }

    auto plan = einsums::hptt::create_plan(c.perm, rank, alpha, A.data(), c.size, c.outer_a, c.offset_a, T(0), B, c.outer_b, c.offset_b,
                                           einsums::hptt::ESTIMATE, threads);
    plan->execute();

    size_t mismatches = 0;
    for (size_t e = 0; e < storage.size(); ++e)
        mismatches += storage[e] != expected[e] ? 1 : 0;
    CAPTURE(shift, threads);
    REQUIRE(mismatches == 0);
}

template <typename T>
void check_all_shifts(Case const &c) {
    // One whole vector of start offsets, whatever the widest rung is.
    for (size_t shift = 0; shift <= 64 / sizeof(T); ++shift) {
        for (int const threads : {1, 4}) {
            check<T>(c, shift, threads);
        }
    }
}

} // namespace

TEMPLATE_TEST_CASE("HPTT streams the one-dimensional path", "[hptt][stream]", float, double) {
    // Lengths below one vector, around a vector and a cache line, and long enough to split
    // across threads with a head and tail in every share.
    for (size_t n : {1, 3, 7, 8, 16, 17, 33, 1001, 4099}) {
        CAPTURE(n);
        check_all_shifts<TestType>({{0}, {n}, {n}, {0}, {n}, {0}});
    }
    // An identity with no padding fuses to one dimension as well.
    check_all_shifts<TestType>({{0, 1, 2}, {5, 7, 9}, {5, 7, 9}, {0, 0, 0}, {5, 7, 9}, {0, 0, 0}});
    // Offsets into padded buffers move the run off the allocation's alignment.
    check_all_shifts<TestType>({{0}, {37}, {45}, {5}, {41}, {3}});
}

TEMPLATE_TEST_CASE("HPTT streams the two-dimensional path", "[hptt][stream]", float, double) {
    // B's padded leading dimension keeps the two dimensions apart and starts every column at a
    // different alignment.
    for (size_t n0 : {1, 5, 16, 19, 67}) {
        CAPTURE(n0);
        check_all_shifts<TestType>({{0, 1}, {n0, 13}, {n0, 13}, {0, 0}, {n0 + 3, 13}, {0, 0}});
    }
    check_all_shifts<TestType>({{0, 1}, {23, 11}, {29, 14}, {2, 1}, {27, 12}, {3, 1}});
}

TEMPLATE_TEST_CASE("HPTT streams the constant-stride-1 path", "[hptt][stream]", float, double) {
    // The fastest index stays; the other two swap.
    for (size_t n0 : {1, 6, 16, 21, 130}) {
        CAPTURE(n0);
        check_all_shifts<TestType>({{0, 2, 1}, {n0, 9, 11}, {n0, 9, 11}, {0, 0, 0}, {n0, 11, 9}, {0, 0, 0}});
    }
    check_all_shifts<TestType>({{0, 2, 1}, {19, 6, 8}, {23, 7, 9}, {1, 1, 0}, {21, 10, 7}, {2, 1, 0}});
}
