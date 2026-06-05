//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorBase/SymmetryDescriptor.hpp>

namespace einsums {

SymmetryOp SymmetryOp::group_swap(std::initializer_list<int> g1, std::initializer_list<int> g2, int8_t sign, bool conjugate) {
    auto op  = identity();
    auto it1 = g1.begin();
    auto it2 = g2.begin();
    for (; it1 != g1.end() && it2 != g2.end(); ++it1, ++it2) {
        op.permutation[*it1] = static_cast<int8_t>(*it2);
        op.permutation[*it2] = static_cast<int8_t>(*it1);
    }
    op.sign      = sign;
    op.conjugate = conjugate;
    return op;
}

SymmetryDescriptor SymmetryDescriptor::symmetric_pair(int a, int b) {
    return SymmetryDescriptor{{SymmetryOp::swap(a, b, +1)}};
}

SymmetryDescriptor SymmetryDescriptor::antisymmetric_pair(int a, int b) {
    return SymmetryDescriptor{{SymmetryOp::swap(a, b, -1)}};
}

SymmetryDescriptor SymmetryDescriptor::hermitian_pair(int a, int b) {
    return SymmetryDescriptor{{SymmetryOp::swap(a, b, +1, /*conjugate=*/true)}};
}

SymmetryDescriptor SymmetryDescriptor::anti_hermitian_pair(int a, int b) {
    return SymmetryDescriptor{{SymmetryOp::swap(a, b, -1, /*conjugate=*/true)}};
}

SymmetryDescriptor SymmetryDescriptor::eri_8fold() {
    return SymmetryDescriptor{{
        SymmetryOp::swap(0, 1, +1),
        SymmetryOp::swap(2, 3, +1),
        SymmetryOp::group_swap({0, 1}, {2, 3}, +1),
    }};
}

SymmetryDescriptor SymmetryDescriptor::eri_4fold() {
    return SymmetryDescriptor{{
        SymmetryOp::swap(0, 1, +1),
        SymmetryOp::swap(2, 3, +1),
    }};
}

SymmetryDescriptor SymmetryDescriptor::ccsd_t2() {
    return SymmetryDescriptor{{
        SymmetryOp::swap(0, 1, -1),
        SymmetryOp::swap(2, 3, -1),
    }};
}

} // namespace einsums
