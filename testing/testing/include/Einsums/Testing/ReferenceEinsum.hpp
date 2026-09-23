//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file ReferenceEinsum.hpp
/// @brief A deliberately naive einsum, for tests to check the real ones against.
///
/// The library has two einsum engines, and a test that compares one of them with the other, or
/// the string engine's eager path with its own graph replay, checks an engine against itself. This
/// is the oracle instead: nested loops over every assignment of the index letters, element access
/// through the strides, no fast paths, no BLAS, and its own spec parser. It shares no code with
/// either engine, so a bug in one of them cannot hide behind the same code here. It is slow on
/// purpose; keep the tensors small.
///
/// Semantics, for C = c_pf * C + ab_pf * sum A * B:
///
/// - A letter that appears in C is a free index; every other letter is summed.
/// - A letter repeated within one input takes that input's diagonal.
/// - A letter present in only one input and absent from C is summed over that input alone.
/// - A zero extent is valid: a summed zero extent makes the sum empty, so C = c_pf * C; a free zero
///   extent leaves C with no elements.
/// - c_pf == 0 overwrites C without reading it, as BLAS does with beta == 0.
///
/// Specs are written as ``"ij <- ik ; kj"`` or ``"ik ; kj -> ij"``. An operand's letters are single
/// characters, or comma-separated names for multi-character indices (``"mu,nu <- mu,rho ; rho,nu"``).
/// Operators such as ``conj(...)`` or the antisymmetrizers are not part of the grammar; pass
/// conjugation as the flags.

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <algorithm>
#include <cctype>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(testing)

/// The index names of the three operands of one spec, as the reference parses it.
struct ReferenceSpec {
    std::vector<std::string> c, a, b;
};

namespace detail {

inline std::string_view trim(std::string_view text) {
    auto const first = text.find_first_not_of(" \t\n");
    if (first == std::string_view::npos) {
        return {};
    }
    auto const last = text.find_last_not_of(" \t\n");
    return text.substr(first, last - first + 1);
}

inline std::vector<std::string> parse_operand(std::string_view text, std::string_view spec) {
    text = trim(text);
    std::vector<std::string> names;
    if (text.empty()) {
        return names;
    }
    auto check = [&](std::string_view name) {
        if (name.empty()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "reference_einsum: empty index name in '{}'", spec);
        }
        for (char const ch : name) {
            if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_')) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "reference_einsum: '{}' in '{}' is not an index name; operators are not supported", ch, spec);
            }
        }
        names.emplace_back(name);
    };
    if (text.find(',') != std::string_view::npos) {
        while (true) {
            auto const comma = text.find(',');
            check(trim(text.substr(0, comma)));
            if (comma == std::string_view::npos) {
                break;
            }
            text = text.substr(comma + 1);
        }
    } else {
        for (char const ch : text) {
            if (ch != ' ' && ch != '\t') {
                check(std::string_view(&ch, 1));
            }
        }
    }
    return names;
}

template <typename T>
T maybe_conj(T value, bool conjugate) {
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        return conjugate ? std::conj(value) : value;
    } else {
        return value;
    }
}

} // namespace detail

/// Parse a spec into the index names of C, A and B.
inline ReferenceSpec parse_reference_spec(std::string_view spec) {
    std::string_view output, inputs;
    if (auto const at = spec.find("<-"); at != std::string_view::npos) {
        output = spec.substr(0, at);
        inputs = spec.substr(at + 2);
    } else if (auto const at2 = spec.find("->"); at2 != std::string_view::npos) {
        inputs = spec.substr(0, at2);
        output = spec.substr(at2 + 2);
    } else {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "reference_einsum: '{}' has no '<-' or '->'", spec);
    }
    auto const semicolon = inputs.find(';');
    if (semicolon == std::string_view::npos || inputs.find(';', semicolon + 1) != std::string_view::npos) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "reference_einsum: '{}' needs exactly two inputs separated by ';'", spec);
    }
    return ReferenceSpec{.c = detail::parse_operand(output, spec),
                         .a = detail::parse_operand(inputs.substr(0, semicolon), spec),
                         .b = detail::parse_operand(inputs.substr(semicolon + 1), spec)};
}

/// C = c_pf * C + ab_pf * contract(A, B) over the rank-erased tensors, by brute force.
template <typename T>
void reference_einsum(std::string_view spec, T c_pf, einsums::detail::TensorImpl<T> *C, T ab_pf, einsums::detail::TensorImpl<T> const &A,
                      einsums::detail::TensorImpl<T> const &B, bool conj_a = false, bool conj_b = false) {
    ReferenceSpec const parsed = parse_reference_spec(spec);

    if (parsed.a.size() != A.rank() || parsed.b.size() != B.rank()) {
        EINSUMS_THROW_EXCEPTION(RankError, "reference_einsum: '{}' names {} axes of A (rank {}) and {} of B (rank {})", spec,
                                parsed.a.size(), A.rank(), parsed.b.size(), B.rank());
    }
    // A spec with no output letters writes one element: a rank-0 C, or a rank-1 C of extent one.
    bool const scalar_output = parsed.c.empty();
    if (scalar_output ? C->size() != 1 : parsed.c.size() != C->rank()) {
        EINSUMS_THROW_EXCEPTION(RankError, "reference_einsum: '{}' names {} axes of C (rank {})", spec, parsed.c.size(), C->rank());
    }

    // Every distinct name, with its extent, checked for agreement wherever it appears.
    std::vector<std::string> names;
    std::vector<size_t>      extents;
    auto id_of  = [&](std::string const &name) -> size_t { return static_cast<size_t>(std::ranges::find(names, name) - names.begin()); };
    auto record = [&](std::vector<std::string> const &operand, einsums::detail::TensorImpl<T> const &tensor, char const *which) {
        for (size_t axis = 0; axis < operand.size(); ++axis) {
            size_t const id = id_of(operand[axis]);
            if (id == names.size()) {
                names.push_back(operand[axis]);
                extents.push_back(tensor.dim(static_cast<int>(axis)));
            } else if (extents[id] != tensor.dim(static_cast<int>(axis))) {
                EINSUMS_THROW_EXCEPTION(DimensionError, "reference_einsum: '{}' has extent {} in {} but {} elsewhere", operand[axis],
                                        tensor.dim(static_cast<int>(axis)), which, extents[id]);
            }
        }
    };
    record(parsed.a, A, "A");
    record(parsed.b, B, "B");
    size_t const input_names = names.size();
    for (size_t axis = 0; axis < parsed.c.size(); ++axis) {
        if (std::ranges::count(parsed.c, parsed.c[axis]) != 1) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "reference_einsum: '{}' repeats '{}' in C", spec, parsed.c[axis]);
        }
    }
    if (!scalar_output) {
        record(parsed.c, *C, "C");
    }
    if (names.size() != input_names) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "reference_einsum: '{}' has an output index in neither input", spec);
    }

    std::vector<size_t> free_ids, summed_ids;
    for (auto const &name : parsed.c) {
        free_ids.push_back(id_of(name));
    }
    for (size_t id = 0; id < names.size(); ++id) {
        if (std::ranges::find(free_ids, id) == free_ids.end()) {
            summed_ids.push_back(id);
        }
    }

    auto ids_of = [&](std::vector<std::string> const &operand) {
        std::vector<size_t> ids;
        for (auto const &name : operand) {
            ids.push_back(id_of(name));
        }
        return ids;
    };
    std::vector<size_t> const a_ids = ids_of(parsed.a), b_ids = ids_of(parsed.b), c_ids = ids_of(parsed.c);

    std::vector<size_t> value(names.size(), 0);
    auto                offset = [&](std::vector<size_t> const &ids, einsums::detail::TensorImpl<T> const &tensor) {
        size_t off = 0;
        for (size_t axis = 0; axis < ids.size(); ++axis) {
            off += value[ids[axis]] * tensor.stride(static_cast<int>(axis));
        }
        return off;
    };
    // Advance the letters in `ids` like an odometer; false once every assignment has been seen.
    auto advance = [&](std::vector<size_t> const &ids) {
        for (size_t k = ids.size(); k-- > 0;) {
            if (++value[ids[k]] < extents[ids[k]]) {
                return true;
            }
            value[ids[k]] = 0;
        }
        return false;
    };
    auto any_empty = [&](std::vector<size_t> const &ids) { return std::ranges::any_of(ids, [&](size_t id) { return extents[id] == 0; }); };

    if (any_empty(free_ids)) {
        return;
    }
    bool const empty_sum = any_empty(summed_ids);
    do {
        T sum{0};
        if (!empty_sum) {
            // Only the summed letters restart; the free ones say which element of C this is.
            for (size_t const id : summed_ids) {
                value[id] = 0;
            }
            do {
                sum += detail::maybe_conj(A.data()[offset(a_ids, A)], conj_a) * detail::maybe_conj(B.data()[offset(b_ids, B)], conj_b);
            } while (advance(summed_ids));
        }
        T &out = C->data()[scalar_output ? 0 : offset(c_ids, *C)];
        out    = (c_pf == T{0}) ? ab_pf * sum : c_pf * out + ab_pf * sum;
    } while (advance(free_ids));
}

/// The same, for any tensor type that exposes its TensorImpl through ``impl()``.
template <typename CType, typename AType, typename BType>
    requires requires(CType &c, AType const &a, BType const &b) {
        c.impl();
        a.impl();
        b.impl();
    }
void reference_einsum(std::string_view spec, typename AType::ValueType c_pf, CType *C, typename AType::ValueType ab_pf, AType const &A,
                      BType const &B, bool conj_a = false, bool conj_b = false) {
    reference_einsum(spec, c_pf, &C->impl(), ab_pf, A.impl(), B.impl(), conj_a, conj_b);
}

/// C = contract(A, B), overwriting C.
template <typename CType, typename AType, typename BType>
    requires requires(CType &c, AType const &a, BType const &b) {
        c.impl();
        a.impl();
        b.impl();
    }
void reference_einsum(std::string_view spec, CType *C, AType const &A, BType const &B) {
    using T = typename AType::ValueType;
    reference_einsum(spec, T{0}, &C->impl(), T{1}, A.impl(), B.impl());
}

EINSUMS_NAMESPACE_END(testing)
