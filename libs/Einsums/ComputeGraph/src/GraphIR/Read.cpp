//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Read.cpp
/// @brief An ``einsums_graph_ir`` document, read into the intermediate form.
///
/// One of the three passes described in `Common.hpp`, and the reason there are
/// three rather than two: a document is read into the plain `Ir*` form first,
/// collecting EVERY problem, and only a clean read is handed to `Build.cpp`.
/// That split is what lets `validate_graph_ir` report three seeded errors as
/// three messages while `load_graph` stops at the first.
///
/// Every helper here records a problem and returns a harmless default rather
/// than bailing out, which is what lets one pass over a document report every
/// problem it has. Nothing in this file throws on malformed input; the only
/// export is @ref read_document.

#include <Einsums/ComputeGraph/Detail/Json.hpp>
#include <Einsums/ComputeGraph/ElementOps.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "Common.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::graph_ir)

namespace {

void note(Problems &problems, std::string const &path, json::Position where, std::string message) {
    problems.push_back(fmt::format("line {} column {}: {}: {}", where.line, where.column, path, std::move(message)));
}

Object const *as_object(Value const &value, std::string const &path, Problems &problems) {
    if (!value.is_object()) {
        note(problems, path, value.position, fmt::format("expected an object, found {}", value.type_name()));
        return nullptr;
    }
    return &value.as_object();
}

Array const *as_array(Value const &value, std::string const &path, Problems &problems) {
    if (!value.is_array()) {
        note(problems, path, value.position, fmt::format("expected an array, found {}", value.type_name()));
        return nullptr;
    }
    return &value.as_array();
}

/// Consume @p key, reporting its absence.
Value const *field(Object const &object, std::string_view key, std::string const &path, Problems &problems, json::Position where) {
    Value const *value = object.take(key);
    if (value == nullptr) {
        note(problems, path, where, fmt::format("required key '{}' is missing", key));
    }
    return value;
}

/// What @ref read_scalar needs to know about one leaf type: how to recognise it,
/// how to read it, and how a diagnostic names it. Three near-identical readers
/// differing only in that triple is what this replaces.
template <typename T>
struct ScalarLeaf;

template <>
struct ScalarLeaf<std::string> {
    static constexpr std::string_view noun = "a string";
    static bool                       matches(Value const &value) { return value.is_string(); }
    static std::string                read(Value const &value) { return value.as_string(); }
};

template <>
struct ScalarLeaf<std::int64_t> {
    static constexpr std::string_view noun = "an integer";
    static bool                       matches(Value const &value) { return value.is_int(); }
    static std::int64_t               read(Value const &value) { return value.as_int(); }
};

template <>
struct ScalarLeaf<bool> {
    static constexpr std::string_view noun = "a bool";
    static bool                       matches(Value const &value) { return value.is_bool(); }
    static bool                       read(Value const &value) { return value.as_bool(); }
};

/// Consume @p key and read it as a @c T, reporting an absent key and a wrong type.
/// A failure of either kind yields a value-initialized @c T and a problem, so a
/// caller reads on and the load collects every fault rather than the first.
template <typename T>
T read_scalar(Object const &object, std::string_view key, std::string const &path, Problems &problems, json::Position where) {
    Value const *value = field(object, key, path, problems, where);
    if (value == nullptr) {
        return T{};
    }
    if (!ScalarLeaf<T>::matches(*value)) {
        note(problems, fmt::format("{}.{}", path, key), value->position,
             fmt::format("expected {}, found {}", ScalarLeaf<T>::noun, value->type_name()));
        return T{};
    }
    return ScalarLeaf<T>::read(*value);
}

std::string read_string(Object const &object, std::string_view key, std::string const &path, Problems &problems, json::Position where) {
    return read_scalar<std::string>(object, key, path, problems, where);
}

std::int64_t read_int(Object const &object, std::string_view key, std::string const &path, Problems &problems, json::Position where) {
    return read_scalar<std::int64_t>(object, key, path, problems, where);
}

bool read_bool(Object const &object, std::string_view key, std::string const &path, Problems &problems, json::Position where) {
    return read_scalar<bool>(object, key, path, problems, where);
}

/// A tensor's provenance tag, or an empty one when the record carries none.
///
/// OPTIONAL by the compatibility policy: a file written before tags existed has no ``tag`` key
/// and its tensors are untagged, which is exactly what they were. Reading it as anything else
/// would break the golden corpus, and the corpus is right that an added field takes a documented
/// default.
///
/// Every key of the tag object is CONSUMED, including the attribute keys, because an unconsumed
/// key is a load-time error by design: a file carrying a field this build does not understand is
/// a file written by something newer, and reading it silently would be the drift the strict
/// document model exists to prevent.
ProvenanceTag read_provenance_tag(Object const &object, std::string const &path, Problems &problems) {
    ProvenanceTag out;
    Value const  *value = object.take("tag");
    if (value == nullptr || value->is_null()) {
        return out;
    }
    if (!value->is_object()) {
        note(problems, fmt::format("{}.tag", path), value->position, fmt::format("expected an object, found {}", value->type_name()));
        return out;
    }

    Object const     &tag      = value->as_object();
    std::string const tag_path = fmt::format("{}.tag", path);
    out.name                   = read_string(tag, "name", tag_path, problems, value->position);

    if (Value const *attributes = tag.take("attributes"); attributes != nullptr && !attributes->is_null()) {
        if (!attributes->is_object()) {
            note(problems, fmt::format("{}.attributes", tag_path), attributes->position,
                 fmt::format("expected an object, found {}", attributes->type_name()));
        } else {
            Object const &entries = attributes->as_object();
            for (auto const &key : entries.keys()) {
                Value const *entry = entries.take(key);
                if (entry == nullptr) {
                    continue;
                }
                if (!entry->is_string()) {
                    note(problems, fmt::format("{}.attributes.{}", tag_path, key), entry->position,
                         fmt::format("expected a string, found {}", entry->type_name()));
                    continue;
                }
                out.attributes.emplace_back(key, entry->as_string());
            }
            // Sorted, matching what the writer's own sort produced, so two loads of one file
            // compare equal and a tag round-trips to the same bytes.
            std::ranges::sort(out.attributes, [](auto const &lhs, auto const &rhs) { return lhs.first < rhs.first; });
        }
    }

    if (out.name.empty()) {
        note(problems, tag_path, value->position, "a provenance tag with an empty name says nothing; omit the key instead");
    }
    return out;
}

std::vector<std::string> read_string_array(Object const &object, std::string_view key, std::string const &path, Problems &problems,
                                           json::Position where) {
    std::vector<std::string> out;
    Value const             *value = field(object, key, path, problems, where);
    if (value == nullptr) {
        return out;
    }
    std::string const child = fmt::format("{}.{}", path, key);
    Array const      *items = as_array(*value, child, problems);
    if (items == nullptr) {
        return out;
    }
    for (std::size_t i = 0; i < items->size(); ++i) {
        if (!(*items)[i].is_string()) {
            note(problems, fmt::format("{}[{}]", child, i), (*items)[i].position,
                 fmt::format("expected a string, found {}", (*items)[i].type_name()));
            continue;
        }
        out.push_back((*items)[i].as_string());
    }
    return out;
}

/// A permutation operator list: an array of operators, each an array of groups,
/// each an array of index letters.
///
/// OPTIONAL, and absent means the empty list. That is not a defaulted guess the
/// way ElementTransform's ``param`` is: every file written before 1.7.0 predates
/// the capability, so an absent key states positively that the node names no
/// operator. A malformed one is reported and skipped rather than half-read,
/// because a partial operator is a different contraction wearing the same spec.
std::vector<PermutationOperator> read_permutation_operators(Object const &object, std::string const &path, Problems &problems) {
    std::vector<PermutationOperator> out;
    Value const                     *value = object.take("operators");
    if (value == nullptr) {
        return out;
    }

    std::string const child     = fmt::format("{}.operators", path);
    Array const      *operators = as_array(*value, child, problems);
    if (operators == nullptr) {
        return out;
    }

    for (std::size_t op_index = 0; op_index < operators->size(); ++op_index) {
        std::string const op_path = fmt::format("{}[{}]", child, op_index);
        Array const      *groups  = as_array((*operators)[op_index], op_path, problems);
        if (groups == nullptr) {
            continue;
        }

        PermutationOperator op;
        bool                well_formed = true;
        for (std::size_t g = 0; g < groups->size(); ++g) {
            std::string const group_path = fmt::format("{}[{}]", op_path, g);
            Array const      *letters    = as_array((*groups)[g], group_path, problems);
            if (letters == nullptr) {
                well_formed = false;
                break;
            }
            std::vector<std::string> group;
            for (std::size_t l = 0; l < letters->size(); ++l) {
                if (!(*letters)[l].is_string()) {
                    note(problems, fmt::format("{}[{}]", group_path, l), (*letters)[l].position,
                         fmt::format("expected an index letter, found {}", (*letters)[l].type_name()));
                    well_formed = false;
                    continue;
                }
                group.push_back((*letters)[l].as_string());
            }
            if (group.empty()) {
                note(problems, group_path, (*groups)[g].position, "a permutation operator group names no index");
                well_formed = false;
            }
            op.groups.push_back(std::move(group));
        }

        if (op.groups.size() < 2) {
            note(problems, op_path, (*operators)[op_index].position,
                 fmt::format("a permutation operator needs at least two groups, found {}", op.groups.size()));
            well_formed = false;
        }
        if (well_formed) {
            out.push_back(std::move(op));
        }
    }
    return out;
}

/// An array of signed integers, for a descriptor whose entries are not extents.
///
/// Separate from @ref read_extent_array rather than a relaxation of it: an extent that came
/// back negative is a corrupt document and is reported as one, and a helper that accepted both
/// would have to stop saying so.
std::vector<std::int64_t> read_int_array(Object const &object, std::string_view key, std::string const &path, Problems &problems,
                                         json::Position where) {
    std::vector<std::int64_t> out;
    Value const              *value = field(object, key, path, problems, where);
    if (value == nullptr) {
        return out;
    }
    std::string const child = fmt::format("{}.{}", path, key);
    Array const      *items = as_array(*value, child, problems);
    if (items == nullptr) {
        return out;
    }
    for (std::size_t i = 0; i < items->size(); ++i) {
        if (!(*items)[i].is_int()) {
            note(problems, fmt::format("{}[{}]", child, i), (*items)[i].position, "expected an integer");
            out.push_back(0);
            continue;
        }
        out.push_back((*items)[i].as_int());
    }
    return out;
}

std::vector<std::size_t> read_extent_array(Object const &object, std::string_view key, std::string const &path, Problems &problems,
                                           json::Position where) {
    std::vector<std::size_t> out;
    Value const             *value = field(object, key, path, problems, where);
    if (value == nullptr) {
        return out;
    }
    std::string const child = fmt::format("{}.{}", path, key);
    Array const      *items = as_array(*value, child, problems);
    if (items == nullptr) {
        return out;
    }
    for (std::size_t i = 0; i < items->size(); ++i) {
        if (!(*items)[i].is_int() || (*items)[i].as_int() < 0) {
            note(problems, fmt::format("{}[{}]", child, i), (*items)[i].position, "expected a non-negative integer");
            out.push_back(0);
            continue;
        }
        out.push_back(static_cast<std::size_t>((*items)[i].as_int()));
    }
    return out;
}

/// Resolve an already-fetched value as a by-name enumerator, reporting the string
/// that did not resolve and what the alternatives are. This shape - the name plus
/// the known set - is what makes an unresolvable name actionable rather than
/// merely fatal.
///
/// Takes the value rather than the key so that the required and the optional
/// spelling below differ only in how they FETCH it; a null @p value is already
/// either reported (required) or allowed (optional), so it is a silent fallback
/// here either way.
template <typename T, typename Fn>
T resolve_named(Value const *value, std::string_view key, std::string const &path, Problems &problems, Fn &&resolve, std::string_view what,
                T fallback) {
    if (value == nullptr) {
        return fallback;
    }
    if (!value->is_string()) {
        note(problems, fmt::format("{}.{}", path, key), value->position, fmt::format("expected a string, found {}", value->type_name()));
        return fallback;
    }
    if (auto const resolved = resolve(value->as_string()); resolved.has_value()) {
        return *resolved;
    }
    note(problems, fmt::format("{}.{}", path, key), value->position, fmt::format("'{}' is not a known {}", value->as_string(), what));
    return fallback;
}

/// A by-name enumerator under a key the file must carry.
template <typename T, typename Fn>
T read_named(Object const &object, std::string_view key, std::string const &path, Problems &problems, json::Position where, Fn &&resolve,
             std::string_view what, T fallback) {
    return resolve_named(field(object, key, path, problems, where), key, path, problems, std::forward<Fn>(resolve), what, fallback);
}

/// As @ref read_named, but for a key a file is allowed not to have.
///
/// The compatibility policy lets the schema GAIN fields, with an absent one taking its
/// documented default, so a key added after a golden was written must not be demanded of it.
/// ``take`` rather than @ref field: it marks the key consumed for the strict unconsumed-key
/// check without reporting a missing one as a problem.
template <typename T, typename Fn>
T read_named_optional(Object const &object, std::string_view key, std::string const &path, Problems &problems, Fn &&resolve,
                      std::string_view what, T fallback) {
    return resolve_named(object.take(key), key, path, problems, std::forward<Fn>(resolve), what, fallback);
}

/// A typed scalar, back to a @ref PrefactorScalar.
PrefactorScalar read_prefactor(Value const &value, std::string const &path, Problems &problems) {
    PrefactorScalar out{double{0}};
    Object const   *object = as_object(value, path, problems);
    if (object == nullptr) {
        return out;
    }
    auto const dtype = read_named<packed_gemm::ScalarType>(*object, "dtype", path, problems, value.position, scalar_type_from_name, "dtype",
                                                           packed_gemm::ScalarType::Float64);

    auto const component = [&](std::string_view key, bool required) -> double {
        Value const *component_value = object->take(key);
        if (component_value == nullptr) {
            if (required) {
                note(problems, path, value.position, fmt::format("typed scalar is missing '{}'", key));
            }
            return 0.0;
        }
        if (auto const number = tagged_number(*component_value); number.has_value()) {
            return *number;
        }
        note(problems, fmt::format("{}.{}", path, key), component_value->position,
             R"(expected a number or one of the special tags "nan", "inf", "-inf")");
        return 0.0;
    };

    double const re = component("re", true);
    switch (dtype) {
    case packed_gemm::ScalarType::Float32:
        return PrefactorScalar{static_cast<float>(re)};
    case packed_gemm::ScalarType::Float64:
        return PrefactorScalar{re};
    case packed_gemm::ScalarType::Complex64:
        return PrefactorScalar{std::complex<float>{static_cast<float>(re), static_cast<float>(component("im", true))}};
    case packed_gemm::ScalarType::Complex128:
        return PrefactorScalar{std::complex<double>{re, component("im", true)}};
    default:
        note(problems, path, value.position, "a typed scalar cannot have dtype 'unknown'");
        return out;
    }
}

BoundExpr read_bound_expr(Value const &value, std::string const &path, Problems &problems) {
    Object const *object = as_object(value, path, problems);
    if (object == nullptr) {
        return BoundExpr{std::int64_t{0}};
    }
    if (Value const *literal = object->take("const"); literal != nullptr) {
        if (!literal->is_int()) {
            note(problems, fmt::format("{}.const", path), literal->position, "expected an integer");
            return BoundExpr{std::int64_t{0}};
        }
        return BoundExpr{literal->as_int()};
    }
    if (Value const *param = object->take("param"); param != nullptr) {
        if (!param->is_string()) {
            note(problems, fmt::format("{}.param", path), param->position, "expected a string");
            return BoundExpr{std::int64_t{0}};
        }
        return BoundExpr{param->as_string()};
    }
    note(problems, path, value.position, R"(a BoundExpr must be {"const": <integer>} or {"param": "<name>"})");
    return BoundExpr{std::int64_t{0}};
}

PredExpr read_pred_expr(Value const &value, std::string const &path, Problems &problems, GateFlagTable const &gates) {
    Object const *object = as_object(value, path, problems);
    if (object == nullptr) {
        return PredExpr{true};
    }
    if (Value const *literal = object->take("const"); literal != nullptr) {
        if (!literal->is_bool()) {
            note(problems, fmt::format("{}.const", path), literal->position, "expected a bool");
            return PredExpr{true};
        }
        return PredExpr{literal->as_bool()};
    }
    if (Value const *compare = object->take("compare"); compare != nullptr) {
        std::string const child = fmt::format("{}.compare", path);
        Object const     *body  = as_object(*compare, child, problems);
        if (body == nullptr) {
            return PredExpr{true};
        }
        Value const *lhs = field(*body, "lhs", child, problems, compare->position);
        auto const   op =
            read_named<CmpOp>(*body, "op", child, problems, compare->position, cmp_op_from_name, "comparison operator", CmpOp::Lt);
        Value const *rhs = field(*body, "rhs", child, problems, compare->position);
        if (lhs == nullptr || rhs == nullptr) {
            return PredExpr{true};
        }
        return PredExpr::compare(read_bound_expr(*lhs, child + ".lhs", problems), op, read_bound_expr(*rhs, child + ".rhs", problems));
    }
    if (Value const *iteration = object->take("iteration"); iteration != nullptr) {
        std::string const child = fmt::format("{}.iteration", path);
        Object const     *body  = as_object(*iteration, child, problems);
        if (body == nullptr) {
            return PredExpr{true};
        }
        auto const op =
            read_named<CmpOp>(*body, "op", child, problems, iteration->position, cmp_op_from_name, "comparison operator", CmpOp::Lt);
        Value const *rhs = field(*body, "rhs", child, problems, iteration->position);
        if (rhs == nullptr) {
            return PredExpr{true};
        }
        return PredExpr::iteration(op, read_bound_expr(*rhs, child + ".rhs", problems));
    }
    if (Value const *flag = object->take("flag"); flag != nullptr) {
        std::string const child = fmt::format("{}.flag", path);
        Object const     *body  = as_object(*flag, child, problems);
        if (body == nullptr) {
            return PredExpr{true};
        }
        std::string const name  = read_string(*body, "name", child, problems, flag->position);
        auto const        index = read_int(*body, "index", child, problems, flag->position);
        auto const        found = gates.find(name);
        if (found == gates.end()) {
            std::vector<std::string> known;
            known.reserve(gates.size());
            for (auto const &[gate_name, buffer] : gates) {
                known.push_back(gate_name);
            }
            std::ranges::sort(known);
            note(problems, child, flag->position,
                 fmt::format("gate-flag array '{}' is not declared by this file. Declared: [{}]", name, fmt::join(known, ", ")));
            return PredExpr{true};
        }
        if (index < 0) {
            note(problems, child, flag->position, "a gate-flag index must not be negative");
            return PredExpr{true};
        }
        return PredExpr::flag(found->second, static_cast<std::size_t>(index));
    }
    note(problems, path, value.position,
         R"(a PredExpr must be one of {"const": <bool>}, {"compare": ...}, {"iteration": ...} or {"flag": ...})");
    return PredExpr{true};
}

/// One tensor record. A MANIFEST entry and an intermediate share most of their
/// shape and differ in the two directions the schema keeps apart: a manifest
/// entry declares a direction and an alias parent, an intermediate declares
/// whether it is graph-owned and how it is initialized.
IrTensor read_tensor(Value const &value, std::string const &path, Problems &problems, bool manifest_entry) {
    IrTensor      out;
    Object const *object = as_object(value, path, problems);
    if (object == nullptr) {
        return out;
    }
    auto const id = read_int(*object, "id", path, problems, value.position);
    if (id < 0) {
        note(problems, path, value.position, "a tensor id must not be negative");
    }
    out.id          = static_cast<std::size_t>(std::max<std::int64_t>(id, 0));
    out.name        = read_string(*object, "name", path, problems, value.position);
    out.dtype       = read_named<packed_gemm::ScalarType>(*object, "dtype", path, problems, value.position, scalar_type_from_name, "dtype",
                                                          packed_gemm::ScalarType::Float64);
    out.rank        = static_cast<std::size_t>(std::max<std::int64_t>(read_int(*object, "rank", path, problems, value.position), 0));
    out.dims        = read_extent_array(*object, "dims", path, problems, value.position);
    out.dim_symbols = read_string_array(*object, "dim_symbols", path, problems, value.position);
    out.spaces      = read_string_array(*object, "spaces", path, problems, value.position);
    out.spaces_inferred = read_bool(*object, "spaces_inferred", path, problems, value.position);
    out.tag             = read_provenance_tag(*object, path, problems);
    out.scope = read_named<TensorOwnership>(*object, "scope", path, problems, value.position, tensor_ownership_from_name, "ownership scope",
                                            TensorOwnership::Graph);
    if (manifest_entry) {
        out.direction = read_named<ManifestDirection>(*object, "direction", path, problems, value.position, manifest_direction_from_name,
                                                      "manifest direction", ManifestDirection::Input);
        if (Value const *alias = field(*object, "aliases_input", path, problems, value.position); alias != nullptr) {
            if (alias->is_string()) {
                out.aliases_input = alias->as_string();
            } else if (!alias->is_null()) {
                note(problems, fmt::format("{}.aliases_input", path), alias->position, "expected a manifest name or null");
            }
        }
    } else {
        out.intermediate = read_bool(*object, "intermediate", path, problems, value.position);
        out.init         = read_named<InitKind>(*object, "init", path, problems, value.position, init_kind_from_name, "initialization kind",
                                                InitKind::None);
        // Optional: a file written before this key existed means Materialized, which is what
        // its graph's intermediates were.
        out.alloc = read_named_optional<AllocState>(*object, "alloc", path, problems, alloc_state_from_name, "allocation state",
                                                    AllocState::Materialized);
    }

    if (out.dims.size() != out.rank) {
        note(problems, path, value.position, fmt::format("rank is {} but {} dims are given", out.rank, out.dims.size()));
    }
    if (!out.dim_symbols.empty() && out.dim_symbols.size() != out.rank) {
        note(problems, path, value.position,
             fmt::format("rank is {} but {} dim symbols are given; the annotation is all axes or none", out.rank, out.dim_symbols.size()));
    }
    if (!out.spaces.empty() && out.spaces.size() != out.rank) {
        note(problems, path, value.position,
             fmt::format("rank is {} but {} index spaces are given; the annotation is all axes or none", out.rank, out.spaces.size()));
    }

    if (Value const *outer = object->take("outer"); outer != nullptr) {
        if (!outer->is_int() || outer->as_int() < 0) {
            note(problems, fmt::format("{}.outer", path), outer->position, "expected a non-negative integer");
        } else {
            out.outer = static_cast<std::size_t>(outer->as_int());
        }
    }
    return out;
}

IrFragment read_fragment(Value const &value, std::string const &path, Problems &problems, GateFlagTable const &gates,
                         SpaceRegistry const &registry);

/// One node's descriptor, per kind. The coverage is exactly the reconstructible
/// set, which is what makes an unknown kind here a file problem rather than a
/// gap: the writer could not have produced one.
// NOLINTNEXTLINE(misc-no-recursion): control-flow descriptors hold fragments.
void read_descriptor(IrNode &node, Value const &value, std::string const &path, Problems &problems, GateFlagTable const &gates,
                     SpaceRegistry const &registry) {
    Object const *object = as_object(value, path, problems);
    if (object == nullptr) {
        return;
    }
    auto const scalar = [&](std::string_view key, PrefactorScalar fallback) {
        Value const *entry = field(*object, key, path, problems, value.position);
        return entry != nullptr ? read_prefactor(*entry, fmt::format("{}.{}", path, key), problems) : fallback;
    };
    auto const transpose_char = [&](std::string_view key, char fallback) {
        std::string const text = read_string(*object, key, path, problems, value.position);
        if (text.size() != 1) {
            if (!text.empty()) {
                note(problems, fmt::format("{}.{}", path, key), value.position, "a BLAS transpose flag is one character");
            }
            return fallback;
        }
        return text[0];
    };

    switch (node.kind) {
    case OpKind::Transpose:
        node.descriptor = std::monostate{};
        return;
    case OpKind::Scale: {
        ScaleDescriptor desc;
        desc.factor     = scalar("factor", PrefactorScalar{double{1}});
        desc.params     = make_elementwise_params(desc.factor);
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Permute: {
        PermuteDescriptor desc;
        auto const        alpha = scalar("alpha", PrefactorScalar{double{1}});
        auto const        beta  = scalar("beta", PrefactorScalar{double{0}});
        desc.alpha              = as<std::complex<double>>(alpha);
        desc.beta               = as<std::complex<double>>(beta);
        desc.c_indices          = read_string_array(*object, "c_indices", path, problems, value.position);
        desc.a_indices          = read_string_array(*object, "a_indices", path, problems, value.position);
        desc.operators          = read_permutation_operators(*object, path, problems);
        desc.params             = make_elementwise_params(alpha, beta);
        node.descriptor         = std::move(desc);
        return;
    }
    case OpKind::Axpby: {
        AxpbyDescriptor desc;
        desc.alpha      = scalar("alpha", PrefactorScalar{double{1}});
        desc.beta       = scalar("beta", PrefactorScalar{double{0}});
        desc.params     = make_elementwise_params(desc.alpha, desc.beta);
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::DirectProduct:
    case OpKind::DirectDivision: {
        ElementwiseBinaryDescriptor desc;
        desc.alpha      = scalar("alpha", PrefactorScalar{double{1}});
        desc.beta       = scalar("beta", PrefactorScalar{double{0}});
        desc.params     = make_elementwise_params(desc.alpha, desc.beta);
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Einsum: {
        EinsumDescriptor desc;
        desc.spec.c_indices      = read_string_array(*object, "c_indices", path, problems, value.position);
        desc.spec.a_indices      = read_string_array(*object, "a_indices", path, problems, value.position);
        desc.spec.b_indices      = read_string_array(*object, "b_indices", path, problems, value.position);
        desc.spec.link_indices   = read_string_array(*object, "link_indices", path, problems, value.position);
        desc.spec.target_indices = read_string_array(*object, "target_indices", path, problems, value.position);
        desc.spec.all_indices    = read_string_array(*object, "all_indices", path, problems, value.position);
        desc.spec.scalar_output  = read_bool(*object, "scalar_output", path, problems, value.position);
        desc.conj_a              = read_bool(*object, "conj_a", path, problems, value.position);
        desc.conj_b              = read_bool(*object, "conj_b", path, problems, value.position);
        desc.spec.conj_a         = desc.conj_a;
        desc.spec.conj_b         = desc.conj_b;
        desc.operators           = read_permutation_operators(*object, path, problems);
        desc.spec.scalar_type    = node.dtype;
        desc.c_prefactor         = scalar("c_prefactor", PrefactorScalar{double{0}});
        desc.ab_prefactor        = scalar("ab_prefactor", PrefactorScalar{double{1}});

        if (Value const *letters = field(*object, "letter_spaces", path, problems, value.position); letters != nullptr) {
            std::string const child = fmt::format("{}.letter_spaces", path);
            if (Array const *items = as_array(*letters, child, problems); items != nullptr) {
                for (std::size_t i = 0; i < items->size(); ++i) {
                    std::string const entry_path = fmt::format("{}[{}]", child, i);
                    Object const     *entry      = as_object((*items)[i], entry_path, problems);
                    if (entry == nullptr) {
                        continue;
                    }
                    std::string const letter = read_string(*entry, "letter", entry_path, problems, (*items)[i].position);
                    std::string const space  = read_string(*entry, "space", entry_path, problems, (*items)[i].position);
                    auto const        id     = registry.find(space);
                    if (!id.has_value()) {
                        note(problems, entry_path, (*items)[i].position,
                             fmt::format("index space '{}' is not registered in this process", space));
                        continue;
                    }
                    desc.letter_spaces.emplace_back(letter, *id);
                }
            }
        }

        if (Value const *hint = object->take("gemm_hint"); hint != nullptr) {
            std::string const child = fmt::format("{}.gemm_hint", path);
            if (Object const *body = as_object(*hint, child, problems); body != nullptr) {
                auto record          = std::make_shared<GemmHint>();
                record->m            = static_cast<int>(read_int(*body, "m", child, problems, hint->position));
                record->n            = static_cast<int>(read_int(*body, "n", child, problems, hint->position));
                record->k            = static_cast<int>(read_int(*body, "k", child, problems, hint->position));
                std::string const ta = read_string(*body, "trans_a", child, problems, hint->position);
                std::string const tb = read_string(*body, "trans_b", child, problems, hint->position);
                record->trans_a      = ta.size() == 1 ? ta[0] : 'N';
                record->trans_b      = tb.size() == 1 ? tb[0] : 'N';
                auto const operand   = [&](std::string_view key, GemmOperand &slot) {
                    Value const *entry = field(*body, key, child, problems, hint->position);
                    if (entry == nullptr) {
                        node.hint_ids.push_back(0);
                        return;
                    }
                    std::string const operand_path = fmt::format("{}.{}", child, key);
                    Object const     *fields       = as_object(*entry, operand_path, problems);
                    if (fields == nullptr) {
                        node.hint_ids.push_back(0);
                        return;
                    }
                    node.hint_ids.push_back(static_cast<std::size_t>(
                        std::max<std::int64_t>(read_int(*fields, "id", operand_path, problems, entry->position), 0)));
                    slot.leading_dim = static_cast<int>(read_int(*fields, "leading_dim", operand_path, problems, entry->position));
                };
                operand("a", record->a);
                operand("b", record->b);
                operand("c", record->c);
                desc.gemm_hint = std::move(record);
            }
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Dot: {
        DotDescriptor desc;
        desc.conjugated = read_bool(*object, "conjugated", path, problems, value.position);
        node.descriptor = desc;
        return;
    }
    case OpKind::Trace:
        node.descriptor = TraceDescriptor{};
        return;
    case OpKind::Gemm: {
        GemmDescriptor desc;
        desc.alpha      = scalar("alpha", PrefactorScalar{double{1}});
        desc.beta       = scalar("beta", PrefactorScalar{double{0}});
        desc.trans_a    = transpose_char("trans_a", 'n');
        desc.trans_b    = transpose_char("trans_b", 'n');
        node.descriptor = desc;
        return;
    }
    case OpKind::Syev: {
        SyevDescriptor desc;
        // REQUIRED, not defaulted, even though the field arrived with this kind. No file
        // predates it: no earlier build could write a Syev node at all, so an absent key is a
        // malformed file rather than an older one, and the two LAPACK jobs leave A holding
        // different matrices. Guessing the default would silently change what a graph computes.
        desc.compute_eigenvectors = read_bool(*object, "compute_eigenvectors", path, problems, value.position);
        node.descriptor           = desc;
        return;
    }
    case OpKind::ElementTransform: {
        ElementTransformDescriptor desc;
        desc.op_name = read_string(*object, "op", path, problems, value.position);
        if (!desc.op_name.empty() && !element_ops::global_element_op_registry().contains(desc.op_name)) {
            note(problems, fmt::format("{}.op", path), value.position,
                 fmt::format("element op '{}' is not registered in this process. Registered: [{}]", desc.op_name,
                             fmt::join(element_ops::global_element_op_registry().names(), ", ")));
        }
        // OPTIONAL, and absent means the op's documented default rather than
        // zero: an older file predates the key entirely, and a node written by
        // this build omits it whenever the capture site named no number.
        if (Value const *param = object->take("param"); param != nullptr) {
            if (auto const parsed = tagged_number(*param); parsed.has_value()) {
                desc.param = *parsed;
            } else {
                note(problems, fmt::format("{}.param", path), param->position, "expected a number");
            }
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::WriteParam: {
        WriteParamDescriptor desc;
        desc.name        = read_string(*object, "param", path, problems, value.position);
        desc.source_type = read_named<ParamSourceType>(*object, "source_type", path, problems, value.position, param_source_type_from_name,
                                                       "parameter source type", ParamSourceType::Int64);
        if (Value const *expr = object->take("source_expr"); expr != nullptr) {
            desc.source_expr = read_bound_expr(*expr, fmt::format("{}.source_expr", path), problems);
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Conditional: {
        ConditionalDescriptor desc;
        if (Value const *predicate = field(*object, "predicate", path, problems, value.position); predicate != nullptr) {
            desc.predicate = read_pred_expr(*predicate, fmt::format("{}.predicate", path), problems, gates);
        }
        if (Value const *then_branch = field(*object, "then", path, problems, value.position); then_branch != nullptr) {
            node.then_branch =
                std::make_shared<IrFragment>(read_fragment(*then_branch, fmt::format("{}.then", path), problems, gates, registry));
        }
        if (Value const *else_branch = field(*object, "else", path, problems, value.position);
            else_branch != nullptr && !else_branch->is_null()) {
            node.else_branch =
                std::make_shared<IrFragment>(read_fragment(*else_branch, fmt::format("{}.else", path), problems, gates, registry));
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Loop: {
        LoopDescriptor desc;
        auto const     limit = read_int(*object, "max_iterations", path, problems, value.position);
        if (limit < 0) {
            note(problems, path, value.position, "max_iterations must not be negative");
        }
        desc.max_iterations = static_cast<std::size_t>(std::max<std::int64_t>(limit, 0));
        if (Value const *condition = field(*object, "condition", path, problems, value.position); condition != nullptr) {
            desc.condition = read_pred_expr(*condition, fmt::format("{}.condition", path), problems, gates);
        }
        if (Value const *body = field(*object, "body", path, problems, value.position); body != nullptr) {
            node.body = std::make_shared<IrFragment>(read_fragment(*body, fmt::format("{}.body", path), problems, gates, registry));
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::LaplaceQuadrature: {
        LaplaceQuadratureDescriptor desc;
        // Required rather than defaulted, and for the reason Syev's job flag is: no file
        // predates the kind, so an absent key is a malformed document rather than an older
        // one, and a guessed tolerance or point count is a different approximation wearing
        // the same graph.
        if (Value const *epsilon = field(*object, "epsilon", path, problems, value.position); epsilon != nullptr) {
            if (auto const parsed = tagged_number(*epsilon); parsed.has_value()) {
                desc.epsilon = *parsed;
            } else {
                note(problems, fmt::format("{}.epsilon", path), epsilon->position, "expected a number");
            }
        }
        desc.points = read_int(*object, "points", path, problems, value.position);
        for (std::int64_t sign : read_int_array(*object, "signs", path, problems, value.position)) {
            desc.signs.push_back(static_cast<std::int8_t>(sign >= 0 ? 1 : -1));
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Setup: {
        SetupDescriptor desc;
        if (Value const *body = field(*object, "body", path, problems, value.position); body != nullptr) {
            node.body = std::make_shared<IrFragment>(read_fragment(*body, fmt::format("{}.body", path), problems, gates, registry));
        }
        node.descriptor = std::move(desc);
        return;
    }
    default:
        note(problems, path, value.position,
             fmt::format("op kind '{}' is not one this schema can describe; the reconstructible set is what a file may contain",
                         op_kind_name(node.kind)));
        return;
    }
}

// NOLINTNEXTLINE(misc-no-recursion): see read_descriptor.
IrNode read_node(Value const &value, std::string const &path, Problems &problems, GateFlagTable const &gates,
                 SpaceRegistry const &registry) {
    IrNode        out;
    Object const *object = as_object(value, path, problems);
    if (object == nullptr) {
        return out;
    }
    out.id    = static_cast<std::size_t>(std::max<std::int64_t>(read_int(*object, "id", path, problems, value.position), 0));
    out.kind  = read_named<OpKind>(*object, "kind", path, problems, value.position, op_kind_from_name, "op kind", OpKind::Custom);
    out.label = read_string(*object, "label", path, problems, value.position);

    auto const ids = [&](std::string_view key) {
        std::vector<std::size_t> list;
        Value const             *entry = field(*object, key, path, problems, value.position);
        if (entry == nullptr) {
            return list;
        }
        std::string const child = fmt::format("{}.{}", path, key);
        Array const      *items = as_array(*entry, child, problems);
        if (items == nullptr) {
            return list;
        }
        for (std::size_t i = 0; i < items->size(); ++i) {
            if (!(*items)[i].is_int() || (*items)[i].as_int() < 0) {
                note(problems, fmt::format("{}[{}]", child, i), (*items)[i].position, "expected a non-negative tensor id");
                list.push_back(0);
                continue;
            }
            list.push_back(static_cast<std::size_t>((*items)[i].as_int()));
        }
        return list;
    };
    out.inputs  = ids("inputs");
    out.outputs = ids("outputs");
    out.dtype   = read_named<packed_gemm::ScalarType>(*object, "dtype", path, problems, value.position, scalar_type_from_name, "dtype",
                                                      packed_gemm::ScalarType::Unknown);
    out.rank    = static_cast<std::size_t>(std::max<std::int64_t>(read_int(*object, "rank", path, problems, value.position), 0));

    if (Value const *descriptor = field(*object, "descriptor", path, problems, value.position); descriptor != nullptr) {
        read_descriptor(out, *descriptor, fmt::format("{}.descriptor", path), problems, gates, registry);
    }
    return out;
}

// NOLINTNEXTLINE(misc-no-recursion): fragments nest.
IrFragment read_fragment(Value const &value, std::string const &path, Problems &problems, GateFlagTable const &gates,
                         SpaceRegistry const &registry) {
    IrFragment    out;
    Object const *object = as_object(value, path, problems);
    if (object == nullptr) {
        return out;
    }
    out.name = read_string(*object, "name", path, problems, value.position);
    if (Value const *tensors = field(*object, "tensors", path, problems, value.position); tensors != nullptr) {
        std::string const child = fmt::format("{}.tensors", path);
        if (Array const *items = as_array(*tensors, child, problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                out.tensors.push_back(read_tensor((*items)[i], fmt::format("{}[{}]", child, i), problems, /*manifest_entry=*/false));
            }
        }
    }
    if (Value const *nodes = field(*object, "nodes", path, problems, value.position); nodes != nullptr) {
        std::string const child = fmt::format("{}.nodes", path);
        if (Array const *items = as_array(*nodes, child, problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                out.nodes.push_back(read_node((*items)[i], fmt::format("{}[{}]", child, i), problems, gates, registry));
            }
        }
    }
    return out;
}

/// Compare two ``major.minor.patch`` strings.
/// @return -1, 0 or 1, or nullopt when either is not a semver triple.
std::optional<int> compare_semver(std::string_view lhs, std::string_view rhs) {
    auto const split = [](std::string_view text) -> std::optional<std::array<long, 3>> {
        std::array<long, 3> parts{};
        std::size_t         start = 0;
        for (std::size_t part = 0; part < 3; ++part) {
            std::size_t const dot = part < 2 ? text.find('.', start) : text.size();
            if (dot == std::string_view::npos || dot == start) {
                return std::nullopt;
            }
            std::string_view const field_text = text.substr(start, dot - start);
            long                   value      = 0;
            for (char const digit : field_text) {
                if (digit < '0' || digit > '9') {
                    return std::nullopt;
                }
                value = value * 10 + (digit - '0');
            }
            parts[part] = value;
            start       = dot + 1;
        }
        return parts;
    };
    auto const left  = split(lhs);
    auto const right = split(rhs);
    if (!left.has_value() || !right.has_value()) {
        return std::nullopt;
    }
    for (std::size_t i = 0; i < 3; ++i) {
        if ((*left)[i] != (*right)[i]) {
            return (*left)[i] < (*right)[i] ? -1 : 1;
        }
    }
    return 0;
}

} // namespace

/// Read the whole document. Reports every problem it can see; a document that
/// reports none is one the builder may run on.
IrDocument read_document(Value const &root, Problems &problems, SpaceRegistry const &registry) {
    IrDocument    out;
    Object const *object = as_object(root, "$", problems);
    if (object == nullptr) {
        return out;
    }

    // The version gate runs FIRST and, when it refuses, nothing else is read:
    // every message a newer schema would produce would be about fields this
    // build does not understand, which buries the one message that matters.
    out.version = read_string(*object, key_version, "$", problems, root.position);
    if (out.version.empty()) {
        return out;
    }
    auto const order = compare_semver(out.version, graph_ir_schema_version);
    if (!order.has_value()) {
        note(problems, "$", root.position, fmt::format("'{}' is not a major.minor.patch schema version", out.version));
        return out;
    }
    if (*order > 0) {
        note(problems, "$", root.position,
             fmt::format("this file is einsums_graph_ir {} and this build understands up to {}; a newer build reads an older IR, "
                         "never the reverse",
                         out.version, graph_ir_schema_version));
        return out;
    }

    // Provenance is DATA. Every field is consumed so the strict audit passes,
    // and not one of them is acted on.
    if (Value const *provenance = field(*object, key_provenance, "$", problems, root.position); provenance != nullptr) {
        if (Object const *body = as_object(*provenance, "$.provenance", problems); body != nullptr) {
            body->mark_consumed("library_version");
            body->mark_consumed("config_fingerprint");
            // Read rather than merely consumed, and reading is not acting: nothing behaves
            // differently for what is in here. A malformed entry is skipped rather than
            // failing the load, because provenance is not structure and a file whose history
            // is unreadable still describes a perfectly good graph.
            if (Value const *passes = body->take("structural_passes"); passes != nullptr && passes->is_array()) {
                for (auto const &entry : passes->as_array()) {
                    if (entry.is_string()) {
                        out.structural_passes.push_back(entry.as_string());
                    }
                }
            }
        }
    }

    out.name = read_string(*object, "name", "$", problems, root.position);

    // Gate-flag declarations are read before anything that can reference one.
    if (Value const *gates = field(*object, "gate_flags", "$", problems, root.position); gates != nullptr) {
        if (Array const *items = as_array(*gates, "$.gate_flags", problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                std::string const path  = fmt::format("$.gate_flags[{}]", i);
                Object const     *entry = as_object((*items)[i], path, problems);
                if (entry == nullptr) {
                    continue;
                }
                std::string const name = read_string(*entry, "name", path, problems, (*items)[i].position);
                auto const        size = read_int(*entry, "size", path, problems, (*items)[i].position);
                if (size < 0) {
                    note(problems, path, (*items)[i].position, "a gate-flag array size must not be negative");
                    continue;
                }
                out.gate_flags.emplace_back(name, static_cast<std::size_t>(size));
            }
        }
    }

    for (auto const &[name, size] : out.gate_flags) {
        out.gate_buffers.emplace(name, std::make_shared<std::vector<std::uint8_t>>(size, std::uint8_t{0}));
    }
    GateFlagTable const &gates = out.gate_buffers;

    if (Value const *manifest = field(*object, "manifest", "$", problems, root.position); manifest != nullptr) {
        if (Array const *items = as_array(*manifest, "$.manifest", problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                out.manifest.push_back(read_tensor((*items)[i], fmt::format("$.manifest[{}]", i), problems, /*manifest_entry=*/true));
            }
        }
    }

    if (Value const *spaces = field(*object, "spaces", "$", problems, root.position); spaces != nullptr) {
        if (Object const *body = as_object(*spaces, "$.spaces", problems); body != nullptr) {
            out.space_names = read_string_array(*body, "names", "$.spaces", problems, spaces->position);
            for (auto const &name : out.space_names) {
                if (!registry.find(name).has_value()) {
                    std::vector<std::string> known;
                    for (SpaceId const id : registry.ids()) {
                        known.push_back(registry.space(id).name);
                    }
                    note(problems, "$.spaces.names", spaces->position,
                         fmt::format("index space '{}' is not registered in this process. Registered: [{}]", name, fmt::join(known, ", ")));
                }
            }
            if (Value const *ties = field(*body, "symbol_ties", "$.spaces", problems, spaces->position); ties != nullptr) {
                if (Array const *items = as_array(*ties, "$.spaces.symbol_ties", problems); items != nullptr) {
                    for (std::size_t i = 0; i < items->size(); ++i) {
                        std::string const path  = fmt::format("$.spaces.symbol_ties[{}]", i);
                        Object const     *entry = as_object((*items)[i], path, problems);
                        if (entry == nullptr) {
                            continue;
                        }
                        out.symbol_ties.emplace_back(read_string(*entry, "symbol", path, problems, (*items)[i].position),
                                                     read_string(*entry, "space", path, problems, (*items)[i].position));
                    }
                }
            }
        }
    }

    // OPTIONAL, and the compatibility policy says an added field takes a documented default:
    // a file written before this section existed describes a graph nothing approximated,
    // which is an empty list and is exactly right.
    if (Value const *approximations = object->take("approximations"); approximations != nullptr) {
        if (Array const *items = as_array(*approximations, "$.approximations", problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                std::string const path  = fmt::format("$.approximations[{}]", i);
                Object const     *entry = as_object((*items)[i], path, problems);
                if (entry == nullptr) {
                    continue;
                }
                ApproximationRecord record;
                record.pass_name = read_string(*entry, "pass_name", path, problems, (*items)[i].position);
                // The effect has no safe default: a bound whose units are unreadable cannot be
                // composed or compared, and guessing one would silently produce a number in
                // the wrong scale. read_named reports the unresolvable name and the load fails.
                record.effect =
                    read_named<ApproximationEffect>(*entry, "effect", path, problems, (*items)[i].position, approximation_effect_from_name,
                                                    "approximation effect", ApproximationEffect::NormRelative);
                if (Value const *tolerance = field(*entry, "tolerance", path, problems, (*items)[i].position); tolerance != nullptr) {
                    if (auto const parsed = tagged_number(*tolerance); parsed.has_value()) {
                        record.tolerance = *parsed;
                    } else {
                        note(problems, fmt::format("{}.tolerance", path), tolerance->position, "expected a number");
                    }
                }
                if (Value const *bound = field(*entry, "bound", path, problems, (*items)[i].position); bound != nullptr) {
                    if (auto const parsed = tagged_number(*bound); parsed.has_value()) {
                        record.bound = *parsed;
                    } else {
                        note(problems, fmt::format("{}.bound", path), bound->position, "expected a number");
                    }
                }
                // OPTIONAL, and defaulted to asserted rather than measured. A file written
                // before this key existed carries a number whose provenance nobody recorded,
                // and reading it as evidence would promote a guess by nothing more than a
                // newer build having opened it.
                record.origin  = read_named_optional<ApproximationOrigin>(*entry, "origin", path, problems, approximation_origin_from_name,
                                                                          "approximation origin", ApproximationOrigin::Asserted);
                record.outputs = read_string_array(*entry, "outputs", path, problems, (*items)[i].position);
                record.spaces  = read_string_array(*entry, "spaces", path, problems, (*items)[i].position);
                record.setup   = read_string(*entry, "setup", path, problems, (*items)[i].position);
                // OPTIONAL, for the reason `origin` is: a file written before the key existed
                // named no parameter, which is exactly what an empty string says.
                if (Value const *measurement = entry->take("measurement"); measurement != nullptr) {
                    if (measurement->is_string()) {
                        record.measurement = measurement->as_string();
                    } else {
                        note(problems, fmt::format("{}.measurement", path), measurement->position, "expected a string");
                    }
                }
                out.approximations.push_back(std::move(record));
            }
        }
    }

    if (Value const *params = field(*object, "params", "$", problems, root.position); params != nullptr) {
        if (Array const *items = as_array(*params, "$.params", problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                std::string const path  = fmt::format("$.params[{}]", i);
                Object const     *entry = as_object((*items)[i], path, problems);
                if (entry == nullptr) {
                    continue;
                }
                out.params.emplace_back(read_string(*entry, "name", path, problems, (*items)[i].position),
                                        read_int(*entry, "value", path, problems, (*items)[i].position));
            }
        }
    }

    if (Value const *tensors = field(*object, "tensors", "$", problems, root.position); tensors != nullptr) {
        if (Array const *items = as_array(*tensors, "$.tensors", problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                out.tensors.push_back(read_tensor((*items)[i], fmt::format("$.tensors[{}]", i), problems, /*manifest_entry=*/false));
            }
        }
    }

    if (Value const *redirects = field(*object, "slot_redirects", "$", problems, root.position); redirects != nullptr) {
        if (Array const *items = as_array(*redirects, "$.slot_redirects", problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                std::string const path  = fmt::format("$.slot_redirects[{}]", i);
                Object const     *entry = as_object((*items)[i], path, problems);
                if (entry == nullptr) {
                    continue;
                }
                out.slot_redirects.emplace_back(
                    static_cast<std::size_t>(std::max<std::int64_t>(read_int(*entry, "from", path, problems, (*items)[i].position), 0)),
                    static_cast<std::size_t>(std::max<std::int64_t>(read_int(*entry, "to", path, problems, (*items)[i].position), 0)));
            }
        }
    }

    if (Value const *nodes = field(*object, "nodes", "$", problems, root.position); nodes != nullptr) {
        if (Array const *items = as_array(*nodes, "$.nodes", problems); items != nullptr) {
            for (std::size_t i = 0; i < items->size(); ++i) {
                out.nodes.push_back(read_node((*items)[i], fmt::format("$.nodes[{}]", i), problems, gates, registry));
            }
        }
    }
    return out;
}

EINSUMS_NAMESPACE_END(compute_graph::graph_ir)
