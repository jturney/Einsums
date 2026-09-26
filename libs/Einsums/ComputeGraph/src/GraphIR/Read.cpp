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

#include <Einsums/ComputeGraph/DescriptorRegistry.hpp>
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
#include <array>
#include <complex>
#include <cstdint>
#include <exception>
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

/// What a scalar field read needs to know about one leaf type: how to recognise it, how to read
/// it, and how a diagnostic names it.
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

/// One JSON object being read, with everything a field read needs to report a problem: the
/// object, its path in the document, the position a missing key is reported at, and the
/// problem list.
///
/// Every read consumes its key with ``take``, so the document-wide unconsumed-key audit sees
/// exactly what was understood. A read that fails records a problem and returns a harmless
/// default, so a caller reads on and the load collects every fault rather than the first.
class Fields {
  public:
    Fields(Object const &object, std::string path, Problems &problems, json::Position at)
        : _object(object), _path(std::move(path)), _problems(problems), _at(at) {}

    /// The object @p value holds, or nothing (and a problem) when it is not one.
    static std::optional<Fields> of(Value const &value, std::string path, Problems &problems) {
        if (!value.is_object()) {
            graph_ir::note(problems, path, value.position, fmt::format("expected an object, found {}", value.type_name()));
            return std::nullopt;
        }
        return Fields{value.as_object(), std::move(path), problems, value.position};
    }

    [[nodiscard]] Object const      &object() const { return _object; }
    [[nodiscard]] std::string const &path() const { return _path; }
    [[nodiscard]] json::Position     position() const { return _at; }
    [[nodiscard]] Problems          &problems() const { return _problems; }
    [[nodiscard]] std::string        child(std::string_view key) const { return fmt::format("{}.{}", _path, key); }

    /// Report a problem against this object.
    void note(std::string message) const { graph_ir::note(_problems, _path, _at, std::move(message)); }

    /// Report a problem against @p key's value.
    void note(std::string_view key, json::Position where, std::string message) const {
        graph_ir::note(_problems, child(key), where, std::move(message));
    }

    /// Consume @p key, reporting its absence.
    [[nodiscard]] Value const *required(std::string_view key) const {
        Value const *value = _object.take(key);
        if (value == nullptr) {
            note(fmt::format("required key '{}' is missing", key));
        }
        return value;
    }

    /// Consume @p key, which a file is allowed not to have. Absence is not a problem: the
    /// compatibility policy lets the schema GAIN fields, each with a documented default.
    [[nodiscard]] Value const *optional(std::string_view key) const { return _object.take(key); }

    template <typename T>
    [[nodiscard]] T scalar(std::string_view key) const {
        Value const *value = required(key);
        if (value == nullptr) {
            return T{};
        }
        if (!ScalarLeaf<T>::matches(*value)) {
            note(key, value->position, fmt::format("expected {}, found {}", ScalarLeaf<T>::noun, value->type_name()));
            return T{};
        }
        return ScalarLeaf<T>::read(*value);
    }

    [[nodiscard]] std::string  str(std::string_view key) const { return scalar<std::string>(key); }
    [[nodiscard]] std::int64_t integer(std::string_view key) const { return scalar<std::int64_t>(key); }
    [[nodiscard]] bool         flag(std::string_view key) const { return scalar<bool>(key); }

    /// An integer that counts or identifies something, so a negative one is a corrupt file.
    /// Reported rather than clamped: 0 is a real node id and the "no tensor" id, so a silent
    /// clamp would read a corrupt file as a different, valid-looking graph.
    [[nodiscard]] std::optional<std::size_t> count(std::string_view key) const {
        Value const *value = _object.peek(key);
        auto const   read  = integer(key);
        if (value == nullptr || !value->is_int()) {
            return std::nullopt;
        }
        if (read < 0) {
            note(key, value->position, "must not be negative");
            return std::nullopt;
        }
        return static_cast<std::size_t>(read);
    }

    /// Resolve @p key's string as a by-name enumerator, reporting the string that did not
    /// resolve. A missing key is reported unless @p optional.
    template <typename T, typename Fn>
    [[nodiscard]] T named(std::string_view key, Fn &&resolve, std::string_view what, T fallback, bool optional = false) const {
        Value const *value = optional ? this->optional(key) : required(key);
        if (value == nullptr) {
            return fallback;
        }
        if (!value->is_string()) {
            note(key, value->position, fmt::format("expected a string, found {}", value->type_name()));
            return fallback;
        }
        if (auto const resolved = resolve(value->as_string()); resolved.has_value()) {
            return *resolved;
        }
        note(key, value->position, fmt::format("'{}' is not a known {}", value->as_string(), what));
        return fallback;
    }

    /// A number, or one of the special tags ``nan``, ``inf``, ``-inf``. Nothing when the key is
    /// absent (reported unless @p optional) or does not hold one.
    [[nodiscard]] std::optional<double> number(std::string_view key, bool optional = false) const {
        Value const *value = optional ? this->optional(key) : required(key);
        if (value == nullptr) {
            return std::nullopt;
        }
        auto const parsed = tagged_number(*value);
        if (!parsed.has_value()) {
            note(key, value->position, "expected a number");
        }
        return parsed;
    }

    /// A one-character BLAS transpose flag.
    [[nodiscard]] char transpose(std::string_view key, char fallback) const {
        std::string const text = str(key);
        if (text.size() != 1) {
            if (!text.empty()) {
                note(key, _at, "a BLAS transpose flag is one character");
            }
            return fallback;
        }
        return text[0];
    }

    [[nodiscard]] PrefactorScalar prefactor(std::string_view key, PrefactorScalar fallback) const;

    /// Walk the array under @p key, calling @p visit with each element and its path. A missing
    /// key is reported unless @p optional; a key that is not an array is always reported.
    template <typename F>
    void each(std::string_view key, F &&visit, bool optional = false) const {
        Value const *value = optional ? this->optional(key) : required(key);
        if (value == nullptr) {
            return;
        }
        std::string const array_path = child(key);
        if (!value->is_array()) {
            graph_ir::note(_problems, array_path, value->position, fmt::format("expected an array, found {}", value->type_name()));
            return;
        }
        Array const &items = value->as_array();
        for (std::size_t i = 0; i < items.size(); ++i) {
            visit(items[i], fmt::format("{}[{}]", array_path, i));
        }
    }

    /// As @ref each, for an array of objects: @p visit gets each element's Fields, and an
    /// element that is not an object is reported and skipped.
    template <typename F>
    void each_object(std::string_view key, F &&visit, bool optional = false) const {
        each(
            key,
            [&](Value const &item, std::string item_path) {
                if (auto entry = Fields::of(item, std::move(item_path), _problems)) {
                    visit(*entry);
                }
            },
            optional);
    }

    /// The object under @p key, or nothing when it is absent (reported unless @p optional) or
    /// not an object.
    [[nodiscard]] std::optional<Fields> object_at(std::string_view key, bool optional = false) const {
        Value const *value = optional ? this->optional(key) : required(key);
        if (value == nullptr) {
            return std::nullopt;
        }
        return Fields::of(*value, child(key), _problems);
    }

    [[nodiscard]] std::vector<std::string> strings(std::string_view key) const {
        std::vector<std::string> out;
        each(key, [&](Value const &item, std::string const &item_path) {
            if (!item.is_string()) {
                graph_ir::note(_problems, item_path, item.position, fmt::format("expected a string, found {}", item.type_name()));
                return;
            }
            out.push_back(item.as_string());
        });
        return out;
    }

    /// An array of signed integers, for a field whose entries are not counts.
    [[nodiscard]] std::vector<std::int64_t> integers(std::string_view key) const {
        std::vector<std::int64_t> out;
        each(key, [&](Value const &item, std::string const &item_path) {
            if (!item.is_int()) {
                graph_ir::note(_problems, item_path, item.position, "expected an integer");
                out.push_back(0);
                return;
            }
            out.push_back(item.as_int());
        });
        return out;
    }

    /// An array of counts or ids; a negative or non-integer entry is reported with @p noun.
    [[nodiscard]] std::vector<std::size_t> counts(std::string_view key, std::string_view noun = "a non-negative integer") const {
        std::vector<std::size_t> out;
        each(key, [&](Value const &item, std::string const &item_path) {
            if (!item.is_int() || item.as_int() < 0) {
                graph_ir::note(_problems, item_path, item.position, fmt::format("expected {}", noun));
                out.push_back(0);
                return;
            }
            out.push_back(static_cast<std::size_t>(item.as_int()));
        });
        return out;
    }

  private:
    Object const  &_object;
    std::string    _path;
    Problems      &_problems;
    json::Position _at;
};

/// A typed scalar, back to a @ref PrefactorScalar.
PrefactorScalar read_prefactor(Fields const &scalar) {
    auto const dtype = scalar.named<packed_gemm::ScalarType>("dtype", scalar_type_from_name, "dtype", packed_gemm::ScalarType::Float64);

    auto const component = [&](std::string_view key) -> double {
        Value const *value = scalar.optional(key);
        if (value == nullptr) {
            scalar.note(fmt::format("typed scalar is missing '{}'", key));
            return 0.0;
        }
        if (auto const number = tagged_number(*value); number.has_value()) {
            return *number;
        }
        scalar.note(key, value->position, R"(expected a number or one of the special tags "nan", "inf", "-inf")");
        return 0.0;
    };

    double const re = component("re");
    switch (dtype) {
    case packed_gemm::ScalarType::Float32:
        return PrefactorScalar{static_cast<float>(re)};
    case packed_gemm::ScalarType::Float64:
        return PrefactorScalar{re};
    case packed_gemm::ScalarType::Complex64:
        return PrefactorScalar{std::complex<float>{static_cast<float>(re), static_cast<float>(component("im"))}};
    case packed_gemm::ScalarType::Complex128:
        return PrefactorScalar{std::complex<double>{re, component("im")}};
    default:
        scalar.note("a typed scalar cannot have dtype 'unknown'");
        return PrefactorScalar{double{0}};
    }
}

PrefactorScalar Fields::prefactor(std::string_view key, PrefactorScalar fallback) const {
    auto const scalar = object_at(key);
    return scalar.has_value() ? read_prefactor(*scalar) : fallback;
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
ProvenanceTag read_provenance_tag(Fields const &record) {
    ProvenanceTag out;
    Value const  *value = record.optional("tag");
    if (value == nullptr || value->is_null()) {
        return out;
    }
    auto const tag = Fields::of(*value, record.child("tag"), record.problems());
    if (!tag.has_value()) {
        return out;
    }
    out.name = tag->str("name");

    if (Value const *attributes = tag->optional("attributes"); attributes != nullptr && !attributes->is_null()) {
        if (auto const entries = Fields::of(*attributes, tag->child("attributes"), record.problems())) {
            for (auto const &key : entries->object().keys()) {
                Value const *entry = entries->optional(key);
                if (entry == nullptr) {
                    continue;
                }
                if (!entry->is_string()) {
                    entries->note(key, entry->position, fmt::format("expected a string, found {}", entry->type_name()));
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
        tag->note("a provenance tag with an empty name says nothing; omit the key instead");
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
std::vector<PermutationOperator> read_permutation_operators(Fields const &descriptor) {
    std::vector<PermutationOperator> out;
    Problems                        &problems = descriptor.problems();
    descriptor.each(
        "operators",
        [&](Value const &op_value, std::string const &op_path) {
            if (!op_value.is_array()) {
                note(problems, op_path, op_value.position, fmt::format("expected an array, found {}", op_value.type_name()));
                return;
            }
            Array const        &groups = op_value.as_array();
            PermutationOperator op;
            bool                well_formed = true;
            for (std::size_t g = 0; g < groups.size(); ++g) {
                std::string const group_path = fmt::format("{}[{}]", op_path, g);
                if (!groups[g].is_array()) {
                    note(problems, group_path, groups[g].position, fmt::format("expected an array, found {}", groups[g].type_name()));
                    well_formed = false;
                    break;
                }
                Array const             &letters = groups[g].as_array();
                std::vector<std::string> group;
                for (std::size_t l = 0; l < letters.size(); ++l) {
                    if (!letters[l].is_string()) {
                        note(problems, fmt::format("{}[{}]", group_path, l), letters[l].position,
                             fmt::format("expected an index letter, found {}", letters[l].type_name()));
                        well_formed = false;
                        continue;
                    }
                    group.push_back(letters[l].as_string());
                }
                if (group.empty()) {
                    note(problems, group_path, groups[g].position, "a permutation operator group names no index");
                    well_formed = false;
                }
                op.groups.push_back(std::move(group));
            }

            if (op.groups.size() < 2) {
                note(problems, op_path, op_value.position,
                     fmt::format("a permutation operator needs at least two groups, found {}", op.groups.size()));
                well_formed = false;
            }
            if (well_formed) {
                out.push_back(std::move(op));
            }
        },
        /*optional=*/true);
    return out;
}

BoundExpr read_bound_expr(Value const &value, std::string const &path, Problems &problems) {
    auto const object = Fields::of(value, path, problems);
    if (!object.has_value()) {
        return BoundExpr{std::int64_t{0}};
    }
    if (Value const *literal = object->optional("const"); literal != nullptr) {
        if (!literal->is_int()) {
            object->note("const", literal->position, "expected an integer");
            return BoundExpr{std::int64_t{0}};
        }
        return BoundExpr{literal->as_int()};
    }
    if (Value const *param = object->optional("param"); param != nullptr) {
        if (!param->is_string()) {
            object->note("param", param->position, "expected a string");
            return BoundExpr{std::int64_t{0}};
        }
        return BoundExpr{param->as_string()};
    }
    object->note(R"(a BoundExpr must be {"const": <integer>} or {"param": "<name>"})");
    return BoundExpr{std::int64_t{0}};
}

PredExpr read_pred_expr(Value const &value, std::string const &path, Problems &problems, GateFlagTable const &gates) {
    auto const object = Fields::of(value, path, problems);
    if (!object.has_value()) {
        return PredExpr{true};
    }
    if (Value const *literal = object->optional("const"); literal != nullptr) {
        if (!literal->is_bool()) {
            object->note("const", literal->position, "expected a bool");
            return PredExpr{true};
        }
        return PredExpr{literal->as_bool()};
    }
    // A present key that holds no object is reported once, by Fields::of, and yields the
    // harmless default; it does not fall through to the "must be one of" message.
    auto const body = [&](std::string_view key) -> std::optional<std::optional<Fields>> {
        Value const *entry = object->optional(key);
        if (entry == nullptr) {
            return std::nullopt;
        }
        return Fields::of(*entry, object->child(key), problems);
    };
    if (auto const present = body("compare")) {
        if (!present->has_value()) {
            return PredExpr{true};
        }
        auto const  &compare = **present;
        Value const *lhs     = compare.required("lhs");
        auto const   op      = compare.named<CmpOp>("op", cmp_op_from_name, "comparison operator", CmpOp::Lt);
        Value const *rhs     = compare.required("rhs");
        if (lhs == nullptr || rhs == nullptr) {
            return PredExpr{true};
        }
        return PredExpr::compare(read_bound_expr(*lhs, compare.child("lhs"), problems), op,
                                 read_bound_expr(*rhs, compare.child("rhs"), problems));
    }
    if (auto const present = body("iteration")) {
        if (!present->has_value()) {
            return PredExpr{true};
        }
        auto const  &iteration = **present;
        auto const   op        = iteration.named<CmpOp>("op", cmp_op_from_name, "comparison operator", CmpOp::Lt);
        Value const *rhs       = iteration.required("rhs");
        if (rhs == nullptr) {
            return PredExpr{true};
        }
        return PredExpr::iteration(op, read_bound_expr(*rhs, iteration.child("rhs"), problems));
    }
    if (auto const present = body("flag")) {
        if (!present->has_value()) {
            return PredExpr{true};
        }
        auto const       &flag  = **present;
        std::string const name  = flag.str("name");
        auto const        index = flag.count("index");
        auto const        found = gates.find(name);
        if (found == gates.end()) {
            std::vector<std::string> known;
            known.reserve(gates.size());
            for (auto const &[gate_name, buffer] : gates) {
                known.push_back(gate_name);
            }
            std::ranges::sort(known);
            flag.note(fmt::format("gate-flag array '{}' is not declared by this file. Declared: [{}]", name, fmt::join(known, ", ")));
            return PredExpr{true};
        }
        if (!index.has_value()) {
            return PredExpr{true};
        }
        return PredExpr::flag(found->second, *index);
    }
    object->note(R"(a PredExpr must be one of {"const": <bool>}, {"compare": ...}, {"iteration": ...} or {"flag": ...})");
    return PredExpr{true};
}

/// One tensor record. A MANIFEST entry and an intermediate share most of their
/// shape and differ in the two directions the schema keeps apart: a manifest
/// entry declares a direction and an alias parent, an intermediate declares
/// whether it is graph-owned and how it is initialized.
IrTensor read_tensor(Fields const &record, bool manifest_entry) {
    IrTensor out;
    out.id              = record.count("id").value_or(0);
    out.name            = record.str("name");
    out.dtype           = record.named<packed_gemm::ScalarType>("dtype", scalar_type_from_name, "dtype", packed_gemm::ScalarType::Float64);
    out.rank            = record.count("rank").value_or(0);
    out.dims            = record.counts("dims");
    out.dim_symbols     = record.strings("dim_symbols");
    out.spaces          = record.strings("spaces");
    out.spaces_inferred = record.flag("spaces_inferred");
    out.tag             = read_provenance_tag(record);
    out.scope           = record.named<TensorOwnership>("scope", tensor_ownership_from_name, "ownership scope", TensorOwnership::Graph);
    if (manifest_entry) {
        out.direction =
            record.named<ManifestDirection>("direction", manifest_direction_from_name, "manifest direction", ManifestDirection::Input);
        if (Value const *alias = record.required("aliases_input"); alias != nullptr) {
            if (alias->is_string()) {
                out.aliases_input = alias->as_string();
            } else if (!alias->is_null()) {
                record.note("aliases_input", alias->position, "expected a manifest name or null");
            }
        }
    } else {
        out.intermediate = record.flag("intermediate");
        out.init         = record.named<InitKind>("init", init_kind_from_name, "initialization kind", InitKind::None);
        // Optional: a file written before this key existed means Materialized, which is what
        // its graph's intermediates were.
        out.alloc = record.named<AllocState>("alloc", alloc_state_from_name, "allocation state", AllocState::Materialized,
                                             /*optional=*/true);
    }

    if (out.dims.size() != out.rank) {
        record.note(fmt::format("rank is {} but {} dims are given", out.rank, out.dims.size()));
    }
    if (!out.dim_symbols.empty() && out.dim_symbols.size() != out.rank) {
        record.note(
            fmt::format("rank is {} but {} dim symbols are given; the annotation is all axes or none", out.rank, out.dim_symbols.size()));
    }
    if (!out.spaces.empty() && out.spaces.size() != out.rank) {
        record.note(
            fmt::format("rank is {} but {} index spaces are given; the annotation is all axes or none", out.rank, out.spaces.size()));
    }

    // Optional: before 1.9.0 a rank-0 record was always a bare element, which is what an absent
    // key still means.
    if (Value const *kind = record.optional("rank0"); kind != nullptr) {
        if (!kind->is_string() || (kind->as_string() != "tensor" && kind->as_string() != "scalar")) {
            record.note("rank0", kind->position, R"(expected "tensor" or "scalar")");
        } else if (out.rank != 0) {
            record.note("rank0", kind->position,
                        fmt::format("only a rank-0 record says what it holds, and this one has rank {}", out.rank));
        } else {
            out.rank0_tensor = kind->as_string() == "tensor";
        }
    }

    if (Value const *outer = record.optional("outer"); outer != nullptr) {
        if (!outer->is_int() || outer->as_int() < 0) {
            record.note("outer", outer->position, "expected a non-negative integer");
        } else {
            out.outer = static_cast<std::size_t>(outer->as_int());
        }
    }
    return out;
}

IrFragment read_fragment(Value const &value, std::string const &path, Problems &problems, GateFlagTable const &gates,
                         SpaceRegistry const &registry);

/// A child fragment, as a Conditional's branch or a Loop or Setup body holds one.
// NOLINTNEXTLINE(misc-no-recursion): fragments nest.
std::shared_ptr<IrFragment> read_child_fragment(Value const &value, std::string const &path, Problems &problems, GateFlagTable const &gates,
                                                SpaceRegistry const &registry) {
    return std::make_shared<IrFragment>(read_fragment(value, path, problems, gates, registry));
}

/// A GEMM hint and the operand ids it names, which the builder resolves once every tensor is
/// registered.
void read_gemm_hint(IrNode &node, EinsumDescriptor &desc, Fields const &hint) {
    auto record     = std::make_shared<GemmHint>();
    record->m       = static_cast<int>(hint.integer("m"));
    record->n       = static_cast<int>(hint.integer("n"));
    record->k       = static_cast<int>(hint.integer("k"));
    record->trans_a = hint.transpose("trans_a", 'N');
    record->trans_b = hint.transpose("trans_b", 'N');
    for (auto const &[key, slot] : {std::pair{"a", &record->a}, std::pair{"b", &record->b}, std::pair{"c", &record->c}}) {
        auto const operand = hint.object_at(key);
        if (!operand.has_value()) {
            node.hint_ids.push_back(0);
            continue;
        }
        node.hint_ids.push_back(operand->count("id").value_or(0));
        slot->leading_dim = static_cast<int>(operand->integer("leading_dim"));
    }
    desc.gemm_hint = std::move(record);
}

/// One node's descriptor, per kind. The coverage is exactly the reconstructible
/// set, which is what makes an unknown kind here a file problem rather than a
/// gap: the writer could not have produced one.
// NOLINTNEXTLINE(misc-no-recursion): control-flow descriptors hold fragments.
void read_descriptor(IrNode &node, Fields const &fields, GateFlagTable const &gates, SpaceRegistry const &registry) {
    Problems &problems = fields.problems();

    switch (node.kind) {
    case OpKind::Transpose:
        node.descriptor = OpData{};
        return;
    case OpKind::Scale: {
        ScaleDescriptor desc;
        desc.factor     = fields.prefactor("factor", PrefactorScalar{double{1}});
        desc.params     = make_elementwise_params(desc.factor);
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Permute: {
        PermuteDescriptor desc;
        auto const        alpha = fields.prefactor("alpha", PrefactorScalar{double{1}});
        auto const        beta  = fields.prefactor("beta", PrefactorScalar{double{0}});
        desc.alpha              = as<std::complex<double>>(alpha);
        desc.beta               = as<std::complex<double>>(beta);
        desc.c_indices          = fields.strings("c_indices");
        desc.a_indices          = fields.strings("a_indices");
        desc.operators          = read_permutation_operators(fields);
        desc.params             = make_elementwise_params(alpha, beta);
        node.descriptor         = std::move(desc);
        return;
    }
    case OpKind::Axpby:
    case OpKind::DirectProduct:
    case OpKind::DirectDivision: {
        auto const read_scalars = [&]<typename Descriptor>(Descriptor desc) {
            desc.alpha      = fields.prefactor("alpha", PrefactorScalar{double{1}});
            desc.beta       = fields.prefactor("beta", PrefactorScalar{double{0}});
            desc.params     = make_elementwise_params(desc.alpha, desc.beta);
            node.descriptor = std::move(desc);
        };
        if (node.kind == OpKind::Axpby) {
            read_scalars(AxpbyDescriptor{});
        } else {
            read_scalars(ElementwiseBinaryDescriptor{});
        }
        return;
    }
    case OpKind::Einsum: {
        EinsumDescriptor desc;
        desc.spec.c_indices      = fields.strings("c_indices");
        desc.spec.a_indices      = fields.strings("a_indices");
        desc.spec.b_indices      = fields.strings("b_indices");
        desc.spec.link_indices   = fields.strings("link_indices");
        desc.spec.target_indices = fields.strings("target_indices");
        desc.spec.all_indices    = fields.strings("all_indices");
        desc.spec.scalar_output  = fields.flag("scalar_output");
        desc.conj_a              = fields.flag("conj_a");
        desc.conj_b              = fields.flag("conj_b");
        desc.spec.conj_a         = desc.conj_a;
        desc.spec.conj_b         = desc.conj_b;
        desc.operators           = read_permutation_operators(fields);
        desc.spec.scalar_type    = node.dtype;
        desc.c_prefactor         = fields.prefactor("c_prefactor", PrefactorScalar{double{0}});
        desc.ab_prefactor        = fields.prefactor("ab_prefactor", PrefactorScalar{double{1}});

        fields.each_object("letter_spaces", [&](Fields const &entry) {
            std::string const letter = entry.str("letter");
            std::string const space  = entry.str("space");
            auto const        id     = registry.find(space);
            if (!id.has_value()) {
                entry.note(fmt::format("index space '{}' is not registered in this process", space));
                return;
            }
            desc.letter_spaces.emplace_back(letter, *id);
        });

        if (auto const hint = fields.object_at("gemm_hint", /*optional=*/true)) {
            read_gemm_hint(node, desc, *hint);
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Dot: {
        DotDescriptor desc;
        desc.conjugated = fields.flag("conjugated");
        node.descriptor = desc;
        return;
    }
    case OpKind::Trace:
        node.descriptor = TraceDescriptor{};
        return;
    case OpKind::Gemm: {
        GemmDescriptor desc;
        desc.alpha      = fields.prefactor("alpha", PrefactorScalar{double{1}});
        desc.beta       = fields.prefactor("beta", PrefactorScalar{double{0}});
        desc.trans_a    = fields.transpose("trans_a", 'n');
        desc.trans_b    = fields.transpose("trans_b", 'n');
        node.descriptor = desc;
        return;
    }
    case OpKind::Syev: {
        SyevDescriptor desc;
        // REQUIRED, not defaulted, even though the field arrived with this kind. No file
        // predates it: no earlier build could write a Syev node at all, so an absent key is a
        // malformed file rather than an older one, and the two LAPACK jobs leave A holding
        // different matrices. Guessing the default would silently change what a graph computes.
        desc.compute_eigenvectors = fields.flag("compute_eigenvectors");
        node.descriptor           = desc;
        return;
    }
    case OpKind::ElementTransform: {
        ElementTransformDescriptor desc;
        desc.op_name = fields.str("op");
        if (!desc.op_name.empty() && !element_ops::global_element_op_registry().contains(desc.op_name)) {
            fields.note("op", fields.position(),
                        fmt::format("element op '{}' is not registered in this process. Registered: [{}]", desc.op_name,
                                    fmt::join(element_ops::global_element_op_registry().names(), ", ")));
        }
        // OPTIONAL, and absent means the op's documented default rather than
        // zero: an older file predates the key entirely, and a node written by
        // this build omits it whenever the capture site named no number.
        if (auto const param = fields.number("param", /*optional=*/true)) {
            desc.param = *param;
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::WriteParam: {
        WriteParamDescriptor desc;
        desc.name = fields.str("param");
        desc.source_type =
            fields.named<ParamSourceType>("source_type", param_source_type_from_name, "parameter source type", ParamSourceType::Int64);
        if (Value const *expr = fields.optional("source_expr"); expr != nullptr) {
            desc.source_expr = read_bound_expr(*expr, fields.child("source_expr"), problems);
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Conditional: {
        ConditionalDescriptor desc;
        if (Value const *predicate = fields.required("predicate"); predicate != nullptr) {
            desc.predicate = read_pred_expr(*predicate, fields.child("predicate"), problems, gates);
        }
        if (Value const *then_branch = fields.required("then"); then_branch != nullptr) {
            node.then_branch = read_child_fragment(*then_branch, fields.child("then"), problems, gates, registry);
        }
        if (Value const *else_branch = fields.required("else"); else_branch != nullptr && !else_branch->is_null()) {
            node.else_branch = read_child_fragment(*else_branch, fields.child("else"), problems, gates, registry);
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Loop: {
        LoopDescriptor desc;
        desc.max_iterations = fields.count("max_iterations").value_or(0);
        if (Value const *condition = fields.required("condition"); condition != nullptr) {
            desc.condition = read_pred_expr(*condition, fields.child("condition"), problems, gates);
        }
        if (Value const *body = fields.required("body"); body != nullptr) {
            node.body = read_child_fragment(*body, fields.child("body"), problems, gates, registry);
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
        if (auto const epsilon = fields.number("epsilon")) {
            desc.epsilon = *epsilon;
        }
        desc.points = fields.integer("points");
        for (std::int64_t const sign : fields.integers("signs")) {
            desc.signs.push_back(static_cast<std::int8_t>(sign >= 0 ? 1 : -1));
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Setup: {
        SetupDescriptor desc;
        if (Value const *body = fields.required("body"); body != nullptr) {
            node.body = read_child_fragment(*body, fields.child("body"), problems, gates, registry);
        }
        node.descriptor = std::move(desc);
        return;
    }
    case OpKind::Custom: {
        // A descriptor declared outside this library, read by the codec registered for it. The
        // codec reads its fields with take(), so the document-wide unconsumed-key audit covers
        // them as it covers this library's own.
        std::string const      name  = fields.str("name");
        DescriptorCodec const *codec = find_descriptor_codec(name);
        if (codec == nullptr) {
            fields.note(
                "name", fields.position(),
                fmt::format("no descriptor is registered under '{}'; register its codec (register_descriptor) before loading", name));
            return;
        }
        auto const value = fields.object_at("value");
        if (!value.has_value()) {
            return;
        }
        try {
            node.descriptor = codec->read(value->object());
        } catch (std::exception const &error) {
            graph_ir::note(problems, value->path(), value->position(), fmt::format("descriptor '{}': {}", name, error.what()));
        }
        return;
    }
    default:
        fields.note(
            fmt::format("op kind '{}' is not one this schema can describe; the reconstructible set is what a file may contain", node.kind));
        return;
    }
}

// NOLINTNEXTLINE(misc-no-recursion): see read_descriptor.
IrNode read_node(Fields const &record, GateFlagTable const &gates, SpaceRegistry const &registry) {
    IrNode out;
    out.id      = record.count("id").value_or(0);
    out.kind    = record.named<OpKind>("kind", op_kind_from_name, "op kind", OpKind::Custom);
    out.label   = record.str("label");
    out.inputs  = record.counts("inputs", "a non-negative tensor id");
    out.outputs = record.counts("outputs", "a non-negative tensor id");
    out.dtype   = record.named<packed_gemm::ScalarType>("dtype", scalar_type_from_name, "dtype", packed_gemm::ScalarType::Unknown);
    out.rank    = record.count("rank").value_or(0);

    if (auto const descriptor = record.object_at("descriptor")) {
        read_descriptor(out, *descriptor, gates, registry);
    }
    return out;
}

// NOLINTNEXTLINE(misc-no-recursion): fragments nest.
IrFragment read_fragment(Value const &value, std::string const &path, Problems &problems, GateFlagTable const &gates,
                         SpaceRegistry const &registry) {
    IrFragment out;
    auto const fragment = Fields::of(value, path, problems);
    if (!fragment.has_value()) {
        return out;
    }
    out.name = fragment->str("name");
    fragment->each_object("tensors", [&](Fields const &record) { out.tensors.push_back(read_tensor(record, /*manifest_entry=*/false)); });
    fragment->each_object("nodes", [&](Fields const &record) { out.nodes.push_back(read_node(record, gates, registry)); });
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
    IrDocument out;
    auto const document = Fields::of(root, "$", problems);
    if (!document.has_value()) {
        return out;
    }

    // The version gate runs FIRST and, when it refuses, nothing else is read:
    // every message a newer schema would produce would be about fields this
    // build does not understand, which buries the one message that matters.
    out.version = document->str(key_version);
    if (out.version.empty()) {
        return out;
    }
    auto const order = compare_semver(out.version, graph_ir_schema_version);
    if (!order.has_value()) {
        document->note(fmt::format("'{}' is not a major.minor.patch schema version", out.version));
        return out;
    }
    if (*order > 0) {
        document->note(fmt::format("this file is einsums_graph_ir {} and this build understands up to {}; a newer build reads an older IR, "
                                   "never the reverse",
                                   out.version, graph_ir_schema_version));
        return out;
    }

    // Provenance is DATA. Every field is consumed so the strict audit passes,
    // and not one of them is acted on.
    if (auto const provenance = document->object_at(key_provenance)) {
        provenance->object().mark_consumed("library_version");
        provenance->object().mark_consumed("config_fingerprint");
        // Read rather than merely consumed, and reading is not acting: nothing behaves
        // differently for what is in here. A malformed entry is skipped rather than
        // failing the load, because provenance is not structure and a file whose history
        // is unreadable still describes a perfectly good graph.
        if (Value const *passes = provenance->optional("structural_passes"); passes != nullptr && passes->is_array()) {
            for (auto const &entry : passes->as_array()) {
                if (entry.is_string()) {
                    out.structural_passes.push_back(entry.as_string());
                }
            }
        }
    }

    out.name = document->str("name");

    // Gate-flag declarations are read before anything that can reference one.
    document->each_object("gate_flags", [&](Fields const &entry) {
        std::string const name = entry.str("name");
        if (auto const size = entry.count("size")) {
            out.gate_flags.emplace_back(name, *size);
        }
    });

    for (auto const &[name, size] : out.gate_flags) {
        out.gate_buffers.emplace(name, std::make_shared<std::vector<std::uint8_t>>(size, std::uint8_t{0}));
    }
    GateFlagTable const &gates = out.gate_buffers;

    document->each_object("manifest", [&](Fields const &record) { out.manifest.push_back(read_tensor(record, /*manifest_entry=*/true)); });

    if (auto const spaces = document->object_at("spaces")) {
        out.space_names = spaces->strings("names");
        for (auto const &name : out.space_names) {
            if (!registry.find(name).has_value()) {
                std::vector<std::string> known;
                for (SpaceId const id : registry.ids()) {
                    known.push_back(registry.space(id).name);
                }
                spaces->note(
                    "names", spaces->position(),
                    fmt::format("index space '{}' is not registered in this process. Registered: [{}]", name, fmt::join(known, ", ")));
            }
        }
        spaces->each_object("symbol_ties",
                            [&](Fields const &entry) { out.symbol_ties.emplace_back(entry.str("symbol"), entry.str("space")); });
    }

    // OPTIONAL, and the compatibility policy says an added field takes a documented default:
    // a file written before this section existed describes a graph nothing approximated,
    // which is an empty list and is exactly right.
    document->each_object(
        "approximations",
        [&](Fields const &entry) {
            ApproximationRecord record;
            record.pass_name = entry.str("pass_name");
            // The effect has no safe default: a bound whose units are unreadable cannot be
            // composed or compared, and guessing one would silently produce a number in
            // the wrong scale. named() reports the unresolvable name and the load fails.
            record.effect = entry.named<ApproximationEffect>("effect", approximation_effect_from_name, "approximation effect",
                                                             ApproximationEffect::NormRelative);
            if (auto const tolerance = entry.number("tolerance")) {
                record.tolerance = *tolerance;
            }
            if (auto const bound = entry.number("bound")) {
                record.bound = *bound;
            }
            // OPTIONAL, and defaulted to asserted rather than measured. A file written
            // before this key existed carries a number whose provenance nobody recorded,
            // and reading it as evidence would promote a guess by nothing more than a
            // newer build having opened it.
            record.origin  = entry.named<ApproximationOrigin>("origin", approximation_origin_from_name, "approximation origin",
                                                              ApproximationOrigin::Asserted, /*optional=*/true);
            record.outputs = entry.strings("outputs");
            record.spaces  = entry.strings("spaces");
            record.setup   = entry.str("setup");
            // OPTIONAL, for the reason `origin` is: a file written before the key existed
            // named no parameter, which is exactly what an empty string says.
            if (Value const *measurement = entry.optional("measurement"); measurement != nullptr) {
                if (measurement->is_string()) {
                    record.measurement = measurement->as_string();
                } else {
                    entry.note("measurement", measurement->position, "expected a string");
                }
            }
            out.approximations.push_back(std::move(record));
        },
        /*optional=*/true);

    document->each_object("params", [&](Fields const &entry) { out.params.emplace_back(entry.str("name"), entry.integer("value")); });

    document->each_object("tensors", [&](Fields const &record) { out.tensors.push_back(read_tensor(record, /*manifest_entry=*/false)); });

    document->each_object("slot_redirects", [&](Fields const &entry) {
        auto const from = entry.count("from");
        auto const to   = entry.count("to");
        if (from.has_value() && to.has_value()) {
            out.slot_redirects.emplace_back(*from, *to);
        }
    });

    document->each_object("nodes", [&](Fields const &record) { out.nodes.push_back(read_node(record, gates, registry)); });
    return out;
}

EINSUMS_NAMESPACE_END(compute_graph::graph_ir)
