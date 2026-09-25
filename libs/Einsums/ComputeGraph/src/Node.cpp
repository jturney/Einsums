//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Node.hpp>

#include <string_view>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

std::optional<SpaceId> EinsumDescriptor::space_for_letter(std::string_view letter) const {
    for (auto const &entry : letter_spaces) {
        if (entry.first == letter) {
            return entry.second;
        }
    }
    return std::nullopt;
}

PrefactorScalar const &live_c_prefactor(EinsumDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->c_pf : desc.c_prefactor;
}

PrefactorScalar const &live_ab_prefactor(EinsumDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->ab_pf : desc.ab_prefactor;
}

bool live_conj_a(EinsumDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->conj_a : desc.conj_a;
}

bool live_conj_b(EinsumDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->conj_b : desc.conj_b;
}

PrefactorScalar const &live_alpha(AxpbyDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->alpha : desc.alpha;
}

PrefactorScalar const &live_beta(AxpbyDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->beta : desc.beta;
}

PrefactorScalar const &live_alpha(ElementwiseBinaryDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->alpha : desc.alpha;
}

PrefactorScalar const &live_beta(ElementwiseBinaryDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->beta : desc.beta;
}

PrefactorScalar const &live_factor(ScaleDescriptor const &desc) noexcept {
    return desc.params != nullptr ? desc.params->alpha : desc.factor;
}

std::shared_ptr<ElementwiseParams> make_elementwise_params(PrefactorScalar const &alpha, PrefactorScalar const &beta) {
    auto params   = std::make_shared<ElementwiseParams>();
    params->alpha = alpha;
    params->beta  = beta;
    return params;
}

std::shared_ptr<ElementwiseParams> live_or_private_params(std::shared_ptr<ElementwiseParams> const &declared, PrefactorScalar const &alpha,
                                                          PrefactorScalar const &beta) {
    if (declared != nullptr) {
        return declared;
    }
    return make_elementwise_params(alpha, beta);
}

std::string_view param_source_type_name(ParamSourceType type) noexcept {
    switch (type) {
    case ParamSourceType::Bool:
        return "bool";
    case ParamSourceType::Char:
        return "char";
    case ParamSourceType::SChar:
        return "signed char";
    case ParamSourceType::UChar:
        return "unsigned char";
    case ParamSourceType::Short:
        return "short";
    case ParamSourceType::UShort:
        return "unsigned short";
    case ParamSourceType::Int:
        return "int";
    case ParamSourceType::UInt:
        return "unsigned int";
    case ParamSourceType::Long:
        return "long";
    case ParamSourceType::ULong:
        return "unsigned long";
    case ParamSourceType::LongLong:
        return "long long";
    case ParamSourceType::ULongLong:
        return "unsigned long long";
    case ParamSourceType::Float:
        return "float";
    case ParamSourceType::Double:
        return "double";
    case ParamSourceType::LongDouble:
        return "long double";
    }
    return "int";
}

std::optional<ParamSourceType> param_source_type_from_name(std::string_view name) noexcept {
    for (auto const type : {ParamSourceType::Bool, ParamSourceType::Char, ParamSourceType::SChar, ParamSourceType::UChar,
                            ParamSourceType::Short, ParamSourceType::UShort, ParamSourceType::Int, ParamSourceType::UInt,
                            ParamSourceType::Long, ParamSourceType::ULong, ParamSourceType::LongLong, ParamSourceType::ULongLong,
                            ParamSourceType::Float, ParamSourceType::Double, ParamSourceType::LongDouble}) {
        if (param_source_type_name(type) == name) {
            return type;
        }
    }
    return std::nullopt;
}

std::vector<std::string> param_writes(Node const &node) {
    if (auto const *wd = node.op_data.get_if<WriteParamDescriptor>()) {
        return {wd->name};
    }
    return {};
}

std::vector<std::string> param_reads(Node const &node) {
    std::vector<std::string> names;
    auto const               add = [&names](BoundExpr const &bound) {
        if (bound.is_param()) {
            names.push_back(bound.param_name());
        }
    };

    if (auto const *vd = node.op_data.get_if<ViewDescriptor>()) {
        for (auto const &ax : vd->axes) {
            add(ax.lo);
            if (ax.kind == ViewAxis::Kind::Range) {
                add(ax.hi);
            }
        }
    } else if (auto const *cd = node.op_data.get_if<ConditionalDescriptor>()) {
        cd->predicate.collect_param_names(names);
    } else if (auto const *ld = node.op_data.get_if<LoopDescriptor>()) {
        ld->condition.collect_param_names(names);
    } else if (auto const *wd = node.op_data.get_if<WriteParamDescriptor>()) {
        if (wd->source_expr.has_value()) {
            add(*wd->source_expr);
        }
    }
    return names;
}

bool has_runtime_view_bounds(Node const &node) {
    if (node.kind != OpKind::View) {
        return false;
    }
    auto const *vd = node.op_data.get_if<ViewDescriptor>();
    if (vd == nullptr) {
        return true; // no descriptor to inspect: assume the slice moves
    }
    return std::ranges::any_of(
        vd->axes, [](ViewAxis const &ax) { return !ax.lo.is_const() || (ax.kind == ViewAxis::Kind::Range && !ax.hi.is_const()); });
}

EINSUMS_NAMESPACE_END(compute_graph)

EINSUMS_NAMESPACE_BEGIN(compute_graph)
namespace detail {

EinsumDescriptor build_einsum_descriptor(ParsedEinsumSpec const &parsed, PrefactorScalar c_pf, PrefactorScalar ab_pf, bool conj_a,
                                         bool conj_b) {
    EinsumDescriptor desc;
    desc.c_prefactor         = c_pf;
    desc.ab_prefactor        = ab_pf;
    desc.conj_a              = conj_a;
    desc.conj_b              = conj_b;
    desc.operators           = parsed.operators;
    desc.spec.c_indices      = parsed.c_indices;
    desc.spec.a_indices      = parsed.a_indices;
    desc.spec.b_indices      = parsed.b_indices;
    desc.spec.link_indices   = parsed.link_indices();
    desc.spec.target_indices = parsed.target_indices();
    desc.spec.all_indices    = desc.spec.target_indices;
    desc.spec.all_indices.insert(desc.spec.all_indices.end(), desc.spec.link_indices.begin(), desc.spec.link_indices.end());
    return desc;
}

std::string space_label(SpaceRegistry const *registry, SpaceId id) {
    if (registry != nullptr && id.valid() && id.value() < registry->size()) {
        return registry->space(id).name;
    }
    return "#" + std::to_string(id.value());
}

std::vector<std::pair<std::string, SpaceId>> build_letter_spaces(std::span<LetterSpaceOperand const> operands,
                                                                 SpaceRegistry const *registry, std::string_view context) {
    std::vector<std::pair<std::string, SpaceId>> bound;
    std::vector<char const *>                    origin; // parallel to `bound`: operand each entry came from

    for (auto const &operand : operands) {
        if (operand.indices == nullptr || operand.spaces == nullptr || operand.spaces->empty()) {
            continue;
        }
        std::size_t const slots = std::min(operand.indices->size(), operand.spaces->size());
        for (std::size_t slot = 0; slot < slots; ++slot) {
            SpaceId const id = (*operand.spaces)[slot];
            if (!id.valid()) {
                continue; // a partially annotated tensor: this axis simply says nothing
            }
            std::string const &letter = (*operand.indices)[slot];

            auto const existing = std::ranges::find_if(bound, [&letter](auto const &e) { return e.first == letter; });
            if (existing == bound.end()) {
                bound.emplace_back(letter, id);
                origin.push_back(operand.label);
                continue;
            }
            if (existing->second != id) {
                std::size_t const at = static_cast<std::size_t>(existing - bound.begin());
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "{}: index letter '{}' binds space '{}' on operand {} and space '{}' on operand {} within one "
                                        "contraction; a letter ranges over exactly one space per contraction",
                                        context, letter, space_label(registry, existing->second), origin[at], space_label(registry, id),
                                        operand.label);
            }
        }
    }

    std::ranges::sort(bound, [](auto const &lhs, auto const &rhs) { return lhs.first < rhs.first; });
    return bound;
}

std::vector<SpaceId> spaces_from_letters(std::vector<std::string> const                     &c_indices,
                                         std::vector<std::pair<std::string, SpaceId>> const &letter_spaces) {
    if (c_indices.empty() || letter_spaces.empty()) {
        return {};
    }
    std::vector<SpaceId> out;
    out.reserve(c_indices.size());
    for (auto const &letter : c_indices) {
        auto const found = std::ranges::find_if(letter_spaces, [&letter](auto const &e) { return e.first == letter; });
        if (found == letter_spaces.end()) {
            return {};
        }
        out.push_back(found->second);
    }
    return out;
}

} // namespace detail
EINSUMS_NAMESPACE_END(compute_graph)
