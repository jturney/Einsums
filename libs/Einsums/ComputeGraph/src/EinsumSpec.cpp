//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EinsumSpec.hpp>

#include <fmt/format.h>

#include <set>
#include <string>

namespace einsums::compute_graph {

namespace {

std::string strip_whitespace(std::string_view s) {
    std::string result;
    result.reserve(s.size());
    for (char const c : s) {
        if (c != ' ' && c != '\t') {
            result += c;
        }
    }
    return result;
}

bool has_commas(std::string_view s) {
    return s.find(',') != std::string_view::npos;
}

std::vector<std::string> parse_index_group(std::string_view group, bool multi_char_mode) {
    std::vector<std::string> indices;
    if (group.empty())
        return indices;

    if (!multi_char_mode) {
        for (char c : group)
            indices.emplace_back(1, c);
    } else {
        size_t start = 0;
        while (start <= group.size()) {
            size_t comma = group.find(',', start);
            if (comma == std::string_view::npos)
                comma = group.size();
            std::string_view idx = group.substr(start, comma - start);
            if (!idx.empty())
                indices.emplace_back(idx);
            start = comma + 1;
        }
    }
    return indices;
}

} // namespace

expected<ParsedEinsumSpec, GraphError> parse_einsum_spec(std::string_view spec) {
    std::string const stripped = strip_whitespace(spec);

    auto left_pos  = stripped.find("<-");
    auto right_pos = stripped.find("->");

    bool const has_left  = (left_pos != std::string::npos);
    bool const has_right = (right_pos != std::string::npos);

    if (!has_left && !has_right) {
        return unexpected(GraphError::parse(fmt::format("einsum spec '{}': missing '<-' or '->' arrow", spec)));
    }
    if (has_left && has_right) {
        return unexpected(GraphError::parse(fmt::format("einsum spec '{}': contains both '<-' and '->'", spec)));
    }

    std::string_view output_part;
    std::string_view inputs_part;

    if (has_left) {
        output_part = std::string_view(stripped).substr(0, left_pos);
        inputs_part = std::string_view(stripped).substr(left_pos + 2);
    } else {
        inputs_part = std::string_view(stripped).substr(0, right_pos);
        output_part = std::string_view(stripped).substr(right_pos + 2);
    }

    auto semi_pos = inputs_part.find(';');
    if (semi_pos == std::string_view::npos) {
        return unexpected(GraphError::parse(fmt::format("einsum spec '{}': missing ';' between operands", spec)));
    }

    std::string_view const a_part = inputs_part.substr(0, semi_pos);
    std::string_view const b_part = inputs_part.substr(semi_pos + 1);

    // Char-vs-comma is decided PER OPERAND: an operand containing ',' is comma-split
    // (so multi-char index names like "sig"/"lam" work); a comma-less operand is
    // char-split, one index per character. Deciding this globally silently
    // mis-tokenized a comma-less operand as a single multi-char index whenever a
    // sibling operand used commas, e.g. "ijab <- Q,a,i,j,f ; Q,b,f" parsed the
    // output as one index "ijab" instead of [i,j,a,b], giving wrong results (bug-1026).
    ParsedEinsumSpec result;
    result.raw       = std::string(spec);
    result.c_indices = parse_index_group(output_part, has_commas(output_part));
    result.a_indices = parse_index_group(a_part, has_commas(a_part));
    result.b_indices = parse_index_group(b_part, has_commas(b_part));

    if (result.a_indices.empty()) {
        return unexpected(GraphError::parse(fmt::format("einsum spec '{}': first operand has no indices", spec)));
    }
    if (result.b_indices.empty()) {
        return unexpected(GraphError::parse(fmt::format("einsum spec '{}': second operand has no indices", spec)));
    }

    return result;
}

std::vector<std::string> ParsedEinsumSpec::link_indices() const {
    std::set<std::string> const a_set(a_indices.begin(), a_indices.end());
    std::set<std::string> const b_set(b_indices.begin(), b_indices.end());
    std::set<std::string> const c_set(c_indices.begin(), c_indices.end());
    std::vector<std::string>    links;
    for (auto const &idx : a_set) {
        if (b_set.count(idx) && !c_set.count(idx)) {
            links.push_back(idx);
        }
    }
    return links;
}

std::vector<std::string> ParsedEinsumSpec::target_indices() const {
    std::set<std::string> c_set(c_indices.begin(), c_indices.end());
    return {c_set.begin(), c_set.end()};
}

expected<ParsedPermuteSpec, GraphError> parse_permute_spec(std::string_view spec) {
    std::string const stripped = strip_whitespace(spec);

    auto left_pos  = stripped.find("<-");
    auto right_pos = stripped.find("->");

    bool const has_left  = (left_pos != std::string::npos);
    bool const has_right = (right_pos != std::string::npos);

    if (!has_left && !has_right) {
        return unexpected(GraphError::parse(fmt::format("permute spec '{}': missing '<-' or '->' arrow", spec)));
    }
    if (has_left && has_right) {
        return unexpected(GraphError::parse(fmt::format("permute spec '{}': contains both '<-' and '->'", spec)));
    }

    // Check for semicolons (not allowed in permute)
    if (stripped.find(';') != std::string::npos) {
        return unexpected(GraphError::parse(fmt::format("permute spec '{}': semicolons are not allowed (only one input tensor)", spec)));
    }

    std::string_view output_part;
    std::string_view input_part;

    if (has_left) {
        output_part = std::string_view(stripped).substr(0, left_pos);
        input_part  = std::string_view(stripped).substr(left_pos + 2);
    } else {
        input_part  = std::string_view(stripped).substr(0, right_pos);
        output_part = std::string_view(stripped).substr(right_pos + 2);
    }

    // Per-operand char-vs-comma split (see parse_einsum_spec for the rationale).
    ParsedPermuteSpec result;
    result.raw       = std::string(spec);
    result.c_indices = parse_index_group(output_part, has_commas(output_part));
    result.a_indices = parse_index_group(input_part, has_commas(input_part));

    if (result.a_indices.empty()) {
        return unexpected(GraphError::parse(fmt::format("permute spec '{}': input has no indices", spec)));
    }
    if (result.c_indices.empty()) {
        return unexpected(GraphError::parse(fmt::format("permute spec '{}': output has no indices", spec)));
    }
    if (result.c_indices.size() != result.a_indices.size()) {
        return unexpected(GraphError::parse(fmt::format("permute spec '{}': output has {} indices but input has {}", spec,
                                                        result.c_indices.size(), result.a_indices.size())));
    }

    return result;
}

} // namespace einsums::compute_graph
