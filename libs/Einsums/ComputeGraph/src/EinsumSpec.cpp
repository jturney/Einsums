//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

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

// Detect and strip a ``conj(...)`` wrapper around an operand's index block. The
// input is already whitespace-stripped. Sets @p conj and returns the inner
// index string; returns the input unchanged when not wrapped. The wrapper marks
// the operand for conjugation in the contraction, a no-op for real dtypes.
std::string_view unwrap_conj(std::string_view part, bool &conj) {
    conj = false;
    if (part.size() >= 6 && part.starts_with("conj(") && part.back() == ')') {
        conj = true;
        return part.substr(5, part.size() - 6);
    }
    return part;
}

// Parse the leading run of ``P(...)`` operators off a term. The input is already
// whitespace-stripped. Appends to @p ops and returns the remainder; on failure
// sets @p error and returns the input unchanged.
//
// Comma-vs-character mode is decided across the WHOLE operator interior, not per
// group: ``P(mu,nu/rho)`` has to read ``rho`` as one index, which the per-group
// rule the operand list uses would char-split into three.
std::string_view parse_permutation_prefix(std::string_view part, std::vector<PermutationOperator> &ops, std::string &error) {
    while (part.size() >= 2 && part[0] == 'P' && part[1] == '(') {
        std::size_t const close = part.find(')');
        if (close == std::string_view::npos) {
            error = "unterminated 'P('";
            return part;
        }

        std::string_view const interior = part.substr(2, close - 2);
        if (interior.empty()) {
            error = "empty 'P()'";
            return part;
        }

        bool const          multi_char_mode = has_commas(interior);
        PermutationOperator op;
        std::size_t         start = 0;
        while (start <= interior.size()) {
            std::size_t slash = interior.find('/', start);
            if (slash == std::string_view::npos) {
                slash = interior.size();
            }
            std::string_view const group = interior.substr(start, slash - start);
            if (group.empty()) {
                error = fmt::format("'P({})' has an empty group", interior);
                return part;
            }
            op.groups.push_back(parse_index_group(group, multi_char_mode));
            start = slash + 1;
        }

        // ``P(ij)`` is sugar for ``P(i/j)``, character mode only: ``P(mu,nu)``
        // is ambiguous between one group of two and two groups of one, and
        // guessing either way would silently compute the wrong operator.
        if (op.groups.size() == 1) {
            if (multi_char_mode) {
                error = fmt::format("'P({})' is ambiguous: write the groups explicitly, as 'P({}/...)'", interior, op.groups[0].front());
                return part;
            }
            if (op.groups[0].size() < 2) {
                error = fmt::format("'P({})' needs at least two groups", interior);
                return part;
            }
            std::vector<std::string> const letters = std::move(op.groups[0]);
            op.groups.clear();
            for (auto const &letter : letters) {
                op.groups.push_back({letter});
            }
        }

        ops.push_back(std::move(op));
        part = part.substr(close + 1);
    }
    return part;
}

// Semantic validation of a parsed operator list against the output index list.
// Returns a message naming the offender, or nullopt when the list is sound.
std::optional<std::string> validate_operators(std::vector<PermutationOperator> const &ops, std::vector<std::string> const &c_indices) {
    std::set<std::string> claimed; // letters already named by an earlier operator

    for (auto const &op : ops) {
        if (op.groups.size() < 2) {
            return fmt::format("operator '{}' needs at least two groups", op.render());
        }

        std::set<std::string> within;
        for (auto const &group : op.groups) {
            if (group.empty()) {
                return fmt::format("operator '{}' has an empty group", op.render());
            }
            for (auto const &letter : group) {
                if (!within.insert(letter).second) {
                    return fmt::format("operator '{}' names index '{}' twice", op.render(), letter);
                }
                if (!claimed.insert(letter).second) {
                    return fmt::format("operator '{}' names index '{}', which another operator already permutes", op.render(), letter);
                }
                // A permutation operator reorders OUTPUT axes. A letter that is
                // summed away has no axis to reorder, and `P(ik)` with `k`
                // summed is a plausible typo for `P(ij)`, so say so rather than
                // silently ignoring it.
                auto const occurrences = std::count(c_indices.begin(), c_indices.end(), letter);
                if (occurrences == 0) {
                    return fmt::format("operator '{}' names index '{}', which is not an output index", op.render(), letter);
                }
                if (occurrences > 1) {
                    return fmt::format("operator '{}' names index '{}', which the output repeats", op.render(), letter);
                }
            }
        }
    }
    return std::nullopt;
}

// Expand ONE operator into (letter substitution, sign) pairs.
//
// Each term is a coset of the Young subgroup S(g0) x ... x S(g[k-1]) in S(L),
// so there is one term per distribution of L's letters into the group shapes and
// the count is n! / prod(|g|!). Within a coset the representative taken is the
// one that displaces the fewest letters, ties broken by pairing the displaced
// letters between groups in their original order. That is what reproduces the
// literature's `P(i/jk) f = f(ijk) - f(jik) - f(kji)` instead of an
// equally-valid-on-valid-input set carrying different parities. See the warning
// on PermutationOperator: the choice is a convention, not a correctness
// property, because the operator presumes within-group antisymmetry.
std::vector<std::pair<std::map<std::string, std::string>, double>> expand_one(PermutationOperator const &op) {
    std::vector<std::string> letters;  // L, the groups concatenated
    std::vector<std::size_t> group_of; // parallel to L: which group a slot belongs to
    for (std::size_t t = 0; t < op.groups.size(); ++t) {
        for (auto const &letter : op.groups[t]) {
            letters.push_back(letter);
            group_of.push_back(t);
        }
    }
    std::size_t const n = letters.size();

    // Slot positions of each group. L is the concatenation, so a letter's
    // original slot is its own index and every group's slots are contiguous.
    std::vector<std::vector<std::size_t>> group_slots(op.groups.size());
    for (std::size_t p = 0; p < n; ++p) {
        group_slots[group_of[p]].push_back(p);
    }

    std::vector<std::pair<std::map<std::string, std::string>, double>> out;

    // A distribution (which letters end up in which group) IS a distinct
    // permutation of the group-label multiset, so `next_permutation` over the
    // sorted labels walks the cosets exactly once each, in a deterministic order
    // that starts at the identity. No recursion and no dedup pass, and the
    // n! / prod(|g|!) count falls out rather than being asserted after the fact.
    std::vector<std::size_t> labels = group_of; // sorted already: L is the concatenation

    do {
        // arrangement[p] = index in L of the letter that ends up at slot p
        std::vector<std::size_t> arrangement(n, 0);
        for (std::size_t t = 0; t < op.groups.size(); ++t) {
            std::vector<std::size_t> incomers;
            std::vector<bool>        slot_used(group_slots[t].size(), false);
            for (std::size_t letter_index = 0; letter_index < n; ++letter_index) {
                if (labels[letter_index] != t) {
                    continue;
                }
                if (group_of[letter_index] == t) {
                    // A letter that did not change group keeps its own slot,
                    // which is what minimizes the number of displaced letters.
                    arrangement[letter_index]                                          = letter_index;
                    auto const slot                                                    = std::ranges::find(group_slots[t], letter_index);
                    slot_used[static_cast<std::size_t>(slot - group_slots[t].begin())] = true;
                } else {
                    incomers.push_back(letter_index); // ascending, so in original order
                }
            }
            std::size_t next = 0;
            for (std::size_t s = 0; s < group_slots[t].size(); ++s) {
                if (!slot_used[s]) {
                    arrangement[group_slots[t][s]] = incomers[next++];
                }
            }
        }

        // sign = parity of the arrangement, as (-1)^(n - cycles).
        std::vector<bool> seen(n, false);
        std::size_t       cycles = 0;
        for (std::size_t p = 0; p < n; ++p) {
            if (seen[p]) {
                continue;
            }
            ++cycles;
            for (std::size_t q = p; !seen[q]; q = arrangement[q]) {
                seen[q] = true;
            }
        }
        double const sign = ((n - cycles) % 2 == 0) ? 1.0 : -1.0;

        std::map<std::string, std::string> substitution;
        for (std::size_t p = 0; p < n; ++p) {
            substitution[letters[p]] = letters[arrangement[p]];
        }
        out.emplace_back(std::move(substitution), sign);
    } while (std::ranges::next_permutation(labels).found);

    return out;
}

/// A spec's two sides, split at its one arrow: output is the side the arrow points at.
struct ArrowSides {
    std::string_view output;
    std::string_view inputs;
};

/// Split @p stripped (whitespace already removed) at its arrow. @p what names the spec kind in the
/// error ("einsum", "permute") and @p spec is the text as written, for the message.
expected<ArrowSides, GraphError> split_arrow(std::string_view stripped, std::string_view what, std::string_view spec) {
    auto const left  = stripped.find("<-");
    auto const right = stripped.find("->");
    if (left == std::string_view::npos && right == std::string_view::npos) {
        return unexpected(GraphError::parse(fmt::format("{} spec '{}': missing '<-' or '->' arrow", what, spec)));
    }
    if (left != std::string_view::npos && right != std::string_view::npos) {
        return unexpected(GraphError::parse(fmt::format("{} spec '{}': contains both '<-' and '->'", what, spec)));
    }
    if (left != std::string_view::npos) {
        return ArrowSides{.output = stripped.substr(0, left), .inputs = stripped.substr(left + 2)};
    }
    return ArrowSides{.output = stripped.substr(right + 2), .inputs = stripped.substr(0, right)};
}

/// The first index label containing a character that is not a letter or digit.
/// Letters plus digits allow numbered names like "i1"/"i2". A comma-less operand
/// is char-split, so without this a stray '@' / '$' / '.' silently becomes an
/// index label and the operation runs on a malformed spec.
std::optional<std::string> first_non_alphanumeric(std::vector<std::string> const &group) {
    for (auto const &idx : group) {
        if (!std::ranges::all_of(idx, [](unsigned char ch) { return std::isalnum(ch) != 0; })) {
            return idx;
        }
    }
    return std::nullopt;
}

} // namespace

std::string PermutationOperator::render() const {
    std::string out = "P(";
    for (std::size_t t = 0; t < groups.size(); ++t) {
        if (t != 0) {
            out += '/';
        }
        out += fmt::format("{}", fmt::join(groups[t], ","));
    }
    out += ')';
    return out;
}

std::vector<std::string> PermutationOperator::letters() const {
    std::vector<std::string> out;
    for (auto const &group : groups) {
        out.insert(out.end(), group.begin(), group.end());
    }
    return out;
}

std::vector<PermutationTerm> expand_permutation_operators(std::vector<std::string> const         &c_indices,
                                                          std::vector<PermutationOperator> const &operators) {
    // The identity term, which is also the whole answer when there is no operator.
    std::vector<PermutationTerm> terms{PermutationTerm{.c_indices = c_indices, .sign = 1.0}};

    // Operators name disjoint letter sets, so one's substitution leaves the
    // others' letters alone and the product is the Cartesian product of the
    // per-operator expansions. The FIRST operator varies slowest.
    for (auto const &op : operators) {
        auto const                   substitutions = expand_one(op);
        std::vector<PermutationTerm> next;
        next.reserve(terms.size() * substitutions.size());
        for (auto const &term : terms) {
            for (auto const &[substitution, sign] : substitutions) {
                PermutationTerm permuted;
                permuted.c_indices.reserve(term.c_indices.size());
                for (auto const &letter : term.c_indices) {
                    auto const it = substitution.find(letter);
                    permuted.c_indices.push_back(it == substitution.end() ? letter : it->second);
                }
                permuted.sign = term.sign * sign;
                next.push_back(std::move(permuted));
            }
        }
        terms = std::move(next);
    }

    return terms;
}

expected<ParsedEinsumSpec, GraphError> parse_einsum_spec(std::string_view spec) {
    std::string const stripped = strip_whitespace(spec);
    auto const        sides    = split_arrow(stripped, "einsum", spec);
    if (!sides) {
        return unexpected(sides.error());
    }
    std::string_view const output_part = sides->output;
    std::string_view       inputs_part = sides->inputs;

    // Permutation operators prefix the TERM, so they sit at the head of the
    // input side under either arrow. Strip them BEFORE the operand split and
    // before the per-operand comma-mode decision, which the commas inside
    // ``P(mu,nu/rho)`` would otherwise flip for the operand carrying the prefix.
    std::vector<PermutationOperator> operators;
    {
        std::string error;
        inputs_part = parse_permutation_prefix(inputs_part, operators, error);
        if (!error.empty()) {
            return unexpected(GraphError::parse(fmt::format("einsum spec '{}': {}", spec, error)));
        }
    }

    auto semi_pos = inputs_part.find(';');
    if (semi_pos == std::string_view::npos) {
        return unexpected(GraphError::parse(fmt::format("einsum spec '{}': missing ';' between operands", spec)));
    }

    std::string_view a_part = inputs_part.substr(0, semi_pos);
    std::string_view b_part = inputs_part.substr(semi_pos + 1);

    // An operator names output axes, so it belongs in front of the whole term.
    // Caught here rather than left to the alphanumeric check below, which would
    // report an index named "P(i" and send the reader looking in the wrong place.
    for (auto const &[part, which] : {std::pair{a_part, "first"}, std::pair{b_part, "second"}}) {
        if (part.find("P(") != std::string_view::npos) {
            return unexpected(GraphError::parse(fmt::format(
                "einsum spec '{}': permutation operator inside the {} operand; it prefixes the whole term, as 'C <- P(i/j) A ; B'", spec,
                which)));
        }
    }

    // A ``conj(...)`` wrapper around an operand marks it for conjugation in the
    // contraction (e.g. ``"ij <- conj(ki) ; kj"`` is A^H @ B). Strip it and record
    // the flag; the einsum op ORs these with any conj_a/conj_b kwargs.
    bool conj_a = false;
    bool conj_b = false;
    a_part      = unwrap_conj(a_part, conj_a);
    b_part      = unwrap_conj(b_part, conj_b);

    // Char-vs-comma is decided PER OPERAND: an operand containing ',' is comma-split
    // (so multi-char index names like "sig"/"lam" work); a comma-less operand is
    // char-split, one index per character. Deciding this globally silently
    // mis-tokenized a comma-less operand as a single multi-char index whenever a
    // sibling operand used commas, e.g. "ijab <- Q,a,i,j,f ; Q,b,f" parsed the
    // output as one index "ijab" instead of [i,j,a,b], giving wrong results.
    ParsedEinsumSpec result;
    result.raw       = std::string(spec);
    result.c_indices = parse_index_group(output_part, has_commas(output_part));
    result.a_indices = parse_index_group(a_part, has_commas(a_part));
    result.b_indices = parse_index_group(b_part, has_commas(b_part));
    result.conj_a    = conj_a;
    result.conj_b    = conj_b;
    result.operators = std::move(operators);

    for (auto const &[group, which] :
         {std::pair{std::cref(result.c_indices), "output"}, std::pair{std::cref(result.a_indices), "first operand"},
          std::pair{std::cref(result.b_indices), "second operand"}}) {
        if (auto const bad = first_non_alphanumeric(group)) {
            return unexpected(
                GraphError::parse(fmt::format("einsum spec '{}': {} index '{}' has a non-letter character", spec, which, *bad)));
        }
    }

    if (result.a_indices.empty()) {
        return unexpected(GraphError::parse(fmt::format("einsum spec '{}': first operand has no indices", spec)));
    }
    if (result.b_indices.empty()) {
        return unexpected(GraphError::parse(fmt::format("einsum spec '{}': second operand has no indices", spec)));
    }

    // An output index has to come from an input. The engine has no rule for one that does not: the generic loop
    // broadcasts the result along it while a BLAS route writes one element, so two routes disagreed. numpy
    // rejects it too.
    for (auto const &idx : result.c_indices) {
        if (std::ranges::find(result.a_indices, idx) == result.a_indices.end() &&
            std::ranges::find(result.b_indices, idx) == result.b_indices.end()) {
            return unexpected(GraphError::parse(fmt::format("einsum spec '{}': output index '{}' appears in neither operand", spec, idx)));
        }
    }

    if (auto const bad = validate_operators(result.operators, result.c_indices)) {
        return unexpected(GraphError::parse(fmt::format("einsum spec '{}': {}", spec, *bad)));
    }

    return result;
}

namespace {

// The ``P(...) P(...) `` prefix a rendered term carries, empty when there is
// none, which is what keeps every spec written before operators existed
// rendering byte-identically.
std::string render_operators(std::vector<PermutationOperator> const &operators) {
    std::string out;
    for (auto const &op : operators) {
        out += op.render();
        out += ' ';
    }
    return out;
}

} // namespace

std::string ParsedEinsumSpec::render() const {
    return fmt::format("{} <- {}{} ; {}", fmt::join(c_indices, ","), render_operators(operators), fmt::join(a_indices, ","),
                       fmt::join(b_indices, ","));
}

std::string ParsedPermuteSpec::render() const {
    return fmt::format("{} <- {}{}", fmt::join(c_indices, ","), render_operators(operators), fmt::join(a_indices, ","));
}

std::vector<std::string> ParsedEinsumSpec::link_indices() const {
    // Index lists are rank-bounded, so linear scans beat building three sets. Sorted and
    // duplicate-free, which is what makes the result independent of axis order.
    auto contains = [](std::vector<std::string> const &list, std::string const &idx) { return std::ranges::find(list, idx) != list.end(); };
    std::vector<std::string> links;
    for (auto const &idx : a_indices) {
        if (contains(b_indices, idx) && !contains(c_indices, idx) && !contains(links, idx)) {
            links.push_back(idx);
        }
    }
    std::ranges::sort(links);
    return links;
}

std::vector<std::string> ParsedEinsumSpec::target_indices() const {
    std::set<std::string> c_set(c_indices.begin(), c_indices.end());
    return {c_set.begin(), c_set.end()};
}

IndexRole index_role(std::string_view letter, std::vector<std::string> const &c, std::vector<std::string> const &a,
                     std::vector<std::string> const &b) noexcept {
    auto const in   = [letter](std::vector<std::string> const &list) { return std::ranges::find(list, letter) != list.end(); };
    bool const in_a = in(a);
    bool const in_b = in(b);
    bool const in_c = in(c);
    if (in_a && in_b) {
        return in_c ? IndexRole::Batch : IndexRole::Link;
    }
    if (in_a) {
        return in_c ? IndexRole::AFree : IndexRole::ALone;
    }
    if (in_b) {
        return in_c ? IndexRole::BFree : IndexRole::BLone;
    }
    return IndexRole::OutputOnly;
}

LinkPlacement link_placement(std::vector<std::string> const &indices, std::vector<std::string> const &link_indices) {
    std::set<std::string> const link_set(link_indices.begin(), link_indices.end());

    size_t nlink = 0;
    for (auto const &idx : indices) {
        if (link_set.count(idx) != 0) {
            nlink++;
        }
    }

    LinkPlacement placement;
    placement.prefix = true;
    placement.suffix = true;
    for (size_t pos = 0; pos < indices.size(); pos++) {
        bool const is_link = link_set.count(indices[pos]) != 0;
        if ((pos < nlink) != is_link) {
            placement.prefix = false;
        }
        if ((pos >= indices.size() - nlink) != is_link) {
            placement.suffix = false;
        }
    }
    return placement;
}

expected<ParsedPermuteSpec, GraphError> parse_permute_spec(std::string_view spec) {
    std::string const stripped = strip_whitespace(spec);

    auto const sides = split_arrow(stripped, "permute", spec);
    if (!sides) {
        return unexpected(sides.error());
    }
    if (stripped.find(';') != std::string::npos) {
        return unexpected(GraphError::parse(fmt::format("permute spec '{}': semicolons are not allowed (only one input tensor)", spec)));
    }
    std::string_view const output_part = sides->output;
    std::string_view       input_part  = sides->inputs;

    // Operators prefix the term, the same as in an einsum spec. C and A name the
    // same letters here, so there is no ambiguity about which list they permute.
    std::vector<PermutationOperator> operators;
    {
        std::string error;
        input_part = parse_permutation_prefix(input_part, operators, error);
        if (!error.empty()) {
            return unexpected(GraphError::parse(fmt::format("permute spec '{}': {}", spec, error)));
        }
    }

    // Per-operand char-vs-comma split (see parse_einsum_spec for the rationale).
    ParsedPermuteSpec result;
    result.raw       = std::string(spec);
    result.c_indices = parse_index_group(output_part, has_commas(output_part));
    result.a_indices = parse_index_group(input_part, has_commas(input_part));
    result.operators = std::move(operators);

    if (result.a_indices.empty()) {
        return unexpected(GraphError::parse(fmt::format("permute spec '{}': input has no indices", spec)));
    }
    if (result.c_indices.empty()) {
        return unexpected(GraphError::parse(fmt::format("permute spec '{}': output has no indices", spec)));
    }
    for (auto const &[group, which] : {std::pair{std::cref(result.c_indices), "output"}, std::pair{std::cref(result.a_indices), "input"}}) {
        if (auto const bad = first_non_alphanumeric(group)) {
            return unexpected(
                GraphError::parse(fmt::format("permute spec '{}': {} index '{}' has a non-letter character", spec, which, *bad)));
        }
    }
    if (result.c_indices.size() != result.a_indices.size()) {
        return unexpected(GraphError::parse(fmt::format("permute spec '{}': output has {} indices but input has {}", spec,
                                                        result.c_indices.size(), result.a_indices.size())));
    }

    for (auto const &idx : result.c_indices) {
        if (std::ranges::find(result.a_indices, idx) == result.a_indices.end()) {
            return unexpected(
                GraphError::parse(fmt::format("permute spec '{}': output index '{}' does not appear in the input", spec, idx)));
        }
    }

    if (auto const bad = validate_operators(result.operators, result.c_indices)) {
        return unexpected(GraphError::parse(fmt::format("permute spec '{}': {}", spec, *bad)));
    }

    return result;
}

EINSUMS_NAMESPACE_END(compute_graph)
