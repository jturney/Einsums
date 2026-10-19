//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file Decision.hpp
/// @brief What an optimizer pass did, declined or wanted said, as records a program can read.

#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief Whether a pass applied something, declined a candidate, or only had something to say.
 * @versionadded{2.0.0}
 */
enum class APIARY_EXPOSE APIARY_MODULE("graph") DecisionKind : std::uint8_t{
    Applied,  ///< The pass changed the graph, or chose something it reports.
    Declined, ///< The pass looked at a candidate and left it, for a reason.
    Note,     ///< Something the pass reported that is neither.
};

/**
 * @brief A reason a pass gives, declared once: a stable code and the words a person reads.
 *
 * The code is the contract: tests and tools match it, so the words can be improved without
 * breaking anyone, and a test naming a code that no longer exists stops compiling. Codes are
 * ``<pass>.<what>`` in lower case with hyphens, as in ``slab-streaming.read-after-update``.
 *
 * @code
 * inline constexpr Reason ReadAfterUpdate{"slab-streaming.read-after-update",
 *                                         "the update-backed tensor is read after a write of it, so it is read from memory"};
 * note_skip(ReadAfterUpdate, where);
 * @endcode
 * @versionadded{2.0.0}
 */
struct Reason {
    std::string_view code; ///< Stable, matched by programs.
    std::string_view text; ///< The shape-independent phrase a person reads, and skip_reasons() counts.
};

/**
 * @brief A named quantity a decision carries: a block size, a byte count, a run count.
 * @versionadded{2.0.0}
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") Fact {
    APIARY_EXPOSE APIARY_READONLY std::string name; ///< What it is, as in ``"block"``.
    APIARY_EXPOSE APIARY_READONLY double      value{0.0};
    APIARY_EXPOSE APIARY_READONLY std::string unit; ///< ``"B"`` for bytes, empty for a count.
};

/**
 * @brief One thing a pass did, declined or said, as a record.
 *
 * What @ref OptimizerPass::explain and @ref OptimizerPass::skip_reasons print is rendered from
 * these; a program reads the fields instead. A record from a call a pass has not yet given a
 * @ref Reason has an empty @ref code and carries its words in @ref reason and @ref text.
 * @versionadded{2.0.0}
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") Decision {
    APIARY_EXPOSE APIARY_READONLY DecisionKind kind{DecisionKind::Note};
    /// The pass that made it.
    APIARY_EXPOSE APIARY_READONLY std::string pass_name;
    /// The stable code of its @ref Reason; empty for a call that gave only words.
    APIARY_EXPOSE APIARY_READONLY std::string code;
    /// The shape-independent phrase: what @ref OptimizerPass::skip_reasons counts for a decline.
    APIARY_EXPOSE APIARY_READONLY std::string reason;
    /// The specifics: which tensor, which node, the numbers that decided it.
    APIARY_EXPOSE APIARY_READONLY std::string detail;
    /// The line a person reads, as the pass's report prints it.
    APIARY_EXPOSE APIARY_READONLY std::string text;
    /// The name of the graph or body it was made in.
    APIARY_EXPOSE APIARY_READONLY std::string graph;
    /// The node it is about, 0 when it is about none.
    APIARY_EXPOSE APIARY_READONLY NodeId node{0};
    /// That node's label.
    APIARY_EXPOSE APIARY_READONLY std::string node_label;
    /// The tensors it is about, by name.
    APIARY_EXPOSE APIARY_READONLY std::vector<std::string> tensors;
    /// The quantities it carries.
    APIARY_EXPOSE APIARY_READONLY std::vector<Fact> facts;
    /// The verbosity at which the report shows it.
    APIARY_EXPOSE APIARY_READONLY int level{1};

    /// @brief The value of the fact named @p name.
    /// @param[in] name The fact's name.
    /// @return Its value, or 0 when the decision carries no such fact.
    APIARY_EXPOSE [[nodiscard]] double fact(std::string_view name) const {
        for (auto const &f : facts) {
            if (f.name == name) {
                return f.value;
            }
        }
        return 0.0;
    }
};

EINSUMS_NAMESPACE_END(compute_graph)
