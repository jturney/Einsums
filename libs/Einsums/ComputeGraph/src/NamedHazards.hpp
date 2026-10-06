//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file NamedHazards.hpp
 * @brief Ordering edges between accesses to named resources, for the schedulers.
 *
 * Private to the sources that derive dependency edges: the graph's hazard scan and Reorder. Both
 * order the parameters and disk datasets a node reads and writes (see @ref named_reads) as they
 * order buffers, and the wildcard an opaque node carries changes what "the same resource" means,
 * so the relation is written once, here.
 */

#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief RAW, WAW and WAR edges between named-resource accesses, fed nodes in program order.
 *
 * A read follows the last write of its key; a write follows the last write and every read since.
 * A read of @ref any_named_key follows the last write of every key, and a write of it follows
 * every access before it and precedes every access after it, so an opaque node is ordered against
 * each named resource without unrelated resources gaining edges between themselves.
 */
class NamedHazards {
  public:
    /// Report through @p emit, once per producer, every edge into @p node from an earlier node.
    template <typename Emit>
    void visit(std::size_t node, std::vector<std::string> const &reads, std::vector<std::string> const &writes, Emit &&emit) {
        std::vector<std::size_t> from;
        auto const               add = [&from, node](std::size_t producer) {
            if (producer != node) {
                from.push_back(producer);
            }
        };
        auto const add_all = [&add](std::vector<std::size_t> const &producers) {
            for (std::size_t const producer : producers) {
                add(producer);
            }
        };

        for (auto const &key : reads) {
            if (_any_writer) {
                add(*_any_writer);
            }
            if (key == any_named_key) {
                for (auto const &[name, writer] : _writer) {
                    add(writer);
                }
                _any_readers.push_back(node);
            } else {
                if (auto const it = _writer.find(key); it != _writer.end()) {
                    add(it->second);
                }
                _readers[key].push_back(node);
            }
        }
        for (auto const &key : writes) {
            if (_any_writer) {
                add(*_any_writer);
            }
            add_all(_any_readers);
            if (key == any_named_key) {
                for (auto const &[name, writer] : _writer) {
                    add(writer);
                }
                for (auto const &[name, readers] : _readers) {
                    add_all(readers);
                }
                _writer.clear();
                _readers.clear();
                _any_readers.clear();
                _any_writer = node;
            } else {
                if (auto const it = _writer.find(key); it != _writer.end()) {
                    add(it->second);
                }
                if (auto const it = _readers.find(key); it != _readers.end()) {
                    add_all(it->second);
                    it->second.clear();
                }
                _writer[key] = node;
            }
        }

        std::ranges::sort(from);
        auto const [first, last] = std::ranges::unique(from);
        from.erase(first, last);
        for (std::size_t const producer : from) {
            emit(producer, node);
        }
    }

  private:
    std::unordered_map<std::string, std::size_t>              _writer;      ///< Last writer of each key.
    std::unordered_map<std::string, std::vector<std::size_t>> _readers;     ///< Readers of each key since its last write.
    std::optional<std::size_t>                                _any_writer;  ///< Last node that wrote every key.
    std::vector<std::size_t>                                  _any_readers; ///< Readers of every key since then.
};

EINSUMS_NAMESPACE_END(compute_graph::detail)
