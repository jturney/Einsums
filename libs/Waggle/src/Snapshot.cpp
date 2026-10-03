//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "Snapshot.hpp"

#include <Waggle/Config.hpp>

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <utility>

WAGGLE_NAMESPACE_BEGIN

namespace {

/// @p from's statistics and annotations, without its children.
void copy_node(AggNode const &from, waggle_node &to) {
    to.name     = from.name;
    to.file     = from.file;
    to.line     = from.line;
    to.function = from.function;

    auto &s             = to.stats;
    s.size              = sizeof(waggle_node_stats);
    s.call_count        = from.call_count;
    s.exclusive_ns      = from.total_exclusive.count();
    s.exclusive_min_ns  = from.call_count == 0 ? 0 : from.exclusive_min.count();
    s.exclusive_max_ns  = from.exclusive_max.count();
    s.exclusive_mean_ns = from.total_exclusive_mean;
    s.exclusive_m2      = from.total_exclusive_M2;
    s.mem_alloc_count   = from.mem_alloc_count;
    s.mem_free_count    = from.mem_free_count;
    s.mem_alloc_bytes   = from.mem_alloc_bytes;
    s.mem_free_bytes    = from.mem_free_bytes;
    s.mem_peak_bytes    = from.mem_peak_bytes;
    std::copy_n(from.histogram, WAGGLE_HISTOGRAM_BUCKETS, s.histogram);

    for (auto const &[key, value] : from.annotations) {
        to.annotations.emplace_back(key, value);
    }
    for (auto const &[key, n] : from.numeric_annotations) {
        to.numeric_annotations.push_back({.key = key, .total = n.total, .min = n.min_val, .max = n.max_val, .count = n.count});
    }
}

/// Fold @p from into @p into, which holds the same zone from another thread.
void merge_node(waggle_node const &from, waggle_node &into) {
    auto       &a = into.stats;
    auto const &b = from.stats;
    if (b.call_count == 0) {
        // Only its memory and its children can differ.
    } else if (a.call_count == 0) {
        a.exclusive_min_ns  = b.exclusive_min_ns;
        a.exclusive_max_ns  = b.exclusive_max_ns;
        a.exclusive_mean_ns = b.exclusive_mean_ns;
        a.exclusive_m2      = b.exclusive_m2;
    } else {
        // Chan's pairwise update of the mean and the sum of squared deviations.
        auto const   na    = static_cast<double>(a.call_count);
        auto const   nb    = static_cast<double>(b.call_count);
        double const delta = b.exclusive_mean_ns - a.exclusive_mean_ns;
        a.exclusive_m2 += b.exclusive_m2 + delta * delta * na * nb / (na + nb);
        a.exclusive_mean_ns += delta * nb / (na + nb);
        a.exclusive_min_ns = std::min(a.exclusive_min_ns, b.exclusive_min_ns);
        a.exclusive_max_ns = std::max(a.exclusive_max_ns, b.exclusive_max_ns);
    }
    a.call_count += b.call_count;
    a.exclusive_ns += b.exclusive_ns;
    a.mem_alloc_count += b.mem_alloc_count;
    a.mem_free_count += b.mem_free_count;
    a.mem_alloc_bytes += b.mem_alloc_bytes;
    a.mem_free_bytes += b.mem_free_bytes;
    a.mem_peak_bytes = std::max(a.mem_peak_bytes, b.mem_peak_bytes);
    for (int i = 0; i < WAGGLE_HISTOGRAM_BUCKETS; ++i) {
        a.histogram[i] += b.histogram[i];
    }

    // An annotation's text keeps the lowest-numbered thread's value: across threads there is no
    // latest. Numeric ones combine.
    for (auto const &annotation : from.annotations) {
        if (std::ranges::none_of(into.annotations, [&](auto const &have) { return have.first == annotation.first; })) {
            into.annotations.push_back(annotation);
        }
    }
    for (auto const &n : from.numeric_annotations) {
        auto it = std::ranges::find(into.numeric_annotations, n.key, &waggle_node::Numeric::key);
        if (it == into.numeric_annotations.end()) {
            into.numeric_annotations.push_back(n);
        } else {
            it->total += n.total;
            it->min = std::min(it->min, n.min);
            it->max = std::max(it->max, n.max);
            it->count += n.count;
        }
    }
}

/// Copy the tree below @p from into @p snap, under @p to.
void copy_tree(waggle_snapshot &snap, AggNode const &from, waggle_node &to) {
    // An explicit stack: the trees can be deeper than the call stack allows.
    std::vector<std::pair<AggNode const *, waggle_node *>> pending{{&from, &to}};
    while (!pending.empty()) {
        auto [src, dst] = pending.back();
        pending.pop_back();
        for (auto const &child : src->children) {
            auto &copy = snap.nodes.emplace_back();
            copy_node(*child.second, copy);
            dst->children.push_back(&copy);
            pending.emplace_back(child.second.get(), &copy);
        }
    }
}

/// Merge the tree below @p from into the tree below @p into, matching children by name.
void merge_tree(waggle_snapshot &snap, waggle_node const &from, waggle_node &into) {
    std::vector<std::pair<waggle_node const *, waggle_node *>> pending{{&from, &into}};
    while (!pending.empty()) {
        auto [src, dst] = pending.back();
        pending.pop_back();
        for (auto const *child : src->children) {
            auto it = std::ranges::find(dst->children, child->name, &waggle_node::name);
            if (it == dst->children.end()) {
                auto &copy      = snap.nodes.emplace_back();
                copy.name       = child->name;
                copy.file       = child->file;
                copy.line       = child->line;
                copy.function   = child->function;
                copy.stats.size = sizeof(waggle_node_stats);
                dst->children.push_back(&copy);
                it = std::prev(dst->children.end());
            }
            merge_node(*child, **it);
            pending.emplace_back(child, *it);
        }
    }
}

/// Fill in every node's inclusive time below @p root, children before parents.
void compute_inclusive(waggle_node &root) {
    std::vector<waggle_node *> order{&root};
    // By index: the loop appends to what it walks.
    for (size_t i = 0; i < order.size(); ++i) { // NOLINT(modernize-loop-convert)
        for (auto *child : order[i]->children) {
            order.push_back(child);
        }
    }
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        auto &node              = **it;
        node.stats.inclusive_ns = node.stats.exclusive_ns;
        for (auto const *child : node.children) {
            node.stats.inclusive_ns += child->stats.inclusive_ns;
        }
    }
}

} // namespace

auto take_snapshot(Consumer &consumer, bool merge_threads) -> waggle_snapshot * {
    auto snap = std::make_unique<waggle_snapshot>();
    {
        auto const lock = consumer.lock_shared();
        // By thread id, so the order does not depend on the hash map's.
        std::map<uint32_t, ThreadState const *> threads;
        for (auto const &[id, state] : consumer.thread_data()) {
            threads.emplace(id, &state);
        }
        for (auto const &[id, state] : threads) {
            auto &root = snap->nodes.emplace_back();
            copy_node(state->root, root);
            copy_tree(*snap, state->root, root);
            snap->threads.push_back({.id = id, .name = consumer.thread_name(id), .root = &root});
        }
    }

    if (merge_threads) {
        auto  merged    = std::make_unique<waggle_snapshot>();
        auto &root      = merged->nodes.emplace_back();
        root.stats.size = sizeof(waggle_node_stats);
        for (auto const &thread : snap->threads) {
            merge_node(*thread.root, root);
            merge_tree(*merged, *thread.root, root);
        }
        merged->threads.push_back({.id = 0, .name = {}, .root = &root});
        snap = std::move(merged);
    }

    for (auto &thread : snap->threads) {
        compute_inclusive(*thread.root);
    }
    return snap.release();
}

auto find_node(waggle_node const *root, std::string_view path) -> waggle_node const * {
    waggle_node const *node = root;
    while (node != nullptr && !path.empty()) {
        auto const             slash = path.find('/');
        std::string_view const name  = path.substr(0, slash);
        path                         = slash == std::string_view::npos ? std::string_view{} : path.substr(slash + 1);
        auto const it                = std::ranges::find(node->children, name, &waggle_node::name);
        node                         = it == node->children.end() ? nullptr : *it;
    }
    return node;
}

WAGGLE_NAMESPACE_END
