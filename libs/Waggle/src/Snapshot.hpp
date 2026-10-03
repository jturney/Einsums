//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Waggle/Config.hpp>

#include <Waggle/Waggle.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Consumer.hpp"

/// One zone of a snapshot: a copy of an AggNode, its inclusive time computed once.
struct waggle_node {
    std::string name;
    std::string file;
    int         line = 0;
    std::string function;
    std::string domain;

    waggle_node_stats stats{};

    std::vector<std::pair<std::string, std::string>> annotations;
    struct Numeric {
        std::string key;
        double      total;
        double      min;
        double      max;
        uint64_t    count;
    };
    std::vector<Numeric> numeric_annotations;

    std::vector<waggle_node *> children;
};

/// The copied trees. Nodes live in a deque, so pointers to them stay valid as it grows.
struct waggle_snapshot {
    struct Thread {
        uint32_t     id = 0;
        std::string  name;
        waggle_node *root = nullptr;
    };

    std::deque<waggle_node> nodes;
    std::vector<Thread>     threads;
};

WAGGLE_NAMESPACE_BEGIN

/// Copy @p consumer's trees, one per thread or, with @p merge_threads, merged into one. Takes the
/// consumer's shared lock; the caller flushes first.
WAGGLE_EXPORT auto take_snapshot(Consumer &consumer, DomainTable const &domains, bool merge_threads) -> waggle_snapshot *;

/// The node at the '/'-separated @p path below @p root, or null.
WAGGLE_EXPORT auto find_node(waggle_node const *root, std::string_view path) -> waggle_node const *;

WAGGLE_NAMESPACE_END
