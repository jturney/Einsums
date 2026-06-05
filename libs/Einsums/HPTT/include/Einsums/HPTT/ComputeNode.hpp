//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <limits>
#include <memory>

namespace hptt {

/**
 * \brief A ComputeNode encodes a loop.
 *
 * Nodes form a singly-linked list via the \c next pointer. The root node
 * is owned by the Plan (in a \c std::vector<ComputeNode>), and each node
 * owns its successor via \c std::unique_ptr.
 */
class ComputeNode {
  public:
    ptrdiff_t start{-1};                                        //!< start index for at the current loop
    ptrdiff_t end{-1};                                          //!< end index for at the current loop
    ptrdiff_t inc{0};                                           //!< increment for at the current loop
    size_t    lda{0};                                           //!< stride of A w.r.t. the loop index
    size_t    ldb{0};                                           //!< stride of B w.r.t. the loop index
    bool      indexA{false};                                    //!< true if index of A is innermost (0)
    bool      indexB{false};                                    //!< true if index of B is innermost (0)
    ptrdiff_t offDiffAB{std::numeric_limits<ptrdiff_t>::min()}; //!< difference in offset A and B (i.e., A - B) at the current loop
    std::unique_ptr<ComputeNode> next;                          //!< next ComputeNode, or nullptr if this is the last (macro-kernel call)
};

} // namespace hptt
