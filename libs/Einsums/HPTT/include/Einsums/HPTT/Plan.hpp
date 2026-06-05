//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/*
  Copyright 2018 Paul Springer

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <Einsums/HPTT/ComputeNode.hpp>

#include <vector>

namespace hptt {

/**
 * @class Plan
 *
 * \brief A plan encodes the execution of a tensor transposition.
 *
 * It stores the loop order and parallelizes each loop.
 */
class Plan {
  public:
    Plan() : _rootNodes(), _numTasks(0) {}

    /**
     * Construct a plan. Initialize the loop order and the number of threads for each level of loop.
     *
     * @param loopOrder Holds the indices in the order they are to be updated.
     * @param numThreadsAtLoop Contains the number of threads for each level of loop. Used to find the number of tasks as well.
     */
    Plan(std::vector<int> loopOrder, std::vector<int> numThreadsAtLoop);

    Plan(std::FILE *fp, bool swap_endian);

    ~Plan() = default;

    /**
     * Get the root node, but get a const pointer to it.
     */
    [[nodiscard]] ComputeNode const *get_root_node(int threadId) const;

    /**
     * Get the root node, but get a non-const pointer to it.
     */
    ComputeNode *get_root_node(int threadId);

    /**
     * Get the number of tasks for the plan.
     */
    [[nodiscard]] int get_num_tasks() const;

    /**
     * Print the loop order and number of threads for each level of loop.
     */
    void print() const;

    void write_to_file(std::FILE *fp) const;

  private:
    /**
     * @var numTasks_
     *
     * The number of tasks to spawn for the calculation.
     */
    int _numTasks;

    /**
     * @var loopOrder_
     *
     * Holds the indices in the order that they will be used.
     *
     * For example, if \f$ B_{1,0,2} \gets A_{0,1,2}\f$. loopOrder_ = {1,0,2} denotes that B is
     * traversed in a linear fashion.
     */
    std::vector<int> _loopOrder;

    /**
     * @var numThreadsAtLoop_
     *
     * Holds the number of threads for each level of the loop.
     */
    std::vector<int> _numThreadsAtLoop;

    /**
     * @var rootNodes_
     *
     * Holds the nodes used to compute the transposition.
     */
    std::vector<ComputeNode> _rootNodes;
};

} // namespace hptt
