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

#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/HPTT/ComputeNode.hpp>
#include <Einsums/HPTT/Plan.hpp>
#include <Einsums/HPTT/Utils.hpp>
#include <Einsums/Logging.hpp>

#include <cerrno>
#include <cstring>
#include <utility>

#include "Einsums/HPTT/Files.hpp"

namespace hptt {

Plan::Plan(std::vector<int> loopOrder, std::vector<int> numThreadsAtLoop)
    : _rootNodes(), _loopOrder(std::move(loopOrder)), _numThreadsAtLoop(numThreadsAtLoop) {
    _numTasks = 1;
    for (auto nt : numThreadsAtLoop)
        _numTasks *= nt;
    _rootNodes.resize(_numTasks);
}

ComputeNode const *Plan::get_root_node(int threadId) const {
    return &_rootNodes.at(threadId);
}
ComputeNode *Plan::get_root_node(int threadId) {
    return &_rootNodes.at(threadId);
}

void Plan::print() const {
    print_vector(_loopOrder, "LoopOrder");
    print_vector(_numThreadsAtLoop, "Parallelization");
}

int Plan::get_num_tasks() const {
    return _rootNodes.size();
}

void Plan::write_to_file(std::FILE *fp) const {
    size_t error = fwrite(&_numTasks, sizeof(int), 1, fp);

    if (error < 1) {
        EINSUMS_LOG_ERROR("HPTT: error writing to file: {}", std::strerror(errno));
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error.");
    }

    std::fflush(fp);

    size_t size = _loopOrder.size();

    error = fwrite(&size, sizeof(size_t), 1, fp);

    if (error < 1) {
        EINSUMS_LOG_ERROR("HPTT: error writing to file: {}", std::strerror(errno));
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error.");
    }

    std::fflush(fp);

    error = fwrite(_loopOrder.data(), sizeof(int), size, fp);

    if (error < size) {
        EINSUMS_LOG_ERROR("HPTT: error writing to file: {}", std::strerror(errno));
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error.");
    }

    std::fflush(fp);

    size = _numThreadsAtLoop.size();

    error = fwrite(&size, sizeof(size_t), 1, fp);

    if (error < 1) {
        EINSUMS_LOG_ERROR("HPTT: error writing to file: {}", std::strerror(errno));
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error.");
    }

    std::fflush(fp);

    error = fwrite(_numThreadsAtLoop.data(), sizeof(int), size, fp);

    if (error < size) {
        EINSUMS_LOG_ERROR("HPTT: error writing to file: {}", std::strerror(errno));
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error.");
    }

    std::fflush(fp);

    // Offsets for the root nodes.
    std::vector<uint32_t> offsets(_numTasks);

    long offset_table_pos = ftell(fp);

    int error2 = fseek(fp, (long)_numTasks * sizeof(uint32_t), SEEK_CUR);

    if (error2 != 0) {
        EINSUMS_LOG_ERROR("HPTT: error seeking in file: {}", std::strerror(errno));
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error.");
    }

    // Write the root nodes.
    for (int i = 0; i < _numTasks; i++) {
        auto *curr_node = &_rootNodes[i];
        // Get the offset of the root node.
        offsets[i] = ftell(fp);

        while (curr_node != nullptr) {
            NodeConstants constants{.start     = curr_node->start,
                                    .end       = curr_node->end,
                                    .inc       = curr_node->inc,
                                    .offDiffAB = curr_node->offDiffAB,
                                    .lda       = curr_node->lda,
                                    .ldb       = curr_node->ldb,
                                    .indexA    = curr_node->indexA,
                                    .indexB    = curr_node->indexB,
                                    .has_next  = (uint16_t)((curr_node->next == nullptr) ? 0 : 1),
                                    .pad       = 0};
            error = fwrite(&constants, sizeof(NodeConstants), 1, fp);

            if (error < 1) {
                EINSUMS_LOG_ERROR("HPTT: error writing to file: {}", std::strerror(errno));
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error");
            }

            std::fflush(fp);

            curr_node = curr_node->next.get();
        }
    }

    error2 = fseek(fp, offset_table_pos, SEEK_SET);

    if (error2 != 0) {
        EINSUMS_LOG_ERROR("HPTT: error seeking in file: {}", std::strerror(errno));
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error.");
    }

    error = fwrite(offsets.data(), sizeof(uint32_t), _numTasks, fp);

    std::fflush(fp);

    if (error < _numTasks) {
        EINSUMS_LOG_ERROR("HPTT: error writing to file: {}", std::strerror(errno));
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error.");
    }
}

Plan::Plan(std::FILE *fp, bool swap_endian) {
    size_t error = fread(&_numTasks, sizeof(int), 1, fp);

    if (error < 1) {
        EINSUMS_LOG_ERROR("HPTT: error reading from file: {}", std::strerror(errno));
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error.");
    }

    if (swap_endian) {
        _numTasks = byteswap(_numTasks);
    }

    size_t size;

    error = fread(&size, sizeof(size_t), 1, fp);

    if (error < 1) {
        EINSUMS_LOG_ERROR("HPTT: error reading from file: {}", std::strerror(errno));
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error.");
    }

    if (swap_endian) {
        size = byteswap(size);
    }

    _loopOrder.resize(size);

    error = fread(_loopOrder.data(), sizeof(int), size, fp);

    if (error < size) {
        EINSUMS_LOG_ERROR("HPTT: error reading from file: {}", std::strerror(errno));
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error.");
    }

    if (swap_endian) {
        for (int &i : _loopOrder) {
            i = byteswap(i);
        }
    }

    error = fread(&size, sizeof(size_t), 1, fp);

    if (error < 1) {
        EINSUMS_LOG_ERROR("HPTT: error reading from file: {}", std::strerror(errno));
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error.");
    }

    if (swap_endian) {
        size = byteswap(size);
    }

    _numThreadsAtLoop.resize(size);

    error = fread(_numThreadsAtLoop.data(), sizeof(int), size, fp);

    if (error < size) {
        EINSUMS_LOG_ERROR("HPTT: error reading from file: {}", std::strerror(errno));
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error.");
    }

    if (swap_endian) {
        for (int &i : _numThreadsAtLoop) {
            i = byteswap(i);
        }
    }

    // Get the offsets for each root node.
    std::vector<uint32_t> offsets;
    offsets.resize(_numTasks);

    error = fread(offsets.data(), sizeof(uint32_t), _numTasks, fp);

    if (error < _numTasks) {
        EINSUMS_LOG_ERROR("HPTT: error reading from file: {}", std::strerror(errno));
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error");
    }

    if (swap_endian) {
        for (int i = 0; i < _numTasks; i++) {
            offsets[i] = byteswap(offsets[i]);
        }
    }

    _rootNodes.resize(_numTasks);

    if (swap_endian) {
        for (int i = 0; i < _numTasks; i++) {
            NodeConstants constants;
            auto         *curr_node = &_rootNodes[i];

            int error2 = fseek(fp, offsets[i], SEEK_SET);

            if (error2 < 0) {
                EINSUMS_LOG_ERROR("HPTT: error while seeking: {}", std::strerror(errno));
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error");
            }

            do {
                error = fread(&constants, sizeof(NodeConstants), 1, fp);

                if (error < 1) {
                    EINSUMS_LOG_ERROR("HPTT: error reading from file: {}", std::strerror(errno));
                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error");
                }

                curr_node->start     = byteswap(constants.start);
                curr_node->end       = byteswap(constants.end);
                curr_node->inc       = byteswap(constants.inc);
                curr_node->lda       = byteswap(constants.lda);
                curr_node->ldb       = byteswap(constants.ldb);
                curr_node->indexA    = byteswap(constants.indexA);
                curr_node->indexB    = byteswap(constants.indexB);
                curr_node->offDiffAB = byteswap(constants.offDiffAB);

                if (constants.has_next) {
                    curr_node->next = std::make_unique<ComputeNode>();
                } else {
                    curr_node->next.reset();
                }
                curr_node = curr_node->next.get();
            } while (constants.has_next);
        }
    } else {
        for (int i = 0; i < _numTasks; i++) {
            NodeConstants constants;
            auto         *curr_node = &_rootNodes[i];

            int error2 = fseek(fp, offsets[i], SEEK_SET);

            if (error2 < 0) {
                EINSUMS_LOG_ERROR("HPTT: error while seeking: {}", std::strerror(errno));
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error");
            }

            do {
                error = fread(&constants, sizeof(NodeConstants), 1, fp);

                if (error < 1) {
                    EINSUMS_LOG_ERROR("HPTT: error reading from file: {}", std::strerror(errno));
                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error");
                }

                curr_node->start     = constants.start;
                curr_node->end       = constants.end;
                curr_node->inc       = constants.inc;
                curr_node->lda       = constants.lda;
                curr_node->ldb       = constants.ldb;
                curr_node->indexA    = constants.indexA;
                curr_node->indexB    = constants.indexB;
                curr_node->offDiffAB = constants.offDiffAB;

                if (constants.has_next) {
                    curr_node->next = std::make_unique<ComputeNode>();
                } else {
                    curr_node->next.reset();
                }
                curr_node = curr_node->next.get();
            } while (constants.has_next);
        }
    }
}
} // namespace hptt
