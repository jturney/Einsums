//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/Timer.hpp"

#include "einsums/OpenMP.hpp"
#include "einsums/Print.hpp"

#include <fmt/chrono.h>

#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace einsums::timer {

using clock      = std::chrono::high_resolution_clock;
using time_point = std::chrono::time_point<clock>;

namespace {

std::mutex lock;

struct TimerDetail {
    // Description of the timing block
    std::string name{"(no name)"};

    // Accumulated runtime
    clock::duration total_time{0};

    // Number of times the timer has been called
    size_t total_calls{0};

    TimerDetail                       *parent;
    std::map<std::string, TimerDetail> children;
    std::vector<std::string>           order;

    time_point start_time;
};

TimerDetail *current_timer;
TimerDetail *root;

} // namespace

void initialize() {
    root              = new TimerDetail();
    root->name        = "Total Run Time";
    root->total_calls = 1;

    current_timer = root;

    // Determine timer overhead
    for (size_t i = 0; i < 1000; i++) {
        push("Timer Overhead");
        pop();
    }
}

void finalize() {
    assert(root == current_timer);
    delete root;
    root = current_timer = nullptr;
}

namespace {
using std::chrono::duration_cast;
using std::chrono::milliseconds;

void print_timer_info(TimerDetail *timer) { // NOLINT
    if (timer != root) {
        std::string buffer;
        if (timer->total_calls != 0) {
            buffer = fmt::format("{:>5} : {:>5} calls : {:>5} per call", duration_cast<milliseconds>(timer->total_time), timer->total_calls,
                                 duration_cast<milliseconds>(timer->total_time) / timer->total_calls);
        } else {
            buffer = "total_calls == 0!!!";
        }
        println("{0:<{1}} : {3: <{4}}{2}", buffer, 70 - print::current_indent_level(), timer->name, "", print::current_indent_level());
    } else {
        println();
        println();
        println("Timing information:");
        println();
    }

    if (!timer->children.empty()) {
        print::indent();

        for (auto &child : timer->order) {
            print_timer_info(&timer->children[child]);
        }

        print::deindent();
    }
}

} // namespace

void report() {
    print_timer_info(root);
}

void push(std::string name) {
    // assert(current_timer != nullptr);
    static bool already_warned{false};

    std::lock_guard<std::mutex> guard(lock);

    if (omp_get_thread_num() == 0) {
        if (omp_in_parallel()) {
            name = fmt::format("{} (master thread only)", name);
        }

        if (current_timer == nullptr) {
            if (already_warned == false) {
                println("Timer::push: Timer was not initialized prior to calling `push`. This is the only warning you will receive.");
                already_warned = true;
            }
            return;
        }

        if (current_timer->children.count(name) == 0) {
            current_timer->children[name].name   = name;
            current_timer->children[name].parent = current_timer;
            current_timer->order.push_back(name);
        }

        current_timer             = &current_timer->children[name];
        current_timer->start_time = clock::now();
    }
}

void pop() {
    static bool already_warned{false};

    std::lock_guard<std::mutex> guard(lock);

    if (omp_get_thread_num() == 0) {
        if (current_timer == nullptr) {
            if (already_warned == false) {
                println(
                    "Timer::pop: current_timer is already nullptr; something might be wrong. This is the only warning you will receive.");
                already_warned = true;
            }
            return;
        }

        current_timer->total_time += clock::now() - current_timer->start_time;
        current_timer->total_calls++;
        current_timer = current_timer->parent;
    }
}

} // namespace einsums::timer