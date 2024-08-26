// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#include <einsums/print.hpp>
#include <einsums/profile/timer.hpp>

#include <fmt/chrono.h>

#include <cassert>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <omp.h>
#include <vector>

namespace einsums::profile::timer {

namespace {

std::mutex lock;

struct timer_detail {
    /// Description of the timing block
    std::string name{"(no name)"};

    /// Accumulated runtime
    clock::duration total_time{0};

    /// Number of times the timer_detail has been called
    size_t total_calls{0};

    timer_detail                       *parent{nullptr};
    std::map<std::string, timer_detail> children;
    std::vector<std::string>            order;

    time_point start_time;
};

timer_detail *current_timer{nullptr};
timer_detail *root{nullptr};

using std::chrono::duration_cast;
using std::chrono::milliseconds;

void print_timer_info(timer_detail *timer, std::FILE *fp) { // NOLINT
    if (timer && timer != root) {
        std::string buffer;
        if (timer->total_calls != 0) {
            buffer = fmt::format("{:>5} : {:>5} calls : {:>5} per call", duration_cast<milliseconds>(timer->total_time), timer->total_calls,
                                 duration_cast<milliseconds>(timer->total_time) / timer->total_calls);
        } else {
            buffer = "total_calls == 0!!!";
        }
        int width = 70 - print::current_indent_level();
        if (width < 0) {
            width = 0;
        }
        fprintln(fp, "{0:<{1}} : {3: <{4}}{2}", buffer, width, timer->name, "", print::current_indent_level());
    } else {
        fprintln(fp);
        fprintln(fp);
        fprintln(fp, "Timing information:");
        fprintln(fp);
    }

    if (timer && !timer->children.empty()) {
        print::indent();

        for (auto &child : timer->order) {
            print_timer_info(&timer->children[child], fp);
        }

        print::deindent();
    }
}

void print_timer_info(timer_detail *timer, std::ostream &os) { // NOLINT
    if (timer != root) {
        std::string buffer;
        if (timer->total_calls != 0) {
            buffer = fmt::format("{:>5} : {:>5} calls : {:>5} per call", duration_cast<milliseconds>(timer->total_time), timer->total_calls,
                                 duration_cast<milliseconds>(timer->total_time) / timer->total_calls);
        } else {
            buffer = "total_calls == 0!!!";
        }
        fprintln(os, "{0:<{1}} : {3: <{4}}{2}", buffer, 70 - print::current_indent_level(), timer->name, "", print::current_indent_level());
    } else {
        fprintln(os);
        fprintln(os);
        fprintln(os, "Timing information:");
        fprintln(os);
    }

    if (!timer->children.empty()) {
        print::indent();

        for (auto &child : timer->order) {
            print_timer_info(&timer->children[child], os);
        }

        print::deindent();
    }
}

} // namespace

void initialize() {
    using namespace detail;
    root              = new timer_detail();
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
    using namespace detail;
    assert(root == current_timer);
    delete root;
    root = current_timer = nullptr;
}

void report() {
    print_timer_info(root, stdout);
}

void report(const std::string &fname) {
    std::FILE *fp = std::fopen(fname.c_str(), "w+");

    print_timer_info(root, fp);

    std::fflush(fp);
    std::fclose(fp);
}

void report(std::FILE *fp) {
    print_timer_info(root, fp);
}

void report(std::ostream &os) {
    print_timer_info(root, os);
}

namespace detail {
void push(std::string name) {
    using namespace detail;
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

        if (!current_timer->children.contains(name)) {
            current_timer->children[name].name   = name;
            current_timer->children[name].parent = current_timer;
            current_timer->order.push_back(name);
        }

        current_timer             = &current_timer->children[name];
        current_timer->start_time = clock::now();
    }
}
} // namespace detail

void pop() {
    using namespace detail;
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

void pop(duration elapsed) {
    using namespace detail;
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

        current_timer->total_time += elapsed;
        current_timer->total_calls++;
        current_timer = current_timer->parent;
    }
}

} // namespace einsums::profile::timer