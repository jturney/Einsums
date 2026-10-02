//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/TypeSupport/JsonEscape.hpp>

#include <iomanip>
#include <memory>
#include <mutex>

#if defined(__x86_64__) || defined(_M_X64)
#    if defined(_MSC_VER) && !defined(__clang__)
#        include <intrin.h>
#    else
#        include <cpuid.h>
#    endif
#endif

#if defined(EINSUMS_HAVE_PROFILER)
EINSUMS_NAMESPACE_BEGIN(profile)

namespace {

auto strip_ansi_sequences(std::string const &s) -> std::string {
    std::string out;
    out.reserve(s.size());

    for (size_t i = 0; i < s.size();) {
        unsigned char const c = s[i];
        if (c == '\x1b') { // ESC
            if (i + 1 >= s.size()) {
                ++i;
                break;
            }

            unsigned char const c1 = s[i + 1];

            if (c1 == '[') {
                // CSI: ESC [ ... final byte in @-~
                i += 2;
                while (i < s.size()) {
                    unsigned char const cc = s[i++];
                    if (cc >= '@' && cc <= '~')
                        break; // final byte
                }
                continue;
            } else if (c1 == ']') {
                // OSC: ESC ] ... terminated by BEL or ESC '\'
                i += 2;
                while (i < s.size()) {
                    if (s[i] == '\x07') {
                        ++i;
                        break;
                    } // BEL
                    if (s[i] == '\x1b' && i + 1 < s.size() && s[i + 1] == '\\') {
                        i += 2;
                        break;
                    } // ESC '\'
                    if (s[i] == '\x1b')
                        break; // new ESC -> bail out
                    ++i;
                }
                continue;
            } else {
                i += 2;
                continue;
            }
        } else {
            out.push_back(static_cast<char>(c));
            ++i;
        }
    }
    return out;
}

auto visible_width(std::string const &s) -> size_t {
    return strip_ansi_sequences(s).size();
}

auto make_clickable_file_line(std::string const &file, int line, std::string const &display) -> std::string {
    if (file.empty() || line <= 0)
        return display;
    std::string const uri = "file://" + file + ":" + std::to_string(line);
    std::string const esc = "\x1b]8;;";
    std::string const st  = "\x1b\\";
    return esc + uri + st + display + esc + st;
}

auto ns_to_ms(ns const t) -> double {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t).count();
}

// Use the shared json_escape from TypeSupport.
auto const &escape_json = ::einsums::json_escape;

} // namespace

auto Profiler::instance() -> Profiler & {
    static Profiler p;
    return p;
}

auto Profiler::thread_channel() -> ThreadChannel & {
    // Constant-initialized, so reading it needs no initialization guard.
    static thread_local ThreadChannel *channel = nullptr;
    if (channel == nullptr) [[unlikely]] {
        channel = &instance().register_thread();
    }
    return *channel;
}

auto Profiler::register_thread() -> ThreadChannel & {
    auto       channel = std::make_shared<ThreadChannel>();
    auto const tid     = thread_key();
    channel->counters  = get_counter_backend().available();

    // The consumer drains the ring, and shares the channel's ownership through it.
    _consumer->register_thread(tid, std::shared_ptr<EventRingBuffer>(channel, &channel->ring));
    get_counter_backend().open_thread_counters();

    // Auto-name the thread: the first thread to initialize is "main"
    static std::atomic<bool> first_thread{true};
    if (first_thread.exchange(false, std::memory_order_acq_rel)) {
        _consumer->set_thread_name(tid, "main");
    } else {
        _consumer->set_thread_name(tid, "thread-" + std::to_string(tid));
    }

    std::scoped_lock const lock(_channels_mutex);
    _channels.push_back(channel);
    return *channel;
}

auto Profiler::total_push_count() const -> uint64_t {
    std::scoped_lock const lock(_channels_mutex);
    uint64_t               total = 0;
    for (auto const &ch : _channels) {
        total += ch->pushes.load(std::memory_order_relaxed);
    }
    return total;
}

auto Profiler::total_pop_count() const -> uint64_t {
    std::scoped_lock const lock(_channels_mutex);
    uint64_t               total = 0;
    for (auto const &ch : _channels) {
        total += ch->pops.load(std::memory_order_relaxed);
    }
    return total;
}

auto Profiler::calibrated_overhead() -> Overhead const & {
    std::call_once(_calibration_once, [this] {
        // The real write_push/write_pop into an unregistered channel. 16384 zones are 32768 events,
        // half the ring, so none is dropped.
        constexpr int kZones  = 16384;
        auto          scratch = std::make_unique<ThreadChannel>();
        scratch->counters     = get_counter_backend().available();
        auto const elapsed    = [](auto t0, auto t1) { return std::chrono::duration<double, std::nano>(t1 - t0).count(); };
        auto const t0         = std::chrono::steady_clock::now();
        for (int i = 0; i < kZones; ++i) {
            write_push(*scratch, 0, 0, 0, 0);
        }
        auto const t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < kZones; ++i) {
            write_pop(*scratch);
        }
        auto const t2        = std::chrono::steady_clock::now();
        _calibration.push_ns = elapsed(t0, t1) / kZones;
        _calibration.pop_ns  = elapsed(t1, t2) / kZones;
    });
    return _calibration;
}

auto TickClock::instance() -> TickClock const & {
    static TickClock const clock;
    return clock;
}

namespace {
#    if defined(__x86_64__) || defined(_M_X64)
/// Whether the TSC ticks at a constant rate whatever the core's frequency and power state
/// (CPUID leaf 0x80000007, EDX bit 8). Without that its ticks are not a clock.
bool invariant_tsc() {
#        if defined(_MSC_VER) && !defined(__clang__)
    int regs[4];
    __cpuid(regs, 0x80000000);
    if (static_cast<unsigned>(regs[0]) < 0x80000007u) {
        return false;
    }
    __cpuid(regs, 0x80000007);
    return (regs[3] & (1 << 8)) != 0;
#        else
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_max(0x80000000u, nullptr) < 0x80000007u) {
        return false;
    }
    __get_cpuid(0x80000007u, &eax, &ebx, &ecx, &edx);
    return (edx & (1u << 8)) != 0;
#        endif
}
#    endif
} // namespace

TickClock::TickClock() {
    // Reads the counters directly: now() goes through instance() on x86, which is this object.
#    if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(_MSC_VER)
    std::uint64_t freq = 0;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    ns_per_tick     = 1e9 / static_cast<double>(freq);
    source          = "cntvct_el0";
    auto const read = [] {
        std::uint64_t v;
        asm volatile("mrs %0, cntvct_el0" : "=r"(v));
        return v;
    };
#    elif defined(__x86_64__) || defined(_M_X64)
    uses_tsc        = invariant_tsc();
    auto const read = [this] { return uses_tsc ? static_cast<std::uint64_t>(__rdtsc()) : fallback_now(); };
    if (uses_tsc) {
        // The TSC's rate is not architectural, so measure it against steady_clock over a couple of
        // milliseconds. Each end is a TSC read bracketed by two steady_clock reads and placed at
        // their midpoint, retaken if the bracket exceeds a microsecond (the thread was descheduled).
        struct Pair {
            std::chrono::steady_clock::time_point time;
            std::uint64_t                         ticks;
        };
        auto const pair = [&] {
            Pair best{};
            auto width = std::chrono::steady_clock::duration::max();
            for (int attempt = 0; attempt < 100; ++attempt) {
                auto const          before = std::chrono::steady_clock::now();
                std::uint64_t const ticks  = read();
                auto const          after  = std::chrono::steady_clock::now();
                if (after - before < width) {
                    width = after - before;
                    best  = {before + (after - before) / 2, ticks};
                }
                if (width < std::chrono::microseconds(1)) {
                    break;
                }
            }
            return best;
        };
        Pair const start = pair();
        while (std::chrono::steady_clock::now() - start.time < std::chrono::milliseconds(2)) {
        }
        Pair const end = pair();
        ns_per_tick =
            std::chrono::duration<double, std::nano>(end.time - start.time).count() / static_cast<double>(end.ticks - start.ticks);
        source = "rdtsc";
    }
#    else
    auto const read = [] { return fallback_now(); };
#    endif
    anchor_time  = std::chrono::steady_clock::now();
    anchor_ticks = read();
}

void Profiler::print(bool detailed, std::ostream &os) {
    // Flush pending events before reading the tree
    flush();
    // Acquire shared lock on consumer's tree
    auto        lock       = _consumer->lock_shared();
    auto const &thread_map = _consumer->thread_data();

    for (auto const &tkv : thread_map) {
        auto const &thread_id = tkv.first;
        auto const &ts        = tkv.second;

        double thread_total_ms = 0.0;
        for (auto const &c : ts.root.children) {
            thread_total_ms += ns_to_ms(c.second->total_exclusive);
        }

        // header
        std::string       tname        = _consumer->thread_name(thread_id);
        std::string const thread_label = tname.empty() ? fmt::format("{}", thread_id) : fmt::format("{} ({})", tname, thread_id);
        fprintln(os);
        fprintln(os, fmt::emphasis::bold | fg(fmt::color::white), "Thread: {}  (total exclusive: {:-7.3f} ms)", thread_label,
                 thread_total_ms);
        fprintln(os, "{:-^157}", "");

        if (!detailed) {
            fprintln(os, " {:>10}  {:^10}  {:^13}  {:<60}  {:<30}  {:<}", "total(ms)", "count", "mean(ms)", "name", "file:line",
                     "function");
            fprintln(os, "{:-^157}", "");
        } else {
            fprintln(os, " {:>10}  {:<60}  {:<30}  {:<20}  {:>8} {:>8} {:>8}", "total(ms)", "name", "file:line", "function", "min", "max",
                     "avg");
            fprintln(os, "{:-^120}", "");
        }

        std::vector<AggNode const *> nodes;
        for (auto const &c : ts.root.children)
            nodes.push_back(c.second.get());

        for (auto const *n : nodes) {
            print_node_recursive(os, n, thread_total_ms, 0, detailed);
        }
        fprintln(os);
    }

    // Print profiler overhead summary
    fprintln(os);
    fprintln(os, fmt::emphasis::bold | fg(fmt::color::white), "Profiler overhead");
    fprintln(os, "{:-^80}", "");
    // Estimates: the calibrated per-call cost times the number of calls.
    auto const pushes = total_push_count();
    auto const pops   = total_pop_count();
    fprintln(os, "  push():  {:.1f} ns each, calibrated  ({} calls, ~{:.3f} ms total)", avg_push_overhead_ns(), pushes,
             avg_push_overhead_ns() * static_cast<double>(pushes) / 1'000'000.0);
    fprintln(os, "  pop():   {:.1f} ns each, calibrated  ({} calls, ~{:.3f} ms total)", avg_pop_overhead_ns(), pops,
             avg_pop_overhead_ns() * static_cast<double>(pops) / 1'000'000.0);
    fprintln(os, "  clock:   {}, {:.3f} ns per tick", TickClock::instance().source, TickClock::instance().ns_per_tick);
    auto dropped = _consumer->dropped_count();
    if (dropped > 0) {
        fprintln(os, fg(fmt::color::red), "  dropped events: {}", dropped);
    }
    // A zone missing its Push or Pop contributes no time, so totals above it are short.
    if (auto const unmatched = _consumer->unmatched_zone_count(); unmatched > 0) {
        fprintln(os, fg(fmt::color::red), "  zones left unmeasured by those drops: {}", unmatched);
    }
    fprintln(os);
}

auto Profiler::export_json(std::string const &path) -> std::optional<std::string> {
    flush();
    auto        lock       = _consumer->lock_shared();
    auto const &thread_map = _consumer->thread_data();

    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs)
        return std::nullopt;
    ofs << "{\n";
    bool first_thread = true;
    for (auto const &tkv : thread_map) {
        if (!first_thread)
            ofs << ",\n";
        first_thread = false;
        ofs << fmt::format("  \"{}\": ", tkv.first);
        write_node_json(ofs, tkv.second.root, 2);
    }
    ofs << "\n}\n";
    return path;
}

void Profiler::write_node_json(std::ostream &ofs, AggNode const &n, int indent) { // NOLINT
    std::string const ind(indent, ' ');
    ofs << ind << "{\n";
    ofs << ind << R"(  "name": ")" << escape_json(n.name) << "\",\n";
    ofs << ind << "  \"call_count\": " << n.call_count << ",\n";
    ofs << ind << "  \"exclusive_ms\": " << std::fixed << std::setprecision(6) << ns_to_ms(n.total_exclusive) << ",\n";
    ofs << ind << "  \"exclusive_min_ms\": " << ns_to_ms(n.exclusive_min) << ",\n";
    ofs << ind << "  \"exclusive_max_ms\": " << ns_to_ms(n.exclusive_max) << ",\n";

    // annotations
    ofs << ind << "  \"annotations\": {";
    {
        bool first = true;
        for (auto const &a : n.annotations) {
            if (!first)
                ofs << ", ";
            first = false;
            // Check if there's a numeric annotation for this key
            auto nit = n.numeric_annotations.find(a.first);
            if (nit != n.numeric_annotations.end()) {
                ofs << "\"" << escape_json(a.first)
                    << "\": " << nit->second.total / static_cast<double>(std::max(static_cast<uint64_t>(1), nit->second.count));
            } else {
                ofs << "\"" << escape_json(a.first) << "\": \"" << escape_json(a.second) << "\"";
            }
        }
    }
    ofs << "},\n";

    // counters
    ofs << ind << "  \"counters\": {";
    {
        bool first = true;
        for (auto const &c : n.counters_total) {
            if (!first)
                ofs << ", ";
            first              = false;
            uint64_t const tot = c.second;
            uint64_t const mn  = n.counters_min.at(c.first);
            uint64_t const mx  = n.counters_max.at(c.first);
            ofs << "\"" << escape_json(c.first) << R"(": {"total": )" << tot << ", \"min\": " << mn << ", \"max\": " << mx << "}";
        }
    }
    ofs << "},\n";

    ofs << ind << "  \"children\": [\n";
    bool first_child = true;
    for (auto const &ch : n.children) {
        if (!first_child)
            ofs << ",\n";
        first_child = false;
        write_node_json(ofs, *ch.second, indent + 4);
    }
    ofs << "\n" << ind << "  ]\n";
    ofs << ind << "}";
}

// NOLINTNEXTLINE
void Profiler::print_node_recursive(std::ostream &os, AggNode const *n, double /*thread_total_ms*/, int depth, bool detailed) {
    // Despite the name, iterative with an explicit stack: recursion overflowed the stack under TSan
    // on deep trees.
    auto variance = [](uint64_t cnt, double M2) -> double { return (cnt > 1) ? M2 / static_cast<double>(cnt - 1) : 0.0; };
    auto stddev   = [variance](uint64_t cnt, double M2) -> double { return sqrt(variance(cnt, M2)); };

    struct Frame {
        AggNode const *node;
        int            depth;
    };
    std::vector<Frame> work;
    work.push_back({n, depth});

    while (!work.empty()) {
        Frame const f = work.back();
        work.pop_back();
        AggNode const *node = f.node;

        std::string const indent(static_cast<size_t>(2) * static_cast<size_t>(f.depth), ' ');
        double const      excl_ms = ns_to_ms(node->total_exclusive);

        std::string name = indent + node->name;
        if (name.size() > 60)
            name = name.substr(0, 57) + "...";

        std::string const mean_str = fmt::format("{:7.3f}\u00B1{:3.3f}", node->total_exclusive_mean / 1'000'000.0,
                                                 stddev(node->call_count, node->total_exclusive_M2) / 1'000'000.0);

        // Build file:line field
        std::string file_field;
        if (!node->file.empty()) {
            std::string shortname;
            try {
                shortname = std::filesystem::path(node->file).filename().string();
            } catch (...) {
                shortname = node->file;
            }
            std::string const file_display = fmt::format("{}:{}", shortname, node->line);
            if (detail::is_terminal(os)) {
                std::string const clickable = make_clickable_file_line(node->file, node->line, file_display);
                // Pad based on visible width (excludes ANSI escape sequences)
                size_t const vlen = visible_width(clickable);
                file_field        = clickable;
                if (vlen < 30)
                    file_field += std::string(30 - vlen, ' ');
            } else {
                file_field = fmt::format("{:<30}", file_display);
            }
        } else {
            file_field = fmt::format("{:<30}", "");
        }

        // Build annotations string
        std::string annotations_str;
        if (!node->annotations.empty()) {
            annotations_str = "  ";
            bool first      = true;
            for (auto const &a : node->annotations) {
                if (!first)
                    annotations_str += " ";
                first = false;
                annotations_str += fmt::format("{}={}", a.first, a.second);
            }
        }

        fprintln(os, " {:10.3f}  {:10}  {:13}  {:<60}  {}  {:<}{}", excl_ms, node->call_count, mean_str, name, file_field, node->function,
                 annotations_str);

        if (detailed) {
            double const min_ms = ns_to_ms(node->exclusive_min);
            double const max_ms = ns_to_ms(node->exclusive_max);
            double const avg_ms = (node->call_count > 0) ? (ns_to_ms(node->total_exclusive) / static_cast<double>(node->call_count)) : 0.0;
            fprintln(os, "{:6}   {:>10.3f}  (min {:>6.3f}  max {:>6.3f}  avg {:>6.3f})", "", excl_ms, min_ms, max_ms, avg_ms);
            if (!node->counters_total.empty()) {
                std::string counters = fmt::format("{:6}   Counters:", "");
                for (auto const &c : node->counters_total) {
                    uint64_t tot = c.second;
                    uint64_t mn  = node->counters_min.at(c.first);
                    uint64_t mx  = node->counters_max.at(c.first);
                    double   avg = (node->call_count > 0) ? static_cast<double>(tot) / static_cast<double>(node->call_count) : 0.0;
                    counters += fmt::format(" {}(tot={},min={},max={},avg={:.1f})", c.first, tot, mn, mx, avg);
                }
                fprintln(os, counters);
            }
            // Show numeric annotation stats in detailed mode
            if (!node->numeric_annotations.empty()) {
                std::string annot_stats = fmt::format("{:6}   Annotations:", "");
                for (auto const &na : node->numeric_annotations) {
                    double avg = (na.second.count > 0) ? na.second.total / static_cast<double>(na.second.count) : 0.0;
                    annot_stats +=
                        fmt::format(" {}(avg={:.1f},min={:.1f},max={:.1f})", na.first, avg, na.second.min_val, na.second.max_val);
                }
                fprintln(os, annot_stats);
            }
        }

        // Pushed in reverse so they print in order, depth-first.
        std::vector<AggNode const *> children;
        children.reserve(node->children.size());
        for (auto const &c : node->children)
            children.push_back(c.second.get());
        for (auto it = children.rbegin(); it != children.rend(); ++it) {
            work.push_back({*it, f.depth + 1});
        }
    }
}

EINSUMS_NAMESPACE_END(profile)
#endif
