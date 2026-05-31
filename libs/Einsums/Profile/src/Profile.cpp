//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/TypeSupport/JsonEscape.hpp>

#if defined(EINSUMS_HAVE_PROFILER)
namespace einsums::profile {

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
    fprintln(os, "  push():  avg {:.1f} ns  ({} calls, {:.3f} ms total)", avg_push_overhead_ns(), total_push_count(),
             static_cast<double>(_push_overhead_ns.load(std::memory_order_relaxed)) / 1'000'000.0);
    fprintln(os, "  pop():   avg {:.1f} ns  ({} calls, {:.3f} ms total)", avg_pop_overhead_ns(), total_pop_count(),
             static_cast<double>(_pop_overhead_ns.load(std::memory_order_relaxed)) / 1'000'000.0);
    auto dropped = _consumer->dropped_count();
    if (dropped > 0) {
        fprintln(os, fg(fmt::color::red), "  dropped events: {}", dropped);
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
void Profiler::print_node_recursive(std::ostream &os, AggNode const *n, double thread_total_ms, int depth, bool detailed) {
    // The name is misleading \u2014 this prints `n` and its descendants but does
    // it ITERATIVELY with an explicit stack. The original recursive
    // implementation overflowed the 8 MB Linux stack under TSan
    // instrumentation when ~6340 fuzz-test cases produced a deeply-nested
    // profile tree (CI run 26696948385: SIGSEGV inside fmt::format'ing a
    // double, with 30+ identical print_node_recursive frames above it).
    // Each frame here holds several std::string locals and TSan multiplies
    // frame size, so deep recursion exhausts stack even at modest tree
    // depth. Switching to an explicit work stack puts the per-node
    // bookkeeping on the heap and keeps a constant call-stack budget.
    auto variance = [](uint64_t cnt, int64_t M2) -> double {
        return (cnt > 1) ? static_cast<double>(M2) / static_cast<double>(cnt - 1) : 0.0;
    };
    auto stddev = [variance](uint64_t cnt, int64_t M2) -> double { return sqrt(variance(cnt, M2)); };

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

        std::string const mean_str = fmt::format("{:7.3f}\u00B1{:3.3f}", static_cast<double>(node->total_exclusive_mean) / 1'000'000.0,
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

        // Push children in REVERSE order so they pop in declaration order \u2014
        // preserves the depth-first preorder output of the original
        // recursive implementation.
        std::vector<AggNode const *> children;
        children.reserve(node->children.size());
        for (auto const &c : node->children)
            children.push_back(c.second.get());
        for (auto it = children.rbegin(); it != children.rend(); ++it) {
            work.push_back({*it, f.depth + 1});
        }
    }
}

} // namespace einsums::profile
#endif
