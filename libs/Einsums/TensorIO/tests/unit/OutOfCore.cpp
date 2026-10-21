//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file OutOfCore.cpp
/// @brief Random programs over backed tensors, run through the default pipeline under a budget
///        every backed tensor exceeds its share of.
///
/// Two things are checked of every program. The replay equals the same statements evaluated by
/// reference_einsum, twice over, so a pass that streams, folds, hoists or reorders around a backed
/// tensor cannot change an answer. And no backed tensor over its share is read whole unless a pass
/// recorded a decline that names it: a pass that holds such a tensor whole without saying why has
/// defeated the budget, as LinearCombinationContractionFolding once did by building a whole L from
/// one. The programs draw the shapes the passes that copy operands look for: pairs that read one
/// tensor with permuted letters, sums over two tensors of one shape, repeated statements, and
/// chains through an intermediate.

#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Observer.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/LinearCombinationContractionFolding.hpp>
#include <Einsums/ComputeGraph/Passes/MemoryPlanning.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorIO/Backing.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;
using tensor_io::TensorFile;

namespace {

/// Sets --einsums:max-memory for one case.
struct MemoryBudget {
    std::string previous{config::get(option::MaxMemory)};
    explicit MemoryBudget(std::size_t bytes) { config::set(option::MaxMemory, std::to_string(bytes) + "b"); }
    ~MemoryBudget() { config::set(option::MaxMemory, previous); }
    MemoryBudget(MemoryBudget const &)            = delete;
    MemoryBudget &operator=(MemoryBudget const &) = delete;
};

/// Sets --einsums:graph:tiling-budget-share for one case.
struct BudgetShare {
    double previous{config::get(option::GraphTilingBudgetShare)};
    explicit BudgetShare(double fraction) { config::set(option::GraphTilingBudgetShare, fraction); }
    ~BudgetShare() { config::set(option::GraphTilingBudgetShare, previous); }
    BudgetShare(BudgetShare const &)            = delete;
    BudgetShare &operator=(BudgetShare const &) = delete;
};

struct TempFile {
    std::string path;
    explicit TempFile(std::string const &name) : path((std::filesystem::temp_directory_path() / ("einsums_out_of_core_" + name)).string()) {
        std::remove(path.c_str());
    }
    ~TempFile() { std::remove(path.c_str()); }
    TempFile(TempFile const &)            = delete;
    TempFile &operator=(TempFile const &) = delete;
};

enum class Role : std::uint8_t { Backed, Memory, Output, Intermediate };

struct Operand {
    std::string              name;
    Role                     role;
    std::vector<std::string> letters; ///< The letter on each axis.
};

struct Statement {
    std::size_t              out, x, y;
    std::vector<std::string> c, a, b; ///< The letters each operand is read with.
    double                   c_pf{0.0}, ab_pf{1.0};
};

std::string join(std::vector<std::string> const &letters) {
    std::string out;
    for (std::size_t p = 0; p < letters.size(); ++p) {
        out += (p == 0 ? "" : ",") + letters[p];
    }
    return out;
}

std::string spec_of(Statement const &s) {
    return join(s.c) + " <- " + join(s.a) + " ; " + join(s.b);
}

/// A program and everything its statements name.
struct Program {
    std::map<std::string, std::size_t> extent;
    std::vector<Operand>               operands;
    std::vector<Statement>             statements;

    [[nodiscard]] std::vector<std::size_t> dims_of(std::vector<std::string> const &letters) const {
        std::vector<std::size_t> out;
        for (auto const &l : letters) {
            out.push_back(extent.at(l));
        }
        return out;
    }

    [[nodiscard]] std::string describe() const {
        std::ostringstream out;
        out << "  extents";
        for (auto const &[letter, n] : extent) {
            out << " " << letter << "=" << n;
        }
        out << "\n";
        for (auto const &op : operands) {
            static std::array<char const *, 4> const roles{"backed", "memory", "output", "intermediate"};
            out << "  " << roles[static_cast<std::size_t>(op.role)] << " " << op.name << "[" << join(op.letters) << "]\n";
        }
        for (auto const &s : statements) {
            out << "  " << operands[s.out].name << " = " << s.c_pf << " " << operands[s.out].name << " + " << s.ab_pf << " "
                << operands[s.x].name << " * " << operands[s.y].name << "   \"" << spec_of(s) << "\"\n";
        }
        return out.str();
    }
};

/// Draws a program. Letters take extents of 3 or 4, so two axes of one tensor often share an extent
/// and a permuted read of it is a legal spec.
Program draw(std::mt19937 &rng) {
    auto const pick = [&](std::size_t n) { return std::uniform_int_distribution<std::size_t>(0, n - 1)(rng); };
    auto const coin = [&](double p) { return std::bernoulli_distribution(p)(rng); };

    Program                        prog;
    std::vector<std::string> const pool{"a", "b", "c", "d", "e", "f"};
    for (auto const &l : pool) {
        prog.extent[l] = 3 + pick(2);
    }
    auto const letters_of_rank = [&](std::size_t rank) {
        std::vector<std::string> shuffled = pool;
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        shuffled.resize(rank);
        return shuffled;
    };
    std::size_t const n_backed = 2 + pick(3);
    for (std::size_t k = 0; k < n_backed; ++k) {
        prog.operands.push_back({.name = "B" + std::to_string(k), .role = Role::Backed, .letters = letters_of_rank(2 + pick(2))});
    }
    // A twin of the first backed tensor, the same letters, for a sum over two tensors of one shape.
    prog.operands.push_back({.name = "Btwin", .role = Role::Backed, .letters = prog.operands[0].letters});
    for (std::size_t k = 0; k < 2; ++k) {
        prog.operands.push_back({.name = "M" + std::to_string(k), .role = Role::Memory, .letters = letters_of_rank(1 + pick(2))});
    }
    std::vector<std::size_t> inputs(prog.operands.size());
    for (std::size_t k = 0; k < inputs.size(); ++k) {
        inputs[k] = k;
    }

    // The contraction of x and y into a new tensor: every letter they share is summed, unless kept.
    auto const contract = [&](std::size_t x, std::size_t y, Role role) {
        // Copies: the push_back below may move the operands these name.
        auto const               lx = prog.operands[x].letters;
        auto const               ly = prog.operands[y].letters;
        std::vector<std::string> c;
        for (auto const &l : lx) {
            bool const shared = std::ranges::find(ly, l) != ly.end();
            if (!shared || coin(0.3)) {
                c.push_back(l);
            }
        }
        for (auto const &l : ly) {
            if (std::ranges::find(lx, l) == lx.end()) {
                c.push_back(l);
            }
        }
        if (c.empty()) {
            c.push_back(lx.front());
        }
        std::shuffle(c.begin(), c.end(), rng);
        std::size_t const out = prog.operands.size();
        prog.operands.push_back({.name = (role == Role::Output ? "C" : "T") + std::to_string(out), .role = role, .letters = c});
        return Statement{.out = out, .x = x, .y = y, .c = c, .a = lx, .b = ly};
    };
    std::array<double, 3> const pf{1.0, -0.5, 2.0};

    std::size_t const n_statements = 3 + pick(5);
    for (std::size_t s = 0; s < n_statements; ++s) {
        std::size_t const x = pick(n_backed);
        std::size_t const y = inputs[pick(inputs.size())];
        switch (pick(5)) {
        case 0: { // a pair reading x with two of its letters exchanged, which LCCF folds
            auto const              &lx = prog.operands[x].letters;
            std::vector<std::size_t> same;
            for (std::size_t p = 0; p < lx.size(); ++p) {
                for (std::size_t q = p + 1; q < lx.size(); ++q) {
                    if (prog.extent[lx[p]] == prog.extent[lx[q]]) {
                        same = {p, q};
                    }
                }
            }
            Statement first = contract(x, y, Role::Output);
            first.ab_pf     = 2.0;
            prog.statements.push_back(first);
            if (!same.empty()) {
                Statement second = first;
                std::swap(second.a[same[0]], second.a[same[1]]);
                second.c_pf  = 1.0;
                second.ab_pf = -1.0;
                prog.statements.push_back(second);
            }
            break;
        }
        case 1: { // a sum over x and its twin, against one operand, which DistributiveFactoring factors
            Statement const first = contract(0, y, Role::Output);
            prog.statements.push_back(first);
            Statement second = first;
            second.x         = n_backed; // Btwin
            second.c_pf      = 1.0;
            second.ab_pf     = pf[pick(3)];
            prog.statements.push_back(second);
            break;
        }
        case 2: { // the same statement twice, into two outputs, which CSE shares
            Statement const first = contract(x, y, Role::Output);
            prog.statements.push_back(first);
            Statement copy = first;
            copy.out       = prog.operands.size();
            prog.operands.push_back({.name = "C" + std::to_string(copy.out), .role = Role::Output, .letters = first.c});
            prog.statements.push_back(copy);
            break;
        }
        case 3: { // a chain through an intermediate
            Statement const first = contract(x, y, Role::Intermediate);
            prog.statements.push_back(first);
            prog.statements.push_back(contract(first.out, inputs[pick(inputs.size())], Role::Output));
            break;
        }
        default: {
            Statement plain = contract(x, y, Role::Output);
            plain.ab_pf     = pf[pick(3)];
            prog.statements.push_back(plain);
            break;
        }
        }
    }
    return prog;
}

/// Reproducible values for an input.
RuntimeTensor<double> filled(std::string const &name, std::vector<std::size_t> const &dims, std::size_t salt) {
    RuntimeTensor<double> out(name, dims);
    for (std::size_t n = 0; n < out.size(); ++n) {
        out.data()[n] = static_cast<double>((n * 7 + salt * 13) % 23) / 11.0 - 1.0;
    }
    return out;
}

} // namespace

TEST_CASE("Random programs over backed tensors replay correctly and hold no streamed tensor whole without saying why",
          "[TensorIO][Backing][OutOfCore]") {
    auto const seed = GENERATE(range(0, 120));
    INFO("seed " << seed);
    std::mt19937  rng(static_cast<std::mt19937::result_type>(seed));
    Program const prog = draw(rng);
    INFO("program:\n" << prog.describe());

    // A share of 64 bytes: every backed tensor (at least 3 x 3 doubles) is over it. The share is a
    // fraction of the budget, and a budget these programs meet keeps the share that small without
    // a budget no program can meet, which the replay reports as a missed budget.
    MemoryBudget const budget(std::size_t{1} << 20);
    BudgetShare const  share(64.0 / static_cast<double>(std::size_t{1} << 20));

    TempFile const                               tmp("program_" + std::to_string(seed) + ".etn");
    std::map<std::string, RuntimeTensor<double>> values; // what the oracle reads and writes
    std::map<std::string, std::size_t>           stored_bytes;
    {
        TensorFile file(tmp.path, TensorFile::Mode::Write);
        for (std::size_t k = 0; k < prog.operands.size(); ++k) {
            auto const &op   = prog.operands[k];
            auto const  dims = prog.dims_of(op.letters);
            if (op.role == Role::Backed || op.role == Role::Memory) {
                values.emplace(op.name, filled(op.name, dims, k));
                if (op.role == Role::Backed) {
                    file.write(op.name, values.at(op.name));
                    stored_bytes[op.name] = values.at(op.name).size() * sizeof(double);
                }
            } else {
                RuntimeTensor<double> zero(op.name, dims);
                zero.zero();
                values.emplace(op.name, std::move(zero));
            }
        }
    }

    // The graph: backed inputs and intermediates declared on it, memory inputs and outputs the caller's.
    tensor_io::EtnFile const                     file(tmp.path);
    cg::Graph                                    graph("out_of_core");
    std::map<std::string, RuntimeTensor<double>> caller;
    std::vector<RuntimeTensor<double> *>         handle(prog.operands.size(), nullptr);
    for (std::size_t k = 0; k < prog.operands.size(); ++k) {
        auto const &op = prog.operands[k];
        switch (op.role) {
        case Role::Backed:
            handle[k] = &graph.declare_runtime_tensor<double>(op.name, file.entry(op.name));
            break;
        case Role::Intermediate:
            handle[k] = &graph.declare_runtime_tensor<double>(op.name, prog.dims_of(op.letters), /*intermediate=*/true);
            break;
        default:
            handle[k] = &caller.emplace(op.name, values.at(op.name)).first->second;
            break;
        }
    }
    {
        cg::CaptureGuard const guard(graph);
        for (auto const &s : prog.statements) {
            std::string const spec = spec_of(s);
            cg::einsum(cg::EinsumFormatString(spec), s.c_pf, handle[s.out], s.ab_pf, *handle[s.x], *handle[s.y]);
        }
    }
    auto manager = cg::PassManager::create_default();
    REQUIRE_NOTHROW(graph.apply(manager));
    auto const log = std::make_shared<cg::EventLog>();
    graph.set_observer(log);

    // The oracle, statement by statement.
    for (auto const &s : prog.statements) {
        auto &out = values.at(prog.operands[s.out].name);
        testing::reference_einsum(spec_of(s), s.c_pf, &out, s.ab_pf, values.at(prog.operands[s.x].name),
                                  values.at(prog.operands[s.y].name));
    }

    for (int replay = 0; replay < 2; ++replay) {
        INFO("replay " << replay);
        log->clear();
        REQUIRE_NOTHROW(graph.execute());
        for (auto const &op : prog.operands) {
            if (op.role != Role::Output) {
                continue;
            }
            auto const &got  = caller.at(op.name);
            auto const &want = values.at(op.name);
            double      peak = 1.0;
            for (std::size_t n = 0; n < want.size(); ++n) {
                peak = std::max(peak, std::abs(want.data()[n]));
            }
            for (std::size_t n = 0; n < want.size(); ++n) {
                INFO(op.name << " element " << n);
                REQUIRE(std::abs(got.data()[n] - want.data()[n]) <= 1e-12 * peak);
            }
        }
    }

    auto const decisions = manager.decisions();

    // The plan's peak is what the passes believe the program holds, and a budget is only as good as
    // that belief: every pass that sizes a loop by what the budget leaves trusts it. MemoryPlanning
    // predicts the replay's own count from the lifecycle nodes, so the two agree to the byte. They
    // disagreed twice: the planned peak left out the arena, which is held whole, and FreeInsertion
    // left any number of small buffers held that the plan took for freed.
    {
        auto const peak =
            std::ranges::find_if(decisions, [](cg::Decision const &d) { return d.code == cg::passes::MemoryPlanning::Peak.code; });
        REQUIRE(peak != decisions.end());
        auto const managed = std::ranges::find_if(peak->facts, [](cg::Fact const &f) { return f.name == "managed_peak"; });
        REQUIRE(managed != peak->facts.end());
        CHECK(static_cast<double>(graph.last_run().peak_bytes) == managed->value);

        // Beside it, what the replay holds throughout: the caller's tensors and nothing backed,
        // since a backed tensor is moved through boxes and slabs the lifecycle nodes manage. The
        // budget bounds the two together.
        auto const resident = std::ranges::find_if(peak->facts, [](cg::Fact const &f) { return f.name == "resident"; });
        REQUIRE(resident != peak->facts.end());
        CHECK(static_cast<double>(graph.last_run().resident_bytes) == resident->value);
        // A drawn operand no statement names never reaches the graph.
        std::set<std::string> named;
        for (auto const &st : prog.statements) {
            for (std::size_t const k : {st.out, st.x, st.y}) {
                named.insert(prog.operands[k].name);
            }
        }
        std::uint64_t callers = 0;
        for (auto const &[name, tensor] : caller) {
            if (named.contains(name)) {
                callers += tensor.size() * sizeof(double);
            }
        }
        CHECK(graph.last_run().resident_bytes == callers);
    }

    // Every whole read of a backed tensor over its share, unless SlabStreaming, the pass that decides
    // what is read a slab at a time, declined to stream it and named it, or a pass that reads an
    // operand whole said it held one beside a larger tensor streamed. A whole copy some other pass
    // makes where SlabStreaming never sees the tensor has neither.
    auto const explained = [&](std::string const &name) {
        return std::ranges::any_of(decisions, [&](cg::Decision const &d) {
            bool const says = (d.kind == cg::DecisionKind::Declined && d.pass_name == "SlabStreaming") ||
                              d.code == cg::passes::LinearCombinationContractionFolding::HeldBeside.code;
            return says && std::ranges::find(d.tensors, name) != d.tensors.end();
        });
    };
    std::set<std::string> unexplained;
    for (auto const &event : log->events()) {
        if (event.kind != cg::EventKind::Read) {
            continue;
        }
        auto const size = stored_bytes.find(event.tensor);
        if (size != stored_bytes.end() && event.bytes == size->second && !explained(event.tensor)) {
            unexplained.insert(event.tensor);
        }
    }
    std::ostringstream said;
    for (auto const &d : decisions) {
        if (!d.tensors.empty() && (d.kind == cg::DecisionKind::Declined || !d.code.empty())) {
            said << "  " << d.pass_name << " " << d.code << " [" << join(d.tensors) << "]: " << d.detail << "\n";
        }
    }
    INFO("declines naming tensors:\n" << said.str());
    INFO("held whole without SlabStreaming naming it: " << join(std::vector<std::string>(unexplained.begin(), unexplained.end())));
    CHECK(unexplained.empty());
}
