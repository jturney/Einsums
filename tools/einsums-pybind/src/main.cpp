//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// einsums-pybind — Phase 3 driver.
//
// Walks the headers given on the command line, builds a Module IR, and
// emits a pybind11 binding TU. The output is post-clang-format using the
// project's .clang-format if reachable from the output path; otherwise
// LLVM style.
//
// Invocation:
//     einsums-pybind --module <name> --output <path> <header>... [-- <args>]
//     einsums-pybind --module <name> --output <path> -p <build-dir> <header>...
//
// Without --output, the formatted source is written to stdout. Without
// --module, "einsums" is used.

#include <string>
#include <unordered_set>

#include "Emitter.hpp"
#include "IR.hpp"
#include "Properties.hpp"
#include "PyiEmitter.hpp"
#include "PythonOverloads.hpp"
#include "Visitor.hpp"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::tooling;

namespace {

// CommonOptionsParser owns -p and the trailing-arg conventions; we add
// tool-specific flags here. Static init is the libtooling pattern.
// NOLINTBEGIN(cert-err58-cpp,bugprone-throwing-static-initialization)
llvm::cl::OptionCategory g_tool_category("einsums-pybind options");

llvm::cl::opt<std::string> g_module_name("module", llvm::cl::desc("Python module name (PYBIND11_MODULE arg)"),
                                         llvm::cl::cat(g_tool_category), llvm::cl::init("einsums"));

llvm::cl::opt<std::string> g_output_path("output", llvm::cl::desc("Generated .cpp output path (default: stdout)"),
                                         llvm::cl::cat(g_tool_category), llvm::cl::init(""));

llvm::cl::opt<std::string> g_stub_output("stub-output",
                                         llvm::cl::desc("Path to write a Python type-stub (.pyi) file. "
                                                        "When set the .cpp emitter still runs (write to --output as usual); "
                                                        "the stub is written here. Without --stub-output no stub is produced."),
                                         llvm::cl::cat(g_tool_category), llvm::cl::init(""));

llvm::cl::opt<bool> g_dump_ir("dump-ir", llvm::cl::desc("Dump the parsed IR instead of emitting pybind11 (Phase 2 mode)"),
                              llvm::cl::cat(g_tool_category), llvm::cl::init(false));

llvm::cl::list<std::string> g_source_includes("source-include",
                                              llvm::cl::desc("Header path to emit as `#include \"...\"` in the generated TU. "
                                                             "Repeat for multiple headers. Required so generated bindings can "
                                                             "name the C++ types they bind."),
                                              llvm::cl::cat(g_tool_category));

llvm::cl::opt<einsums::pybind::Target>
    g_target("target", llvm::cl::desc("Binding library to emit code against"),
             llvm::cl::values(clEnumValN(einsums::pybind::Target::Pybind11, "pybind11", "pybind11 (default)"),
                              clEnumValN(einsums::pybind::Target::Nanobind, "nanobind", "nanobind")),
             llvm::cl::cat(g_tool_category), llvm::cl::init(einsums::pybind::Target::Pybind11));

llvm::cl::opt<std::string> g_register_fn("register-function",
                                         llvm::cl::desc("Emit a free function with this name that takes "
                                                        "(py::module_ &m) and registers all bindings, instead of "
                                                        "emitting a PYBIND11_MODULE block. Used by the "
                                                        "einsums_add_module autogen path so per-module TUs can be "
                                                        "aggregated under a single ``import einsums``."),
                                         llvm::cl::cat(g_tool_category), llvm::cl::init(""));
// NOLINTEND(cert-err58-cpp,bugprone-throwing-static-initialization)

// Per-translation-unit IR built up across all source files in a single
// run. Headers that get included by multiple of the module's other
// headers (e.g. Tensor.hpp pulled in via ArithmeticTensor.hpp +
// TensorForward.hpp + ...) cause the same annotated declaration to be
// visited once per TU. Dedupe by qualified_name so the emitter sees
// each declaration once.
einsums::pybind::Module         g_module;
std::unordered_set<std::string> g_seen_classes;
std::unordered_set<std::string> g_seen_functions;
std::unordered_set<std::string> g_seen_enums;
int                             g_error_count = 0;

class IrConsumer : public ASTConsumer {
  public:
    void HandleTranslationUnit(ASTContext &ctx) override {
        einsums::pybind::Visitor visitor(ctx);
        // Constrain the visitor to only emit bindings for declarations that
        // live in this module's own headers. Without this, transitively
        // included headers from other modules (e.g. RuntimeTensor.hpp
        // pulled in via Operations.hpp) get re-bound here, causing
        // "an object with that name is already defined" at import time.
        std::vector<std::string> filter;
        for (std::string const &p : g_source_includes) {
            filter.push_back(p);
        }
        visitor.set_module_header_filter(filter);
        visitor.TraverseDecl(ctx.getTranslationUnitDecl());
        einsums::pybind::Module local = std::move(visitor).take();
        g_error_count += visitor.error_count();
        for (auto &c : local.classes) {
            if (g_seen_classes.insert(c.qualified_name).second) {
                g_module.classes.push_back(std::move(c));
            }
        }
        for (auto &f : local.functions) {
            // Key on qualified name PLUS parameter signature so that
            // overloaded free functions sharing a name (e.g. one
            // templated, one not — common for Python-bindable wrappers
            // around C++ overload sets) all survive the cross-TU dedupe.
            std::string key = f.qualified_name;
            key += '(';
            for (std::size_t i = 0; i < f.params.size(); ++i) {
                if (i != 0) {
                    key += ',';
                }
                key += f.params[i].type;
            }
            key += ')';
            if (g_seen_functions.insert(std::move(key)).second) {
                g_module.functions.push_back(std::move(f));
            }
        }
        for (auto &e : local.enums) {
            if (g_seen_enums.insert(e.qualified_name).second) {
                g_module.enums.push_back(std::move(e));
            }
        }
    }
};

class IrAction : public ASTFrontendAction {
  protected:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance & /*ci*/, llvm::StringRef /*file*/) override {
        return std::make_unique<IrConsumer>();
    }
};

int write_output(std::string const &content) {
    if (g_output_path.empty()) {
        llvm::outs() << content;
        return 0;
    }
    std::error_code      ec;
    llvm::raw_fd_ostream out(g_output_path, ec);
    if (ec) {
        llvm::errs() << "einsums-pybind: cannot open output '" << g_output_path << "': " << ec.message() << "\n";
        return 1;
    }
    out << content;
    return 0;
}

} // namespace

int main(int argc, char const **argv) {
    auto expected = CommonOptionsParser::create(argc, argv, g_tool_category);
    if (!expected) {
        llvm::errs() << llvm::toString(expected.takeError());
        return 1;
    }
    CommonOptionsParser &options = *expected;
    ClangTool            tool(options.getCompilations(), options.getSourcePathList());
    int const            rc = tool.run(newFrontendActionFactory<IrAction>().get());

    // Resolve the dispatcher-grouping decisions for every templated free
    // function. Both the C++ emitter and the (future) .pyi emitter
    // consume the precomputed view via BoundFunction.python_overloads.
    einsums::pybind::compute_python_overloads(g_module);

    // Collapse @getter/@setter pairs into BoundClass.properties so
    // the .pyi emitter can render Python @property entries directly.
    einsums::pybind::compute_properties(g_module);

    if (g_dump_ir) {
        return write_output(einsums::pybind::dump(g_module));
    }

    einsums::pybind::EmitOptions opts;
    opts.module_name            = g_module_name;
    opts.register_function_name = g_register_fn;
    opts.source_path_for_format = g_output_path.empty() ? std::string("generated.cpp") : g_output_path;
    opts.target                 = g_target;
    for (std::string const &p : g_source_includes) {
        opts.source_includes.push_back(p);
    }
    std::string const generated = einsums::pybind::emit(g_module, opts);
    int const         write_rc  = write_output(generated);

    // Optionally also emit a Python stub (.pyi) file for pyright. The
    // stub mirrors what the C++ binding TU exposes — same py_names,
    // same submodule routing — so adding the flag to a build just adds
    // a sibling .pyi alongside the .cpp.
    if (!g_stub_output.empty()) {
        einsums::pybind::PyiOptions stub_opts;
        stub_opts.banner          = "module: " + std::string{g_module_name};
        std::string const    stub = einsums::pybind::emit_pyi(g_module, stub_opts);
        std::error_code      ec;
        llvm::raw_fd_ostream out(g_stub_output, ec);
        if (ec) {
            llvm::errs() << "einsums-pybind: cannot open stub output '" << g_stub_output << "': " << ec.message() << "\n";
            return 1;
        }
        out << stub;
    }

    if (g_error_count > 0) {
        llvm::errs() << "einsums-pybind: " << g_error_count << " error(s) — bindings may be incomplete.\n";
        return 1;
    }
    return rc != 0 ? rc : write_rc;
}
