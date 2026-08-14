//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// -------------------------- Example ---------------------------- //

#include <Einsums/CommandLine/CommandLine.hpp>

using namespace einsums::cl;

OptionCategory Cat{"General"};
Flag           Verbose{"verbose", {'v'}, "Enable verbose output", Cat};
Opt<int>       Iters{"iters",
                     {'i'},
               5,
               "Iteration count",
               Cat,
               Visibility::Normal,
               Occurrence::Optional,
               ValueExpected::ValueRequired,
               RangeBetween(1, 1000000)};

OptionCategory   ToolsCat{"Tools"};
Opt<std::string> ToolName{"tool", {'t'}, "Which tool", ToolsCat};

List<std::string> Inputs{"inputs", Positional{}, "Input files"};

int main(int argc, char **argv) {
    std::vector<std::string> args(argv, argv + argc);

    // Every option above also reads an environment variable derived from its
    // name, so ETOOL_ITERS=10 sets --iters without touching the command line.
    // Values resolve default < config file < environment < command line.
    Registry::instance().set_env_prefix("ETOOL");

    auto pr = parse_with_config(args, "ein-tool", "1.2.3", "config.json");
    if (!pr.ok)
        return pr.exit_code;
    fmt::print("verbose={} iters={} (from {}) inputs={}\n", Verbose.get(), Iters.get(), to_string(Iters.value_source),
               Inputs.values().size());
    return 0;
}
