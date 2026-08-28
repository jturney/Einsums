//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Factorization.hpp>

#include <memory>
#include <string>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// A provider that offers nothing, which is all these cases need: the registry's job is to
/// hold providers and hand back the ones claiming a tag, and it does that without ever
/// calling propose.
class NamedProvider : public cg::FactorizationProvider {
  public:
    NamedProvider(std::string name, std::string tag) : _name(std::move(name)), _tag(std::move(tag)) {}

    [[nodiscard]] std::string name() const override { return _name; }
    [[nodiscard]] std::string tag() const override { return _tag; }

    [[nodiscard]] expected<cg::FactorizationPlan, std::string> propose(cg::Graph const & /*graph*/,
                                                                       cg::TensorId /*tensor*/) const override {
        return unexpected(std::string{"this provider offers nothing"});
    }

  private:
    std::string _name;
    std::string _tag;
};

std::shared_ptr<cg::FactorizationProvider> provider(std::string name, std::string tag) {
    return std::make_shared<NamedProvider>(std::move(name), std::move(tag));
}

} // namespace

TEST_CASE("Factorization - a registry hands back every provider claiming a tag, in registration order", "[ComputeGraph][Factorization]") {
    cg::FactorizationRegistry registry;
    REQUIRE(registry.size() == 0);
    REQUIRE_FALSE(registry.claims("eri"));

    registry.add(provider("AutoDF", "eri"));
    registry.add(provider("AutoCholesky", "eri"));
    registry.add(provider("AutoTHC", "eri_df_b"));

    REQUIRE(registry.size() == 3);
    REQUIRE(registry.claims("eri"));
    REQUIRE(registry.claims("eri_df_b"));
    REQUIRE_FALSE(registry.claims("fock"));

    auto const eri = registry.for_tag("eri");
    REQUIRE(eri.size() == 2);
    // Registration order, not some container's iteration order: a pass that costs several
    // offers and finds two of them equal breaks the tie somehow, and a tie broken by
    // iteration order makes the optimizer's output depend on nothing anyone can see.
    REQUIRE(eri[0]->name() == "AutoDF");
    REQUIRE(eri[1]->name() == "AutoCholesky");

    REQUIRE(registry.for_tag("fock").empty());
}

TEST_CASE("Factorization - two providers cannot share a name", "[ComputeGraph][Factorization]") {
    cg::FactorizationRegistry registry;
    registry.add(provider("AutoDF", "eri"));

    // Not a tidiness rule. The name is what the approximation record carries and what the
    // report names, and two providers answering to one of them makes which ran a function of
    // registration order.
    REQUIRE_THROWS_WITH(registry.add(provider("AutoDF", "coulomb_metric")),
                        Catch::Matchers::ContainsSubstring("already registered") && Catch::Matchers::ContainsSubstring("'eri'"));
    REQUIRE(registry.size() == 1);
}

TEST_CASE("Factorization - a provider can be removed and the registry emptied", "[ComputeGraph][Factorization]") {
    cg::FactorizationRegistry registry;
    registry.add(provider("AutoDF", "eri"));
    registry.add(provider("AutoTHC", "eri"));

    REQUIRE(registry.remove("AutoDF"));
    REQUIRE_FALSE(registry.remove("AutoDF"));
    REQUIRE(registry.size() == 1);
    REQUIRE(registry.for_tag("eri").front()->name() == "AutoTHC");

    registry.clear();
    REQUIRE(registry.size() == 0);
    REQUIRE_FALSE(registry.claims("eri"));
}

TEST_CASE("Factorization - a null provider is refused rather than stored", "[ComputeGraph][Factorization]") {
    cg::FactorizationRegistry registry;
    REQUIRE_THROWS_WITH(registry.add(nullptr), Catch::Matchers::ContainsSubstring("null"));
    REQUIRE(registry.size() == 0);
}

TEST_CASE("Factorization - the process registry is one object", "[ComputeGraph][Factorization]") {
    auto             &registry = cg::global_factorization_registry();
    std::size_t const before   = registry.size();

    registry.add(provider("TestOnlyProvider", "test_only_tag"));
    REQUIRE(cg::global_factorization_registry().size() == before + 1);
    REQUIRE(cg::global_factorization_registry().claims("test_only_tag"));

    // Put it back. A process-global registry a test adds to and leaves is a test that changes
    // what every later one sees.
    REQUIRE(registry.remove("TestOnlyProvider"));
    REQUIRE(cg::global_factorization_registry().size() == before);
}
