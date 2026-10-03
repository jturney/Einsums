//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Profile/Profile.hpp>

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <thread>

using namespace waggle;

TEST_CASE("A site is registered once, by its whole description", "[profiler][sites]") {
    SiteTable  sites;
    Site const a{.name_id = 1, .file_id = 2, .func_id = 3, .line = 10, .domain = 0};
    Site       b = a;
    b.line       = 11;

    uint32_t const id_a = sites.add(a);
    CHECK(id_a != 0); // 0 is no site
    // A second registration of the same site, as another shared object's copy of an inline
    // function's static would make, gets the same id.
    CHECK(sites.add(a) == id_a);
    CHECK(sites.add(b) != id_a);
    CHECK(sites.get(id_a).line == 10);
    CHECK(sites.get(0).name_id == 0);
    CHECK(sites.get(999).name_id == 0); // an id this table never issued
}

TEST_CASE("Domains are named once, and the empty name is domain 0", "[profiler][sites]") {
    DomainTable domains;
    CHECK(domains.add("") == 0);
    uint32_t const einsums = domains.add("einsums");
    CHECK(einsums != 0);
    CHECK(domains.add("einsums") == einsums);
    CHECK(domains.add("nectar") != einsums);
    CHECK(domains.name(einsums) == "einsums");
    CHECK(domains.name(0).empty());
}

namespace {

AggNode const *find_named(AggNode const &node, std::string const &name) {
    if (node.name == name) {
        return &node;
    }
    for (auto const &child : node.children) {
        if (auto const *found = find_named(*child.second, name)) {
            return found;
        }
    }
    return nullptr;
}

} // namespace

// An event carries a site and, for a name built at run time, the name: the node takes its file,
// line and function from the site either way.
TEST_CASE("A zone's file and line come from its site, its name from the event when it has one", "[profiler][sites]") {
    auto          &prof    = Profiler::instance();
    uint32_t const site_id = prof.register_site("sites: static name", "sites_test.cpp", 77, "site_function");
    uint32_t       tid     = 0;
    std::thread    worker([&] {
        tid = Profiler::current_thread_id();
        prof.push_interned(site_id, 0);
        prof.pop();
        prof.push_interned(site_id, prof.string_table().intern("sites: run-time name"));
        prof.pop();
    });
    worker.join();
    prof.flush();

    auto lock = prof.consumer()->lock_shared();
    auto it   = prof.consumer()->thread_data().find(tid);
    REQUIRE(it != prof.consumer()->thread_data().end());
    for (char const *name : {"sites: static name", "sites: run-time name"}) {
        AggNode const *node = find_named(it->second.root, name);
        REQUIRE(node != nullptr);
        CHECK(node->file == "sites_test.cpp");
        CHECK(node->line == 77);
        CHECK(node->function == "site_function");
    }
}
