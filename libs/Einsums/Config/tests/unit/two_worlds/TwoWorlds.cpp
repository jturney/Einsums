//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file
/// Two copies of one library in one process, and each consumer reaching its own.
///
/// This tests the MECHANISM rather than a claim about it. The interesting case
/// is the RTLD_GLOBAL one below: on ELF, without a linker version script, the
/// first-loaded copy's symbols win the global lookup and the second consumer
/// silently calls into the wrong library. With version scripts, the dynamic
/// linker rejects the version mismatch and keeps searching, so each consumer
/// reaches its own copy regardless of load order.
///
/// Platform reality, stated because the test passes everywhere and only earns
/// its keep on one:
///
/// - ELF (Linux): the flat symbol namespace makes interposition possible. This
///   is where the version script does the work, and where removing it should
///   turn the RTLD_GLOBAL case red.
/// - Mach-O (macOS): two-level namespace records the source dylib per binding
///   at link time, so there is no name-based race to lose.
/// - PE (Windows): imports bind per-DLL by name, likewise.
///
/// So a green run here on macOS says the test harness is sound; a green run on
/// Linux says the isolation actually works.

#include <string>

#include <Einsums/Testing.hpp>

#if defined(_WIN32)
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace {

using value_fn    = int (*)(void);
using identity_fn = void const *(*)(void);

/// A loaded consumer, closed on scope exit so each case starts clean.
class Loaded {
  public:
    /// @param global When true, load so the library's symbols join the global
    ///               lookup scope. That is the pollution case: some Python
    ///               packages set RTLD_GLOBAL process-wide, and once one copy
    ///               is global it is searched before any consumer's own group.
    Loaded(std::string const &path, bool global) {
#if defined(_WIN32)
        // PE has no equivalent knob; binding is per-DLL either way.
        (void)global;
        _handle = static_cast<void *>(LoadLibraryA(path.c_str()));
        if (_handle == nullptr) {
            _error = "LoadLibraryA failed for " + path;
        }
#else
        _handle = dlopen(path.c_str(), RTLD_NOW | (global ? RTLD_GLOBAL : RTLD_LOCAL));
        if (_handle == nullptr) {
            // Captured here, not on demand: dlerror() CLEARS the error, so a
            // second load between the failure and the report would erase it.
            char const *e = dlerror();
            _error        = path + ": " + (e != nullptr ? e : "unknown dlopen failure");
        }
#endif
    }

    ~Loaded() {
        if (_handle != nullptr) {
#if defined(_WIN32)
            FreeLibrary(static_cast<HMODULE>(_handle));
#else
            dlclose(_handle);
#endif
        }
    }

    Loaded(Loaded const &)            = delete;
    Loaded &operator=(Loaded const &) = delete;

    [[nodiscard]] bool ok() const { return _handle != nullptr; }

    [[nodiscard]] std::string const &error() const { return _error; }

    template <typename Fn>
    [[nodiscard]] Fn symbol(char const *name) const {
#if defined(_WIN32)
        return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(_handle), name));
#else
        // Handle-scoped lookup: searches this object and its dependencies, so
        // it finds THIS consumer's function even when another copy is global.
        return reinterpret_cast<Fn>(dlsym(_handle, name));
#endif
    }

  private:
    std::string _error;
    void       *_handle = nullptr;
};

/// Load both consumers together and report what each one reached.
struct Observation {
    int         value_a;
    int         value_b;
    void const *identity_a;
    void const *identity_b;
};

Observation observe_both(bool global) {
    Loaded const a(CONSUMER_A_PATH, global);
    Loaded const b(CONSUMER_B_PATH, global);
    INFO(a.error());
    REQUIRE(a.ok());
    INFO(b.error());
    REQUIRE(b.ok());

    auto const value_a    = a.symbol<value_fn>("consumer_value");
    auto const value_b    = b.symbol<value_fn>("consumer_value");
    auto const identity_a = a.symbol<identity_fn>("consumer_identity");
    auto const identity_b = b.symbol<identity_fn>("consumer_identity");
    REQUIRE(value_a != nullptr);
    REQUIRE(value_b != nullptr);
    REQUIRE(identity_a != nullptr);
    REQUIRE(identity_b != nullptr);

    return Observation{value_a(), value_b(), identity_a(), identity_b()};
}

} // namespace

TEST_CASE("each consumer reaches its own world when loaded privately", "[abi][two_worlds]") {
    // The baseline every platform should manage: two RTLD_LOCAL loads, each
    // resolving inside its own dependency group. If THIS fails, the harness is
    // wrong, not the isolation.
    Observation const o = observe_both(/*global=*/false);

    REQUIRE(o.value_a == MINIWORLD_A_VALUE);
    REQUIRE(o.value_b == MINIWORLD_B_VALUE);
    REQUIRE(o.identity_a != o.identity_b);
}

TEST_CASE("each consumer reaches its own world even in the global scope", "[abi][two_worlds]") {
    // The case the version script exists for. Loading with RTLD_GLOBAL puts the
    // first copy's symbols ahead of the second consumer's own group, which is
    // exactly what happens when any package in a process calls
    // sys.setdlopenflags(RTLD_GLOBAL). Unversioned, `consumer_value` in B ends
    // up calling A's `miniworld_value` and this reports A's constant twice.
    Observation const o = observe_both(/*global=*/true);

    INFO("a=" << o.value_a << " b=" << o.value_b << " (equal values mean the two copies interposed)");
    REQUIRE(o.value_a == MINIWORLD_A_VALUE);
    REQUIRE(o.value_b == MINIWORLD_B_VALUE);
    REQUIRE(o.identity_a != o.identity_b);
}

TEST_CASE("a statically linked consumer is a world of its own", "[abi][two_worlds]") {
    // The static-link case the handshake has to refuse. A consumer that
    // absorbed the library rather than linking it shared carries its own copy
    // of the identity, so it can never match the shared one no matter what the
    // loader does. That is the whole reason a build-time guard against a
    // static-only install is not enough on its own.
    Loaded const shared_consumer(CONSUMER_A_PATH, /*global=*/false);
    Loaded const static_consumer(CONSUMER_STATIC_PATH, /*global=*/false);
    INFO(shared_consumer.error());
    REQUIRE(shared_consumer.ok());
    INFO(static_consumer.error());
    REQUIRE(static_consumer.ok());

    auto const shared_value    = shared_consumer.symbol<value_fn>("consumer_value");
    auto const static_value    = static_consumer.symbol<value_fn>("consumer_value");
    auto const shared_identity = shared_consumer.symbol<identity_fn>("consumer_identity");
    auto const static_identity = static_consumer.symbol<identity_fn>("consumer_identity");
    REQUIRE(shared_value != nullptr);
    REQUIRE(static_value != nullptr);

    REQUIRE(shared_value() == MINIWORLD_A_VALUE);
    REQUIRE(static_value() == MINIWORLD_STATIC_VALUE);
    REQUIRE(shared_identity() != static_identity());
}
