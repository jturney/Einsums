//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/ElementOps.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cmath>
#include <concepts>

EINSUMS_NAMESPACE_BEGIN(compute_graph::element_ops)

void register_builtin_element_ops(ElementOpRegistry &registry) {
    // Each kernel is written once, generically, and the registry instantiates
    // it per dtype. `sqrt_or_zero` is the one that is CONSTRAINED rather than
    // merely generic: "greater than zero" is not a question a complex number
    // answers, so the constraint is what tells the registry the op is real-only
    // instead of leaving it to fail at instantiation.
    registry.register_op(
        "recip", []<typename T>(T x) { return T{1} / x; },
        ElementOpSignature{.arity = 1, .domain = ElementOpDomain::AllDtypes, .description = "1 / x"});

    registry.register_op(
        "square", []<typename T>(T x) { return x * x; },
        ElementOpSignature{.arity = 1, .domain = ElementOpDomain::AllDtypes, .description = "x * x"});

    registry.register_op(
        "negate", []<typename T>(T x) { return -x; },
        ElementOpSignature{.arity = 1, .domain = ElementOpDomain::AllDtypes, .description = "-x"});

    registry.register_op(
        "sqrt_or_zero", []<std::floating_point T>(T x) { return x > T{0} ? std::sqrt(x) : T{0}; },
        ElementOpSignature{.arity = 1, .domain = ElementOpDomain::RealOnly, .description = "sqrt(x) for x > 0, else 0"});

    // The kernel a symmetric inverse square root is made of. Guarded the same way its
    // partner is, and for a reason rather than for symmetry: the matrices this is applied
    // to are metrics, a metric of a nearly linearly dependent basis has eigenvalues at or
    // below zero, and 1/sqrt of one of those is an infinity that propagates through
    // everything the factorization then touches. Zeroing the offending direction is what
    // every orthogonalization does with it.
    registry.register_op(
        "inv_sqrt_or_zero", []<std::floating_point T>(T x) { return x > T{0} ? T{1} / std::sqrt(x) : T{0}; },
        ElementOpSignature{.arity = 1, .domain = ElementOpDomain::RealOnly, .description = "1/sqrt(x) for x > 0, else 0"});
}

ElementOpRegistry &global_element_op_registry() {
    // Function-local statics, in order: the registry is constructed on first
    // use, then seeded exactly once. Two statics rather than a seeded factory
    // because the registry owns a mutex and is therefore neither copyable nor
    // movable. Never destroyed before the graphs that name its ops stop being
    // replayed, which is why this is a reference rather than an object with a
    // destructor ordered against other statics.
    static ElementOpRegistry registry;
    static bool const        seeded = (register_builtin_element_ops(registry), true);
    (void)seeded;
    return registry;
}

EINSUMS_NAMESPACE_END(compute_graph::element_ops)
