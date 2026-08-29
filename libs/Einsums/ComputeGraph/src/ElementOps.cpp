//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/ElementOps.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>
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
    //
    // The threshold is a POLICY NUMBER and therefore a parameter rather than a constant.
    // Testing x > 0 catches the direction that has already gone negative and misses the one
    // that has not quite: an eigenvalue at +1e-17 passes the guard and comes back as a 1/sqrt
    // of about 3e8, which is worse than the infinity because nothing downstream looks wrong.
    // Real orthogonalizations drop BELOW A THRESHOLD, and the number is the caller's because
    // it depends on what their metric is a metric of.
    //
    // The default is zero, which is exactly the guard this op has always had. That is the
    // documented default of the compatibility policy rather than a recommendation: a file
    // written before a node could carry a threshold has to keep computing what it computed,
    // and a caller who wants a real one says so. MetricFitFactorization does.
    registry.register_op(
        "inv_sqrt_or_zero",
        []<std::floating_point T>(T x, double threshold) {
            // The floor is never below zero, whatever the caller passed: this op's whole
            // contract is that it does not return a non-finite value, and a negative
            // threshold would let a negative eigenvalue through to std::sqrt.
            T const floor = std::max(static_cast<T>(threshold), T{0});
            return x > floor ? T{1} / std::sqrt(x) : T{0};
        },
        ElementOpSignature{.arity         = 1,
                           .domain        = ElementOpDomain::RealOnly,
                           .parameterized = true,
                           .default_param = 0.0,
                           .description   = "1/sqrt(x) for x above the drop threshold, else 0"});

    // The indicator that counts what a guarded kernel threw away. Exact equality with zero
    // looks like the usual floating-point mistake and is not one HERE: the entries this is
    // applied to are zero only because a guard such as `inv_sqrt_or_zero` ASSIGNED T{0} to
    // them, and no surviving value can drift into looking dropped, since the inverse square
    // root of a small positive is enormous rather than tiny. The guard and the counter read
    // the same decision.
    registry.register_op(
        "is_zero", []<std::floating_point T>(T x) { return x == T{0} ? T{1} : T{0}; },
        ElementOpSignature{.arity = 1, .domain = ElementOpDomain::RealOnly, .description = "1 when x is exactly zero, else 0"});
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
