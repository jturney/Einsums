//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file
/// pybind11 casters for the DLPNO contract types.
///
/// One caster per `@contract` dataclass. With these included, a stage entry
/// point is `m.def("stage_x", &dlpno::x)` and the conversion is invisible; the
/// binding's signature is the C++ API verbatim, which is what `promote` should
/// be emitting rather than a field-by-field read at every call site.
///
/// The direction is fixed and worth stating, because the "obvious" improvement
/// is the one wrong answer. Python OWNS the contract: `plan_pno_couplings`
/// builds a `@contract` dataclass and this side reads it. Binding
/// `CouplingPlan` to Python and having the planner construct it instead would
/// make the pure-Python path require this module to exist, which breaks the
/// promise that a method runs all-Python before anyone writes C++. The
/// awkwardness of reading a foreign object field by field is the price of that,
/// and it is the right price - it is just paid once, here, instead of at every
/// entry point.
///
/// Cost note for whoever needs it later: `load()` copies each field into a
/// `std::vector`. For a stage called once per calculation, as this one is, that
/// is not worth avoiding. A stage that took a plan-shaped argument inside an
/// iteration loop would want the flat arrays to arrive as numpy buffers with
/// the caster holding spans instead - zero copy, at the cost of the C++ side no
/// longer owning its inputs. This is the only file that would change.

#pragma once

#include <dlpno/PnoOverlaps.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace pybind11::detail {

namespace dlpno_detail {

/// Read one field, and say which one when it is missing or the wrong shape.
///
/// The whole reason this is a function rather than an inline `.cast<>()`: a
/// bare cast failure partway down a twelve-field contract reports the C++ type
/// it wanted and nothing about where it was, which turns a five-second fix into
/// a bisect. `promote` will emit contracts wider than this one.
template <typename T>
void read_field(handle src, char const *contract, char const *field, T &out) {
    if (!hasattr(src, field)) {
        throw value_error(std::string(contract) + ": object has no field '" + field +
                          "'. The C++ struct and the @contract dataclass have drifted apart.");
    }
    try {
        out = getattr(src, field).template cast<T>();
    } catch (cast_error const &) {
        throw value_error(std::string(contract) + "." + field + ": cannot convert to its C++ type. " +
                          "Check the field's element type against the contract.");
    }
}

} // namespace dlpno_detail

template <>
struct type_caster<dlpno::CouplingPlan> {
    PYBIND11_TYPE_CASTER(dlpno::CouplingPlan, const_name("CouplingPlan"));

    /// Python `@contract` dataclass -> the C++ aggregate.
    ///
    /// Throws rather than returning false. Returning false is pybind's "try the
    /// next overload", and there is no next overload: it would surface as a
    /// generic "incompatible function arguments" listing every parameter, which
    /// hides the one field that was actually wrong.
    bool load(handle src, bool) {
        if (src.is_none()) {
            return false;
        }
        using dlpno_detail::read_field;
        char const *c = "CouplingPlan";

        read_field(src, c, "classes", value.classes);
        read_field(src, c, "slots_pair", value.slots_pair);
        read_field(src, c, "slots_partner", value.slots_partner);
        read_field(src, c, "inv_perm", value.inv_perm);
        read_field(src, c, "pair", value.pair);
        read_field(src, c, "partner", value.partner);
        read_field(src, c, "cls", value.cls);
        read_field(src, c, "dest_slot", value.dest_slot);
        read_field(src, c, "src_slot", value.src_slot);
        read_field(src, c, "sign", value.sign);
        read_field(src, c, "factor", value.factor);
        read_field(src, c, "coupled_pairs", value.coupled_pairs);

        // The per-coupling arrays are parallel by construction on the Python
        // side. Checked here anyway because every loop in the stage indexes all
        // seven with one ordinal, and a short array would be a silent
        // out-of-bounds read rather than an exception.
        auto const n = value.pair.size();
        for (auto const &[name, size] : {std::pair{"partner", value.partner.size()}, std::pair{"cls", value.cls.size()},
                                         std::pair{"dest_slot", value.dest_slot.size()}, std::pair{"src_slot", value.src_slot.size()},
                                         std::pair{"sign", value.sign.size()}, std::pair{"factor", value.factor.size()}}) {
            if (size != n) {
                throw value_error(std::string(c) + ": per-coupling arrays must be parallel, but 'pair' has " + std::to_string(n) +
                                  " entries and '" + name + "' has " + std::to_string(size) + ".");
            }
        }
        return true;
    }
};

} // namespace pybind11::detail
