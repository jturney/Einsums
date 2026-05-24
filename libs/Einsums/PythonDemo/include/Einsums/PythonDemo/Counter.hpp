//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Python/Annotations.hpp>

#include <string>

namespace einsums::pythondemo {

/// A tiny counter — proof-of-life for the einsums-pybind autogen path.
///
/// This class exists purely so the autogen pipeline can be exercised
/// end-to-end against a real Einsums module. It pulls in no other
/// Einsums modules and uses no template machinery beyond what the
/// emitter already handles.
class EINSUMS_EXPORT EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_RENAME("Counter") Counter {
  public:
    /// Default-construct a counter starting at zero.
    EINSUMS_PYBIND_EXPOSE Counter();

    /// Construct a counter with an explicit initial value and label.
    EINSUMS_PYBIND_EXPOSE Counter(long start, std::string label);

    /// Increase the count by one and return the new value.
    EINSUMS_PYBIND_EXPOSE long bump();

    /// Increase the count by `delta` and return the new value.
    EINSUMS_PYBIND_EXPOSE long bump_by(long delta);

    /// Reset the count to zero.
    EINSUMS_PYBIND_EXPOSE void reset();

    /// Read-only access to the current value.
    EINSUMS_PYBIND_GETTER("value") [[nodiscard]] long get_value() const;

    /// Read-only access to the label.
    EINSUMS_PYBIND_GETTER("label") [[nodiscard]] std::string const &get_label() const;

  private:
    long        _value = 0;
    std::string _label;
};

/// Free function — proof that m.def() emission works through the autogen
/// path, not just class methods.
EINSUMS_PYBIND_EXPOSE EINSUMS_EXPORT long sum_of_squares(long n);

} // namespace einsums::pythondemo
