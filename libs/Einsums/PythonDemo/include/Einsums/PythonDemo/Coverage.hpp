//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Python/Annotations.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace einsums::pythondemo {

/// Vec3 — exercises operator binding and read-write fields.
class EINSUMS_EXPORT EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_RENAME("Vec3") Vec3 {
  public:
    EINSUMS_PYBIND_EXPOSE Vec3();

    EINSUMS_PYBIND_EXPOSE Vec3(double x, double y, double z);

    /// Component-wise addition; bound as Python's __add__.
    EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_OPERATOR("__add__") Vec3 operator+(Vec3 const &other) const;

    /// Scalar multiplication; bound as Python's __mul__.
    EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_OPERATOR("__mul__") Vec3 operator*(double scalar) const;

    /// Equality; bound as Python's __eq__.
    EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_OPERATOR("__eq__") bool operator==(Vec3 const &other) const;

    EINSUMS_PYBIND_EXPOSE double                         x;
    EINSUMS_PYBIND_EXPOSE double                         y;
    EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_READONLY double z;
};

/// Resource — exercises HOLDER (shared_ptr) + RVP (reference_internal).
/// The factory returns a pointer that pybind11 keeps alive via the
/// shared_ptr holder; the borrow() method exposes the embedded child by
/// reference, with the parent kept alive for the borrow's lifetime.
class EINSUMS_EXPORT EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_RENAME("Resource") EINSUMS_PYBIND_HOLDER(std::shared_ptr) Resource {
  public:
    /// Inner type returned by reference; a separate small bound class so we
    /// can prove the @rvp(reference_internal) policy is honored.
    class EINSUMS_EXPORT EINSUMS_PYBIND_EXPOSE Slot {
      public:
        EINSUMS_PYBIND_EXPOSE explicit Slot(int v);

        EINSUMS_PYBIND_GETTER("value") [[nodiscard]] int value() const;

      private:
        int _value = 0;
    };

    EINSUMS_PYBIND_EXPOSE Resource(std::string label, int slots);

    /// Borrow the i-th slot; parent must outlive the returned reference,
    /// hence reference_internal.
    EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_RVP(reference_internal) Slot &slot(int i);

    EINSUMS_PYBIND_GETTER("label") [[nodiscard]] std::string const &label() const;

  private:
    std::string       _label;
    std::vector<Slot> _slots;
};

/// CounterError — exercises @exception. Lives in the ``demo`` submodule
/// to also exercise @module.
class EINSUMS_EXPORT EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_EXCEPTION EINSUMS_PYBIND_MODULE("demo") CounterError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/// Helper that the binding raises so the smoke test can catch it.
/// Lives in the ``demo`` submodule alongside its exception type.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("demo") EINSUMS_EXPORT void raise_counter_error(std::string const &msg);

/// Worker — exercises RELEASE_GIL on a long-running method.
class EINSUMS_EXPORT EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_RENAME("Worker") Worker {
  public:
    EINSUMS_PYBIND_EXPOSE Worker();

    /// "Heavy" computation that should release the GIL while running.
    EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_RELEASE_GIL long crunch(long n);
};

} // namespace einsums::pythondemo
