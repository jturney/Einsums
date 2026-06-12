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

/// Vec3: exercises operator binding and read-write fields.
class EINSUMS_EXPORT APIARY_EXPOSE APIARY_RENAME("Vec3") Vec3 {
  public:
    APIARY_EXPOSE Vec3();

    APIARY_EXPOSE Vec3(double x, double y, double z);

    /// Component-wise addition; bound as Python's __add__.
    APIARY_EXPOSE APIARY_OPERATOR("__add__") Vec3 operator+(Vec3 const &other) const;

    /// Scalar multiplication; bound as Python's __mul__.
    APIARY_EXPOSE APIARY_OPERATOR("__mul__") Vec3 operator*(double scalar) const;

    /// Equality; bound as Python's __eq__.
    APIARY_EXPOSE APIARY_OPERATOR("__eq__") bool operator==(Vec3 const &other) const;

    APIARY_EXPOSE double                 x;
    APIARY_EXPOSE double                 y;
    APIARY_EXPOSE APIARY_READONLY double z;
};

/// Resource: exercises the shared_ptr HOLDER and the reference_internal RVP.
/// The factory returns a pointer that pybind11 keeps alive via the
/// shared_ptr holder, and the borrow() method exposes the embedded child by
/// reference, with the parent kept alive for the borrow's lifetime.
class EINSUMS_EXPORT APIARY_EXPOSE APIARY_RENAME("Resource") APIARY_HOLDER(std::shared_ptr) Resource {
  public:
    /// Inner type returned by reference; a separate small bound class so we
    /// can prove the @rvp(reference_internal) policy is honored.
    class EINSUMS_EXPORT APIARY_EXPOSE Slot {
      public:
        APIARY_EXPOSE explicit Slot(int v);

        APIARY_GETTER("value") [[nodiscard]] int value() const;

      private:
        int _value = 0;
    };

    APIARY_EXPOSE Resource(std::string label, int slots);

    /// Borrow the i-th slot; parent must outlive the returned reference,
    /// hence reference_internal.
    APIARY_EXPOSE APIARY_RVP(reference_internal) Slot &slot(int i);

    APIARY_GETTER("label") [[nodiscard]] std::string const &label() const;

  private:
    std::string       _label;
    std::vector<Slot> _slots;
};

/// CounterError: exercises @exception. Lives in the ``demo`` submodule
/// to also exercise @module.
class EINSUMS_EXPORT APIARY_EXPOSE APIARY_EXCEPTION APIARY_MODULE("demo") CounterError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/// Helper that the binding raises so the smoke test can catch it.
/// Lives in the ``demo`` submodule alongside its exception type.
APIARY_EXPOSE APIARY_MODULE("demo") EINSUMS_EXPORT void raise_counter_error(std::string const &msg);

/// Worker: exercises RELEASE_GIL on a long-running method.
class EINSUMS_EXPORT APIARY_EXPOSE APIARY_RENAME("Worker") Worker {
  public:
    APIARY_EXPOSE Worker();

    /// "Heavy" computation that should release the GIL while running.
    APIARY_EXPOSE APIARY_RELEASE_GIL long crunch(long n);
};

} // namespace einsums::pythondemo
