//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/config/ExportDefinitions.hpp"

#include <memory>
#include <string>

/**
 * @brief Convenience class for timing blocks of code.
 *
 * The Section class is a wrapper to the timer system used in Einsums.
 * Additionally, if at configure time VTune is detected then sections
 * are also registered with VTune to aid in analysis.
 *
 * @code
 * void test_code() {
 *   Section timingSection{"test_code"};
 *
 *   // Perform some time consuming action.
 * }
 * @endcode
 *
 */
struct EINSUMS_EXPORT Section {
    struct Impl;

    /**
     * @brief Construct a new Section object
     *
     * @param name Name of the section to be timed. If VTune is available then \p name becomes the label in VTune.
     * @param pushTimer Enable Einsums timing mechanism for this section. Default is true.
     */
    Section(std::string const &name, bool pushTimer = true);

    /**
     * @brief Construct a new Section object
     *
     * @param name Name of the section to be timed.
     * @param domain If VTune is available then this is the label used in VTune.
     * @param pushTimer Enable Einsums timing mechanism for this section. Default is true.
     */
    Section(std::string const &name, std::string const &domain, bool pushTimer = true);

    /**
     * @brief Destroy the Section object
     */
    ~Section();

    /**
     * @brief Manually stop the section.
     *
     * If for some reason you need to prematurely end a timing section then use this function.
     * This is automatically called by the destructor when Section goes out of scope.
     *
     */
    void end();

  private:
    void begin();

    std::unique_ptr<Impl> _impl;
};

// Use of LabeledSection requires fmt/format.h to be included and the use of
// (BEGIN|END)_EINSUMS_NAMESPACE_(CPP|HPP)() defined in _Common.hpp

/**
 * @brief Convenience wrapper to Section.
 *
 * Use of this macro requires that (BEGIN|END)_EINSUMS_NAMESPACE_(CPP|HPP)() defined
 * in _Common.hpp is included and used.
 * Constructs a label that includes the encompassing namespace and function names.
 * This macro also includes an extra label that will be appended to the section name.
 */
#define LabeledSection1(x) const Section _section(fmt::format("{}::{} {}", detail::s_Namespace, __func__, x))

/**
 * @brief Convenience wrapper to Section.
 *
 * Use of this macro requires that (BEGIN|END)_EINSUMS_NAMESPACE_(CPP|HPP)() defined
 * in _Common.hpp is included and used.
 * Constructs a label that includes the encompassing namespace and function names.
 */
#define LabeledSection0() const Section _section(fmt::format("{}::{}", detail::s_Namespace, __func__))
