//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/HPTT/Options.hpp>
#include <Einsums/Options/Declare.hpp>

EINSUMS_NAMESPACE_BEGIN()

hptt::SelectionMethod hptt_plan_selection() {
    auto const val = config::get(option::HpttSelectionMethod);
    if (val == "measure") {
        return hptt::MEASURE;
    }
    if (val == "patient") {
        return hptt::PATIENT;
    }
    if (val == "crazy") {
        return hptt::CRAZY;
    }
    return hptt::ESTIMATE;
}

int register_Einsums_HPTT_options() {
    cl::register_option(option::HpttSelectionMethod);
    return 0;
}

EINSUMS_NAMESPACE_END()
