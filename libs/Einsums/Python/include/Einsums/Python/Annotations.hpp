//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// The binding-annotation contract, the APIARY_* macros, is owned by
// Apiary, the standalone libclang codegen tool vendored at
// external/apiary. This thin shim stays at the historical
// ``Einsums/Python/Annotations.hpp`` path so existing includes across the
// codebase keep working unchanged. The macros themselves live in
// <apiary/Annotations.hpp>.
#include <apiary/Annotations.hpp>
