//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The HeaderClosure gate run on a probe that does reach TensorAlgebra, and expected to fail. A gate
// that cannot fail is not a gate: this is what shows the include listing is read and matched.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/TensorAlgebra/Backends/ElementTransform.hpp>
