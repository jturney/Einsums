//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// dataflow(): run a task once all its input futures are ready. Included after TaskPool.hpp,
// which it and TaskPool::submit both depend on.
//
// Usage:
//   auto result = pool.dataflow("step2",
//       [](int a, double b) { return a + b; },
//       handle_a, handle_b);

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TaskPool/TaskHandle.hpp>

EINSUMS_NAMESPACE_BEGIN(task_pool)

// Forward declaration; defined in TaskPool.hpp
class TaskPool;

EINSUMS_NAMESPACE_END(task_pool)
