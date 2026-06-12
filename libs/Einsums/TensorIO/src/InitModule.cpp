//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// TensorIO module initialization. Currently a no-op.
// Future: register runtime config options for default I/O paths.

namespace {
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
[[maybe_unused]] auto init_Einsums_TensorIO = []() { return 0; }();
} // namespace
