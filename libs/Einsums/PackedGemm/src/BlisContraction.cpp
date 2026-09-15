//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>

#include <complex>

// The one place the packed engine is code-generated.
//
// blis_contraction used to be templated on the tensor TYPES as well as the
// element type, so every call site produced its own copy and gcc optimized each
// one from scratch. Measured at -O3, one 189-line test file carried 724
// instantiations - 207 KB of generated kernel - and spent 70% of its 34.9 s
// compile in the optimizer expanding them.
//
// Nothing in the engine needed those types: it reads a data pointer, a rank,
// and dims and strides, all of which the PackingPlan already carries as runtime
// values. Keyed on the element type alone it is four functions, compiled here,
// with -O3 exactly as before - the code is optimized once rather than less.
//
// Keep this list in step with the extern template declarations at the bottom of
// EinsumPackedGemm.hpp.

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

template EINSUMS_EXPORT void blis_contraction<float>(PackingPlan const &, float *, OperandView<float> const &, OperandView<float> const &,
                                                     float, float, bool, bool, bool);
template EINSUMS_EXPORT void blis_contraction<double>(PackingPlan const &, double *, OperandView<double> const &,
                                                      OperandView<double> const &, double, double, bool, bool, bool);
template EINSUMS_EXPORT void blis_contraction<std::complex<float>>(PackingPlan const &, std::complex<float> *,
                                                                   OperandView<std::complex<float>> const &,
                                                                   OperandView<std::complex<float>> const &, std::complex<float>,
                                                                   std::complex<float>, bool, bool, bool);
template EINSUMS_EXPORT void blis_contraction<std::complex<double>>(PackingPlan const &, std::complex<double> *,
                                                                    OperandView<std::complex<double>> const &,
                                                                    OperandView<std::complex<double>> const &, std::complex<double>,
                                                                    std::complex<double>, bool, bool, bool);

EINSUMS_NAMESPACE_END(packed_gemm)
