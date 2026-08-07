//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Defines.hpp>
#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>

#if defined(EINSUMS_HAVE_MKL)
typedef void (*XerblaEntry)(char const *Name, int const *Num, int const len);
extern "C" {
XerblaEntry mkl_set_xerbla(XerblaEntry xerbla);
}
#endif

EINSUMS_NAMESPACE_BEGIN(blas::vendor)

namespace {
extern "C" void xerbla(char const *srname, int const *info, int const /*len*/) {
    if (*info == 1001) {
        EINSUMS_LOG_WARN("BLAS/LAPACK: Incompatible optional parameters on entry to {}", srname);
    } else if (*info == 1000 || *info == 1089) {
        EINSUMS_LOG_WARN("BLAS/LAPACK: Insufficient workspace available in function {}.", srname);
    } else if (*info < 0) {
        EINSUMS_LOG_WARN("BLAS/LAPACK: Condition {} detected in function {}.", -(*info), srname);
    } else {
        EINSUMS_LOG_WARN("BLAS/LAPACK: The value of parameter {} is invalid in function call to {}.", *info, srname);
    }
}

} // namespace

void initialize() {
#if defined(EINSUMS_HAVE_MKL)
    mkl_set_xerbla(&xerbla);
#endif
}

void finalize() {
}

EINSUMS_NAMESPACE_END(blas::vendor)