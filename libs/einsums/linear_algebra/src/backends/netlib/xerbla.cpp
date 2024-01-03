//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/Print.hpp"
#include "internal.hpp"

namespace einsums::backend::linear_algebra::netlib {

/* Subroutine */ auto xerbla(const char *srname, int *info) -> int {
    /*  -- LAPACK auxiliary routine (version 2.0) --
           Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
           Courant Institute, Argonne National Lab, and Rice University
           September 30, 1994


        Purpose
        =======

        XERBLA  is an error handler for the LAPACK routines.
        It is called by an LAPACK routine if an input parameter has an
        invalid value.  A message is printed and execution stops.

        Installers may consider modifying the STOP statement in order to
        call system-specific exception-handling facilities.

        Arguments
        =========

        SRNAME  (input) CHARACTER*6
                The name of the routine which called XERBLA.

        INFO    (input) INTEGER
                The position of the invalid parameter in the parameter list

                of the calling routine.

       =====================================================================
    */

    println("** On entry to %6s, parameter number %2i had an illegal value.", srname, *info);

    /*     End of XERBLA */

    return 0;
} /* xerbla_ */
} // namespace einsums::backend::linear_algebra::netlib