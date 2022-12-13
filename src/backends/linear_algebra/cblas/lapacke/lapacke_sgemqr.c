/*****************************************************************************
  Copyright (c) 2014, Intel Corp.
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
  THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************
* Contents: Native high-level C interface to LAPACK function sgemqr
* Author: Intel Corporation
*****************************************************************************/

#include "lapacke_utils.h"

lapack_int LAPACKE_sgemqr(int matrix_layout, char side, char trans, lapack_int m, lapack_int n, lapack_int k, const float *a,
                          lapack_int lda, const float *t, lapack_int tsize, float *c, lapack_int ldc) {
    lapack_int info = 0;
    lapack_int lwork = -1;
    float *work = NULL;
    float work_query;
    lapack_int r;
    if (matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR) {
        LAPACKE_xerbla("LAPACKE_sgemqr", -1);
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    if (LAPACKE_get_nancheck()) {
        /* Optionally check input matrices for NaNs */
        r = LAPACKE_lsame(side, 'l') ? m : n;
        if (LAPACKE_sge_nancheck(matrix_layout, r, k, a, lda)) {
            return -7;
        }
        if (LAPACKE_sge_nancheck(matrix_layout, m, n, c, ldc)) {
            return -10;
        }
        if (LAPACKE_s_nancheck(tsize, t, 1)) {
            return -9;
        }
    }
#endif
    /* Query optimal working array(s) size */
    info = LAPACKE_sgemqr_work(matrix_layout, side, trans, m, n, k, a, lda, t, tsize, c, ldc, &work_query, lwork);
    if (info != 0) {
        goto exit_level_0;
    }
    lwork = (lapack_int)work_query;
    /* Allocate memory for work arrays */
    work = (float *)LAPACKE_malloc(sizeof(float) * lwork);
    if (work == NULL) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    /* Call middle-level interface */
    info = LAPACKE_sgemqr_work(matrix_layout, side, trans, m, n, k, a, lda, t, tsize, c, ldc, work, lwork);
    /* Release memory and exit */
    LAPACKE_free(work);
exit_level_0:
    if (info == LAPACK_WORK_MEMORY_ERROR) {
        LAPACKE_xerbla("LAPACKE_sgemqr", info);
    }
    return info;
}