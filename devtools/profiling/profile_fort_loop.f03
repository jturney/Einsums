! J and K builds from plain Fortran loop nests, serial and with OpenMP: the
! Fortran rows of the J/K breakdown figure (see profile_jk_breakdown.cpp for the
! other rows).
!
!   J(mu,nu) =  2 sum_{lam,sig} (mu nu|lam sig) D(lam,sig)
!   K(mu,nu) = -1 sum_{lam,sig} (mu lam|nu sig) D(lam,sig)
!
! The operands are the deterministic symmetric values profile_jk_breakdown
! fills, so the checksums printed here must match the ones it prints. Times
! are wall clock, the median of the trials after one untimed warm-up call.
!
! Usage: profile_fort_loop -n <norbs> -t <trials> [-c]
!   -c prints "fortran_loops,j_ms,k_ms", "fortran_omp_loops,j_ms,k_ms" and a
!      "checksum,J,K" line.
MODULE loops
   USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY : INT64, REAL64
   IMPLICIT NONE

CONTAINS

   PURE INTEGER(INT64) FUNCTION pair_index(a, b)
      INTEGER(INT64), INTENT(IN) :: a, b
      pair_index = MAX(a, b) * (MAX(a, b) + 1) / 2 + MIN(a, b)
   END FUNCTION pair_index

   ! Zero-based indices, matching tei_value in profile_jk_breakdown.cpp.
   PURE REAL(REAL64) FUNCTION tei_value(a, b, c, d)
      INTEGER(INT64), INTENT(IN) :: a, b, c, d
      INTEGER(INT64) :: idx
      idx = pair_index(pair_index(a, b), pair_index(c, d))
      tei_value = REAL(MOD(idx * 2654435761_INT64, 1000003_INT64), REAL64) / 500001.5_REAL64 - 1.0_REAL64
   END FUNCTION tei_value

   PURE REAL(REAL64) FUNCTION density_value(a, b)
      INTEGER(INT64), INTENT(IN) :: a, b
      density_value = REAL(MOD(pair_index(a, b) * 40503_INT64, 10007_INT64), REAL64) / 5003.5_REAL64 - 1.0_REAL64
   END FUNCTION density_value

   SUBROUTINE create_j(j, d, tei, n)
      INTEGER, INTENT(IN) :: n
      REAL(REAL64), INTENT(OUT) :: j(n, n)
      REAL(REAL64), INTENT(IN) :: d(n, n), tei(n, n, n, n)
      INTEGER :: mu, nu, lam, sig

      ! Storage order: the TEI streams through once, unit stride, while the
      ! n x n output stays in cache - the Fortran counterpart of the C nests
      ! reading their last index fastest.
      j(:, :) = 0
      DO sig = 1, n
         DO lam = 1, n
            DO nu = 1, n
               DO mu = 1, n
                  j(mu, nu) = j(mu, nu) + 2 * d(lam, sig) * tei(mu, nu, lam, sig)
               END DO
            END DO
         END DO
      END DO
   END SUBROUTINE create_j

   SUBROUTINE create_k(k, d, tei, n)
      INTEGER, INTENT(IN) :: n
      REAL(REAL64), INTENT(OUT) :: k(n, n)
      REAL(REAL64), INTENT(IN) :: d(n, n), tei(n, n, n, n)
      INTEGER :: mu, nu, lam, sig

      k(:, :) = 0
      DO sig = 1, n
         DO nu = 1, n
            DO lam = 1, n
               DO mu = 1, n
                  k(mu, nu) = k(mu, nu) - d(lam, sig) * tei(mu, lam, nu, sig)
               END DO
            END DO
         END DO
      END DO
   END SUBROUTINE create_k

   ! The same nests, OpenMP over the outermost (sig) loop. Every thread streams
   ! its own slab of the TEI and accumulates into a private copy of the n x n
   ! output, which the reduction sums at the end.
   SUBROUTINE create_j_omp(j, d, tei, n)
      INTEGER, INTENT(IN) :: n
      REAL(REAL64), INTENT(OUT) :: j(n, n)
      REAL(REAL64), INTENT(IN) :: d(n, n), tei(n, n, n, n)
      INTEGER :: mu, nu, lam, sig

      j(:, :) = 0
      !$omp parallel do reduction(+:j) private(lam, nu, mu)
      DO sig = 1, n
         DO lam = 1, n
            DO nu = 1, n
               DO mu = 1, n
                  j(mu, nu) = j(mu, nu) + 2 * d(lam, sig) * tei(mu, nu, lam, sig)
               END DO
            END DO
         END DO
      END DO
      !$omp end parallel do
   END SUBROUTINE create_j_omp

   SUBROUTINE create_k_omp(k, d, tei, n)
      INTEGER, INTENT(IN) :: n
      REAL(REAL64), INTENT(OUT) :: k(n, n)
      REAL(REAL64), INTENT(IN) :: d(n, n), tei(n, n, n, n)
      INTEGER :: mu, nu, lam, sig

      k(:, :) = 0
      !$omp parallel do reduction(+:k) private(nu, lam, mu)
      DO sig = 1, n
         DO nu = 1, n
            DO lam = 1, n
               DO mu = 1, n
                  k(mu, nu) = k(mu, nu) - d(lam, sig) * tei(mu, lam, nu, sig)
               END DO
            END DO
         END DO
      END DO
      !$omp end parallel do
   END SUBROUTINE create_k_omp

   REAL(REAL64) FUNCTION now_ms()
      INTEGER(INT64) :: count, rate
      CALL SYSTEM_CLOCK(count, rate)
      now_ms = 1000.0_REAL64 * REAL(count, REAL64) / REAL(rate, REAL64)
   END FUNCTION now_ms

   ! Median of v; sorts v in place (insertion sort, the trial count is small).
   REAL(REAL64) FUNCTION median(v)
      REAL(REAL64), INTENT(INOUT) :: v(:)
      INTEGER :: i, m, n
      REAL(REAL64) :: x
      n = SIZE(v)
      DO i = 2, n
         x = v(i)
         m = i - 1
         DO WHILE (m >= 1)
            IF (v(m) <= x) EXIT
            v(m + 1) = v(m)
            m = m - 1
         END DO
         v(m + 1) = x
      END DO
      IF (MOD(n, 2) == 1) THEN
         median = v(n / 2 + 1)
      ELSE
         median = 0.5_REAL64 * (v(n / 2) + v(n / 2 + 1))
      END IF
   END FUNCTION median

   ! Layout-independent for a symmetric matrix; the C++ driver computes the same sum.
   REAL(REAL64) FUNCTION checksum(x, n)
      INTEGER, INTENT(IN) :: n
      REAL(REAL64), INTENT(IN) :: x(n, n)
      INTEGER :: a, b
      checksum = 0
      DO b = 1, n
         DO a = 1, n
            checksum = checksum + x(a, b) * REAL(1 + (a - 1) + 3 * (b - 1), REAL64)
         END DO
      END DO
   END FUNCTION checksum

   SUBROUTINE parse_args(n, trials, csv)
      INTEGER, INTENT(OUT) :: n, trials
      LOGICAL, INTENT(OUT) :: csv
      INTEGER :: i, stat
      CHARACTER(len=64) :: arg

      n = 100
      trials = 20
      csv = .FALSE.
      i = 1
      DO WHILE (i <= COMMAND_ARGUMENT_COUNT())
         CALL GET_COMMAND_ARGUMENT(i, arg)
         SELECT CASE (TRIM(arg))
         CASE ("-n")
            i = i + 1
            CALL GET_COMMAND_ARGUMENT(i, arg)
            READ (arg, *, iostat=stat) n
            IF (stat /= 0 .OR. n < 1) STOP "-n needs a positive integer"
         CASE ("-t")
            i = i + 1
            CALL GET_COMMAND_ARGUMENT(i, arg)
            READ (arg, *, iostat=stat) trials
            IF (stat /= 0 .OR. trials < 1) STOP "-t needs a positive integer"
         CASE ("-c")
            csv = .TRUE.
         CASE DEFAULT
            STOP "usage: profile_fort_loop -n <norbs> -t <trials> [-c]"
         END SELECT
         i = i + 1
      END DO
   END SUBROUTINE parse_args

END MODULE loops

PROGRAM time_loops
   USE loops
   IMPLICIT NONE

   INTEGER :: n, trials, i
   INTEGER(INT64) :: a, b, c, d
   LOGICAL :: csv
   REAL(REAL64), ALLOCATABLE :: jm(:, :), km(:, :), dm(:, :), tei(:, :, :, :), tj(:), tk(:)
   REAL(REAL64) :: t0, mean_j, mean_k, sum_j, sum_k

   CALL parse_args(n, trials, csv)

   ALLOCATE (jm(n, n), km(n, n), dm(n, n), tei(n, n, n, n), tj(trials), tk(trials))

   DO d = 0, n - 1
      DO c = 0, n - 1
         DO b = 0, n - 1
            DO a = 0, n - 1
               tei(a + 1, b + 1, c + 1, d + 1) = tei_value(a, b, c, d)
            END DO
         END DO
      END DO
   END DO
   DO b = 0, n - 1
      DO a = 0, n - 1
         dm(a + 1, b + 1) = density_value(a, b)
      END DO
   END DO

   CALL create_j(jm, dm, tei, n)
   DO i = 1, trials
      t0 = now_ms()
      CALL create_j(jm, dm, tei, n)
      tj(i) = now_ms() - t0
   END DO

   CALL create_k(km, dm, tei, n)
   DO i = 1, trials
      t0 = now_ms()
      CALL create_k(km, dm, tei, n)
      tk(i) = now_ms() - t0
   END DO

   mean_j = median(tj)
   mean_k = median(tk)
   sum_j  = checksum(jm, n)
   sum_k  = checksum(km, n)

   CALL create_j_omp(jm, dm, tei, n)
   DO i = 1, trials
      t0 = now_ms()
      CALL create_j_omp(jm, dm, tei, n)
      tj(i) = now_ms() - t0
   END DO

   CALL create_k_omp(km, dm, tei, n)
   DO i = 1, trials
      t0 = now_ms()
      CALL create_k_omp(km, dm, tei, n)
      tk(i) = now_ms() - t0
   END DO

   ! The threaded nests must reproduce the serial ones' checksums: the reduction
   ! only reorders the summation, far inside this tolerance.
   IF (ABS(checksum(jm, n) - sum_j) > 1.0E-10_REAL64 * ABS(sum_j) .OR. &
       ABS(checksum(km, n) - sum_k) > 1.0E-10_REAL64 * ABS(sum_k)) THEN
      ERROR STOP "MISMATCH: the OpenMP nests disagree with the serial ones"
   END IF

   IF (csv) THEN
      WRITE (*, '(A,F0.4,A,F0.4)') "fortran_loops,", mean_j, ",", mean_k
      WRITE (*, '(A,F0.4,A,F0.4)') "fortran_omp_loops,", median(tj), ",", median(tk)
      WRITE (*, '(A,ES22.15,A,ES22.15)') "checksum,", sum_j, ",", sum_k
   ELSE
      WRITE (*, '(A,I0,A,I0,A)') "n=", n, ", ", trials, " trials, median wall time"
      WRITE (*, '(A,F9.2,A,F9.2,A)') "  fortran_loops      J ", mean_j, " ms   K ", mean_k, " ms"
      WRITE (*, '(A,F9.2,A,F9.2,A)') "  fortran_omp_loops  J ", median(tj), " ms   K ", median(tk), " ms"
      WRITE (*, '(A,ES22.15,A,ES22.15)') "  checksum J ", sum_j, "  K ", sum_k
   END IF

   DEALLOCATE (jm, km, dm, tei, tj, tk)
END PROGRAM time_loops
