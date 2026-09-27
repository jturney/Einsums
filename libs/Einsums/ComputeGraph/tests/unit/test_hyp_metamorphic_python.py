# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Metamorphic / algebraic property tests for the linalg surface.

Unlike the differential-vs-numpy harnesses, these assert identities the library
must satisfy, which also covers properties a value-equality oracle misses -- most
importantly DECOMPOSITION RECONSTRUCTION: eig/svd/qr were previously checked only
on eigenvalues / singular values, never on the returned VECTORS. Here:

  * einsum linearity:           einsum(A, aB+bC) == a*einsum(A,B) + b*einsum(A,C)
  * gemm transpose identity:    (A@B)^T == B^T @ A^T
  * invert round-trip:          A @ inv(A) == I
  * solve round-trip:           A @ gesv(A,B) == B
  * eig reconstruction:         V diag(w) V^(T/H) == A, with V orthonormal
  * svd reconstruction:         U diag(S) Vh == A, with U, Vh orthonormal
  * qr  reconstruction:         Q R == A, Q orthonormal, R upper-triangular
  * conjugation:                element_transform(conj) == numpy.conj, involutive

Complex variants use the conjugate-transpose (V^H, Vh, etc.). Note: einsums has no
dedicated conj/real/imag/abs op bound to Python -- conjugation is reachable only
via element_transform with a callable, which is what the conj test uses.

Every property is drawn over all four dtypes (conjugation over the two complex
ones) and judged at the rounding of the drawn dtype, scaled by the size of the
terms behind each element and, for the LAPACK round trips, by the condition
number or the dimension that the backward error bound carries.
"""

import itertools, numpy as np
import einsums
from hypothesis import HealthCheck, assume, given, settings, strategies as st
from _sanitizer_scaling import sanitizer_examples
from _dtype_draws import COMPLEX, DTYPES, assert_rounding_close, is_complex, random_array, real_of, rounded, wide
_c = itertools.count()
def mk(a, dt):
    a=np.asarray(a); t=einsums.create_zero_tensor(f"t{next(_c)}", list(a.shape), dtype=dt)
    if a.size: np.asarray(t)[...]=a
    return t
def rnd(shape, dt, rng):
    return random_array(shape, dt, rng)
def dom(n, dt, rng):
    M=rnd((n,n),dt,rng); M[np.diag_indices(n)]=M[np.diag_indices(n)]+(n+2.0); return rounded(M, dt)
def back(t, dt):
    """A tensor's values in double precision, so the checks below do not round at dt again."""
    return np.asarray(t).astype(wide(dt))
SZ=st.integers(1,6); DT=DTYPES
def H(x): return x.conj().T

@given(m=SZ,k=SZ,n=SZ,dt=DT,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_einsum_linearity(m,k,n,dt,seed):
    rng=np.random.default_rng(seed)
    A=rnd((m,k),dt,rng); B=rnd((k,n),dt,rng); C=rnd((k,n),dt,rng); al,be=1.5,-2.0
    def es(X,Y):
        Ct=mk(np.zeros((m,n)),dt); einsums.einsum("ij <- ik ; kj", Ct, mk(X,dt), mk(Y,dt)); return back(Ct,dt)
    lhs=es(A, al*B+be*C); rhs=al*es(A,B)+be*es(A,C)
    scale=np.abs(A)@(abs(al)*np.abs(B)+abs(be)*np.abs(C))
    assert_rounding_close(lhs,rhs,dt,scale,err_msg=f"linearity m={m} k={k} n={n} {dt} s={seed}")

@given(m=SZ,k=SZ,n=SZ,dt=DT,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_gemm_transpose_identity(m,k,n,dt,seed):
    rng=np.random.default_rng(seed)
    A=rnd((m,k),dt,rng); B=rnd((k,n),dt,rng)
    C=mk(np.zeros((m,n)),dt); einsums.linalg.gemm(1.0,mk(A,dt),mk(B,dt),0.0,C)         # C = A@B
    D=mk(np.zeros((n,m)),dt); einsums.linalg.gemm(1.0,mk(B,dt),mk(A,dt),0.0,D,trans_a=True,trans_b=True)  # B^T A^T
    assert_rounding_close(back(C,dt).T, back(D,dt), dt, (np.abs(A)@np.abs(B)).T, err_msg=f"(AB)^T m={m} k={k} n={n} {dt} s={seed}")

@given(n=SZ,dt=DT,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_invert_roundtrip(n,dt,seed):
    rng=np.random.default_rng(seed); A0=dom(n,dt,rng)
    Ai=mk(A0,dt); einsums.linalg.invert(Ai)
    prod=mk(np.zeros((n,n)),dt); einsums.linalg.gemm(1.0,mk(A0,dt),Ai,0.0,prod)
    # The inverse carries the condition number into its error; the product of A with it
    # is then judged against the size of its terms.
    scale=np.abs(A0)@np.abs(np.linalg.inv(A0))
    assert_rounding_close(back(prod,dt), np.eye(n), dt, scale, factor=np.linalg.cond(A0), err_msg=f"A@inv n={n} {dt} s={seed}")

@given(n=SZ,nrhs=st.integers(1,3),dt=DT,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_solve_roundtrip(n,nrhs,dt,seed):
    rng=np.random.default_rng(seed); A0=dom(n,dt,rng); B0=rnd((n,nrhs),dt,rng)
    Bx=mk(B0,dt); einsums.linalg.gesv(mk(A0,dt), Bx)             # Bx = X
    prod=mk(np.zeros((n,nrhs)),dt); einsums.linalg.gemm(1.0,mk(A0,dt),Bx,0.0,prod)
    scale=np.abs(A0)@np.abs(np.linalg.solve(A0,B0))
    assert_rounding_close(back(prod,dt), B0, dt, scale, factor=np.linalg.cond(A0), err_msg=f"A@solve n={n} nrhs={nrhs} {dt} s={seed}")

@given(n=SZ,dt=DT,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_eig_reconstruction(n,dt,seed):
    rng=np.random.default_rng(seed); c=is_complex(dt)
    M=rnd((n,n),dt,rng); A0=rounded((M+H(M))/2.0, dt)   # the matrix the tensor holds
    At=mk(A0,dt); Wt=mk(np.zeros(n),real_of(dt))
    (einsums.linalg.heev if c else einsums.linalg.syev)(At, Wt, compute_eigenvectors=True)
    V=back(At,dt); w=np.asarray(Wt).astype(np.float64)
    recon = V@np.diag(w)@H(V)
    # Backward stability: the residual is a small multiple of eps * ||A|| for the whole
    # matrix, not per element, and the multiple grows with n.
    assert_rounding_close(recon, A0, dt, np.linalg.norm(A0, 2), factor=n, err_msg=f"eig recon n={n} {dt} s={seed}")
    assert_rounding_close(H(V)@V, np.eye(n), dt, 1.0, factor=n, err_msg=f"eig ortho n={n} {dt} s={seed}")

@given(m=SZ,n=SZ,dt=DT,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_svd_reconstruction(m,n,dt,seed):
    rng=np.random.default_rng(seed); A0=rnd((m,n),dt,rng)
    U,S,Vh=einsums.linalg.svd(mk(A0,dt)); U,S,Vh=back(U,dt),np.asarray(S).astype(np.float64),back(Vh,dt)
    k=min(m,n); Smat=np.zeros((m,n),dtype=A0.dtype); Smat[:k,:k]=np.diag(S[:k])
    d=max(m,n); norm=np.linalg.norm(A0, 2)
    assert_rounding_close(U@Smat@Vh, A0, dt, norm, factor=d, err_msg=f"svd recon m={m} n={n} {dt} s={seed}")
    assert_rounding_close(H(U)@U, np.eye(U.shape[0]), dt, 1.0, factor=d, err_msg=f"svd U ortho m={m} n={n} {dt} s={seed}")
    assert_rounding_close(Vh@H(Vh), np.eye(Vh.shape[0]), dt, 1.0, factor=d, err_msg=f"svd V ortho m={m} n={n} {dt} s={seed}")

@given(m=SZ,n=SZ,dt=DT,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_qr_reconstruction(m,n,dt,seed):
    rng=np.random.default_rng(seed); A0=rnd((m,n),dt,rng)
    Q,R=einsums.linalg.qr(mk(A0,dt)); Q,R=back(Q,dt),back(R,dt)
    d=max(m,n); norm=np.linalg.norm(A0, 2)
    assert_rounding_close(Q@R, A0, dt, norm, factor=d, err_msg=f"qr recon m={m} n={n} {dt} s={seed}")
    assert_rounding_close(H(Q)@Q, np.eye(Q.shape[0]), dt, 1.0, factor=d, err_msg=f"qr Q ortho m={m} n={n} {dt} s={seed}")
    tri = np.tril(R, -1)
    assert_rounding_close(tri, np.zeros_like(tri), dt, norm, factor=d, err_msg=f"qr R upper-tri m={m} n={n} {dt} s={seed}")

@given(m=SZ,n=SZ,dt=COMPLEX,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_conj_via_element_transform(m,n,dt,seed):
    # Conjugation only flips a sign, so it is exact in every dtype.
    rng=np.random.default_rng(seed); A0=rnd((m,n),dt,rng)
    t=mk(A0,dt); einsums.linalg.element_transform(t, lambda x: x.conjugate())
    np.testing.assert_array_equal(back(t,dt), A0.conj(), err_msg=f"conj m={m} n={n} {dt} s={seed}")
    einsums.linalg.element_transform(t, lambda x: x.conjugate())  # involution -> back to A0
    np.testing.assert_array_equal(back(t,dt), A0, err_msg=f"conj involution m={m} n={n} {dt} s={seed}")
