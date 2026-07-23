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
"""

import itertools, numpy as np
import einsums
from hypothesis import HealthCheck, assume, given, settings, strategies as st
from _sanitizer_scaling import sanitizer_examples
_c = itertools.count()
def mk(a, dt):
    a=np.asarray(a); t=einsums.create_zero_tensor(f"t{next(_c)}", list(a.shape), dtype=dt)
    if a.size: np.asarray(t)[...]=a
    return t
def rnd(shape, cplx, rng):
    return (rng.standard_normal(shape)+1j*rng.standard_normal(shape)) if cplx else rng.standard_normal(shape)
def dom(n, cplx, rng):
    M=rnd((n,n),cplx,rng); M[np.diag_indices(n)]=M[np.diag_indices(n)]+(n+2.0); return M
SZ=st.integers(1,6); DT=st.sampled_from(["float64","complex128"])
def H(x): return x.conj().T

@given(m=SZ,k=SZ,n=SZ,dt=DT,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_einsum_linearity(m,k,n,dt,seed):
    rng=np.random.default_rng(seed); c=(dt=="complex128")
    A=rnd((m,k),c,rng); B=rnd((k,n),c,rng); C=rnd((k,n),c,rng); al,be=1.5,-2.0
    def es(X,Y):
        Ct=mk(np.zeros((m,n)),dt); einsums.einsum("ij <- ik ; kj", Ct, mk(X,dt), mk(Y,dt)); return np.asarray(Ct).copy()
    lhs=es(A, al*B+be*C); rhs=al*es(A,B)+be*es(A,C)
    np.testing.assert_allclose(lhs,rhs,rtol=1e-9,atol=1e-9,err_msg=f"linearity m={m} k={k} n={n} {dt} s={seed}")

@given(m=SZ,k=SZ,n=SZ,dt=DT,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_gemm_transpose_identity(m,k,n,dt,seed):
    rng=np.random.default_rng(seed); c=(dt=="complex128")
    A=rnd((m,k),c,rng); B=rnd((k,n),c,rng)
    C=mk(np.zeros((m,n)),dt); einsums.linalg.gemm(1.0,mk(A,dt),mk(B,dt),0.0,C)         # C = A@B
    D=mk(np.zeros((n,m)),dt); einsums.linalg.gemm(1.0,mk(B,dt),mk(A,dt),0.0,D,trans_a=True,trans_b=True)  # B^T A^T
    np.testing.assert_allclose(np.asarray(C).T, np.asarray(D), rtol=1e-7,atol=1e-8,err_msg=f"(AB)^T m={m} k={k} n={n} {dt} s={seed}")

@given(n=SZ,dt=DT,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_invert_roundtrip(n,dt,seed):
    rng=np.random.default_rng(seed); c=(dt=="complex128"); A0=dom(n,c,rng)
    Ai=mk(A0,dt); einsums.linalg.invert(Ai)
    prod=mk(np.zeros((n,n)),dt); einsums.linalg.gemm(1.0,mk(A0,dt),Ai,0.0,prod)
    np.testing.assert_allclose(np.asarray(prod), np.eye(n), rtol=1e-5,atol=1e-7,err_msg=f"A@inv n={n} {dt} s={seed}")

@given(n=SZ,nrhs=st.integers(1,3),dt=DT,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_solve_roundtrip(n,nrhs,dt,seed):
    rng=np.random.default_rng(seed); c=(dt=="complex128"); A0=dom(n,c,rng); B0=rnd((n,nrhs),c,rng)
    Bx=mk(B0,dt); einsums.linalg.gesv(mk(A0,dt), Bx)             # Bx = X
    prod=mk(np.zeros((n,nrhs)),dt); einsums.linalg.gemm(1.0,mk(A0,dt),Bx,0.0,prod)
    np.testing.assert_allclose(np.asarray(prod), B0, rtol=1e-5,atol=1e-7,err_msg=f"A@solve n={n} nrhs={nrhs} {dt} s={seed}")

@given(n=SZ,dt=DT,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_eig_reconstruction(n,dt,seed):
    rng=np.random.default_rng(seed); c=(dt=="complex128")
    M=rnd((n,n),c,rng); A0=(M+H(M))/2.0
    At=mk(A0,dt); Wt=mk(np.zeros(n),"float64")
    (einsums.linalg.heev if c else einsums.linalg.syev)(At, Wt, compute_eigenvectors=True)
    V=np.asarray(At).copy(); w=np.asarray(Wt).copy()
    recon = V@np.diag(w)@H(V) if c else V@np.diag(w)@V.T
    np.testing.assert_allclose(recon, A0, rtol=1e-5,atol=1e-7,err_msg=f"eig recon n={n} {dt} s={seed}")
    ortho = H(V)@V if c else V.T@V
    np.testing.assert_allclose(ortho, np.eye(n), rtol=1e-5,atol=1e-7,err_msg=f"eig ortho n={n} {dt} s={seed}")

@given(m=SZ,n=SZ,dt=DT,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_svd_reconstruction(m,n,dt,seed):
    rng=np.random.default_rng(seed); c=(dt=="complex128"); A0=rnd((m,n),c,rng)
    U,S,Vh=einsums.linalg.svd(mk(A0,dt)); U,S,Vh=np.asarray(U),np.asarray(S),np.asarray(Vh)
    k=min(m,n); Smat=np.zeros((m,n),dtype=A0.dtype); Smat[:k,:k]=np.diag(S[:k])
    np.testing.assert_allclose(U@Smat@Vh, A0, rtol=1e-5,atol=1e-7,err_msg=f"svd recon m={m} n={n} {dt} s={seed}")
    np.testing.assert_allclose(H(U)@U, np.eye(U.shape[0]), rtol=1e-5,atol=1e-7,err_msg=f"svd U ortho m={m} n={n} {dt} s={seed}")
    np.testing.assert_allclose(Vh@H(Vh), np.eye(Vh.shape[0]), rtol=1e-5,atol=1e-7,err_msg=f"svd V ortho m={m} n={n} {dt} s={seed}")

@given(m=SZ,n=SZ,dt=DT,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_qr_reconstruction(m,n,dt,seed):
    rng=np.random.default_rng(seed); c=(dt=="complex128"); A0=rnd((m,n),c,rng)
    Q,R=einsums.linalg.qr(mk(A0,dt)); Q,R=np.asarray(Q),np.asarray(R)
    np.testing.assert_allclose(Q@R, A0, rtol=1e-5,atol=1e-7,err_msg=f"qr recon m={m} n={n} {dt} s={seed}")
    np.testing.assert_allclose(H(Q)@Q, np.eye(Q.shape[0]), rtol=1e-5,atol=1e-7,err_msg=f"qr Q ortho m={m} n={n} {dt} s={seed}")
    tri = np.tril(R, -1)
    np.testing.assert_allclose(tri, np.zeros_like(tri), atol=1e-7,err_msg=f"qr R upper-tri m={m} n={n} {dt} s={seed}")

@given(m=SZ,n=SZ,seed=st.integers(0,2**31-1))
@settings(max_examples=sanitizer_examples(300),deadline=None,suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
def test_conj_via_element_transform(m,n,seed):
    rng=np.random.default_rng(seed); A0=rnd((m,n),True,rng)
    t=mk(A0,"complex128"); einsums.linalg.element_transform(t, lambda x: x.conjugate())
    np.testing.assert_allclose(np.asarray(t), A0.conj(), rtol=1e-12,atol=1e-14,err_msg=f"conj m={m} n={n} s={seed}")
    einsums.linalg.element_transform(t, lambda x: x.conjugate())  # involution -> back to A0
    np.testing.assert_allclose(np.asarray(t), A0, rtol=1e-12,atol=1e-14,err_msg=f"conj involution m={m} n={n} s={seed}")
