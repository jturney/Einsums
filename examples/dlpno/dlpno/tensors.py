#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Small tensor helpers shared by the DLPNO port.

Every numeric operation here runs through ``einsums.linalg``; numpy appears only
to move buffers in and out and to carry index lists. The helpers mirror the psi4
utilities the DLPNO source leans on (``linalg::doublet``/``triplet``,
``Matrix::diagonalize``, ``DLPNO::canonicalizer``, ``DLPNO::orthocanonicalizer``)
so the ported code reads like the original.

Note on eigenvalue ordering: psi4's ``diagonalize(..., descending)`` is used
throughout DLPNO, while LAPACK (and therefore ``einsums.linalg.syev``) returns
ascending eigenvalues. :func:`eigh` takes a ``descending`` flag so the ports can
keep psi4's convention, which matters wherever a leading block is truncated
(PNO occupation numbers, near-linear-dependency removal).
"""

import numpy as np

import einsums
from einsums import linalg as la

__all__ = [
    "zeros",
    "scalar",
    "from_numpy",
    "view",
    "shape",
    "transpose",
    "doublet",
    "triplet",
    "eigh",
    "solve",
    "diagonal",
    "reshape",
    "vector_dot",
    "dot_into",
    "canonicalizer",
    "orthocanonicalizer",
]

def zeros(name, shape_):
    """A zeroed dense float64 tensor."""
    return einsums.create_zero_tensor(name, [int(s) for s in shape_], dtype="float64")


def scalar(name="scalar"):
    """A length-1 tensor to receive a captured reduction."""
    return zeros(name, [1])


def from_numpy(name, arr):
    """Copy a numpy array into a fresh dense float64 tensor."""
    arr = np.asarray(arr, dtype=np.float64)
    T = zeros(name, arr.shape)
    if arr.size:
        np.asarray(T)[...] = arr
    return T


def view(T):
    """The writable numpy view of a tensor's storage."""
    return np.asarray(T)


def shape(T):
    return np.asarray(T).shape


def transpose(A, name="transpose"):
    """A fresh tensor holding ``A^T``."""
    m, n = shape(A)
    out = zeros(name, [n, m])
    einsums.permute("ab <- ba", out, A)
    return out


def doublet(A, B, trans_a=False, trans_b=False, name="doublet"):
    """``op(A) op(B)`` as a fresh tensor, the analogue of psi4's ``linalg::doublet``."""
    m = shape(A)[1] if trans_a else shape(A)[0]
    n = shape(B)[0] if trans_b else shape(B)[1]
    out = zeros(name, [m, n])
    la.gemm(1.0, A, B, 0.0, out, trans_a=trans_a, trans_b=trans_b)
    return out


def triplet(A, B, C, trans_a=False, trans_b=False, trans_c=False, name="triplet"):
    """``op(A) op(B) op(C)``, the analogue of psi4's ``linalg::triplet``."""
    return doublet(doublet(A, B, trans_a, trans_b), C, False, trans_c, name=name)


def eigh(A, descending=False, name="eigen"):
    """Symmetric eigendecomposition. Returns ``(evals, evecs)`` with the
    eigenvectors in the columns of ``evecs``, as fresh tensors.

    ``A`` is not modified.
    """
    n = shape(A)[0]
    evecs = from_numpy(name + " vectors", view(A))
    evals = zeros(name + " values", [n])
    la.syev(evecs, evals, compute_eigenvectors=True)
    if descending:
        v = view(evecs)
        w = view(evals)
        v[...] = v[:, ::-1].copy()
        w[...] = w[::-1].copy()
    return evals, evecs


def diagonal(A, name="diagonal"):
    """The diagonal of a rank-2 tensor as a fresh rank-1 tensor."""
    out = zeros(name, [min(shape(A)[0], shape(A)[1])])
    la.diagonal(out, A)
    return out


def reshape(A, new_shape, name="reshape", row_major=True):
    """``A`` with a new shape, as a fresh tensor.

    ``row_major`` defaults to True because every reshape here replaces a numpy
    one, and numpy's default order is C. einsums tensors are column major, so
    the other walk is the natural one for native code - getting it wrong
    transposes blocks silently rather than raising, which is why the underlying
    ``linalg.reshape`` makes the argument mandatory.
    """
    out = zeros(name, list(new_shape))
    la.reshape(out, A, row_major)
    return out


def solve(A, B, name="solve"):
    """``A^-1 B`` as a fresh tensor, leaving ``A`` and ``B`` untouched."""
    Acopy = from_numpy("solve lhs", view(A))
    X = from_numpy(name, view(B))
    info = la.gesv(Acopy, X)
    if info != 0:
        raise RuntimeError(f"gesv failed with info={info}")
    return X


def vector_dot(A, B):
    """``sum_ij A_ij B_ij`` as a python float, psi4's ``Matrix::vector_dot``."""
    return float(la.dot(A, B))


def dot_into(out, A, B):
    """:func:`vector_dot` in pointer-writer form, so it can be captured.

    The returning form has to throw under capture - there is no value to return
    until the graph runs - so a captured phase that needs a scalar writes it
    into a length-1 tensor and reads it back after ``execute()``.
    """
    la.dot(out, A, B)
    return out


def rms(A):
    """Root-mean-square of a tensor's elements."""
    n = int(np.prod(shape(A)))
    if n == 0:
        return 0.0
    return float(np.sqrt(la.dot(A, A) / n))


def scale_columns(X, factors, name=None):
    """Scale column ``k`` of ``X`` by ``factors[k]``, in place. Returns ``X``.

    The in-place elementwise einsum is the aliasing carve-out the dispatcher
    documents: the aliased operand's index list is identical to the output's.
    """
    f = from_numpy("column factors", factors)
    einsums.einsum("ab <- ab ; b", X, X, f)
    return X


def pair_denominator(e, shift, name="denominator"):
    """``D[a,b] = shift - e[a] - e[b]``, the MP2/LMP2 energy denominator.

    ``shift`` is ``F_ii + F_jj`` for the pair.
    """
    n = shape(e)[0]
    D = zeros(name, [n, n])
    la.outer_sum(D, [e, e], [-1.0, -1.0])
    la.shift(shift, D)
    return D


def antisymmetrize(T, name="Tt"):
    """``2 T - T^T``, psi4's ``Tt_iajb_``."""
    out = zeros(name, shape(T))
    einsums.permute("ab <- ba", out, T)
    la.axpby(2.0, T, -1.0, out)
    return out


def canonicalizer(C, F, name="canonical"):
    """psi4's ``DLPNO::canonicalizer``.

    Diagonalizes ``C^T F C`` and returns ``(X, e)`` where the columns of ``X``
    rotate ``C``'s basis to one in which ``F`` is diagonal, eigenvalues
    descending.
    """
    Fmo = triplet(C, F, C, trans_a=True, name="C^T F C")
    e, X = eigh(Fmo, descending=True, name=name)
    return X, e


def orthocanonicalizer(S, F, s_cut, name="orthocanonical"):
    """psi4's ``DLPNO::orthocanonicalizer``.

    Returns ``(X, e)`` such that the columns of ``X`` are orthonormal with
    respect to ``S`` and diagonalize ``F``, with near-linear-dependencies
    (overlap eigenvalues below ``s_cut``) removed. ``X`` is ``nmo x nmo_final``
    with ``nmo_final <= nmo``.

    psi4 removes the linear dependencies with a partial Cholesky decomposition;
    this uses canonical (symmetric) orthogonalization instead. The two span the
    same space, and the subsequent Fock diagonalization fixes the basis up to
    sign, so both give identical energies whenever the same directions survive.
    """
    s, U = eigh(S, descending=True, name="overlap")
    keep = int(np.count_nonzero(view(s) > s_cut))
    if keep == 0:
        return zeros(name, [shape(S)[0], 0]), zeros(name + " energies", [0])

    X = scale_columns(
        from_numpy("orthogonalizer", view(U)[:, :keep]),
        view(s)[:keep] ** -0.5,
        name="orthogonalizer",
    )

    U2, e = canonicalizer(X, F, name=name)
    return doublet(X, U2, name=name), e
