import numpy as np
from .tensor import tprod, layout


def svdtrunc(A: np.ndarray, tol: float = np.finfo(float).eps):
    """Truncated SVD returning left singular vectors and kept singular values."""
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    keep = np.sum(S > tol)
    if keep < len(S):
        tol_next = S[keep]
        S = S[:keep]
        U = U[:, :keep]
    else:
        tol_next = 0.0
    return U, S, tol_next


def hosvd(T: np.ndarray, dim=None, tol=None):
    """High Order SVD."""
    ndim = T.ndim
    if dim is None:
        dim = [1] * ndim
    if tol is None:
        tol = [np.finfo(float).eps] * ndim
    if np.isscalar(tol):
        tol = [tol] * ndim

    U = [None] * ndim
    UT = [None] * ndim
    sv = [None] * ndim
    tol_out = list(tol)

    for i in range(ndim):
        if dim[i]:
            A = layout(T, i)
            Ui, svi, toli = svdtrunc(A, tol[i])
            U[i] = Ui
            UT[i] = Ui.T
            sv[i] = svi
            tol_out[i] = toli
        else:
            U[i] = None
            UT[i] = None
            sv[i] = None
            tol_out[i] = 0.0

    S = tprod(T, UT)
    return S, U, sv, tol_out
