import numpy as np
from .hosvd import hosvd
from .hull import genhullall
from .tensor import tprod
from .utils import weighting_function


def ConvexTPStructure(F, hulltypes, tol, manual=0):
    dim = [0 if ht == "none" else 1 for ht in hulltypes]
    S, U, sv, _ = hosvd(F, dim, tol)

    if manual == 1:
        raise NotImplementedError("Manual truncation not supported in this port.")
    else:
        Uco = genhullall(U, hulltypes)

    PUco = [np.linalg.pinv(Ui) if Ui is not None else None for Ui in Uco]
    Sco = tprod(F, PUco)
    PU = [np.linalg.pinv(Ui) if Ui is not None else None for Ui in U]
    S_core = tprod(F, PU)
    return Sco, Uco, sv, S_core, U


def TPmodel(p, S, U, G):
    N = len(G)
    SS = S.shape
    blockdim = len(SS) - N
    w = weighting_function(np.asarray(p), U, G)
    ygross = tprod(S, w)

    if blockdim == 0:
        return ygross
    if blockdim == 1:
        axes = tuple(range(N))
        return np.squeeze(ygross, axis=axes)
    return np.reshape(ygross, SS[N:])
