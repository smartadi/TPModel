import functools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize


def snnn_decomp(U: np.ndarray):
    sumtol = 1e-5
    Usum = np.sum(U, axis=0)
    X = np.diag(Usum)
    for i in np.where(np.abs(Usum) < sumtol)[0]:
        X[i, i] = X[i, i] + 1
        if i == 0:
            X[i, i + 1] = -1
        else:
            X[i, i - 1] = -1

    if np.all(np.abs(np.sum(U @ U.T, axis=1) - 1) < sumtol):
        Usn = U @ X
    else:
        Usn = np.hstack([U, 1 - np.sum(U, axis=1, keepdims=True)])

    n = Usn.shape[1]
    umin = min(np.min(Usn), 0.0)
    umax = max(np.max(Usn), 2.0 / n)
    c1 = 1 / (1 - umin * n)
    c2 = 1 / (1 - umax * n)
    c = c2 if c1 < -c2 else c1
    d = (1 - c) / n
    M = d * np.ones((n, n)) + c * np.eye(n)
    W = Usn @ M
    V = None
    if U.ndim == 2:
        V = np.linalg.lstsq(W, U, rcond=None)[0]
    return W, V


def box_decomp(U: np.ndarray):
    M = np.vstack([np.min(U, axis=0), np.max(U, axis=0)])
    L = M[1, :] - M[0, :]
    n1, n2 = U.shape
    m = 2 ** n2
    B = np.zeros((m, n2))

    z = np.ones(n2, dtype=int)
    z[0] = 0

    def nexti(zv):
        zv = zv.copy()
        i = 0
        zv[i] += 1
        while zv[i] > 2:
            zv[i] = 1
            i += 1
            zv[i] += 1
        return zv

    for i in range(m):
        z = nexti(z)
        for j in range(n2):
            B[i, j] = M[z[j] - 1, j]

    W = np.zeros((n1, m))
    for j in range(n1):
        q = np.vstack([(M[1, :] - U[j, :]) / L, (U[j, :] - M[0, :]) / L])
        z = np.ones(n2, dtype=int)
        z[0] = 0
        for i in range(m):
            z = nexti(z)
            W[j, i] = np.prod([q[z[k] - 1, k] for k in range(n2)])
    return W, B


def rnodiff(U: np.ndarray, c: float):
    U0 = U[:, :-1]
    n, r = U0.shape
    s = np.sum(U0, axis=1, keepdims=True)
    U1 = U0 + (c - 1) / r * s * np.ones((1, r))
    U2 = U1 - np.min(U1, axis=0, keepdims=True)
    U3 = U2 / np.max(U2, axis=0, keepdims=True)
    U4 = U3 / np.max(np.sum(U3, axis=1))
    return 1 - np.min(np.sum(U4, axis=1)) - np.max(U4[:, 0])


def rnoino(U0: np.ndarray):
    U = U0.copy()
    n, r = U0.shape
    i = r - 1
    vet = U[:, i].copy()
    U = U[:, :i]
    U = U + vet.reshape(-1, 1) * np.ones((1, i)) / i
    U = U - vet.reshape(-1, 1) * np.ones((1, i)) / i
    U = np.hstack([U, 1 - np.sum(U, axis=1, keepdims=True)])

    a = [0.0, 1.0]
    while rnodiff(U, a[1]) < 0:
        a[1] *= 2
    d = [rnodiff(U, a[0]), rnodiff(U, a[1])]

    while abs(d[0]) + abs(d[1]) > 1e-6:
        auj = sum(a) / 2
        if rnodiff(U, auj) > 0:
            a[1] = auj
        else:
            a[0] = auj
        d = [rnodiff(U, a[0]), rnodiff(U, a[1])]

    auj = sum(a) / 2
    U00 = U[:, :-1]
    s = np.sum(U00, axis=1, keepdims=True)
    U1 = U00 + (auj - 1) / i * s * np.ones((1, i))
    U2 = U1 - np.min(U1, axis=0, keepdims=True)
    U25 = np.ones((n, 1)) * np.max(U2, axis=0, keepdims=True)
    U3 = U2 / U25
    U4 = U3 / np.max(np.sum(U3, axis=1))
    U = np.hstack([U4, 1 - np.sum(U4, axis=1, keepdims=True)])
    fi = np.linalg.inv(U[:r, :]) @ U0[:r, :]
    return U, fi


def polarorto(U1: np.ndarray, polarv: np.ndarray):
    n, r = U1.shape
    a = polarv
    v = np.zeros((r, r))
    for i in range(r - 2):
        v[:, i] = np.sin(a[:, i]) * np.prod(np.cos(a[:, :i]), axis=1)
    v[:, r - 2] = np.prod(np.cos(a), axis=1)
    v[:, r - 1] = 1 - np.sum(v[:, : r - 1], axis=1)

    v = v + np.ones((r, 1)) * (U1[0, :] - np.sum(v, axis=0) / r)
    eye_r = np.eye(r)
    for i in range(r):
        col_i = np.linalg.solve(v, eye_r[:, i])
        m = np.min(U1 @ col_i)
        v = v - m * (v - np.ones((r, 1)) * v[i, :])
        v[:, r - 1] = 1 - np.sum(v[:, : r - 1], axis=1)

    U2 = np.linalg.solve(v.T, U1.T).T
    return U2, v


def closeness(polarv, U1, h, hh):
    n, r = U1.shape
    a = polarv
    v = np.zeros((r, r))
    for i in range(r - 2):
        v[:, i] = np.sin(a[:, i]) * np.prod(np.cos(a[:, :i]), axis=1)
    v[:, r - 2] = np.prod(np.cos(a), axis=1)
    v[:, r - 1] = 1 - np.sum(v[:, : r - 1], axis=1)

    v = v + np.ones((r, 1)) * (U1[0, :] - np.sum(v, axis=0) / r)
    eye_r = np.eye(r)
    for i in range(r):
        # Only compute column i of U1 @ inv(v) — avoids full (n,r)@(r,r) matmul
        col_i = np.linalg.solve(v, eye_r[:, i])
        m = np.min(U1 @ col_i)
        v = v - m * (v - np.ones((r, 1)) * v[i, :])
        v[:, r - 1] = 1 - np.sum(v[:, : r - 1], axis=1)

    U2 = np.linalg.solve(v.T, U1.T).T
    norma = np.linalg.norm(np.log(np.max(U2, axis=0)), ord=h)
    vegek = np.log(np.sort(np.max(np.vstack([U2[0, :], U2[-1, :]]), axis=0)))
    norma = norma + hh * np.linalg.norm(vegek[r - 2 : r], ord=h)
    return norma


def _closeness_flat(x, r, U1, h, hh):
    """Picklable wrapper around closeness for multiprocessing."""
    return closeness(x.reshape(r, r - 2), U1, h, hh)


def _run_minimize_cno(args):
    """Single Nelder-Mead run — picklable for ProcessPoolExecutor."""
    x0_ravel, r, U1, h, hh = args
    fn = functools.partial(_closeness_flat, r=r, U1=U1, h=h, hh=hh)
    return minimize(fn, x0_ravel, method="Nelder-Mead")


def _run_perturb_cno(args):
    """Perturbation + local refinement run — picklable for ProcessPoolExecutor."""
    x0_ravel, r, U1, h, hh, nlokalis = args
    fn = functools.partial(_closeness_flat, r=r, U1=U1, h=h, hh=hh)
    res = minimize(fn, x0_ravel, method="Nelder-Mead")
    szamlalo = 0
    while szamlalo < nlokalis:
        res2 = minimize(fn, res.x, method="Nelder-Mead")
        if res2.fun < res.fun - 0.001:
            szamlalo = 0
            res = res2
        else:
            szamlalo += 1
    return res


def cno(U1: np.ndarray, h, hh, nveletlen, nzavar, nlokalis):
    n, r = U1.shape
    if r > n:
        raise ValueError("Input matrix must satisfy n >= r.")

    if r == 2:
        fi2 = U1[np.argmin(U1, axis=0), :]
        U2 = np.linalg.solve(fi2.T, U1.T).T
        return U2, fi2

    center = np.ones((n, 1)) * np.sum(U1[:, : r - 1] / n, axis=0)
    U11 = U1[:, : r - 1] - center
    inercia = U11.T @ U11
    eigvals, eigvecs = np.linalg.eigh(inercia)
    eigvals = np.maximum(eigvals, 1e-12)
    U111 = U11 @ eigvecs @ np.diag(eigvals ** -0.5)
    U1 = np.hstack([U111, 1 - np.sum(U111, axis=1, keepdims=True)])

    rng = np.random.default_rng()
    ertekmin = 1000
    polarmin = None

    # Pre-generate all random starts, then run them in parallel
    starts = []
    for _ in range(int(nveletlen)):
        veletlenhely = (rng.random((r, r - 2)) - 0.5) * np.pi
        veletlenhely[:, r - 3] *= 2
        starts.append((veletlenhely.ravel(), r, U1, h, hh))
    with ProcessPoolExecutor() as executor:
        for res in executor.map(_run_minimize_cno, starts):
            if res.fun < ertekmin:
                ertekmin = res.fun
                polarmin = res.x.reshape(r, r - 2)

    if polarmin is None:
        polarmin = np.zeros((r, r - 2))

    szamlalo = 0
    fn = functools.partial(_closeness_flat, r=r, U1=U1, h=h, hh=hh)
    while szamlalo < nlokalis:
        res = minimize(fn, polarmin.ravel(), method="Nelder-Mead")
        if res.fun < ertekmin - 0.001:
            szamlalo = 0
            ertekmin = res.fun
            polarmin = res.x.reshape(r, r - 2)
        else:
            szamlalo += 1

    # Pre-generate all perturbations, then run them (with local refinement) in parallel
    perturbs = []
    for _ in range(int(nzavar)):
        zavar = (rng.random((r, r - 2)) - 0.5) * np.pi
        zavar[:, r - 3] *= 2
        x0 = (polarmin + 0.1 * rng.random() * zavar).ravel()
        perturbs.append((x0, r, U1, h, hh, nlokalis))
    with ProcessPoolExecutor() as executor:
        for res in executor.map(_run_perturb_cno, perturbs):
            if res.fun < ertekmin:
                ertekmin = res.fun
                polarmin = res.x.reshape(r, r - 2)

    if r > 3:
        for i in range(r):
            for j in range(r - 3):
                if (polarmin[i, j] + np.pi / 2) % (2 * np.pi) > np.pi:
                    polarmin[i, j + 1] = polarmin[i, j + 1] + np.pi
                    polarmin[i, j] = np.pi - polarmin[i, j]
        polarmin[:, r - 3] = np.mod(polarmin[:, r - 3], 2 * np.pi)
    polarmin[:, : r - 2] = (polarmin[:, : r - 2] + np.pi / 2) % (2 * np.pi) - np.pi / 2
    polarmin = polarmin[np.lexsort(polarmin.T[::-1])]

    U2, fi2 = polarorto(U1, polarmin)
    fi2 = fi2[:, : r - 1] @ np.diag(np.sqrt(eigvals)) @ eigvecs.T + center[:r, :]
    fi2 = np.hstack([fi2, 1 - np.sum(fi2, axis=1, keepdims=True)])
    return U2, fi2


def decomp(U: np.ndarray, wtype: str):
    if wtype is None:
        wtype = "snnn"

    if wtype not in {"eye", "ortho"}:
        n1, n2 = U.shape
        Ushift = np.mean(U, axis=0)
        Uu, Su, Vt = np.linalg.svd(U - np.ones((n1, 1)) * Ushift, full_matrices=False)
        sv = Su
        ns = np.sum(sv > 1e-5)
        Uu = Uu[:, :ns]
        Vt = Vt[:ns, :]
        Su_mat = np.diag(Su[:ns])
        if ns < n2:
            U = Uu
    if wtype in {"snnn", "hull"}:
        W, V = snnn_decomp(U)
    elif wtype == "cno":
        W, V1 = snnn_decomp(U)
        W, V2 = cno(W, 1, 5, 50, 15, 10)
        V = V2 @ V1
    elif wtype == "irno":
        W, V1 = snnn_decomp(U)
        W, V2 = rnoino(W)
        V = V2 @ V1
    elif wtype == "eye":
        W = np.eye(U.shape[0])
        V = U
    elif wtype == "ortho":
        W = U
        V = np.eye(U.shape[1])
    elif wtype == "box":
        W, V = box_decomp(U)
    else:
        raise ValueError(f"Unknown wtype: {wtype}")

    if wtype not in {"eye", "ortho"} and 'Ushift' in locals():
        if ns < n2:
            V = V @ Su_mat @ Vt + np.ones((V.shape[0], 1)) * Ushift
    return W, V


def genhull(U, wtype="snnn"):
    if isinstance(U, list):
        return genhullall(U, [wtype] * len(U))
    W, _ = decomp(U, wtype)
    return W


def genhullall(U, types):
    Uco = []
    for Ui, t in zip(U, types):
        if t == "none" or Ui is None:
            Uco.append(None)
        else:
            Uco.append(genhull(Ui, t))
    return Uco
