"""State-space model utilities for TP model transformation examples."""

from __future__ import annotations

from itertools import product as iterproduct

import numpy as np

from .model import ConvexTPStructure
from .utils import gridvectors


def _normalize_density(density, ndim):
    if np.isscalar(density):
        return [int(density)] * ndim
    if len(density) != ndim:
        raise ValueError("density must be a scalar or match the number of dimensions")
    return [int(d) for d in density]


def build_grid(omega, density, center=None):
    """Return grid vectors for a centered hyper-rectangle.

    Args:
        omega: list/array of half-intervals per dimension.
        density: scalar or list with points per dimension.
        center: optional center per dimension (defaults to zeros).

    Returns:
        List of 1D numpy arrays, one per dimension.
    """
    omega = np.asarray(omega, dtype=float)
    ndim = omega.shape[0]
    center = np.zeros(ndim) if center is None else np.asarray(center, dtype=float)
    density = _normalize_density(density, ndim)

    omega_minmax = np.column_stack([center - omega, center + omega])
    grids = gridvectors(omega_minmax, np.asarray(density, dtype=int))
    return grids


def build_grid_tensor(M, Omega, O=None):
    """Build grid tensor and per-dimension grid vectors.

    Returns:
        grid_tensor: shape (M, M, ..., M, ndim)
        grids: list of 1D arrays
    """
    grids = build_grid(Omega, M, center=O)
    mesh = np.meshgrid(*grids, indexing="ij")
    grid_tensor = np.stack(mesh, axis=-1)
    return grid_tensor, grids


def linearise(func, Omega, M, O=None):
    """Numerically reconstruct linear mapping y ≈ S @ x using a grid.

    Args:
        func: callable, maps x -> y
        Omega: half-intervals for each input dimension
        M: grid density per dimension
        O: grid center (defaults to zeros)

    Returns:
        S: matrix R^{O x I}
        error: RMS error over grid
    """
    G, _ = build_grid_tensor(M, Omega, O)
    ndim = G.shape[-1]
    G_flat = G.reshape(-1, ndim)

    y0 = func(G_flat[0])
    out_dim = len(y0)
    F_flat = np.zeros((G_flat.shape[0], out_dim))
    for k, gk in enumerate(G_flat):
        F_flat[k] = func(gk)

    S = F_flat.T @ np.linalg.pinv(G_flat.T)
    residuals = F_flat - (S @ G_flat.T).T
    error = np.sqrt(np.mean(residuals**2))
    return S, error


def extract_lpv(
    func_xp,
    Omega_p,
    M_p,
    Omega_x,
    M_x,
    O_x=None,
    hull_type="snnn",
    tol=1e-11,
    O_p=None,
):
    """Reconstruct S(p) in TP model form from y = f(x, p).

    Returns:
        S_tensor: discretized S(p) tensor
        grids_p: list of parameter grids
        Uco: weighting function matrices per dimension
        Sco: core tensor after convexification
        sv: singular values per dimension
    """
    ndim_p = len(Omega_p)
    ndim_x = len(Omega_x)
    if O_x is None:
        O_x = np.zeros(ndim_x)
    if O_p is None:
        O_p = np.zeros(ndim_p)

    G_p, grids_p = build_grid_tensor(M_p, Omega_p, O_p)

    x0 = np.zeros(ndim_x)
    y0 = func_xp(x0, G_p.reshape(-1, ndim_p)[0])
    out_dim, in_dim = len(y0), ndim_x

    S_shape = (M_p,) * ndim_p + (out_dim, in_dim)
    S_tensor = np.zeros(S_shape)

    idx_list = list(iterproduct(range(M_p), repeat=ndim_p))
    for idx in idx_list:
        p_val = G_p[idx]

        def f_fixed(x):
            return func_xp(x, p_val)

        S_loc, _ = linearise(f_fixed, Omega_x, M_x, O_x)
        S_tensor[idx] = S_loc

    hulltypes = [hull_type] * ndim_p + ["none", "none"]
    Sco, Uco, sv, _, _ = ConvexTPStructure(S_tensor, hulltypes, tol)
    return S_tensor, grids_p, Uco, Sco, sv


def extract_qlpv(
    func,
    Omega_p,
    M_p,
    Omega_x,
    M_x,
    hull_type="snnn",
    tol=1e-11,
):
    """Reconstruct qLPV structure y = S(p)x where p is a subset of x.

    Returns:
        S_tensor, grids_p, Uco, Sco, sv
    """
    ndim_p = len(Omega_p)
    ndim_x = len(Omega_x)

    G_p, grids_p = build_grid_tensor(M_p, Omega_p, np.zeros(ndim_p))

    x0 = np.zeros(ndim_x)
    y0 = func(x0)
    out_dim, in_dim = len(y0), ndim_x

    S_shape = (M_p,) * ndim_p + (out_dim, in_dim)
    S_tensor = np.zeros(S_shape)

    idx_list = list(iterproduct(range(M_p), repeat=ndim_p))
    for idx in idx_list:
        p_val = G_p[idx]
        O_loc = np.zeros(ndim_x)
        O_loc[:ndim_p] = p_val

        S_loc, _ = linearise(func, Omega_x, M_x, O_loc)
        S_tensor[idx] = S_loc

    hulltypes = [hull_type] * ndim_p + ["none", "none"]
    Sco, Uco, sv, _, _ = ConvexTPStructure(S_tensor, hulltypes, tol)
    return S_tensor, grids_p, Uco, Sco, sv
