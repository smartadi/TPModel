"""
Discrete-time dynamical system with high-dimensional image output.

State-space form:
    x[k+1] = A x[k] + B u[k] + w[k],   w[k] ~ N(0, Q)
    y[k]   = C x[k]

The output y is a flattened (10n x 10n) image reconstructed from
Gaussian basis functions placed across the image grid.  Each state
dimension modulates the amplitude of one basis blob, so the image
is a real-time visualisation of the internal state.
"""

from __future__ import annotations

import numpy as np


def _rotation_block(angle: float, decay: float) -> np.ndarray:
    """2x2 discrete rotation with contraction."""
    c, s = np.cos(angle), np.sin(angle)
    return decay * np.array([[c, -s], [s, c]])


def build_system(
    n: int = 4,
    m: int = 2,
    angles: np.ndarray | None = None,
    decay: float = 0.98,
    noise_std: float = 0.05,
    blob_sigma: float | None = None,
    seed: int | None = None,
):
    """Construct the system matrices and image projection operator.

    Parameters
    ----------
    n : int
        State dimension.
    m : int
        Input dimension.
    angles : array-like, optional
        Rotation angles for each 2-block in A.  Length = n // 2.
        Remaining odd dimension (if n is odd) gets a scalar decay entry.
        Defaults to evenly spaced angles in [0.3, 1.2].
    decay : float
        Spectral-radius contraction factor (< 1 for stability).
    noise_std : float
        Standard deviation of process noise  (Q = noise_std^2 * I).
    blob_sigma : float, optional
        Width of each Gaussian basis blob in pixel units.
        Defaults to ``n * 1.2``.
    seed : int, optional
        RNG seed for reproducibility of B and blob centres.

    Returns
    -------
    sys : dict with keys
        A        (n, n)         state matrix
        B        (n, m)         input matrix
        Q        (n, n)         process noise covariance
        C        (img_dim, n)   observation / projection matrix  (img_dim = (10n)^2)
        img_size (int)          side length of the square image = 10*n
        blob_centres (n, 2)     pixel centres of each basis blob
    """
    rng = np.random.default_rng(seed)
    img_size = 10 * n

    # --- A: block-diagonal rotations ---
    if angles is None:
        angles = np.linspace(0.3, 1.2, n // 2)
    A = np.zeros((n, n))
    for i in range(n // 2):
        A[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = _rotation_block(angles[i], decay)
    if n % 2 == 1:
        A[-1, -1] = decay * 0.95  # stable scalar mode for odd dimension

    # --- B: random but normalised input coupling ---
    B = rng.standard_normal((n, m))
    B /= np.linalg.norm(B, axis=0, keepdims=True)
    B *= 0.3

    # --- Q: isotropic process noise ---
    Q = (noise_std ** 2) * np.eye(n)

    # --- C: Gaussian-blob projection to (10n x 10n) image ---
    if blob_sigma is None:
        blob_sigma = n * 1.2

    # place blob centres on a grid or circle depending on n
    centres = np.zeros((n, 2))
    margin = img_size * 0.15
    if n <= 6:
        # arrange on a circle
        for i in range(n):
            theta = 2 * np.pi * i / n
            centres[i] = [
                img_size / 2 + (img_size / 2 - margin) * np.cos(theta),
                img_size / 2 + (img_size / 2 - margin) * np.sin(theta),
            ]
    else:
        # quasi-random Halton-like placement
        centres[:, 0] = margin + (img_size - 2 * margin) * _halton_seq(n, base=2)
        centres[:, 1] = margin + (img_size - 2 * margin) * _halton_seq(n, base=3)

    # build C  (each column is a flattened Gaussian blob)
    px = np.arange(img_size)
    gx, gy = np.meshgrid(px, px, indexing="ij")
    C = np.zeros((img_size * img_size, n))
    for i in range(n):
        dx = gx - centres[i, 0]
        dy = gy - centres[i, 1]
        blob = np.exp(-(dx ** 2 + dy ** 2) / (2 * blob_sigma ** 2))
        C[:, i] = blob.ravel()

    return {
        "A": A,
        "B": B,
        "Q": Q,
        "C": C,
        "img_size": img_size,
        "blob_centres": centres,
    }


def _halton_seq(n: int, base: int = 2) -> np.ndarray:
    """First *n* elements of the Halton sequence in [0, 1)."""
    out = np.zeros(n)
    for i in range(n):
        f, r = 1.0, 0.0
        idx = i + 1
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        out[i] = r
    return out


def simulate(
    sys: dict,
    T: int = 200,
    x0: np.ndarray | None = None,
    u_func=None,
    seed: int = 0,
):
    """Run the system forward for T steps.

    Parameters
    ----------
    sys : dict
        Output of ``build_system``.
    T : int
        Number of time steps.
    x0 : array, optional
        Initial state.  Defaults to zeros.
    u_func : callable(k) -> (m,), optional
        Input signal as a function of time step.
        Defaults to zero input.
    seed : int
        RNG seed for process noise realisation.

    Returns
    -------
    xs   : (T+1, n)   state trajectory
    imgs : (T+1, img_size, img_size)  image sequence
    """
    A, B, Q = sys["A"], sys["B"], sys["Q"]
    C, img_size = sys["C"], sys["img_size"]
    n, m = B.shape
    rng = np.random.default_rng(seed)

    if x0 is None:
        x0 = np.zeros(n)
    if u_func is None:
        def u_func(_k):
            return np.zeros(m)

    noise_chol = np.linalg.cholesky(Q)

    xs = np.zeros((T + 1, n))
    xs[0] = x0

    for k in range(T):
        w = noise_chol @ rng.standard_normal(n)
        xs[k + 1] = A @ xs[k] + B @ u_func(k) + w

    imgs = (C @ xs.T).T.reshape(T + 1, img_size, img_size)
    return xs, imgs
