"""
Cortical-dynamics-inspired discrete-time state-space system with image output.

Improvements over dynamic_image_system.py that make the output resemble
widefield calcium imaging of mouse cortex:

  1. A   -- distance-dependent inter-area network coupling on top of an
             oscillatory base, with three frequency bands:
             slow/delta  (~0.05-0.18 rad/step)
             theta/alpha (~0.35-0.80 rad/step)
             gamma       (~1.50-2.20 rad/step)
  2. Q   -- spatially correlated process noise so neighbouring areas
             co-fluctuate (Gaussian covariance in area-position space)
  3. C   -- bilateral symmetry (left/right hemisphere mirror) with wider,
             overlapping Gaussian blobs for a smooth cortical map
  4. B   -- sensory inputs localised to specific cortical areas with a
             Gaussian spatial spread to neighbouring areas
  5. simulate() -- AR(1)-coloured noise for a 1/f-like temporal power
                   spectrum; spatial colouring via Cholesky of Q

API is identical to dynamic_image_system.py.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _halton_seq(n: int, base: int = 2) -> np.ndarray:
    """First n elements of the Halton low-discrepancy sequence in [0, 1)."""
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


def _rotation_block(angle: float, decay: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return decay * np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------------------
# System construction
# ---------------------------------------------------------------------------

def build_system(
    n: int = 10,
    m: int = 5,
    decay: float = 0.97,
    noise_std: float = 0.05,
    coupling_strength: float = 0.12,
    coupling_range: float = 0.40,       # fraction of img_size
    noise_spatial_scale: float = 0.30,  # fraction of img_size
    noise_temporal_alpha: float = 0.65, # AR(1) coefficient  (0=white, <1)
    blob_sigma: float | None = None,
    seed: int | None = None,
):
    """Construct a cortex-like state-space system.

    Parameters
    ----------
    n : int
        Number of cortical areas (state dimension).
    m : int
        Number of input channels.
    decay : float
        Base spectral-radius contraction (< 1 for stability).
    noise_std : float
        Process noise amplitude.
    coupling_strength : float
        Strength of inter-area network coupling added to the oscillatory base A.
    coupling_range : float
        Spatial range of inter-area coupling (fraction of image width).
    noise_spatial_scale : float
        Spatial correlation length of process noise (fraction of image width).
    noise_temporal_alpha : float
        AR(1) temporal-correlation coefficient for coloured noise.
    blob_sigma : float, optional
        Gaussian blob width in pixels. Defaults to 16 % of image width.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys
        A, B, Q, C, img_size, blob_centres, noise_temporal_alpha
    """
    rng = np.random.default_rng(seed)
    img_size = 10 * n

    # ------------------------------------------------------------------
    # 1. Bilateral cortical area positions
    # ------------------------------------------------------------------
    half = n // 2
    margin = img_size * 0.12
    x_span = 0.5 * img_size - 2 * margin   # one hemisphere x-range

    x_left = margin + x_span * _halton_seq(max(half, 1), base=2)
    y_pos  = margin + (img_size - 2 * margin) * _halton_seq(max(half, 1), base=3)

    left_centres  = np.column_stack([x_left, y_pos])
    right_centres = np.column_stack([img_size - 1 - x_left, y_pos])  # mirror
    centres = np.vstack([left_centres, right_centres])

    if n % 2 == 1:                           # midline area for odd n
        centres = np.vstack([centres, [[img_size / 2, img_size / 2]]])

    # Pairwise Euclidean distances between area centres
    dist = np.linalg.norm(centres[:, None] - centres[None, :], axis=-1)  # (n,n)

    # ------------------------------------------------------------------
    # 2. A: multi-timescale oscillatory base + distance-dependent coupling
    # ------------------------------------------------------------------
    pairs = n // 2

    # Divide oscillator pairs across three frequency bands
    slow_pairs = max(1, pairs // 3)
    fast_pairs = max(1, pairs // 3) if pairs >= 2 else 0
    med_pairs  = pairs - slow_pairs - fast_pairs

    angle_parts = [np.linspace(0.05, 0.18, slow_pairs)]         # delta/slow
    decay_parts = [np.full(slow_pairs, decay)]

    if med_pairs > 0:
        angle_parts.append(np.linspace(0.35, 0.80, med_pairs))  # theta/alpha
        decay_parts.append(np.full(med_pairs, decay))

    if fast_pairs > 0:
        angle_parts.append(np.linspace(1.50, 2.20, fast_pairs)) # gamma
        decay_parts.append(np.full(fast_pairs, max(decay - 0.02, 0.92)))

    angles     = np.concatenate(angle_parts)
    decays_arr = np.concatenate(decay_parts)

    A_base = np.zeros((n, n))
    for i in range(pairs):
        A_base[2*i:2*i+2, 2*i:2*i+2] = _rotation_block(angles[i], decays_arr[i])
    if n % 2 == 1:
        A_base[-1, -1] = decay * 0.95

    # Row-normalised distance-dependent coupling (no self-coupling)
    coupling_sigma = coupling_range * img_size
    coupling_mat = np.exp(-dist**2 / (2 * coupling_sigma**2))
    np.fill_diagonal(coupling_mat, 0.0)
    row_sum = coupling_mat.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)
    coupling_mat /= row_sum

    A = A_base + coupling_strength * coupling_mat

    # Rescale so spectral radius < 1
    sr = np.max(np.abs(np.linalg.eigvals(A)))
    if sr >= 0.995:
        A *= 0.99 / sr

    # ------------------------------------------------------------------
    # 3. Q: spatially correlated process noise covariance
    # ------------------------------------------------------------------
    noise_sigma = noise_spatial_scale * img_size
    Q = noise_std**2 * np.exp(-dist**2 / (2 * noise_sigma**2))
    Q += 1e-8 * np.eye(n)      # ensure positive definiteness

    # ------------------------------------------------------------------
    # 4. C: bilateral, wider overlapping Gaussian blobs
    # ------------------------------------------------------------------
    if blob_sigma is None:
        blob_sigma = img_size * 0.16   # wider than original → smoother map

    px = np.arange(img_size)
    gx, gy = np.meshgrid(px, px, indexing="ij")
    C = np.zeros((img_size * img_size, n))
    for i in range(n):
        dx = gx - centres[i, 0]
        dy = gy - centres[i, 1]
        C[:, i] = np.exp(-(dx**2 + dy**2) / (2 * blob_sigma**2)).ravel()

    # ------------------------------------------------------------------
    # 5. B: inputs localised to target areas with Gaussian spatial spread
    # ------------------------------------------------------------------
    input_sigma   = coupling_range * 0.5 * img_size
    target_areas  = np.round(np.linspace(0, n - 1, m)).astype(int)
    B = np.zeros((n, m))
    for j, tgt in enumerate(target_areas):
        spread       = np.exp(-dist[tgt] ** 2 / (2 * input_sigma**2))
        spread      /= spread.max()
        B[:, j]      = 0.05 * spread   # weak neighbourhood drive
        B[tgt, j]    = 0.30            # strong direct drive

    return {
        "A": A,
        "B": B,
        "Q": Q,
        "C": C,
        "img_size": img_size,
        "blob_centres": centres,
        "noise_temporal_alpha": noise_temporal_alpha,
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    sys: dict,
    T: int = 200,
    x0: np.ndarray | None = None,
    u_func=None,
    seed: int = 0,
):
    """Run the cortical system forward for T steps.

    Noise is spatially coloured via Cholesky(Q) and temporally coloured via
    an AR(1) filter, giving a 1/f-like power spectrum.

    Parameters
    ----------
    sys    : dict  output of build_system
    T      : int   number of time steps
    x0     : (n,)  initial state (zeros if None)
    u_func : callable(k) -> (m,)  input signal (zero if None)
    seed   : int   RNG seed

    Returns
    -------
    xs   : (T+1, n)                   state trajectory
    imgs : (T+1, img_size, img_size)  image sequence
    """
    A, B, Q   = sys["A"], sys["B"], sys["Q"]
    C, img_size = sys["C"], sys["img_size"]
    alpha       = float(sys.get("noise_temporal_alpha", 0.0))
    n, m        = B.shape
    rng         = np.random.default_rng(seed)

    if x0 is None:
        x0 = np.zeros(n)
    if u_func is None:
        def u_func(_k):
            return np.zeros(m)

    noise_chol = np.linalg.cholesky(Q)

    # Pre-compute all inputs
    us_arr = np.array([u_func(k) for k in range(T)])   # (T, m)
    Bu     = us_arr @ B.T                               # (T, n)

    # AR(1)-coloured noise: z_k = alpha * z_{k-1} + sqrt(1 - alpha^2) * eps_k
    # Applied per state dimension (rows), then spatially coloured via noise_chol
    eps   = rng.standard_normal((n, T))
    scale = float(np.sqrt(max(1.0 - alpha**2, 1e-10)))
    z     = lfilter([scale], [1.0, -alpha], eps, axis=1)  # (n, T)
    W     = (noise_chol @ z).T                            # (T, n)

    xs    = np.zeros((T + 1, n))
    xs[0] = x0
    for k in range(T):
        xs[k + 1] = A @ xs[k] + Bu[k] + W[k]

    imgs = (C @ xs.T).T.reshape(T + 1, img_size, img_size)
    return xs, imgs
