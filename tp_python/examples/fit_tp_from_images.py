"""
TP model on 20 cortical pixels selected via Fibonacci spiral.

Pipeline
--------
1. Load SVD session (no full image reconstruction).
2. Trim first 5 minutes (35 Hz nominal rate).
3. Derive cortical mask from mean image percentile threshold.
4. Select 20 pixels via Fibonacci (sunflower) spiral growing from image centre,
   keeping only pixels inside the cortical mask.
5. Extract dF/F time series at those pixels via SVD projection.
6. Fit AR(1): y[k+1] = A y[k]  (A is 20×20, least squares).
7. Apply TP model transform (ConvexTPStructure) for hull types snnn, irno, box.
8. Save plots: point locations, weighting functions, vertex weights over time,
   and TP singular values.

Usage
-----
Set mn / td / en below, then run:
    python tp_python/examples/fit_tp_from_images.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tp_python.utils.widefield_loader import load_svd_session
from tp_python.tptoolbox.model import ConvexTPStructure

# =============================================================================
# USER PARAMETERS
# =============================================================================

mn = "AL_0046"
td = "2026-02-22"
en = "1"

trim_minutes = 5.0      # skip first N minutes of data
fs           = 35.0     # nominal sampling rate (Hz) — used only for logging

mask_pct     = 20       # percentile threshold: pixels above this are cortex

n_points     = 20       # number of cortical pixels to select
# r_step is computed automatically after loading (see below):
# r_step = r_spread * (min(img_H, img_W) / 2) / sqrt(n_points)
# This sets the outermost point at ~r_spread of the image half-size.
r_spread     = 0.65     # fraction of image half-size to reach (0 < r_spread < 1)

hull_types_to_try = ["snnn", "irno"]

# =============================================================================
# HELPERS
# =============================================================================

def fibonacci_cortical_points(
    cortical_mask: np.ndarray,
    n_points: int,
    r_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (ys, xs) of shape (n_points,) — cortical pixel coordinates selected
    by a Fibonacci/sunflower spiral growing outward from the image centre.

    Point k is placed at:
        rho   = r_step * sqrt(k + 1)          equal-area annuli
        theta = k * golden_angle              golden angle ≈ 2.400 rad

    Only pixels that are (1) in bounds, (2) inside cortical_mask, and
    (3) not already selected (deduplication) are accepted.
    r_step is halved up to max_retries times if fewer than n_points valid
    points are found within the image; halving is used here because r_step
    is already set to span most of the image — if that fails (e.g. tight mask)
    the points need to be moved inward, not outward.
    """
    img_H, img_W = cortical_mask.shape
    cy, cx       = img_H // 2, img_W // 2
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ≈ 2.3999 rad

    r_step      = float(r_step)
    max_retries = 20

    for attempt in range(max_retries):
        ys_found: list[int] = []
        xs_found: list[int] = []
        seen: set[tuple[int, int]] = set()
        budget = n_points * 10 * (attempt + 1)

        k = 0
        while len(ys_found) < n_points and k < budget:
            rho   = r_step * np.sqrt(k + 1)
            theta = k * golden_angle
            py    = int(round(cy + rho * np.sin(theta)))
            px    = int(round(cx + rho * np.cos(theta)))

            if (0 <= py < img_H
                    and 0 <= px < img_W
                    and cortical_mask[py, px]
                    and (py, px) not in seen):
                ys_found.append(py)
                xs_found.append(px)
                seen.add((py, px))
            k += 1

        if len(ys_found) == n_points:
            return np.array(ys_found, dtype=int), np.array(xs_found, dtype=int)

        # Too few valid cortical pixels in the spread — shrink inward
        r_step *= 0.75

    raise RuntimeError(
        f"Could not find {n_points} cortical points after {max_retries} retries. "
        f"Last r_step={r_step:.1f}. Try lowering mask_pct or n_points."
    )


# =============================================================================
# LOAD SESSION
# =============================================================================

print(f"Loading session {mn} / {td} / {en} ...")
session    = load_svd_session(mn=mn, td=td, en=en)

spatial    = session["svd_spatial"]    # (Y, X, C)
temporal   = session["svd_temporal"]  # (T_full, C)
timestamps = session["timestamps"]    # (T_full,)  seconds
mean_img   = session["mean_image"]    # (Y, X)

img_H, img_W, C = spatial.shape
T_full = temporal.shape[0]

print(f"Image: {img_H}×{img_W}   SVD comps: {C}   Total frames: {T_full}")
print(f"Session range: {timestamps[0]/60:.2f} – {timestamps[-1]/60:.2f} min")

# =============================================================================
# TRIM FIRST 5 MINUTES
# =============================================================================

trim_secs = trim_minutes * 60.0
trim_idx  = int(np.searchsorted(timestamps, timestamps[0] + trim_secs))

if trim_idx >= T_full:
    raise RuntimeError(
        f"Session too short to trim {trim_minutes} min: only "
        f"{T_full / fs / 60:.1f} min available ({T_full} frames at {fs} Hz)."
    )

temporal   = temporal[trim_idx:]     # (T, C)
timestamps = timestamps[trim_idx:]   # (T,)
T          = temporal.shape[0]
t_axis     = timestamps - timestamps[0]   # seconds from trim start

print(f"After trim: {trim_idx} frames dropped  |  {T} frames remain "
      f"({T / fs / 60:.1f} min from {timestamps[0]/60:.2f} min)")

# =============================================================================
# CORTICAL MASK
# =============================================================================

thresh        = np.percentile(mean_img, mask_pct)
cortical_mask = mean_img > thresh          # (Y, X) bool
n_cortical    = int(cortical_mask.sum())

print(f"Cortical mask: {n_cortical} pixels above {mask_pct}th percentile "
      f"(thresh={thresh:.4f})")

if n_cortical < n_points:
    raise RuntimeError(
        f"Only {n_cortical} pixels pass cortical mask (mask_pct={mask_pct}). "
        "Lower mask_pct."
    )

# =============================================================================
# SELECT 20 CORTICAL PIXELS VIA FIBONACCI SPIRAL
# =============================================================================

# r_step scaled so the outermost point sits at r_spread × (half image size).
# With rho_k = r_step * sqrt(k+1) and k_max = n_points-1:
#   r_step = r_spread * (min(img_H, img_W) / 2) / sqrt(n_points)
r_step = r_spread * (min(img_H, img_W) / 2.0) / np.sqrt(n_points)
print(f"Spiral r_step={r_step:.1f} px  "
      f"(outermost point ≈ {r_step * np.sqrt(n_points):.0f} px from centre)")

ys, xs = fibonacci_cortical_points(cortical_mask, n_points, r_step)
P      = len(ys)   # == n_points

cy_img, cx_img = img_H // 2, img_W // 2
radii = np.sqrt((ys - cy_img) ** 2 + (xs - cx_img) ** 2)

print(f"Selected {P} pixels  |  "
      f"radii: {radii.min():.1f}–{radii.max():.1f} px from centre")
for i in range(P):
    print(f"  pt {i:2d}: y={ys[i]:4d}  x={xs[i]:4d}  r={radii[i]:6.1f} px")

# =============================================================================
# PREVIEW — show selected pixels; confirm before processing
# =============================================================================

fig_pre, ax_pre = plt.subplots(figsize=(6, 6), constrained_layout=True)
vmax_pre = float(np.percentile(mean_img, 99))
ax_pre.imshow(mean_img, cmap="gray", vmin=0, vmax=vmax_pre)
ax_pre.contour(cortical_mask.astype(float), levels=[0.5],
               colors="cyan", linewidths=0.8, alpha=0.7)
sc_pre = ax_pre.scatter(xs, ys,
                        c=np.arange(P), cmap="plasma",
                        s=60, zorder=5,
                        edgecolors="white", linewidths=0.6)
plt.colorbar(sc_pre, ax=ax_pre, label="point order  (0 = nearest centre)")
for i in range(P):
    ax_pre.text(xs[i] + 2, ys[i] - 2, str(i), fontsize=7,
                color="white", zorder=6)
ax_pre.set_title(f"Point preview — {mn} {td} exp {en}  "
                 f"(close window to continue)", fontsize=10)
ax_pre.axis("off")
plt.show(block=True)

try:
    ans = input("Continue with these points? [y]/n: ").strip().lower()
except EOFError:
    ans = "y"   # non-interactive execution (e.g. batch script)
if ans == "n":
    print("Aborted. Adjust r_spread or mask_pct and re-run.")
    raise SystemExit(0)

# =============================================================================
# EXTRACT dF/F PIXEL TIME SERIES VIA SVD PROJECTION
# =============================================================================

spatial_sel = spatial[ys, xs, :]                           # (P, C)
mean_sel    = mean_img[ys, xs]                             # (P,)

# Y_raw[t, p] = pixel p at time t (raw F)
Y_raw = (spatial_sel @ temporal.T).T + mean_sel[None, :]   # (T, P)

# dF/F = (F - F_mean) / F_mean
F_mean = Y_raw.mean(axis=0)                                # (P,)

bad = np.where(F_mean <= 0)[0]
if len(bad) > 0:
    print(f"WARNING: {len(bad)} selected pixel(s) have non-positive mean F "
          f"(indices {bad.tolist()}). Consider raising mask_pct.")

Y = (Y_raw - F_mean) / np.where(F_mean > 0, F_mean, 1.0)  # (T, P)

print(f"dF/F matrix Y: {Y.shape}  (T × P)  "
      f"|Y|_max={np.abs(Y).max():.4f}")

# =============================================================================
# FIT AR(1):  y[k+1] ≈ A y[k]
# =============================================================================

Y_prev = Y[:-1]   # (T-1, P)
Y_next = Y[1:]    # (T-1, P)

A_lstsq, _, _, _ = np.linalg.lstsq(Y_prev, Y_next, rcond=None)
A = A_lstsq.T     # (P, P): convention y[k+1] = A @ y[k]

fit_err  = np.sqrt(np.mean((Y_next - Y_prev @ A.T) ** 2))
spec_rad = np.max(np.abs(np.linalg.eigvals(A)))
print(f"\nAR(1) fit RMS error: {fit_err:.6f}   spectral radius: {spec_rad:.4f}")
if spec_rad > 1.0:
    print("  NOTE: spectral radius > 1 (estimated unstable); "
          "normal for windowed real data.")

# =============================================================================
# TP MODEL TRANSFORM  — loop over hull types
# =============================================================================
# A is (P, P) = (20, 20) treated as a 2D tensor.
# Both dimensions are active (output pixel × input pixel), same as
# fit_tp_whole_image.py.

out_dir = PROJECT_ROOT / "tp_python" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
tag = f"{mn}_{td}_{en}_fromimages_20pts"

results: dict[str, tuple] = {}   # ht -> (Sco, Uco, sv)

for ht in hull_types_to_try:
    hulltypes = [ht, ht]
    print(f"\n--- Hull type: {ht} ---")
    try:
        Sco, Uco, sv, _, _ = ConvexTPStructure(A, hulltypes, tol=1e-8)
        results[ht] = (Sco, Uco, sv)
        print(f"  Core shape: {Sco.shape}")
        for dim_idx, (w, s) in enumerate(zip(Uco, sv)):
            if w is None:
                continue
            print(f"  dim {dim_idx+1}: {w.shape[1]} vertices  "
                  f"top-5 sv = {s[:5].round(4)}")
    except Exception as exc:
        print(f"  FAILED: {exc}")

if not results:
    print("All hull types failed — no plots to generate.")
    raise SystemExit(1)

# =============================================================================
# PLOT 1 — Point Locations (save the preview figure that was already shown)
# =============================================================================

ax_pre.set_title(f"Spiral cortical points — {mn} {td} exp {en}", fontsize=10)
path_pts = out_dir / f"tp_point_locations_{tag}.png"
fig_pre.savefig(path_pts, dpi=150)
print(f"\nSaved point locations → {path_pts}")

# =============================================================================
# PLOT 2 — Weighting Functions  (one figure per hull type)
# =============================================================================

pt_index = np.arange(P)   # x-axis: point index (0 = nearest centre)

for ht, (Sco, Uco, sv) in results.items():
    active = [(i, Uco[i]) for i in range(len(Uco)) if Uco[i] is not None]
    if not active:
        continue

    fig_w, axes_w = plt.subplots(1, len(active),
                                  figsize=(5 * len(active), 4),
                                  constrained_layout=True)
    if len(active) == 1:
        axes_w = [axes_w]

    fig_w.suptitle(
        f"Weighting Functions [{ht}] — {mn} {td} exp {en}", fontsize=11)

    for ax, (dim_idx, W) in zip(axes_w, active):
        for r_idx in range(W.shape[1]):
            ax.plot(pt_index, W[:, r_idx], linewidth=1.8,
                    label=f"w{r_idx+1}")
        ax.set_title(f"dim {dim_idx+1}  "
                     f"({'output' if dim_idx == 0 else 'input'} pixels)")
        ax.set_xlabel("point index  (0 = nearest centre)")
        ax.set_ylabel("weight")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    path_w = out_dir / f"tp_weighting_{tag}_{ht}.png"
    fig_w.savefig(path_w, dpi=150)
    print(f"Saved weighting [{ht}] → {path_w}")

# =============================================================================
# PLOT 3 — Vertex Weights: 3 random 5-min windows, both hull types overlaid
# =============================================================================
# Layout: 2 rows (dim 1 output / dim 2 input) × 3 columns (windows).
# snnn = solid lines, irno = dashed lines, matched colours across hull types.
# Windows are drawn one-per-third of the session so they span the recording.

# --- pre-compute full-session activations for each hull type -----------------
activations: dict[str, dict] = {}   # ht -> {"act_r": (T,r0), "act_s": (T,r1)}
for ht, (Sco, Uco, sv) in results.items():
    W0, W1 = Uco[0], Uco[1]
    if W0 is None or W1 is None:
        continue
    alpha   = Y @ W0                                          # (T, r0)
    beta    = Y @ W1                                          # (T, r1)
    contrib = np.einsum("rs,tr,ts->rst", Sco, alpha, beta)   # (r0, r1, T)
    activations[ht] = {
        "act_r": contrib.sum(axis=1).T,   # (T, r0)
        "act_s": contrib.sum(axis=0).T,   # (T, r1)
        "r0": W0.shape[1],
        "r1": W1.shape[1],
    }

# --- pick 3 random windows, one from each third of the session ---------------
window_sec    = 5.0 * 60.0                    # 300 s
window_frames = int(round(window_sec * fs))   # frames per window
window_frames = min(window_frames, T)         # safety: session shorter than 5 min

rng   = np.random.default_rng(0)
third = T // 3
# start index drawn uniformly from each third, clamped so window fits
win_starts = [
    int(rng.integers(0,              max(1, third - window_frames))),
    int(rng.integers(third,          max(third + 1, 2 * third - window_frames))),
    int(rng.integers(2 * third,      max(2 * third + 1, T - window_frames))),
]
win_labels = [
    f"t={t_axis[s]/60:.1f}–{t_axis[min(s+window_frames-1, T-1)]/60:.1f} min"
    for s in win_starts
]

# style: solid for snnn, dashed for irno
ht_style = {"snnn": "-", "irno": "--"}
# use a fixed colour cycle so vertex i has the same colour in both hull types
cmap_verts = plt.cm.tab10

# --- one figure per dimension (dim1 output, dim2 input) ----------------------
for dim_key, act_key, dim_label in [
    ("r0", "act_r", "dim 1 — output pixels"),
    ("r1", "act_s", "dim 2 — input pixels"),
]:
    # max vertices across hull types for this dimension
    max_verts = max(activations[ht][dim_key] for ht in activations)

    fig_w3, axes_w3 = plt.subplots(
        1, 3, figsize=(18, 4), sharey=True, constrained_layout=True)
    fig_w3.suptitle(
        f"Vertex weights — {dim_label}  ({mn} {td} exp {en})\n"
        f"solid = snnn  |  dashed = irno",
        fontsize=11,
    )

    for col, (s0, wlabel) in enumerate(zip(win_starts, win_labels)):
        s1  = min(s0 + window_frames, T)
        t_w = t_axis[s0:s1] - t_axis[s0]   # seconds from window start
        ax  = axes_w3[col]

        for ht in activations:
            act = activations[ht][act_key][s0:s1]   # (window, r)
            ls  = ht_style.get(ht, "-")
            for v in range(act.shape[1]):
                color = cmap_verts(v % 10)
                label = f"{ht} v{v+1}" if col == 0 else None
                ax.plot(t_w, act[:, v],
                        linestyle=ls, linewidth=0.9,
                        color=color, label=label, alpha=0.85)

        ax.set_title(wlabel, fontsize=9)
        ax.set_xlabel("time in window (s)")
        ax.grid(True, alpha=0.2)
        if col == 0:
            ax.set_ylabel("vertex activation")
            ax.legend(fontsize=7, ncol=max(1, max_verts), loc="upper right")

    dim_tag = dim_key[1]   # "0" or "1"
    path_w3 = out_dir / f"tp_vertex_weights_windows_{tag}_dim{dim_tag}.png"
    fig_w3.savefig(path_w3, dpi=150)
    print(f"Saved vertex weights windows dim{dim_tag} → {path_w3}")

# =============================================================================
# PLOT 4 — TP Singular Values  (all hull types overlaid)
# =============================================================================

fig_sv, axes_sv = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
fig_sv.suptitle(f"TP Singular Values — {mn} {td} exp {en}", fontsize=11)
dim_labels = ["dim 1 (output pixels)", "dim 2 (input pixels)"]

for ax, dim_idx, dlabel in zip(axes_sv, [0, 1], dim_labels):
    for ht, (_, Uco, sv) in results.items():
        if sv[dim_idx] is not None:
            ax.semilogy(sv[dim_idx], "o-", linewidth=2, label=ht)
    ax.set_title(dlabel)
    ax.set_xlabel("index")
    ax.set_ylabel("singular value (log)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

path_sv = out_dir / f"tp_singular_values_{tag}_all.png"
fig_sv.savefig(path_sv, dpi=150)
print(f"Saved TP singular values → {path_sv}")

plt.show()
print("\nDone.")
