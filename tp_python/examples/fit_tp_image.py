"""
Fit a TP model to widefield image time series.

Pipeline
--------
1. Load widefield SVD session (no full image reconstruction).
2. Select an N×N pixel grid centred on the image.
3. Extract pixel time-series at those locations via SVD projection.
4. Fit AR(1): y[k+1] ≈ A y[k]  (least squares).
5. Reshape A → (N, N, N, N) tensor (output-pixel × input-pixel).
6. Apply TP model transform (HOSVD + convex hull) over the spatial dims.
7. Save plots of weighting functions and singular values.

Usage
-----
Set  mn / td / en / N  below, then run:
    python tp_python/examples/fit_tp_image.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tp_python.utils.widefield_loader import load_svd_session
from tp_python.tptoolbox.model import ConvexTPStructure

# =============================================================================
# USER PARAMETERS  — fill these in before running
# =============================================================================

mn = "ZYE_0043"      # mouse name
td = "2021-09-13"    # date
en = "1"             # experiment number

N = 5                # grid side length  →  N×N pixels extracted
spacing = 10         # pixels between neighbouring grid points

# Time window: centre of session ± half_window_min minutes.
half_window_min = 5.0   # → 10 min total

# Hull types to try for the two spatial (output) dimensions.
# snnn : sum=1, non-negative vertices (standard TP)
# cno  : close-to-normal-form optimal (tightest convex hull)
# irno : inner relaxed normal form
# box  : axis-aligned bounding box
hull_types_to_try = ["snnn", "irno", "box"]

# =============================================================================
# LOAD SESSION
# =============================================================================

print(f"Loading session {mn} / {td} / {en} ...")
session = load_svd_session(mn=mn, td=td, en=en)

spatial    = session["svd_spatial"]   # (Y, X, C)
temporal   = session["svd_temporal"]  # (T, C)
timestamps = session["timestamps"]    # (T,)
mean_img   = session["mean_image"]    # (Y, X)

# Select 10-minute window centred on the session midpoint.
t_mid = (timestamps[0] + timestamps[-1]) / 2.0
hw    = half_window_min * 60.0          # seconds
mask  = (timestamps >= t_mid - hw) & (timestamps <= t_mid + hw)
temporal   = temporal[mask]
timestamps = timestamps[mask]

T, C = temporal.shape
img_H, img_W = mean_img.shape
print(f"Image size: {img_H} × {img_W}   |   SVD comps: {C}")
print(f"Session range: {timestamps[0]/60:.2f} – {session['timestamps'][-1]/60:.2f} min")
print(f"Window: {timestamps[0]/60:.2f} – {timestamps[-1]/60:.2f} min  ({T} frames)")

# =============================================================================
# SELECT N×N GRID AROUND IMAGE CENTRE
# =============================================================================

cy, cx = img_H // 2, img_W // 2

# Fixed spacing between neighbours: total span = (N-1) * spacing
half_span = (N - 1) * spacing // 2

ys = np.arange(N, dtype=int) * spacing + (cy - half_span)
xs = np.arange(N, dtype=int) * spacing + (cx - half_span)

# Clamp to valid pixel range.
ys = np.clip(ys, 0, img_H - 1)
xs = np.clip(xs, 0, img_W - 1)

grid_ys, grid_xs = np.meshgrid(ys, xs, indexing="ij")  # (N, N)
P = N * N  # number of selected pixels

print(f"Grid centre: ({cy}, {cx})   |   y range: [{ys[0]}, {ys[-1]}]   "
      f"x range: [{xs[0]}, {xs[-1]}]   |   P = {P} pixels")

# =============================================================================
# EXTRACT PIXEL TIME SERIES  (vectorised SVD projection)
# =============================================================================
# spatial at selected pixels:  (P, C)
spatial_sel = spatial[grid_ys.ravel(), grid_xs.ravel(), :]   # (P, C)
mean_sel    = mean_img[grid_ys.ravel(), grid_xs.ravel()]      # (P,)

# Y[t, p] = pixel p at time t  (raw F)
Y = (spatial_sel @ temporal.T).T + mean_sel[None, :]          # (T, P)

# Convert to dF/F = (F - F_mean) / F_mean
F_mean = Y.mean(axis=0, keepdims=True)
Y = (Y - F_mean) / F_mean

print(f"Pixel time-series matrix Y: {Y.shape}  (T × P)")

# =============================================================================
# FIT AR(1) MODEL:  y[k+1] ≈ A y[k]
# =============================================================================
# Least-squares: A = Y_next.T @ Y_prev.T  ×  pinv(Y_prev @ Y_prev.T)
#   equivalently solve  Y_prev.T @ A.T = Y_next.T

Y_prev = Y[:-1]   # (T-1, P)
Y_next = Y[1:]    # (T-1, P)

# A is (P, P)
A, residuals, rank, _ = np.linalg.lstsq(Y_prev, Y_next, rcond=None)
A = A.T  # convention: y[k+1] = A @ y[k]

fit_err = np.sqrt(np.mean((Y_next - Y_prev @ A.T) ** 2))
print(f"AR(1) fit RMS error: {fit_err:.6f}")
print(f"A shape: {A.shape}   spectral radius: "
      f"{np.max(np.abs(np.linalg.eigvals(A))):.4f}")

# =============================================================================
# TP MODEL TRANSFORM  — loop over hull types
# =============================================================================
# Reshape A (P×P) → (N, N, N, N) tensor.
# dims 0,1 = output pixel (y, x); dims 2,3 = input pixel (y, x).
# Hull applied to dims 0 and 1; dims 2,3 left as "none".

A_tensor = A.reshape(N, N, N, N)

out_dir = PROJECT_ROOT / "tp_python" / "plots" / "mouse"
out_dir.mkdir(parents=True, exist_ok=True)
tag = f"{mn}_{td}_{en}_N{N}"

results = {}   # hull_type -> (Sco, Uco, sv)

for ht in hull_types_to_try:
    hulltypes = [ht, ht, "none", "none"]
    print(f"\n--- Hull type: {ht} ---")
    try:
        Sco, Uco, sv, _, _ = ConvexTPStructure(A_tensor, hulltypes, tol=1e-8)
        results[ht] = (Sco, Uco, sv)
        print(f"  Core tensor: {Sco.shape}")
        for dim_idx, (w, s) in enumerate(zip(Uco, sv)):
            if w is None:
                continue
            print(f"  dim {dim_idx+1}: {w.shape[1]} weighting fns, "
                  f"top-5 sv = {s[:5].round(4)}")
    except Exception as exc:
        print(f"  FAILED: {exc}")

# =============================================================================
# PLOTS
# =============================================================================

# --- Weighting functions: one figure per hull type ---
for ht, (Sco, Uco, sv) in results.items():
    active_dims = [(i, Uco[i], sv[i]) for i in range(len(Uco)) if Uco[i] is not None]
    if not active_dims:
        continue

    fig_w, axes_w = plt.subplots(1, len(active_dims),
                                 figsize=(5 * len(active_dims), 4),
                                 constrained_layout=True)
    if len(active_dims) == 1:
        axes_w = [axes_w]
    fig_w.suptitle(
        f"Weighting Functions [{ht}] — {mn} {td} exp {en}  N={N}", fontsize=11)

    for ax, (dim_idx, W, _) in zip(axes_w, active_dims):
        grid_label = ys if dim_idx == 0 else xs
        for r in range(W.shape[1]):
            ax.plot(grid_label, W[:, r], linewidth=2, label=f"w{r+1}")
        ax.set_title(f"dim {dim_idx+1} ({'y' if dim_idx == 0 else 'x'} pixels)")
        ax.set_xlabel("pixel index")
        ax.set_ylabel("weight")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    path_w = out_dir / f"tp_weighting_{tag}_{ht}.png"
    fig_w.savefig(path_w, dpi=150)
    print(f"Saved weighting [{ht}] → {path_w}")

# --- Singular values: all hull types overlaid, one subplot per spatial dim ---
# Determine active dim indices from first result
first_result = next(iter(results.values()))
active_dim_ids = [i for i, w in enumerate(first_result[1]) if w is not None]

if active_dim_ids:
    fig_sv, axes_sv = plt.subplots(1, len(active_dim_ids),
                                   figsize=(5 * len(active_dim_ids), 4),
                                   constrained_layout=True)
    if len(active_dim_ids) == 1:
        axes_sv = [axes_sv]
    fig_sv.suptitle(f"Singular Values — {mn} {td} exp {en}  N={N}", fontsize=11)

    for ax, dim_idx in zip(axes_sv, active_dim_ids):
        for ht, (_, Uco, sv) in results.items():
            if Uco[dim_idx] is not None:
                ax.semilogy(sv[dim_idx], "o-", linewidth=2, label=ht)
        ax.set_title(f"dim {dim_idx+1} ({'y' if dim_idx == 0 else 'x'} pixels)")
        ax.set_xlabel("index")
        ax.set_ylabel("singular value (log)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    path_sv = out_dir / f"tp_singular_values_{tag}_all.png"
    fig_sv.savefig(path_sv, dpi=150)
    print(f"Saved singular values    → {path_sv}")

# --- Vertex weights over time ---
# For each time t and each vertex (r, s), compute the Frobenius inner product
# of the vertex matrix Sco[r,s,:,:] with the current pixel state Y[t,:,:].
# This is the true TP model activation: how much each vertex is "excited"
# by the current brain state.
#
#   contrib[r, s, t] = <Sco[r,s,:,:], Y[t,:,:]>_F
#                    = sum_{k,l} Sco[r,s,k,l] * Y[t,k,l]
#
# Marginal activations:
#   act_r(t) = sum_s contrib[r, s, t]   (net activation of output vertex r)
#   act_s(t) = sum_r contrib[r, s, t]   (net activation of input vertex s)

Y_cube  = Y.reshape(T, N, N)          # (T, N, N)
t_axis  = timestamps - timestamps[0]  # seconds from window start

for ht, (Sco, Uco, sv) in results.items():
    if Uco[0] is None or Uco[1] is None:
        continue

    r0, r1 = Sco.shape[0], Sco.shape[1]   # number of vertices per dim

    # contrib shape: (r0, r1, T)
    contrib = np.einsum("rskl,tkl->rst", Sco, Y_cube)

    act_r = contrib.sum(axis=1).T   # (T, r0) — sum over input vertices
    act_s = contrib.sum(axis=0).T   # (T, r1) — sum over output vertices

    fig_vt, axes_vt = plt.subplots(2, 1, figsize=(14, 6),
                                    sharex=True, constrained_layout=True)
    fig_vt.suptitle(
        f"Vertex Weights Over Time [{ht}] — {mn} {td} exp {en}  N={N}",
        fontsize=11)

    for i in range(r0):
        axes_vt[0].plot(t_axis, act_r[:, i], linewidth=0.8, label=f"v{i+1}")
    axes_vt[0].set_title(f"dim 1 (output y)  —  {r0} vertices")
    axes_vt[0].set_ylabel("vertex activation")
    axes_vt[0].legend(fontsize=8, ncol=min(r0, 6))
    axes_vt[0].grid(True, alpha=0.2)

    for j in range(r1):
        axes_vt[1].plot(t_axis, act_s[:, j], linewidth=0.8, label=f"v{j+1}")
    axes_vt[1].set_title(f"dim 2 (output x)  —  {r1} vertices")
    axes_vt[1].set_ylabel("vertex activation")
    axes_vt[1].set_xlabel("time (s)")
    axes_vt[1].legend(fontsize=8, ncol=min(r1, 6))
    axes_vt[1].grid(True, alpha=0.2)

    path_vt = out_dir / f"tp_vertex_weights_time_{tag}_{ht}.png"
    fig_vt.savefig(path_vt, dpi=150)
    print(f"Saved vertex weights over time [{ht}] → {path_vt}")

    # --- Weighting functions over time (one curve per spatial grid point) ---
    # Reconstruct the effective TP weight at each of the N grid points:
    #   wf_y[t, k] = sum_r Uco[0][k, r] * act_r[t, r]
    #   wf_x[t, k] = sum_r Uco[1][k, r] * act_s[t, r]
    # This gives N=5 time series — one per grid point — showing how each
    # spatial location is weighted by the TP model over time.

    W_y = Uco[0]   # (N, r0)
    W_x = Uco[1]   # (N, r1)

    wf_y = act_r @ W_y.T   # (T, N)
    wf_x = act_s @ W_x.T   # (T, N)

    fig_wf, axes_wf = plt.subplots(2, 1, figsize=(14, 6),
                                    sharex=True, constrained_layout=True)
    fig_wf.suptitle(
        f"Weighting Functions Over Time [{ht}] — {mn} {td} exp {en}  N={N}",
        fontsize=11)

    for k in range(N):
        axes_wf[0].plot(t_axis, wf_y[:, k], linewidth=0.8,
                        label=f"y={ys[k]}")
    axes_wf[0].set_title("dim 1 — y grid points")
    axes_wf[0].set_ylabel("TP weight")
    axes_wf[0].legend(fontsize=8, ncol=N)
    axes_wf[0].grid(True, alpha=0.2)

    for k in range(N):
        axes_wf[1].plot(t_axis, wf_x[:, k], linewidth=0.8,
                        label=f"x={xs[k]}")
    axes_wf[1].set_title("dim 2 — x grid points")
    axes_wf[1].set_ylabel("TP weight")
    axes_wf[1].set_xlabel("time (s)")
    axes_wf[1].legend(fontsize=8, ncol=N)
    axes_wf[1].grid(True, alpha=0.2)

    path_wf = out_dir / f"tp_wf_over_time_{tag}_{ht}.png"
    fig_wf.savefig(path_wf, dpi=150)
    print(f"Saved weighting functions over time [{ht}] → {path_wf}")

# --- AR(1) pixel time-series preview (first 6 pixels) ---
n_preview = min(6, P)
fig_ts, axes_ts = plt.subplots(n_preview, 1,
                                figsize=(12, 2 * n_preview),
                                sharex=True, constrained_layout=True)
fig_ts.suptitle(f"Pixel Time Series (first {n_preview} of {P}) — {mn}", fontsize=11)
for i, ax in enumerate(axes_ts):
    ax.plot(Y[:, i], linewidth=0.8)
    row, col = divmod(i, N)
    ax.set_ylabel(f"px({ys[row]},{xs[col]})", fontsize=8)
    ax.grid(True, alpha=0.2)
axes_ts[-1].set_xlabel("frame")

path_ts = out_dir / f"tp_pixel_timeseries_{tag}.png"
fig_ts.savefig(path_ts, dpi=150)
print(f"Saved pixel time series  → {path_ts}")

plt.show()
print("\nDone.")
