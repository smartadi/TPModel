"""
Whole-image TP model analysis using SVD temporal components as state.

Pipeline
--------
1. Load widefield SVD session, select 10-min window from session midpoint.
2. SVD analysis: plot singular values and top spatial/temporal components.
3. Fit AR(1) to temporal components: X[k+1] ≈ A X[k]  (A is C×C).
4. TP model transform on A for hull types: snnn, irno, box.
5. Plot vertex spatial maps: Uco[:, i] projected through svd_spatial → (Y, X).
6. Plot vertex weights over time: X[t] · Uco[:, i] for each vertex i.

State space
-----------
  X[t] ∈ R^C  — SVD temporal components (already encode whole-image dF/F dynamics)
  A ∈ R^{C×C} — learned transition matrix
  image(t) ≈ svd_spatial @ X[t] + mean_image   (full reconstruction)
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
# USER PARAMETERS
# =============================================================================

mn = "ZYE_0043"
td = "2021-09-13"
en = "1"

half_window_min = 5.0    # 10-min window centred on session midpoint

hull_types_to_try = ["snnn", "irno", "box"]

# =============================================================================
# LOAD SESSION
# =============================================================================

print(f"Loading session {mn} / {td} / {en} ...")
session = load_svd_session(mn=mn, td=td, en=en)

spatial    = session["svd_spatial"]    # (Y, X, C)
temporal   = session["svd_temporal"]   # (T, C)
timestamps = session["timestamps"]     # (T,)
mean_img   = session["mean_image"]     # (Y, X)

img_H, img_W, C = spatial.shape
print(f"Image: {img_H}×{img_W}   SVD comps: {C}   Total frames: {temporal.shape[0]}")

# Select 10-min window centred on session midpoint
t_mid = (timestamps[0] + timestamps[-1]) / 2.0
hw    = half_window_min * 60.0
mask  = (timestamps >= t_mid - hw) & (timestamps <= t_mid + hw)
X          = temporal[mask]      # (T, C)  — state matrix
timestamps = timestamps[mask]
T          = X.shape[0]

print(f"Window: {timestamps[0]/60:.2f} – {timestamps[-1]/60:.2f} min  ({T} frames)")

t_axis = timestamps - timestamps[0]   # seconds from window start

# =============================================================================
# SVD ANALYSIS
# =============================================================================

out_dir = PROJECT_ROOT / "tp_python" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
tag = f"{mn}_{td}_{en}_whole"

# Singular values of temporal components over the window
_, sv_data, _ = np.linalg.svd(X, full_matrices=False)

fig_sv, ax_sv = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax_sv.semilogy(sv_data, "o-", linewidth=1.5, markersize=4)
ax_sv.set_title(f"Singular values of temporal components — {mn} {td} exp {en}")
ax_sv.set_xlabel("component index")
ax_sv.set_ylabel("singular value (log)")
ax_sv.grid(True, alpha=0.3)
path = out_dir / f"svd_singularvalues_{tag}.png"
fig_sv.savefig(path, dpi=150)
print(f"Saved SVD singular values → {path}")

# Top 6 spatial components as images
n_show = min(6, C)
fig_sp, axes_sp = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
fig_sp.suptitle(f"Top {n_show} spatial SVD components — {mn}", fontsize=11)
for i, ax in enumerate(axes_sp.ravel()):
    img = spatial[:, :, i]
    vmax = np.percentile(np.abs(img), 99)
    ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"comp {i+1}")
    ax.axis("off")
path = out_dir / f"svd_spatial_components_{tag}.png"
fig_sp.savefig(path, dpi=150)
print(f"Saved spatial components  → {path}")

# Top 6 temporal components over the window
fig_tc, axes_tc = plt.subplots(n_show, 1, figsize=(14, 2 * n_show),
                                sharex=True, constrained_layout=True)
fig_tc.suptitle(f"Top {n_show} temporal SVD components — {mn}", fontsize=11)
for i, ax in enumerate(axes_tc):
    ax.plot(t_axis, X[:, i], linewidth=0.6)
    ax.set_ylabel(f"c{i+1}", fontsize=8)
    ax.grid(True, alpha=0.2)
axes_tc[-1].set_xlabel("time (s)")
path = out_dir / f"svd_temporal_components_{tag}.png"
fig_tc.savefig(path, dpi=150)
print(f"Saved temporal components → {path}")

# =============================================================================
# FIT AR(1):  X[k+1] ≈ A X[k]
# =============================================================================

X_prev = X[:-1]    # (T-1, C)
X_next = X[1:]     # (T-1, C)

A, _, _, _ = np.linalg.lstsq(X_prev, X_next, rcond=None)
A = A.T            # (C, C): X[k+1] = A @ X[k]

fit_err = np.sqrt(np.mean((X_next - X_prev @ A.T) ** 2))
spec_rad = np.max(np.abs(np.linalg.eigvals(A)))
print(f"\nAR(1) fit RMS error: {fit_err:.6f}   spectral radius: {spec_rad:.4f}")

# =============================================================================
# TP MODEL TRANSFORM — loop over hull types
# =============================================================================
# A is (C, C) treated as a 2D tensor.
# Both dimensions are active (output components × input components).

results = {}

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
            print(f"  dim {dim_idx+1}: {w.shape[1]} vertices, "
                  f"top-5 sv = {s[:5].round(4)}")
    except Exception as exc:
        print(f"  FAILED: {exc}")

# =============================================================================
# PLOTS
# =============================================================================

for ht, (Sco, Uco, sv) in results.items():

    W0 = Uco[0]   # (C, r0) — output-component vertices
    W1 = Uco[1]   # (C, r1) — input-component vertices
    if W0 is None or W1 is None:
        continue

    r0, r1 = W0.shape[1], W1.shape[1]

    # --- Vertex spatial maps ---
    # Project each vertex weighting vector through svd_spatial → (Y, X) image.
    # vertex_img_i = svd_spatial @ W0[:, i]  (output-space vertices)

    for dim_idx, W, dim_label in [(0, W0, "output"), (1, W1, "input")]:
        r = W.shape[1]
        ncols = min(r, 4)
        nrows = int(np.ceil(r / ncols))
        fig_v, axes_v = plt.subplots(nrows, ncols,
                                      figsize=(4 * ncols, 3.5 * nrows),
                                      constrained_layout=True)
        axes_v = np.array(axes_v).ravel()
        fig_v.suptitle(
            f"Vertex spatial maps [{ht}] dim {dim_idx+1} ({dim_label}) — {mn}",
            fontsize=11)

        for i in range(r):
            img = spatial @ W[:, i]    # (Y, X)
            vmax = np.percentile(np.abs(img), 99)
            axes_v[i].imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes_v[i].set_title(f"vertex {i+1}")
            axes_v[i].axis("off")
        for ax in axes_v[r:]:
            ax.set_visible(False)

        path = out_dir / f"tp_vertex_maps_{tag}_{ht}_dim{dim_idx+1}.png"
        fig_v.savefig(path, dpi=150)
        print(f"Saved vertex maps [{ht}] dim {dim_idx+1} → {path}")

    # --- Vertex weights over time ---
    # Sco has shape (r0, r1) — core tensor after HOSVD of the (C,C) matrix A.
    # At each time t, the activation of vertex (r,s) is:
    #   contrib[r, s, t] = Sco[r,s] * (Uco[0][:,r] · X[t]) * (Uco[1][:,s] · X[t])
    # which decomposes as:
    #   alpha[t, r] = Uco[0][:,r] · X[t]   (projection of state onto output vertex r)
    #   beta[t, s]  = Uco[1][:,s] · X[t]   (projection of state onto input vertex s)
    #   contrib[r,s,t] = Sco[r,s] * alpha[t,r] * beta[t,s]
    # Marginals:
    #   act_r(t) = sum_s contrib[r,s,t]
    #   act_s(t) = sum_r contrib[r,s,t]

    alpha = X @ W0    # (T, r0): state projected onto output vertices
    beta  = X @ W1    # (T, r1): state projected onto input vertices

    # contrib[r,s,t] = Sco[r,s] * alpha[t,r] * beta[t,s]
    contrib = np.einsum("rs,tr,ts->rst", Sco, alpha, beta)  # (r0, r1, T)

    act_r = contrib.sum(axis=1).T   # (T, r0)
    act_s = contrib.sum(axis=0).T   # (T, r1)

    fig_wt, axes_wt = plt.subplots(2, 1, figsize=(14, 6),
                                    sharex=True, constrained_layout=True)
    fig_wt.suptitle(
        f"Vertex weights over time [{ht}] — {mn} {td} exp {en}", fontsize=11)

    for i in range(r0):
        axes_wt[0].plot(t_axis, act_r[:, i], linewidth=0.7, label=f"v{i+1}")
    axes_wt[0].set_title(f"dim 1 (output components)  —  {r0} vertices")
    axes_wt[0].set_ylabel("vertex activation")
    axes_wt[0].legend(fontsize=8, ncol=min(r0, 8))
    axes_wt[0].grid(True, alpha=0.2)

    for j in range(r1):
        axes_wt[1].plot(t_axis, act_s[:, j], linewidth=0.7, label=f"v{j+1}")
    axes_wt[1].set_title(f"dim 2 (input components)  —  {r1} vertices")
    axes_wt[1].set_ylabel("vertex activation")
    axes_wt[1].set_xlabel("time (s)")
    axes_wt[1].legend(fontsize=8, ncol=min(r1, 8))
    axes_wt[1].grid(True, alpha=0.2)

    path = out_dir / f"tp_vertex_weights_time_{tag}_{ht}.png"
    fig_wt.savefig(path, dpi=150)
    print(f"Saved vertex weights over time [{ht}] → {path}")

    # --- Weighting functions over time (one curve per SVD component) ---
    # Reconstruct effective TP weight at each of the C SVD components:
    #   wf_out[t, c] = sum_r Uco[0][c, r] * alpha[t, r]
    #   wf_in[t, c]  = sum_r Uco[1][c, r] * beta[t, r]
    # Show only top n_show components for readability.

    n_show = min(6, C)
    wf_out = alpha @ W0.T   # (T, C)
    wf_in  = beta  @ W1.T   # (T, C)

    fig_wf, axes_wf = plt.subplots(2, 1, figsize=(14, 6),
                                    sharex=True, constrained_layout=True)
    fig_wf.suptitle(
        f"Weighting Functions Over Time [{ht}] — {mn} {td} exp {en}",
        fontsize=11)

    for c in range(n_show):
        axes_wf[0].plot(t_axis, wf_out[:, c], linewidth=0.7, label=f"comp {c+1}")
    axes_wf[0].set_title(f"dim 1 (output) — top {n_show} SVD components")
    axes_wf[0].set_ylabel("TP weight")
    axes_wf[0].legend(fontsize=8, ncol=n_show)
    axes_wf[0].grid(True, alpha=0.2)

    for c in range(n_show):
        axes_wf[1].plot(t_axis, wf_in[:, c], linewidth=0.7, label=f"comp {c+1}")
    axes_wf[1].set_title(f"dim 2 (input) — top {n_show} SVD components")
    axes_wf[1].set_ylabel("TP weight")
    axes_wf[1].set_xlabel("time (s)")
    axes_wf[1].legend(fontsize=8, ncol=n_show)
    axes_wf[1].grid(True, alpha=0.2)

    path_wf = out_dir / f"tp_wf_over_time_{tag}_{ht}.png"
    fig_wf.savefig(path_wf, dpi=150)
    print(f"Saved weighting functions over time [{ht}] → {path_wf}")

    # --- Singular values across hull types (combined) ---

# Singular values: all hull types overlaid
active_dim_ids = [0, 1]
fig_sv2, axes_sv2 = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
fig_sv2.suptitle(f"TP Singular Values — {mn} {td} exp {en}", fontsize=11)
dim_labels = ["dim 1 (output)", "dim 2 (input)"]
for ax, dim_idx, dlabel in zip(axes_sv2, active_dim_ids, dim_labels):
    for ht, (_, Uco, sv) in results.items():
        if Uco[dim_idx] is not None:
            ax.semilogy(sv[dim_idx], "o-", linewidth=2, label=ht)
    ax.set_title(dlabel)
    ax.set_xlabel("index")
    ax.set_ylabel("singular value (log)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

path = out_dir / f"tp_singular_values_{tag}_all.png"
fig_sv2.savefig(path, dpi=150)
print(f"Saved TP singular values  → {path}")

plt.show()
print("\nDone.")
