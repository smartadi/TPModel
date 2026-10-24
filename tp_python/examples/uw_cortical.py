"""
TP model transformation applied to cortical dynamics.

Simulates a mouse-cortex-inspired linear system (cortical_image_system),
then decomposes the state trajectory with the TP model framework:
  SVD → rank truncation → SNNN / CNO / IRNO hull decompositions.

Plots
-----
  1.  Cortical network (A matrix heatmap + area positions)
  2.  State trajectories
  3.  Cortical image snapshots
  4.  Input signals
  5.  SVD spectrum with rank truncation
  6.  Weighting functions (SNNN / CNO / IRNO)
  7.  State reconstruction via SNNN
  8.  Vertex systems projected to cortical image space (all three methods)
  9.  Reconstruction error comparison across methods
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SSmodels.cortical_image_system import build_system, simulate
from tptoolbox.hull import genhull


def main():
    # ---- parameters ---------------------------------------------------------
    n = 10      # cortical areas
    m = 5       # input channels
    T = 1000    # simulation steps

    # ---- build cortical system ----------------------------------------------
    csys = build_system(
        n=n, m=m,
        decay=0.97,
        noise_std=0.05,
        coupling_strength=0.25,
        coupling_range=0.40,
        noise_spatial_scale=0.30,
        noise_temporal_alpha=0.65,
        seed=42,
    )
    A, B, C_obs = csys["A"], csys["B"], csys["C"]
    img_size = csys["img_size"]
    centres  = csys["blob_centres"]

    sr = np.max(np.abs(np.linalg.eigvals(A)))
    print(f"Cortical system  n={n}, m={m},  image {img_size}x{img_size}")
    print(f"Spectral radius of A: {sr:.4f}")

    # ---- input: multi-channel drive to excite all frequency bands ----------
    def u_func(k):
        u = np.zeros(m)
        # channel 0: visual pulse (on for first 30 steps of every 120-step cycle)
        if (k // 30) % 4 == 0:
            u[0] = 1.0
        # channel 1: slow rhythmic somatosensory drive (theta-like)
        if m > 1:
            u[1] = 0.4 * np.sin(2 * np.pi * k / 80)
        # channel 2: brief transient at t=200 + recurring burst at t=500
        if m > 2 and (200 <= k < 215 or 500 <= k < 520):
            u[2] = 0.8
        # channel 3: gamma-band-like rapid alternation
        if m > 3:
            u[3] = 0.5 * np.sin(2 * np.pi * k / 15)
        # channel 4: slow ramp in second half of simulation
        if m > 4 and k > T // 2:
            u[4] = 0.6 * (k - T // 2) / (T // 2)
        return u

    # ---- simulate -----------------------------------------------------------
    rng_x0 = np.random.default_rng(123)
    x0 = 0.5 * rng_x0.standard_normal(n)  # spread energy across all states
    xs, imgs = simulate(csys, T=T, x0=x0, u_func=u_func, seed=7)
    t_axis = np.arange(T + 1)

    vmax = float(np.max(np.abs(imgs)))

    # =========================================================================
    # Plot 1: Cortical network overview
    # =========================================================================
    fig1 = plt.figure(figsize=(13, 4))
    gs = gridspec.GridSpec(1, 3, figure=fig1, wspace=0.35)

    # A heatmap
    ax_A = fig1.add_subplot(gs[0])
    im_A = ax_A.imshow(A, cmap="RdBu_r", aspect="auto",
                       vmin=-np.max(np.abs(A)), vmax=np.max(np.abs(A)))
    plt.colorbar(im_A, ax=ax_A, fraction=0.046, pad=0.04)
    ax_A.set_title("Network matrix A", fontsize=11)
    ax_A.set_xlabel("Area j")
    ax_A.set_ylabel("Area i")

    # Q heatmap
    ax_Q = fig1.add_subplot(gs[1])
    im_Q = ax_Q.imshow(csys["Q"], cmap="viridis", aspect="auto")
    plt.colorbar(im_Q, ax=ax_Q, fraction=0.046, pad=0.04)
    ax_Q.set_title("Noise covariance Q", fontsize=11)
    ax_Q.set_xlabel("Area j")
    ax_Q.set_ylabel("Area i")

    # Area positions on cortex
    ax_pos = fig1.add_subplot(gs[2])
    half = n // 2
    ax_pos.scatter(centres[:half, 0], centres[:half, 1],
                   c=np.arange(half), cmap="tab10", s=120, zorder=3, label="Left hemi")
    ax_pos.scatter(centres[half:, 0], centres[half:, 1],
                   c=np.arange(n - half), cmap="tab10", s=120,
                   marker="^", zorder=3, label="Right hemi")
    for i in range(n):
        ax_pos.annotate(str(i + 1), centres[i], fontsize=7,
                        ha="center", va="center", color="white", fontweight="bold")
    ax_pos.set_xlim(0, img_size)
    ax_pos.set_ylim(0, img_size)
    ax_pos.set_aspect("equal")
    ax_pos.set_title("Cortical area positions", fontsize=11)
    ax_pos.set_xlabel("x (pixels)")
    ax_pos.set_ylabel("y (pixels)")
    ax_pos.legend(fontsize=8)
    ax_pos.grid(True, alpha=0.3)
    fig1.suptitle("Cortical system structure", fontsize=13)

    # =========================================================================
    # Plot 2: State trajectories
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    for i in range(n):
        ax2.plot(xs[:, i], linewidth=1.2, label=f"area {i+1}", alpha=0.85)
    ax2.set_xlabel("Time step k")
    ax2.set_ylabel("State value")
    ax2.set_title(f"Cortical state trajectories  (n={n}, m={m})")
    ax2.legend(ncol=n, fontsize=8)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 3: Cortical image snapshots
    # =========================================================================
    snap_steps = [0, T // 6, T // 3, T // 2, 2 * T // 3, T]
    fig3, axes3 = plt.subplots(1, len(snap_steps), figsize=(3 * len(snap_steps), 3))
    for ax, k in zip(axes3, snap_steps):
        ax.imshow(imgs[k], cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
        ax.set_title(f"k={k}", fontsize=10)
        ax.axis("off")
    fig3.suptitle(f"Cortical image snapshots ({img_size}×{img_size})", fontsize=13)
    fig3.tight_layout()

    # =========================================================================
    # Plot 4: Input signals
    # =========================================================================
    us = np.array([u_func(k) for k in range(T)])
    fig4, ax4 = plt.subplots(figsize=(12, 2.5))
    for j in range(m):
        ax4.step(range(T), us[:, j], where="mid", linewidth=1.2, label=f"u{j+1}")
    ax4.set_xlabel("Time step k")
    ax4.set_ylabel("Input")
    ax4.set_title("Input signals")
    ax4.legend(ncol=m, fontsize=9)
    ax4.grid(True, alpha=0.3)

    # =========================================================================
    # TP MODEL ANALYSIS
    # =========================================================================
    F = xs          # (T+1, n) — rows=time, cols=cortical areas
    print("\n--- TP Model Analysis ---")

    U_svd, D, Vt = np.linalg.svd(F, full_matrices=False)
    print("Singular values:", D.round(4))

    R = int(np.sum(D > 0.01 * D[0]))
    R = max(R, 2)
    print(f"Retaining R={R} components")

    Uc = U_svd[:, :R]

    print("Computing SNNN hull...")
    Usnnn = genhull(Uc, "snnn")
    print("Computing CNO hull...")
    Ucno  = genhull(Uc, "cno")
    print("Computing IRNO hull...")
    Uirno = genhull(Uc, "irno")

    Ssnnn = np.linalg.lstsq(Usnnn, F, rcond=None)[0]
    Scno  = np.linalg.lstsq(Ucno,  F, rcond=None)[0]
    Sirno = np.linalg.lstsq(Uirno, F, rcond=None)[0]

    Esnnn = np.linalg.norm(F - Usnnn @ Ssnnn)
    Ecno  = np.linalg.norm(F - Ucno  @ Scno)
    Eirno = np.linalg.norm(F - Uirno @ Sirno)

    print(f"Reconstruction error  SNNN: {Esnnn:.6f}")
    print(f"Reconstruction error   CNO: {Ecno:.6f}")
    print(f"Reconstruction error  IRNO: {Eirno:.6f}")

    # =========================================================================
    # Plot 5: SVD spectrum
    # =========================================================================
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    ax5.semilogy(D, "o-", markersize=5)
    ax5.axvline(R - 0.5, color="r", linestyle="--", alpha=0.7,
                label=f"truncation R={R}")
    ax5.set_xlabel("Component index")
    ax5.set_ylabel("Singular value")
    ax5.set_title("SVD spectrum of cortical state trajectory")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 6: Weighting functions
    # =========================================================================
    fig6, axes6 = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, W_mat, label in zip(
        axes6,
        [Usnnn, Ucno, Uirno],
        ["SNNN", "CNO", "IRNO"],
    ):
        for r in range(W_mat.shape[1]):
            ax.plot(t_axis, W_mat[:, r], linewidth=1.8, label=f"w{r+1}", alpha=0.85)
        ax.set_xlabel("Time step k")
        ax.set_title(f"{label} weighting functions")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=min(R, 4))
    axes6[0].set_ylabel("Weight")
    fig6.suptitle("TP model weighting functions — cortical trajectory", fontsize=13)
    fig6.tight_layout()

    # =========================================================================
    # Plot 7: State reconstruction via SNNN
    # =========================================================================
    F_recon = Usnnn @ Ssnnn
    fig7, axes7 = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
    if n == 1:
        axes7 = [axes7]
    for i, ax in enumerate(axes7):
        ax.plot(t_axis, F[:, i], "k-",  linewidth=1.5, alpha=0.7, label="original")
        ax.plot(t_axis, F_recon[:, i], "r--", linewidth=1.2, label="SNNN recon")
        ax.set_ylabel(f"area {i+1}", fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    axes7[-1].set_xlabel("Time step k")
    fig7.suptitle(
        f"State reconstruction via SNNN  (R={R}, error={Esnnn:.4f})", fontsize=12
    )
    fig7.tight_layout()

    # =========================================================================
    # Plot 8: Vertex systems as cortical images (all three methods)
    # =========================================================================
    methods   = [("SNNN", Ssnnn, Usnnn), ("CNO", Scno, Ucno), ("IRNO", Sirno, Uirno)]
    n_verts   = Ssnnn.shape[0]           # same for all methods
    # extra column for per-row colorbar
    fig8, axes8 = plt.subplots(
        3, n_verts + 1, figsize=(3.2 * n_verts + 1.2, 9),
        gridspec_kw={"width_ratios": [1] * n_verts + [0.06]},
    )
    if n_verts == 1:
        axes8 = axes8[:, np.newaxis]

    for row, (label, S, _) in enumerate(methods):
        imgs_row = [(C_obs @ S[col]).reshape(img_size, img_size)
                     for col in range(n_verts)]
        vmax_row = max(np.max(np.abs(im)) for im in imgs_row)
        if vmax_row < 1e-12:
            vmax_row = 1.0
        for col in range(n_verts):
            im = axes8[row, col].imshow(
                imgs_row[col], cmap="RdBu_r", origin="lower",
                vmin=-vmax_row, vmax=vmax_row,
            )
            axes8[row, col].axis("off")
            if row == 0:
                axes8[row, col].set_title(f"Vertex {col+1}", fontsize=10)
        # row label on left side
        axes8[row, 0].text(
            -0.08, 0.5, label, transform=axes8[row, 0].transAxes,
            fontsize=13, fontweight="bold", va="center", ha="right",
            rotation=90,
        )
        # per-row colorbar in the last column
        fig8.colorbar(im, cax=axes8[row, -1], label=f"max={vmax_row:.2e}")
    fig8.suptitle(
        "Vertex systems projected to cortical image space", fontsize=13
    )
    fig8.tight_layout()

    # =========================================================================
    # Plot 9: Reconstruction error bar chart
    # =========================================================================
    fig9, ax9 = plt.subplots(figsize=(5, 4))
    labels = ["SNNN", "CNO", "IRNO"]
    errors = [Esnnn, Ecno, Eirno]
    bars = ax9.bar(labels, errors, color=["steelblue", "tomato", "seagreen"], width=0.5)
    for bar, val in zip(bars, errors):
        ax9.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.2e}",
            ha="center", va="bottom", fontsize=10,
        )
    ax9.set_ylabel("||F - W·S||₂")
    ax9.set_title(f"Reconstruction error by hull method  (R={R})")
    ax9.grid(True, axis="y", alpha=0.3)
    fig9.tight_layout()

    # ---- save all figures to plots/ folder ----------------------------------
    plots_dir = Path(__file__).resolve().parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    for i, fig in enumerate(
        [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9], start=1
    ):
        fig.savefig(plots_dir / f"cortical_plot{i}.png", dpi=150, bbox_inches="tight")
    print(f"\nAll plots saved to {plots_dir}")

    plt.show()


if __name__ == "__main__":
    main()
