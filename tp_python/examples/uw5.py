"""
Example: Discrete-time state-space system with image output.

Simulates a stable linear system driven by a pulsed input and
Gaussian process noise, then visualises:
  1. State trajectories over time
  2. Snapshots of the projected (10n x 10n) image at selected steps
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SSmodels.dynamic_image_system import build_system, simulate
from tptoolbox.hull import genhull


def main():
    # ---- tuneable parameters ------------------------------------------------
    n = 10          # state dimensions
    m = 5          # input dimensions
    T = 1000        # simulation steps

    # ---- build system -------------------------------------------------------
    sys = build_system(n=n, m=m, decay=0.97, noise_std=0.08, seed=42)
    A, B = sys["A"], sys["B"]
    print(f"State dim n={n},  Input dim m={m},  Image {sys['img_size']}x{sys['img_size']}")
    print(f"Eigenvalues of A: {np.linalg.eigvals(A).round(4)}")

    # ---- input signal: periodic pulses + ramp -------------------------------
    def u_func(k):
        u = np.zeros(m)
        # channel 0: square pulse every 60 steps
        if (k // 30) % 2 == 0:
            u[0] = 1.0
        # channel 1: slow sine
        if m > 1:
            u[1] = 0.5 * np.sin(2 * np.pi * k / 120)
        return u

    # ---- simulate -----------------------------------------------------------
    x0 = np.zeros(n)
    x0[0] = 1.0  # kick the first mode
    xs, imgs = simulate(sys, T=T, x0=x0, u_func=u_func, seed=7)

    # ---- plot 1: state trajectories -----------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    for i in range(n):
        ax1.plot(xs[:, i], linewidth=1.5, label=f"x{i+1}")
    ax1.set_xlabel("Time step k")
    ax1.set_ylabel("State value")
    ax1.set_title(f"State trajectories  (n={n}, m={m})")
    ax1.legend(ncol=n, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ---- plot 2: image snapshots --------------------------------------------
    snap_steps = [0, T // 6, T // 3, T // 2, 2 * T // 3, T]
    n_snaps = len(snap_steps)
    fig2, axes2 = plt.subplots(1, n_snaps, figsize=(3 * n_snaps, 3))
    for ax, k in zip(axes2, snap_steps):
        ax.imshow(imgs[k], cmap="RdBu_r", origin="lower", vmin=-imgs.max(), vmax=imgs.max())
        ax.set_title(f"k={k}", fontsize=10)
        ax.axis("off")
    fig2.suptitle(f"Image output ({sys['img_size']}×{sys['img_size']})", fontsize=13)
    fig2.tight_layout()

    # ---- plot 3: phase portrait of first two state dims ---------------------
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.plot(xs[:, 0], xs[:, 1], linewidth=0.7, alpha=0.8)
    ax3.plot(xs[0, 0], xs[0, 1], "go", markersize=8, label="start")
    ax3.plot(xs[-1, 0], xs[-1, 1], "rs", markersize=8, label="end")
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.set_title("Phase portrait (x1 vs x2)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect("equal")

    # ---- plot 4: input signals ----------------------------------------------
    us = np.array([u_func(k) for k in range(T)])
    fig4, ax4 = plt.subplots(figsize=(12, 2.5))
    for j in range(m):
        ax4.step(range(T), us[:, j], where="mid", linewidth=1.2, label=f"u{j+1}")
    ax4.set_xlabel("Time step k")
    ax4.set_ylabel("Input")
    ax4.set_title("Input signals")
    ax4.legend(ncol=m, fontsize=9)
    ax4.grid(True, alpha=0.3)

    # =====================================================================
    # TP MODEL ANALYSIS on the state trajectory
    # =====================================================================
    # F = state trajectory matrix (T+1 x n).
    # Rows = time samples, columns = state features.
    # SVD → truncate to R components → hull decompositions.
    F = xs  # (T+1, n)
    t_axis = np.arange(T + 1)

    U_svd, D, Vt = np.linalg.svd(F, full_matrices=False)
    print("\n--- TP Model Analysis on state trajectory ---")
    print("Singular values:", D.round(4))

    # choose rank: keep components above 1% of leading SV
    R = int(np.sum(D > 0.01 * D[0]))
    R = max(R, 2)  # at least 2 for meaningful decomposition
    print(f"Retaining R={R} components  (n={n})")

    Uc = U_svd[:, :R]

    Usnnn = genhull(Uc, "snnn")
    Ucno = genhull(Uc, "cno")
    Uirno = genhull(Uc, "irno")

    # vertex systems (S) and reconstruction errors
    Ssnnn = np.linalg.lstsq(Usnnn, F, rcond=None)[0]
    Scno = np.linalg.lstsq(Ucno, F, rcond=None)[0]
    Sirno = np.linalg.lstsq(Uirno, F, rcond=None)[0]

    Esnnn = np.linalg.norm(F - Usnnn @ Ssnnn)
    Ecno = np.linalg.norm(F - Ucno @ Scno)
    Eirno = np.linalg.norm(F - Uirno @ Sirno)

    print(f"Reconstruction error  SNNN: {Esnnn:.6f}")
    print(f"Reconstruction error   CNO: {Ecno:.6f}")
    print(f"Reconstruction error  IRNO: {Eirno:.6f}")

    # ---- plot 5: singular value spectrum ------------------------------------
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    ax5.semilogy(D, "o-", markersize=5)
    ax5.axvline(R - 0.5, color="r", linestyle="--", alpha=0.6, label=f"truncation R={R}")
    ax5.set_xlabel("Index")
    ax5.set_ylabel("Singular value")
    ax5.set_title("SVD spectrum of state trajectory")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ---- plot 6: weighting functions ----------------------------------------
    fig6, axes6 = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, W, title in zip(axes6, [Usnnn, Ucno, Uirno], ["SNNN", "CNO", "IRNO"]):
        for r in range(W.shape[1]):
            ax.plot(t_axis, W[:, r], linewidth=2, label=f"w{r+1}")
        ax.set_xlabel("Time step k")
        ax.set_title(f"{title} weighting functions")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=min(R, 4))
    axes6[0].set_ylabel("Weight")
    fig6.tight_layout()

    # ---- plot 7: reconstruction comparison ----------------------------------
    fig7, axes7 = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
    if n == 1:
        axes7 = [axes7]
    F_recon = Usnnn @ Ssnnn
    for i, ax in enumerate(axes7):
        ax.plot(t_axis, F[:, i], "k-", linewidth=1.5, alpha=0.7, label="original")
        ax.plot(t_axis, F_recon[:, i], "r--", linewidth=1.2, label="SNNN recon")
        ax.set_ylabel(f"x{i+1}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    axes7[-1].set_xlabel("Time step k")
    fig7.suptitle(f"State reconstruction via SNNN  (R={R}, error={Esnnn:.4f})", fontsize=12)
    fig7.tight_layout()

    # ---- plot 8: vertex systems as images -----------------------------------
    n_vertices = Ssnnn.shape[0]
    fig8, axes8 = plt.subplots(1, n_vertices, figsize=(3.5 * n_vertices, 3))
    if n_vertices == 1:
        axes8 = [axes8]
    C = sys["C"]
    img_size = sys["img_size"]
    for idx in range(n_vertices):
        vertex_img = (C @ Ssnnn[idx]).reshape(img_size, img_size)
        axes8[idx].imshow(vertex_img, cmap="RdBu_r", origin="lower")
        axes8[idx].set_title(f"Vertex {idx+1}", fontsize=10)
        axes8[idx].axis("off")
    fig8.suptitle("SNNN vertex systems projected to image space", fontsize=12)
    fig8.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
