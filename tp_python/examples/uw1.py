import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tptoolbox.hull import genhull


def f(p):
    return np.array([2 + np.sin(p), 2 + np.cos(p)])


def main():
    H = 200
    omega = np.array([0, 2 * np.pi])
    print("omega.shape:", omega.shape)
    R = 2
    g = np.linspace(omega[0], omega[1], H)
    print("g.shape:", g.shape)

    F = np.vstack([f(p) for p in g])
    print("F.shape:", F.shape)

    U, D, Vt = np.linalg.svd(F, full_matrices=False)
    print("U.shape:", U.shape)
    print("D.shape:", D.shape)
    print("Vt.shape:", Vt.shape)
    print("Singular values:", D)

    Uc = U[:, :R]
    print("Uc.shape:", Uc.shape)
    Usnnn = genhull(Uc, "snnn")
    print("Usnnn.shape:", Usnnn.shape)
    Ucno = genhull(Uc, "cno")
    print("Ucno.shape:", Ucno.shape)
    Uirno = genhull(Uc, "irno")
    print("Uirno.shape:", Uirno.shape)

    Ssnnn = np.linalg.pinv(Usnnn) @ F
    print("Ssnnn.shape:", Ssnnn.shape)
    Scno = np.linalg.pinv(Ucno) @ F
    print("Scno.shape:", Scno.shape)
    Sirno = np.linalg.pinv(Uirno) @ F
    print("Sirno.shape:", Sirno.shape)

    Esnnn = np.linalg.norm(F - Usnnn @ Ssnnn)
    print("Esnnn.shape:", np.array(Esnnn).shape)
    Ecno = np.linalg.norm(F - Ucno @ Scno)
    print("Ecno.shape:", np.array(Ecno).shape)
    Eirno = np.linalg.norm(F - Uirno @ Sirno)
    print("Eirno.shape:", np.array(Eirno).shape)

    print("Esnnn:", Esnnn)
    print("Ecno:", Ecno)
    print("Eirno:", Eirno)

    fig1 = plt.figure()
    plt.plot(g, Usnnn, linewidth=3)
    plt.grid(True)
    plt.title("SNNN weighting functions")

    fig2 = plt.figure()
    plt.plot(g, Ucno, linewidth=3)
    plt.grid(True)
    plt.title("CNO weighting functions")

    fig3 = plt.figure()
    plt.plot(g, Uirno, linewidth=3)
    plt.grid(True)
    plt.title("IRNO weighting functions")

    # ---- Vertex plots for each hull method ----------------------------------
    fig4, axes4 = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (label, S) in zip(axes4, [("SNNN", Ssnnn), ("CNO", Scno), ("IRNO", Sirno)]):
        n_verts = S.shape[0]
        for v in range(n_verts):
            ax.plot(S[v, 0], S[v, 1], "o", markersize=10, label=f"Vertex {v+1}")
        # connect vertices to show the polytope
        hull_pts = np.vstack([S, S[:1]])
        ax.plot(hull_pts[:, 0], hull_pts[:, 1], "k--", alpha=0.4)
        # overlay the original trajectory
        ax.plot(F[:, 0], F[:, 1], "-", linewidth=1.5, alpha=0.5, label="f(p)")
        ax.set_xlabel("f₁")
        ax.set_ylabel("f₂")
        ax.set_title(f"{label} vertices ({n_verts})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
    fig4.suptitle("Vertex systems in output space", fontsize=13)
    fig4.tight_layout()

    # ---- Save all plots -----------------------------------------------------
    plots_dir = Path(__file__).resolve().parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    for i, fig in enumerate([fig1, fig2, fig3, fig4], start=1):
        fig.savefig(plots_dir / f"uw1_plot{i}.png", dpi=150, bbox_inches="tight")
    print(f"\nAll plots saved to {plots_dir}")

    plt.show()


if __name__ == "__main__":
    main()
