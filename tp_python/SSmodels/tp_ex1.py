"""
TP Model Transformation Framework
Based on: Baranyi, P. (2019)
"Extracting LPV and qLPV structures from state-space functions:
 a TP model transformation based framework"
IEEE Transactions on Fuzzy Systems, DOI: 10.1109/TFUZZ.2019.2908770

Implements:
  - Method 1: TP Model Transformation (HOSVD-based)
  - Method 2: Linearisation (global or local)
  - Method 3: Parameter Varying Linear Structure
  - Method 4: Quasi-Linear Structure Reconstruction
  - Example: Inverted Pendulum (all 4 examples from paper)

This version integrates with the existing tptoolbox APIs.
"""

from __future__ import annotations

from itertools import product as iterproduct
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tptoolbox.model import ConvexTPStructure
from tptoolbox.ssmodels import (
    build_grid_tensor,
    extract_qlpv,
    linearise,
)


# =============================================================================
# INVERTED PENDULUM MODEL  (Equations 58-64 in paper)
# =============================================================================

Mc = 3.0      # cart mass [kg]
mp = 0.2      # pendulum mass [kg]
l = 0.31      # pendulum length [m]
b = 0.1       # friction [N*m/s]
g = 9.88      # gravity [m/s^2]


def model(X: np.ndarray) -> np.ndarray:
    """
    State-space model of inverted pendulum.
    X = [theta, dtheta, s, ds, u]  (state + input concatenated)
    Returns xdot = [dtheta, ddtheta, ds, dds]
    """
    th, dth, s, ds, u = X
    si, co = np.sin(th), np.cos(th)
    s1l = l * (Mc + mp * si**2)
    s2l = Mc + mp * si**2

    T1 = (Mc + mp) * g * si / s1l
    T2 = b * co * ds / s1l
    T3 = -mp * l * si * co * dth**2 / s1l
    T4 = -co * u / s1l

    P1 = -mp * g * si * co / s2l
    P2 = -b * ds / s2l
    P3 = mp * l * si * dth**2 / s2l
    P4 = u / s2l

    return np.array([dth, T1 + T2 + T3 + T4, ds, P1 + P2 + P3 + P4])


def modelS(P: np.ndarray) -> np.ndarray:
    """
    System matrix S(p1, p2) for inverted pendulum.
    P = [theta, dtheta]  (scheduling parameters)
    Returns S in R^{4x5} such that xdot ≈ S @ [x; u]
    """
    th, dth = P[0], P[1]
    si, co = np.sin(th), np.cos(th)
    s1l = l * (Mc + mp * si**2)
    s2l = Mc + mp * si**2

    T1 = (Mc + mp) * g * si / s1l
    T2 = b * co / s1l
    T3 = -mp * l * si * co / s1l
    T4 = -co / s1l

    P1 = -mp * g * si * co / s2l
    P2 = -b / s2l
    eps = 1e-12
    P3_val = mp * l * si * dth**2 / s2l
    P4 = 1.0 / s2l

    S = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, T1 / max(abs(dth), eps) + T3, 0, T2, T4],
            [0, 0, 0, 1, 0],
            [0, P1 / max(abs(dth), eps) + P3_val, 0, P2, P4],
        ]
    )
    return S


# =============================================================================
# METHOD 2 - LINEARISATION  (Section IV)
# =============================================================================


def example1_global_linearisation() -> np.ndarray:
    """
    Example 1: Reconstruct global linear mapping xdot ≈ S @ [x; u]
    (assuming closed formula is unknown, only I/O pairs available)
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1 - Global Linearisation (Method 2)")
    print("=" * 60)

    Omega = [1.6, 1.6, 1.0, 1.0, 1.0]
    M = 5
    O = [0.0] * 5

    S_global, err_global = linearise(model, Omega, M, O)
    print(f"Global linearisation RMS error: {err_global:.4f}")

    eps = 1e-10
    Omega_local = [eps, eps, 1.0, 1.0, 1.0]
    S_local, err_local = linearise(model, Omega_local, M, O)
    print(f"Local linearisation RMS error: {err_local:.6f}")
    print("\nSystem matrix S at equilibrium (Eq. 65):")
    print(np.round(S_local, 3))
    return S_local


# =============================================================================
# METHOD 4 - QUASI-LINEAR STRUCTURE  (Section VII)
# =============================================================================


def example2_qlpv_extended():
    """
    Example 2: Extended TP model transformation - extract qLPV model
    where all 5 variables [x1..x4, u] are scheduling parameters.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2 - Extended TP Model Transformation (Method 4)")
    print("=" * 60)

    Omega_p = [1.6, 1.6, 1.0, 1.0, 1.0]
    M_p = 6
    Omega_x = [1e-10] * 5
    M_x = 3

    S_tensor, grids_p, Uco, Sco, sv = extract_qlpv(
        model, Omega_p, M_p, Omega_x, M_x, hull_type="snnn", tol=1e-11
    )

    print("Core tensor shape:", Sco.shape)
    print("Number of scheduling dimensions:", len(Uco))
    for i, (wi, si) in enumerate(zip(Uco, sv)):
        if wi is None:
            continue
        print(
            f"  p{i+1}: {wi.shape[1]} weighting functions, "
            f"top-3 sv = {si[:3].round(4)}"
        )

    return Uco, grids_p, sv


# =============================================================================
# METHOD 1 - TP MODEL TRANSFORMATION  (Section III)
# =============================================================================


def example3_known_structure():
    """
    Example 3: TP transformation when system matrix S(p1,p2) is known.
    Scheduling parameters: p1=x1=theta, p2=x2=dtheta.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3 - TP Transform of Known S(p1,p2) (Method 1)")
    print("=" * 60)

    Omega = [1.6, 1.6]
    M = 50
    O = [0.0, 0.0]

    G, grids = build_grid_tensor(M, Omega, O)
    idx_list = list(iterproduct(range(M), repeat=2))

    S_disc = np.zeros((M, M, 4, 5))
    for i, j in idx_list:
        S_disc[i, j] = modelS(G[i, j])

    hulltypes = ["snnn", "snnn", "none", "none"]
    Sco, Uco, sv, _, _ = ConvexTPStructure(S_disc, hulltypes, tol=1e-11)

    print(
        f"Weighting functions - p1: {Uco[0].shape[1]} rules, "
        f"p2: {Uco[1].shape[1]} rules"
    )
    print(f"Total rules: {Uco[0].shape[1] * Uco[1].shape[1]}")
    print(f"Singular values p1: {sv[0][:5].round(4)}")
    print(f"Singular values p2: {sv[1][:5].round(4)}")

    return Uco, grids, Sco, sv


# =============================================================================
# METHOD 3 - PARAMETER SPACE MANIPULATION  (Section VI + Method 3)
# =============================================================================


def example4_modified_parameter_space():
    """
    Example 4: Change parameter space to p1=sin(x1), p2=cos(x1), p3=x2.
    Demonstrates flexible parameter space manipulation.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4 - Modified Parameter Space (Section VI + Method 3)")
    print("=" * 60)

    def modelS_new_params(P):
        """P = [sin(theta), cos(theta), dtheta]"""
        si, co, dth = P[0], P[1], P[2]
        s1l = l * (Mc + mp * si**2)
        s2l = Mc + mp * si**2
        eps = 1e-12
        T1 = (Mc + mp) * g * si / s1l
        T2 = b * co / s1l
        T3 = -mp * l * si * co / s1l
        T4 = -co / s1l
        P1 = -mp * g * si * co / s2l
        P2 = -b / s2l
        P3 = mp * l * si * dth**2 / s2l
        P4 = 1.0 / s2l
        S = np.array(
            [
                [0, 1, 0, 0, 0],
                [0, T1 / max(abs(dth), eps) + T3, 0, T2, T4],
                [0, 0, 0, 1, 0],
                [0, P1 / max(abs(dth), eps) + P3, 0, P2, P4],
            ]
        )
        return S

    Omega = [1.6, 1.6, 0.6]
    M = 50
    O = [0.0, 0.0, 0.0]
    G, grids = build_grid_tensor(M, Omega, O)

    S_disc = np.zeros((M, M, M, 4, 5))
    for idx in iterproduct(range(M), repeat=3):
        S_disc[idx] = modelS_new_params(G[idx])

    hulltypes = ["cno", "cno", "cno", "none", "none"]
    Sco, Uco, sv, _, _ = ConvexTPStructure(S_disc, hulltypes, tol=1e-11)

    print(
        "Weighting functions - "
        f"p1(sin x1): {Uco[0].shape[1]}, "
        f"p2(cos x1): {Uco[1].shape[1]}, "
        f"p3(x2): {Uco[2].shape[1]}"
    )
    print(f"Total rules: {np.prod([w.shape[1] for w in Uco[:3]])}")

    return Uco, grids, Sco, sv


# =============================================================================
# PLOTTING
# =============================================================================


def plot_weighting_functions(W_list, grids_list, titles, suptitle):
    """Plot weighting functions for each scheduling parameter."""
    n = len(W_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    for ax, W, grid, title in zip(axes, W_list, grids_list, titles):
        for r in range(W.shape[1]):
            ax.plot(grid, W[:, r], linewidth=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Weighting function")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)

    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    S_eq = example1_global_linearisation()

    W3, grids3, S_core3, sv3 = example3_known_structure()
    fig3 = plot_weighting_functions(
        W3,
        grids3,
        titles=["p1 = x1 (theta)", "p2 = x2 (dtheta)"],
        suptitle="Example 3 - SNNN Weighting Functions",
    )
    fig3.savefig(output_dir / "example3_weighting.png", dpi=150, bbox_inches="tight")
    print("Saved: example3_weighting.png")

    W4, grids4, S_core4, sv4 = example4_modified_parameter_space()
    fig4 = plot_weighting_functions(
        W4,
        grids4,
        titles=["p1 = sin(x1)", "p2 = cos(x1)", "p3 = x2"],
        suptitle="Example 4 - CNO Weighting Functions",
    )
    fig4.savefig(output_dir / "example4_weighting.png", dpi=150, bbox_inches="tight")
    print("Saved: example4_weighting.png")

    W2, grids2, sv2 = example2_qlpv_extended()

    fig_sv, axes_sv = plt.subplots(1, 2, figsize=(12, 4))
    fig_sv.suptitle("Singular Values by Scheduling Dimension", fontsize=13)
    for ax, sv, title in zip(
        axes_sv, [sv3, sv4[0:1]], ["Example 3 (p1, p2)", "Example 4 (p1)"]
    ):
        for i, s in enumerate(sv if isinstance(sv, list) else [sv]):
            ax.semilogy(s, "o-", label=f"dim {i+1}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular value (log)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_sv.savefig(output_dir / "singular_values.png", dpi=150, bbox_inches="tight")
    print("Saved: singular_values.png")

    plt.show()
    print("\nDone.")
