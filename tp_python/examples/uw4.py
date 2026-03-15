import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tptoolbox.hull import genhull


def sigmoid(v, v0=6.0, r=0.56, e0=2.5):
    """Sigmoid firing response used in Jansen-Rit model."""
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


def ring_coupling(n_regions):
    """Create a ring-lattice coupling matrix with nearest neighbors."""
    W = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        W[i, (i - 1) % n_regions] = 1.0
        W[i, (i + 1) % n_regions] = 1.0
    return W / np.sum(W, axis=1, keepdims=True)


def jansen_rit_network_rhs(t, y, params, W, kappa):
    """Coupled Jansen-Rit network dynamics for N regions.

    State y is shape (N, 6) flattened.
    """
    n_regions = W.shape[0]
    y = y.reshape(n_regions, 6)

    v1 = y[:, 0]
    v2 = y[:, 1]
    v3 = y[:, 2]
    v4 = y[:, 3]
    v5 = y[:, 4]
    v6 = y[:, 5]

    A = params["A"]
    B = params["B"]
    a = params["a"]
    b = params["b"]
    C1 = params["C1"]
    C2 = params["C2"]
    C3 = params["C3"]
    C4 = params["C4"]

    pyramidal = v2 - v3
    coupled_drive = kappa * (W @ sigmoid(pyramidal))

    s1 = sigmoid(v2 - v3)
    s2 = sigmoid(C1 * v1)
    s3 = sigmoid(C3 * v1)

    dv1 = v4
    dv2 = v5
    dv3 = v6

    dv4 = A * a * s1 - 2.0 * a * v4 - a * a * v1
    dv5 = A * a * (params["p_in"] + coupled_drive + C2 * s2) - 2.0 * a * v5 - a * a * v2
    dv6 = B * b * (C4 * s3) - 2.0 * b * v6 - b * b * v3

    dy = np.column_stack([dv1, dv2, dv3, dv4, dv5, dv6])
    return dy.reshape(-1)


def simulate_network(
    n_regions=10,
    t_end=5.0,
    dt=1e-3,
    p_in=120.0,
    kappa=15.0,
    seed=1,
):
    """Simulate a coupled Jansen-Rit network with ring coupling."""
    rng = np.random.default_rng(seed)

    params = {
        "A": 3.25,
        "B": 22.0,
        "a": 100.0,
        "b": 50.0,
        "C": 135.0,
    }
    params["C1"] = params["C"]
    params["C2"] = 0.8 * params["C"]
    params["C3"] = 0.25 * params["C"]
    params["C4"] = 0.25 * params["C"]
    params["p_in"] = p_in

    W = ring_coupling(n_regions)

    t = np.arange(0.0, t_end + dt, dt)
    y = np.zeros((t.size, n_regions * 6))

    for k in range(t.size - 1):
        params["p_in"] = p_in + 5.0 * rng.standard_normal()
        k1 = jansen_rit_network_rhs(t[k], y[k], params, W, kappa)
        k2 = jansen_rit_network_rhs(t[k] + 0.5 * dt, y[k] + 0.5 * dt * k1, params, W, kappa)
        k3 = jansen_rit_network_rhs(t[k] + 0.5 * dt, y[k] + 0.5 * dt * k2, params, W, kappa)
        k4 = jansen_rit_network_rhs(t[k] + dt, y[k] + dt * k3, params, W, kappa)
        y[k + 1] = y[k] + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    y = y.reshape(t.size, n_regions, 6)
    pyramidal = y[:, :, 1] - y[:, :, 2]
    mean_pyramidal = np.mean(pyramidal, axis=1)
    return t, pyramidal, mean_pyramidal


def main():
    t, pyramidal, mean_pyramidal = simulate_network()

    F = pyramidal

    R = 3
    U, D, Vt = np.linalg.svd(F, full_matrices=False)
    print("Singular values:", D)

    Uc = U[:, :R]
    Usnnn = genhull(Uc, "snnn")
    Ucno = genhull(Uc, "cno")
    Uirno = genhull(Uc, "irno")

    Ssnnn = np.linalg.pinv(Usnnn) @ F
    Scno = np.linalg.pinv(Ucno) @ F
    Sirno = np.linalg.pinv(Uirno) @ F

    Esnnn = np.linalg.norm(F - Usnnn @ Ssnnn)
    Ecno = np.linalg.norm(F - Ucno @ Scno)
    Eirno = np.linalg.norm(F - Uirno @ Sirno)

    print("Esnnn:", Esnnn)
    print("Ecno:", Ecno)
    print("Eirno:", Eirno)

    print("\nVertices (Ssnnn):")
    print(Ssnnn)
    print("\nVertices (Scno):")
    print(Scno)
    print("\nVertices (Sirno):")
    print(Sirno)

    plt.figure(figsize=(10, 4))
    plt.plot(t, mean_pyramidal, color="tab:blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean pyramidal potential (mV)")
    plt.title("10-Region Jansen-Rit Network (Ring Coupling)")
    plt.grid(True, alpha=0.3)

    plt.figure()
    plt.plot(t, Usnnn, linewidth=2)
    plt.grid(True)
    plt.title("SNNN")

    plt.figure()
    plt.plot(t, Ucno, linewidth=2)
    plt.grid(True)
    plt.title("CNO")

    plt.figure()
    plt.plot(t, Uirno, linewidth=2)
    plt.grid(True)
    plt.title("IRNO")

    plt.show()


if __name__ == "__main__":
    main()
