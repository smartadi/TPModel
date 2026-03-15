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


def jansen_rit_rhs(t, y, p):
    """Jansen-Rit neural mass model dynamics.

    State y = [v1, v2, v3, v4, v5, v6]
      v1, v2, v3: PSPs (pyramidal, excitatory, inhibitory)
      v4, v5, v6: derivatives of v1, v2, v3
    """
    v1, v2, v3, v4, v5, v6 = y

    A = p["A"]
    B = p["B"]
    a = p["a"]
    b = p["b"]
    C = p["C"]
    C1 = p["C1"]
    C2 = p["C2"]
    C3 = p["C3"]
    C4 = p["C4"]
    p_in = p["p_in"]

    s1 = sigmoid(v2 - v3)
    s2 = sigmoid(C1 * v1)
    s3 = sigmoid(C3 * v1)

    dv1 = v4
    dv2 = v5
    dv3 = v6

    dv4 = A * a * s1 - 2.0 * a * v4 - a * a * v1
    dv5 = A * a * (p_in + C2 * s2) - 2.0 * a * v5 - a * a * v2
    dv6 = B * b * (C4 * s3) - 2.0 * b * v6 - b * b * v3

    return np.array([dv1, dv2, dv3, dv4, dv5, dv6])


def simulate_jansen_rit(
    t_end=5.0,
    dt=1e-3,
    p_in=120.0,
    seed=1,
):
    """Simulate Jansen-Rit model with small stochastic input variations."""
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

    t = np.arange(0.0, t_end + dt, dt)
    y = np.zeros((t.size, 6))

    for k in range(t.size - 1):
        params["p_in"] = p_in + 5.0 * rng.standard_normal()
        k1 = jansen_rit_rhs(t[k], y[k], params)
        k2 = jansen_rit_rhs(t[k] + 0.5 * dt, y[k] + 0.5 * dt * k1, params)
        k3 = jansen_rit_rhs(t[k] + 0.5 * dt, y[k] + 0.5 * dt * k2, params)
        k4 = jansen_rit_rhs(t[k] + dt, y[k] + dt * k3, params)
        y[k + 1] = y[k] + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    v1 = y[:, 0]
    v2 = y[:, 1]
    v3 = y[:, 2]
    e_pop = v2
    i_pop = v3
    pyramidal = v2 - v3
    return t, e_pop, i_pop, pyramidal


def main():
    t, e_pop, i_pop, pyramidal = simulate_jansen_rit()

    F = np.column_stack([e_pop, i_pop])

    R = 2
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

    plt.figure(figsize=(10, 6))
    plt.plot(t, e_pop, label="Excitatory PSP (v2)")
    plt.plot(t, i_pop, label="Inhibitory PSP (v3)")
    plt.xlabel("Time (s)")
    plt.ylabel("Potential (mV)")
    plt.title("Jansen-Rit Neural Mass: E/I Populations")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.figure(figsize=(10, 4))
    plt.plot(t, pyramidal, color="tab:purple")
    plt.xlabel("Time (s)")
    plt.ylabel("Potential (mV)")
    plt.title("Pyramidal Output (v2 - v3)")
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
