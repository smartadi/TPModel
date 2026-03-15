import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tptoolbox.hull import genhull


def oscillating_state(t, n_states=10):
    """Ten-dimensional oscillating state vector with variable frequencies."""
    freqs = np.linspace(0.7, 4.8, n_states)
    amps = 1.0 / (1.0 + 0.2 * np.arange(n_states))
    return amps * np.sin(freqs * t)


def output_from_state(x):
    """One-dimensional output from the 10D state."""
    weights = np.linspace(1.0, 0.3, x.size)
    return float(weights @ x)


def main():
    H = 500
    omega = np.array([0, 5 * np.pi])
    R = 4
    g = np.linspace(omega[0], omega[1], H)

    X = np.vstack([oscillating_state(t, n_states=10) for t in g])
    y = np.array([output_from_state(x) for x in X])

    F = X

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

    plt.figure()
    plt.plot(g, Usnnn, linewidth=3)
    plt.grid(True)
    plt.title("SNNN")

    plt.figure()
    plt.plot(g, Ucno, linewidth=3)
    plt.grid(True)
    plt.title("CNO")

    plt.figure()
    plt.plot(g, Uirno, linewidth=3)
    plt.grid(True)
    plt.title("IRNO")

    plt.figure()
    plt.plot(g, y, linewidth=2)
    plt.grid(True)
    plt.title("Output y(t)")

    plt.show()


if __name__ == "__main__":
    main()
