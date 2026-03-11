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
    R = 2
    g = np.linspace(omega[0], omega[1], H)

    F = np.vstack([f(p) for p in g])

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

    plt.show()


if __name__ == "__main__":
    main()
