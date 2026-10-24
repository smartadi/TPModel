"""
Simulate the dynamic image system and save an animated GIF of image outputs.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SSmodels.dynamic_image_system import build_system, simulate


def main():
    n = 10
    m = 5
    T = 300          # simulation steps
    fps = 20         # output frame rate

    print("Building system...")
    sys_dict = build_system(n=n, m=m, decay=0.97, noise_std=0.08, seed=42)
    print(f"  State dim n={n}, Input dim m={m}, Image {sys_dict['img_size']}x{sys_dict['img_size']}")

    def u_func(k):
        u = np.zeros(m)
        if (k // 30) % 2 == 0:
            u[0] = 1.0
        if m > 1:
            u[1] = 0.5 * np.sin(2 * np.pi * k / 120)
        return u

    x0 = np.zeros(n)
    x0[0] = 1.0

    print("Simulating...")
    xs, imgs = simulate(sys_dict, T=T, x0=x0, u_func=u_func, seed=7)
    print(f"  imgs shape: {imgs.shape}")

    vmax = float(np.max(np.abs(imgs)))

    print("Building animation...")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    im = ax.imshow(imgs[0], cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
    title = ax.set_title("k=0", fontsize=13)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    def update(frame):
        im.set_data(imgs[frame])
        title.set_text(f"k={frame:03d}")
        return im, title

    ani = animation.FuncAnimation(
        fig, update, frames=T + 1, interval=1000 // fps, blit=True
    )

    out_path = Path(__file__).parent / "image_movie.gif"
    print(f"Saving GIF to {out_path}  (this may take a moment)...")
    ani.save(str(out_path), writer="pillow", fps=fps)
    print(f"Done. Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
