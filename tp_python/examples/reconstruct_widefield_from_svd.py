"""
Load widefield SVD outputs and reconstruct image frames.

This script mirrors the MATLAB logic used around line 193 in
primary_mouseDataAnalysis.m:
    [U, V, t, mimg] = loadUVt(serverRoot, nSV)

and frame reconstruction used in input_pixel_test.m:
    Im = reshape(reshape(U, y*x, nSV) * V(:, i), [y, x]) + mimg

Usage example:
    python tp_python/examples/reconstruct_widefield_from_svd.py \
        --exp-root "C:/data/AL_0033/2025-04-20/2" \
        --nsv 500 \
        --frame 100 \
        --save-frame reconstructed_frame_100.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_v_like_matlab(v_path: Path, nsv: int) -> np.ndarray:
    """
    Load temporal components with MATLAB-compatible orientation.

    MATLAB readVfromNPY assumes file shape is (n_time, nsv_total) and returns
    V with shape (nsv, n_time) after transpose.
    """
    v_raw = np.load(v_path)
    if v_raw.ndim != 2:
        raise ValueError(f"Expected 2D temporal components at {v_path}, got shape {v_raw.shape}.")

    # Common case from MATLAB pipeline: (n_time, nsv_total)
    if v_raw.shape[1] >= nsv:
        return v_raw[:, :nsv].T

    # Fallback if file is already stored as (nsv_total, n_time)
    if v_raw.shape[0] >= nsv:
        return v_raw[:nsv, :]

    raise ValueError(
        f"Could not infer temporal component orientation for shape {v_raw.shape} with nsv={nsv}."
    )


def load_uvt(exp_root: Path, nsv: int = 500, movie_suffix: str = "blue"):
    """
    Python equivalent of MATLAB loadUVt(expRoot, nSV).

    Returns:
        U: (y, x, nsv)
        V: (nsv, n_frames)
        t: (n_frames,)
        mimg: (y, x)
        source: str indicating whether corr or uncorrected temporal data was used
    """
    u_path = exp_root / movie_suffix / "svdSpatialComponents.npy"
    mimg_path = exp_root / movie_suffix / "meanImage.npy"

    if not u_path.exists():
        raise FileNotFoundError(f"Spatial component file not found: {u_path}")
    if not mimg_path.exists():
        raise FileNotFoundError(f"Mean image file not found: {mimg_path}")

    U = np.load(u_path)
    if U.ndim != 3:
        raise ValueError(f"Expected U with shape (y, x, nsv_total), got {U.shape}")

    nsv_eff = min(nsv, U.shape[2])
    U = U[:, :, :nsv_eff]

    mimg = np.load(mimg_path)
    if mimg.shape != U.shape[:2]:
        raise ValueError(f"meanImage shape {mimg.shape} does not match U spatial shape {U.shape[:2]}")

    corr_v_path = exp_root / "corr" / "svdTemporalComponents_corr.npy"
    corr_t_path = exp_root / "corr" / "svdTemporalComponents_corr.timestamps.npy"
    blue_v_path = exp_root / movie_suffix / "svdTemporalComponents.npy"
    blue_t_path = exp_root / movie_suffix / "svdTemporalComponents.timestamps.npy"

    if corr_v_path.exists() and corr_t_path.exists():
        V = _load_v_like_matlab(corr_v_path, nsv_eff)
        t = np.load(corr_t_path)
        source = "corr"
    else:
        V = _load_v_like_matlab(blue_v_path, nsv_eff)
        t = np.load(blue_t_path)
        source = movie_suffix

    if t.ndim != 1:
        t = np.ravel(t)

    # Align lengths if there is an extra timestamp or frame.
    n_frames = min(V.shape[1], t.shape[0])
    V = V[:, :n_frames]
    t = t[:n_frames]

    return U, V, t, mimg, source


def reconstruct_frame(U: np.ndarray, V: np.ndarray, mimg: np.ndarray, frame_idx: int, add_mean: bool = True):
    """Reconstruct a single frame with MATLAB-compatible reshape ordering."""
    ypix, xpix, nsv = U.shape
    if not (0 <= frame_idx < V.shape[1]):
        raise IndexError(f"frame_idx={frame_idx} is out of range [0, {V.shape[1] - 1}]")

    # MATLAB uses column-major memory order for reshape.
    U_flat = U.reshape(ypix * xpix, nsv, order="F")
    frame = (U_flat @ V[:, frame_idx]).reshape(ypix, xpix, order="F")

    if add_mean:
        frame = frame + mimg
    return frame


def reconstruct_stack(
    U: np.ndarray,
    V: np.ndarray,
    mimg: np.ndarray,
    start: int,
    stop: int,
    step: int,
    add_mean: bool = True,
):
    """Reconstruct a frame stack in [start:stop:step]."""
    ypix, xpix, nsv = U.shape
    U_flat = U.reshape(ypix * xpix, nsv, order="F")

    frame_idx = np.arange(start, stop, step)
    if frame_idx.size == 0:
        raise ValueError("No frames selected. Check start/stop/step arguments.")

    block = U_flat @ V[:, frame_idx]
    stack = block.reshape(ypix, xpix, frame_idx.size, order="F").transpose(2, 0, 1)

    if add_mean:
        stack = stack + mimg[None, :, :]
    return stack, frame_idx


def parse_args():
    parser = argparse.ArgumentParser(description="Load widefield SVD and reconstruct image frames.")
    parser.add_argument("--exp-root", type=Path, required=True, help="Experiment root directory (serverRoot).")
    parser.add_argument("--nsv", type=int, default=500, help="Number of SVD components to use.")
    parser.add_argument("--movie-suffix", type=str, default="blue", help="Color folder name, usually 'blue'.")
    parser.add_argument("--frame", type=int, default=0, help="Single frame index to reconstruct.")
    parser.add_argument("--save-frame", type=Path, default=None, help="Path to save reconstructed frame (.png/.tif/.npy).")
    parser.add_argument("--save-stack", type=Path, default=None, help="Path to save stack as .npy.")
    parser.add_argument("--start", type=int, default=0, help="Start frame for stack reconstruction.")
    parser.add_argument("--stop", type=int, default=200, help="Stop frame (exclusive) for stack reconstruction.")
    parser.add_argument("--step", type=int, default=1, help="Step for stack reconstruction.")
    parser.add_argument("--no-mean", action="store_true", help="Reconstruct without adding meanImage.")
    parser.add_argument("--show", action="store_true", help="Show the reconstructed single frame.")
    return parser.parse_args()


def main():
    args = parse_args()

    U, V, t, mimg, source = load_uvt(args.exp_root, nsv=args.nsv, movie_suffix=args.movie_suffix)

    print(f"Loaded U shape: {U.shape}")
    print(f"Loaded V shape: {V.shape}")
    print(f"Loaded t shape: {t.shape}")
    print(f"Temporal source: {source}")

    add_mean = not args.no_mean
    frame = reconstruct_frame(U, V, mimg, args.frame, add_mean=add_mean)
    print(f"Reconstructed frame {args.frame} with shape: {frame.shape}")

    if args.save_frame is not None:
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        if args.save_frame.suffix.lower() == ".npy":
            np.save(args.save_frame, frame)
        else:
            plt.imsave(args.save_frame, frame, cmap="gray")
        print(f"Saved frame to: {args.save_frame}")

    if args.save_stack is not None:
        stack, frame_idx = reconstruct_stack(
            U,
            V,
            mimg,
            start=args.start,
            stop=min(args.stop, V.shape[1]),
            step=args.step,
            add_mean=add_mean,
        )
        args.save_stack.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_stack, stack)
        print(f"Saved stack to: {args.save_stack} with shape {stack.shape}")
        print(f"Frame indices: {frame_idx[0]}..{frame_idx[-1]} (step={args.step})")

    if args.show:
        plt.figure(figsize=(6, 6))
        plt.imshow(frame, cmap="gray")
        plt.title(f"Reconstructed frame {args.frame}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
