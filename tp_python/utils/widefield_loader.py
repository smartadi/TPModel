from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.interpolate

DEFAULT_SUBJECTS_ROOT = Path(r"\\sahale.biostr.washington.edu\data\Subjects")


def get_expdir(
    mn: str,
    td: str,
    en: str | int,
    subjects_root: Path | str = DEFAULT_SUBJECTS_ROOT,
) -> Path:
    """Build and validate experiment directory path."""
    expdir = Path(subjects_root) / str(mn) / str(td) / str(en)
    if not expdir.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {expdir}. "
            "Check subjects_root, mn, td, and en."
        )
    return expdir


def _load_temporal_components(
    expdir: Path,
    n_comps: int,
    movie_suffix: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load temporal SVD components (corr first, then uncorrected color path)."""
    corr_temp = expdir / "corr" / "svdTemporalComponents_corr.npy"
    corr_time = expdir / "corr" / "svdTemporalComponents_corr.timestamps.npy"
    base_temp = expdir / movie_suffix / "svdTemporalComponents.npy"
    base_time = expdir / movie_suffix / "svdTemporalComponents.timestamps.npy"

    if corr_temp.exists() and corr_time.exists():
        svd_temp = np.asarray(np.load(corr_temp))
        svd_time = np.asarray(np.load(corr_time))
        source = "corr"
    elif base_temp.exists() and base_time.exists():
        svd_temp = np.asarray(np.load(base_temp))
        svd_time = np.asarray(np.load(base_time))
        source = movie_suffix
    else:
        raise FileNotFoundError(
            "Could not find temporal SVD files in corr/ or "
            f"{movie_suffix}/ under {expdir}"
        )

    if svd_temp.ndim != 2:
        raise ValueError(f"Expected 2D temporal components, got shape {svd_temp.shape}")

    # Common format is (time, comps). Fallback handles (comps, time).
    if svd_temp.shape[1] >= n_comps:
        temp_tc = svd_temp[:, :n_comps]
    elif svd_temp.shape[0] >= n_comps:
        temp_tc = svd_temp[:n_comps, :].T
    else:
        raise ValueError(
            f"Temporal data shape {svd_temp.shape} has fewer than n_comps={n_comps} components."
        )

    svd_time = np.ravel(svd_time)
    n_valid = min(len(svd_time), temp_tc.shape[0])
    return temp_tc[:n_valid], svd_time[:n_valid], source


def load_svd_session(
    mn: str,
    td: str,
    en: str | int,
    n_comps: int = 50,
    subjects_root: Path | str = DEFAULT_SUBJECTS_ROOT,
    movie_suffix: str = "blue",
) -> dict[str, Any]:
    """
    Load widefield SVD arrays for one session.

    Returns a dict with keys:
      - expdir
      - mn, td, en
      - n_comps
      - temporal_source
      - svd_spatial: (y, x, n_comps)
      - svd_temporal: (n_frames, n_comps)
      - timestamps: (n_frames,)
      - mean_image: (y, x)
    """
    expdir = get_expdir(mn, td, en, subjects_root=subjects_root)

    spatial_path = expdir / movie_suffix / "svdSpatialComponents.npy"
    mean_path = expdir / movie_suffix / "meanImage.npy"
    if not spatial_path.exists():
        raise FileNotFoundError(f"Spatial component file not found: {spatial_path}")
    if not mean_path.exists():
        raise FileNotFoundError(f"Mean image file not found: {mean_path}")

    svd_spatial = np.asarray(np.load(spatial_path))
    if svd_spatial.ndim != 3:
        raise ValueError(
            f"Expected spatial components with shape (y, x, comps), got {svd_spatial.shape}"
        )

    n_eff = min(n_comps, svd_spatial.shape[2])
    svd_spatial = svd_spatial[:, :, :n_eff]
    mean_image = np.asarray(np.load(mean_path))

    if mean_image.shape != svd_spatial.shape[:2]:
        raise ValueError(
            f"meanImage shape {mean_image.shape} does not match spatial shape {svd_spatial.shape[:2]}"
        )

    svd_temporal, timestamps, source = _load_temporal_components(expdir, n_eff, movie_suffix)

    return {
        "expdir": expdir,
        "mn": str(mn),
        "td": str(td),
        "en": str(en),
        "n_comps": n_eff,
        "temporal_source": source,
        "svd_spatial": svd_spatial,
        "svd_temporal": svd_temporal,
        "timestamps": timestamps,
        "mean_image": mean_image,
    }


def get_svd_interpolator(session: dict[str, Any]):
    """Create t -> temporal-component interpolator from a loaded session."""
    return scipy.interpolate.interp1d(
        session["timestamps"],
        session["svd_temporal"],
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",
    )


def reconstruct_image_at_time(
    session: dict[str, Any],
    t_sec: float,
    add_mean: bool = True,
) -> np.ndarray:
    """Reconstruct one image at arbitrary time using temporal interpolation."""
    t_to_svd = get_svd_interpolator(session)
    img = session["svd_spatial"] @ t_to_svd(float(t_sec))
    if add_mean:
        img = img + session["mean_image"]
    return img


def reconstruct_time_series_images(
    session: dict[str, Any],
    frame_step: int = 1,
    max_frames: int | None = None,
    add_mean: bool = True,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct a time series image tensor.

    Returns:
      images: (n_selected_frames, y, x)
      times: (n_selected_frames,)
      frame_idx: selected source frame indices
    """
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")

    total_frames = session["svd_temporal"].shape[0]
    frame_idx = np.arange(0, total_frames, frame_step)
    if max_frames is not None:
        frame_idx = frame_idx[: max_frames]

    coeff = session["svd_temporal"][frame_idx]
    # Equivalent to per-frame (svd_spatial @ coeff_t) but vectorized.
    images = np.einsum("xyc,tc->txy", session["svd_spatial"], coeff).astype(dtype, copy=False)
    if add_mean:
        images = images + session["mean_image"][None, :, :]

    times = session["timestamps"][frame_idx]
    return images, times, frame_idx


def load_time_series_images(
    mn: str,
    td: str,
    en: str | int,
    n_comps: int = 50,
    subjects_root: Path | str = DEFAULT_SUBJECTS_ROOT,
    movie_suffix: str = "blue",
    frame_step: int = 1,
    max_frames: int | None = None,
    add_mean: bool = True,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    One-call API: load session and reconstruct image time series.

    Returns:
      images: (n_selected_frames, y, x)
      times: (n_selected_frames,)
      session: loaded session metadata/arrays
    """
    session = load_svd_session(
        mn=mn,
        td=td,
        en=en,
        n_comps=n_comps,
        subjects_root=subjects_root,
        movie_suffix=movie_suffix,
    )
    images, times, _ = reconstruct_time_series_images(
        session=session,
        frame_step=frame_step,
        max_frames=max_frames,
        add_mean=add_mean,
        dtype=dtype,
    )
    return images, times, session


def evenly_spaced_indices(n_frames: int, n_samples: int) -> np.ndarray:
    """Return n_samples evenly spaced frame indices across [0, n_frames-1]."""
    if n_frames < 1:
        return np.array([], dtype=int)
    n_samples = max(1, min(n_samples, n_frames))
    return np.linspace(0, n_frames - 1, n_samples, dtype=int)
