"""
Standalone loader for OSPulse ECoG sessions (no gdar dependency).

Raw files
---------
One .mat file per channel, 96 channels total:
    {base_path}/{date}/{monkey}{date}_ParameterStudy_AllBlocks_{site}_Ch{ch}.mat

Each .mat file contains:
    DataAll          : (n_conditions, n_trials, n_samples_raw)  raw ECoG
    StimFreqs        : (n_conditions, 1)  stimulation frequency per condition (Hz)
    StimPulseWidths  : (n_conditions, 1)  pulse width per condition

Bad channels
------------
Optional pickle file at:
    {base_path}/{monkey}/{date}/bad_channels.pkl

Structure: { site: [ch1, ch2, ...] }  — channel numbers are 1-indexed.
Matching gdar convention: converted to 0-indexed and dropped from the tensor.

Original sampling rate: 3051.76 Hz.
Default ds_factor=3 → decimated fs ≈ 1017 Hz.

Usage
-----
from tp_python.utils.ospulse_loader import load_ospulse_session

session = load_ospulse_session(
    monkey   = "Jalapeno",
    date     = "091114",
    site     = "M1",
    base_path= r"Z:\\Yazdan Lab Members\\Grad\\brianam2\\ParameterStudy\\Data",
)

session["ecog"]          # (n_cond, n_trials, n_good_ch, n_samples)  float32
session["mean"]          # (n_cond, n_good_ch, n_samples)
session["good_channels"] # sorted list of 0-indexed good channel numbers
session["bad_channels"]  # sorted list of 0-indexed bad channel numbers
session["n_channels"]    # int — number of channels after bad-channel removal
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import scipy.signal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CHANNELS_RAW = 96       # fixed by the recording hardware
FS_RAW         = 3051.76  # hardware sampling rate (Hz)
DS_FACTOR      = 3        # default decimation factor → fs ≈ 1017 Hz


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_bad_channels(
    bad_ch_file: str | Path,
    site: str,
    base_path: Path | None = None,
    monkey: str = "",
    date: str = "",
) -> list[int]:
    """
    Load bad channel indices (0-indexed) from a bad_channels pickle.

    bad_ch_file can be:
      - an absolute path (used as-is)
      - a plain filename (resolved relative to base_path/monkey/date/)

    Returns an empty list if the file does not exist or the site key is absent.
    """
    p = Path(bad_ch_file)
    if not p.is_absolute():
        p = base_path / monkey / date / p

    if not p.exists():
        return []

    with open(p, "rb") as f:
        bad_dict = pickle.load(f)

    raw = bad_dict.get(site, [])
    # stored as 1-indexed → convert to 0-indexed
    return sorted(int(ch) - 1 for ch in raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ospulse_session(
    monkey:           str,
    date:             str,
    site:             str,
    base_path:        str | Path,
    sep_stim_params:  str  = "1",
    bad_ch_file:      str | None = "bad_channels.pkl",
    ds_factor:        int  = DS_FACTOR,
    remove_dc:        bool = True,
    verbose:          bool = True,
) -> dict[str, Any]:
    """
    Load one OSPulse ECoG session directly from .mat files.

    Parameters
    ----------
    monkey : str
        Animal name, e.g. "Jalapeno".
    date : str
        Session date string, e.g. "090414".
    site : str
        Recording site, e.g. "M1" or "S1".
    base_path : path-like
        Root of the ParameterStudy data directory.
    sep_stim_params : str
        Value used in the SeparateStimParams-{N} subdirectory, default "1".
    bad_ch_file : str or None
        Filename of the bad channels pickle inside
        {base_path}/{monkey}/{date}/. Pass None to skip bad-channel removal.
    ds_factor : int
        Decimation factor applied to the time axis (default 3).
        Set to 1 to keep the raw 3051.76 Hz rate.
    remove_dc : bool
        If True, subtract the temporal mean from each trial independently
        (matches gdar's remove_mean=True). Default True.
    verbose : bool
        Print per-channel progress.

    Returns
    -------
    dict with keys:
        monkey, date, site
        fs              : float — sampling rate after decimation (Hz)
        ecog            : ndarray (n_cond, n_trials, n_channels, n_samples) float32
        mean            : ndarray (n_cond, n_channels, n_samples) float32
        stim_freq       : ndarray (n_cond,) float  — Hz
        stim_pw         : ndarray (n_cond,) float
        n_conditions    : int
        n_trials        : int
        n_channels      : int  — good channels only
        n_samples       : int
        good_channels   : list[int]  — 0-indexed channel indices kept
        bad_channels    : list[int]  — 0-indexed channel indices removed
        conditions      : list[dict]  — one dict per condition

    Each entry in conditions[i] has:
        index     : int
        stim_freq : float
        stim_pw   : float
        ecog      : (n_trials, n_channels, n_samples) float32
        mean      : (n_channels, n_samples) float32
    """
    base_path = Path(base_path)
    fs = FS_RAW / ds_factor if ds_factor >= 1 else FS_RAW

    # --- bad channels --------------------------------------------------------
    bad_channels: list[int] = []
    if bad_ch_file is not None:
        bad_channels = _load_bad_channels(
            bad_ch_file, site,
            base_path=base_path, monkey=monkey, date=date,
        )
        if verbose and bad_channels:
            print(f"Bad channels (0-indexed): {bad_channels}")
        elif verbose:
            print("No bad channels found.")

    bad_set = set(bad_channels)
    good_channels = [ch for ch in range(N_CHANNELS_RAW) if ch not in bad_set]
    n_good = len(good_channels)

    # --- load each channel file ----------------------------------------------
    ecog_tensor = None
    stim_freq   = None
    stim_pw     = None

    for ch in range(1, N_CHANNELS_RAW + 1):
        ch_idx = ch - 1   # 0-indexed

        mat_path = (
            base_path / monkey / date /
            f"SeparateStimParams-{sep_stim_params}" / "AllBlocks" /
            f"{monkey}{date}_ParameterStudy_allBlocks_{site}_CH{ch}.mat"
        )
        if not mat_path.exists():
            raise FileNotFoundError(
                f"Channel file not found: {mat_path}\n"
                "Check monkey, date, site, and base_path."
            )

        mat     = scipy.io.loadmat(str(mat_path))
        ch_data = mat["DataAll"]   # (n_cond, n_trials, n_samples_raw)

        # allocate tensor on first channel (using full channel count initially)
        if ch == 1:
            stim_freq = mat["StimFreqs"].squeeze().astype(float)
            stim_pw   = mat["StimPulseWidths"].squeeze().astype(float)

            n_cond, n_trials, n_raw = ch_data.shape
            n_samples = n_raw // ds_factor if ds_factor > 1 else n_raw
            ecog_tensor = np.zeros(
                (n_cond, n_trials, N_CHANNELS_RAW, n_samples), dtype=np.float32
            )

        if ds_factor > 1:
            ch_data = scipy.signal.decimate(
                ch_data.astype(np.float64), q=ds_factor, axis=-1
            ).astype(np.float32)

        if remove_dc:
            ch_data = ch_data - ch_data.mean(axis=-1, keepdims=True)

        ecog_tensor[:, :, ch_idx, :] = ch_data

        if verbose:
            print(f"\r  loaded ch {ch:2d}/{N_CHANNELS_RAW}", end="", flush=True)

    if verbose:
        print()

    # --- drop bad channels (axis 2) ------------------------------------------
    if bad_channels:
        ecog_tensor = ecog_tensor[:, :, good_channels, :]

    # ecog_tensor is now (n_cond, n_trials, n_good, n_samples)

    # --- trial average -------------------------------------------------------
    trial_mean = ecog_tensor.mean(axis=1)   # (n_cond, n_good, n_samples)

    # --- per-condition list --------------------------------------------------
    conditions = [
        {
            "index":     i,
            "stim_freq": float(stim_freq[i]),
            "stim_pw":   float(stim_pw[i]),
            "ecog":      ecog_tensor[i],   # (n_trials, n_good, n_samples)
            "mean":      trial_mean[i],    # (n_good, n_samples)
        }
        for i in range(n_cond)
    ]

    if verbose:
        print(f"Session: {monkey} {date} {site}")
        print(f"  raw channels : {N_CHANNELS_RAW}")
        print(f"  bad channels : {len(bad_channels)}  {bad_channels}")
        print(f"  good channels: {n_good}")
        print(f"  ecog shape   : {ecog_tensor.shape}  (cond × trials × ch × samples)")
        print(f"  fs           : {fs:.2f} Hz")
        print(f"  stim conditions:")
        for c in conditions:
            print(f"    [{c['index']:2d}]  {c['stim_freq']:6.1f} Hz  "
                  f"{c['stim_pw']:8.4f}")

    return {
        "monkey":        monkey,
        "date":          date,
        "site":          site,
        "fs":            fs,
        "ecog":          ecog_tensor,
        "mean":          trial_mean,
        "stim_freq":     stim_freq,
        "stim_pw":       stim_pw,
        "n_conditions":  n_cond,
        "n_trials":      n_trials,
        "n_channels":    n_good,
        "n_samples":     n_samples,
        "good_channels": good_channels,
        "bad_channels":  bad_channels,
        "conditions":    conditions,
    }
