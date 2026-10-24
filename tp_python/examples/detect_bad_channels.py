"""
Interactive bad channel detector for OSPulse ECoG sessions.

Plots the trial-averaged response of condition 0 as a spatial trace map,
one trace per electrode at its physical (x, y) position.
Click any trace to toggle it as bad (red). Click again to un-mark.
Press Enter or close the window to save and exit.

Bad channels are saved to:
    {base_path}/{monkey}/{date}/bad_channels.pkl

Format: { site: [ch1, ch2, ...] }  — 1-indexed, matching gdar convention.
If the file already exists it is updated (existing bad channels for OTHER sites
are preserved).

Usage
-----
Set the parameters below, then run:
    python tp_python/examples/detect_bad_channels.py
"""

from __future__ import annotations

import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tp_python.utils.ospulse_loader import load_ospulse_session

# =============================================================================
# USER PARAMETERS
# =============================================================================

monkey       = "Jalapeno"
date         = "091114"
site         = "M1"
base_path    = r"Z:\Yazdan Lab Members\Grad\brianam2\ParameterStudy\Data"
electrode_pos_file = r"Z:\Yazdan Lab Members\Grad\fschwock\8 Reach\electrode_positions.pkl"

ds_factor    = 3      # keep consistent with the rest of the pipeline
condition    = 0      # which condition to display (0 = first)

# visual scaling — tune if traces are too large/small
trace_width_frac  = 0.80   # fraction of inter-electrode x spacing used by each trace
trace_height_frac = 0.35   # fraction of inter-electrode y spacing used by amplitude

# click tolerance: fraction of inter-electrode spacing;
# a click within this radius of an electrode centre is accepted
click_tol_frac = 0.6

# =============================================================================
# HELPERS
# =============================================================================

def median_spacing(vals: np.ndarray) -> float:
    unique = np.unique(np.round(vals, 6))
    if unique.size < 2:
        return 1.0
    diffs = np.diff(unique)
    return float(np.median(diffs[diffs > 0])) if diffs[diffs > 0].size else 1.0


def load_electrode_positions(path: str) -> dict[int, tuple[float, float]]:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {int(k): (float(np.asarray(v).ravel()[0]),
                     float(np.asarray(v).ravel()[1]))
            for k, v in raw.items()}


BAD_CH_DIR = PROJECT_ROOT / "tp_python" / "bad_channels"


def _local_pkl_path(monkey: str, date: str) -> Path:
    return BAD_CH_DIR / f"{monkey}_{date}.pkl"


def save_bad_channels(
    monkey: str,
    date: str,
    site: str,
    bad_ch_1indexed: list[int],
) -> Path:
    """Write (or update) the local bad_channels pickle for this session."""
    pkl_path = _local_pkl_path(monkey, date)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, list] = {}
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            existing = pickle.load(f)

    existing[site] = sorted(bad_ch_1indexed)

    with open(pkl_path, "wb") as f:
        pickle.dump(existing, f)

    return pkl_path


# =============================================================================
# CHECK FOR EXISTING BAD CHANNELS — skip if already done for this session
# =============================================================================

_pkl_path = _local_pkl_path(monkey, date)
if _pkl_path.exists():
    with open(_pkl_path, "rb") as _f:
        _existing = pickle.load(_f)
    if site in _existing:
        print(f"Bad channels already saved for {monkey} {date} {site}:")
        print(f"  {_existing[site]}  (1-indexed)")
        print(f"  File: {_pkl_path}")
        print("Delete the site key from the file or remove it entirely to re-run.")
        raise SystemExit(0)

# =============================================================================
# LOAD DATA  (no bad-channel removal yet — we're building the list)
# =============================================================================

print(f"Loading {monkey} {date} {site} (condition {condition}) ...")
session = load_ospulse_session(
    monkey          = monkey,
    date            = date,
    site            = site,
    base_path       = base_path,
    sep_stim_params = "1",
    bad_ch_file     = None,   # intentionally skip — we're creating the list
    ds_factor       = ds_factor,
    remove_dc       = True,
    verbose         = True,
)

fs         = session["fs"]
n_samples  = session["n_samples"]
# trial-averaged response for the chosen condition: (96, n_samples)
mean_trace = session["conditions"][condition]["mean"]
stim_freq  = session["conditions"][condition]["stim_freq"]
stim_pw    = session["conditions"][condition]["stim_pw"]
t_axis     = np.arange(n_samples) / fs   # seconds

print(f"Condition {condition}: {stim_freq:.2f} Hz  pw={stim_pw:.4f}")

# =============================================================================
# LOAD ELECTRODE POSITIONS
# =============================================================================

elec_pos = load_electrode_positions(electrode_pos_file)

# keep only channels that exist in both the data and the position file
valid_channels = sorted(
    ch for ch in elec_pos
    if 0 <= ch < mean_trace.shape[0]
)
positions = np.array([elec_pos[ch] for ch in valid_channels])   # (N, 2)

x_spacing = median_spacing(positions[:, 0])
y_spacing = median_spacing(positions[:, 1])
click_tol = click_tol_frac * min(x_spacing, y_spacing)

trace_half_w = 0.5 * trace_width_frac  * x_spacing
trace_half_h = 0.5 * trace_height_frac * y_spacing

# normalise time axis to [-trace_half_w, +trace_half_w]
t_norm = (t_axis - t_axis.min()) / (t_axis.max() - t_axis.min())
t_norm = t_norm * 2 * trace_half_w - trace_half_w

# global amplitude scale (robust)
amp_scale = np.nanpercentile(np.abs(mean_trace[valid_channels, :]), 99)
if amp_scale == 0 or not np.isfinite(amp_scale):
    amp_scale = 1.0

# =============================================================================
# BUILD INTERACTIVE FIGURE
# =============================================================================

bad_set: set[int] = set()   # 0-indexed channel numbers marked as bad

fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

# --- draw one trace per electrode -------------------------------------------
trace_lines: dict[int, plt.Line2D] = {}   # ch (0-indexed) -> Line2D
dot_markers: dict[int, plt.Line2D] = {}
label_texts: dict[int, plt.Text]   = {}

COLOR_GOOD = "#4fc3f7"   # light blue
COLOR_BAD  = "#ef5350"   # red

for ch in valid_channels:
    x0, y0 = elec_pos[ch]
    trace   = mean_trace[ch]
    y_trace = y0 + (trace / amp_scale) * trace_half_h
    x_trace = x0 + t_norm

    line, = ax.plot(x_trace, y_trace,
                    color=COLOR_GOOD, linewidth=0.9, alpha=0.85,
                    picker=False)
    dot,  = ax.plot([x0], [y0], "o",
                    color=COLOR_GOOD, markersize=5, zorder=4,
                    picker=False)
    lbl   = ax.text(x0, y0 - 0.42 * y_spacing, str(ch),
                    ha="center", va="top", fontsize=6,
                    color=COLOR_GOOD, zorder=5)

    trace_lines[ch] = line
    dot_markers[ch] = dot
    label_texts[ch] = lbl

# --- title and instructions --------------------------------------------------
title_text = ax.set_title(
    f"{monkey}  {date}  {site}  |  condition {condition}: "
    f"{stim_freq:.1f} Hz  pw={stim_pw:.4f}\n"
    "Click electrode to toggle bad  |  Press Enter to save and exit",
    fontsize=10, color="white", pad=10,
)

status_text = fig.text(
    0.5, 0.01,
    "Bad channels: none",
    ha="center", va="bottom", fontsize=9,
    color=COLOR_BAD,
    transform=fig.transFigure,
)

ax.set_xlabel("Electrode X (mm)", color="white", fontsize=9)
ax.set_ylabel("Electrode Y (mm)", color="white", fontsize=9)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.set_aspect("equal", adjustable="datalim")
ax.grid(True, alpha=0.12, color="white")

# =============================================================================
# INTERACTION
# =============================================================================

def _update_channel(ch: int) -> None:
    """Redraw a single channel in its current good/bad colour."""
    color = COLOR_BAD if ch in bad_set else COLOR_GOOD
    trace_lines[ch].set_color(color)
    trace_lines[ch].set_linewidth(1.5 if ch in bad_set else 0.9)
    dot_markers[ch].set_color(color)
    dot_markers[ch].set_markersize(7 if ch in bad_set else 5)
    label_texts[ch].set_color(color)


def _refresh_status() -> None:
    bad_sorted = sorted(bad_set)
    bad_1idx   = [ch + 1 for ch in bad_sorted]
    if bad_1idx:
        status_text.set_text(f"Bad channels (1-indexed): {bad_1idx}")
    else:
        status_text.set_text("Bad channels: none")
    fig.canvas.draw_idle()


def on_click(event):
    if event.inaxes is not ax:
        return
    if event.xdata is None or event.ydata is None:
        return

    # find nearest electrode to click
    click_xy = np.array([event.xdata, event.ydata])
    dists    = np.linalg.norm(positions - click_xy, axis=1)
    nearest  = int(np.argmin(dists))
    ch       = valid_channels[nearest]

    if dists[nearest] > click_tol:
        return   # click was not close enough to any electrode

    if ch in bad_set:
        bad_set.discard(ch)
    else:
        bad_set.add(ch)

    _update_channel(ch)
    _refresh_status()


def on_key(event):
    if event.key in ("enter", "q"):
        _save_and_close()


def _save_and_close():
    bad_1indexed = sorted(ch + 1 for ch in bad_set)
    pkl_path = save_bad_channels(
        monkey          = monkey,
        date            = date,
        site            = site,
        bad_ch_1indexed = bad_1indexed,
    )
    print(f"\nSaved {len(bad_1indexed)} bad channel(s) → {pkl_path}")
    print(f"  {site}: {bad_1indexed}  (1-indexed)")
    plt.close(fig)


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event",    on_key)
fig.canvas.mpl_connect("close_event",        lambda e: _save_and_close())

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()
