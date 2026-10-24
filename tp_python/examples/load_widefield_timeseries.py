from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tp_python.utils.widefield_loader import (
    evenly_spaced_indices,
    load_time_series_images,
)

# Define only session identifiers.
mn = "AL_0046"
td = "2026-02-22"
en = "1"

# Reconstructing every frame can require very large RAM.
# Increase/decrease these values based on your machine capacity.
frame_step = 100
max_frames = 300

# Load reconstructed image time series.
# If memory is tight, use frame_step > 1 or max_frames (optional args).
images, times, session = load_time_series_images(
    mn=mn,
    td=td,
    en=en,
    frame_step=frame_step,
    max_frames=max_frames,
)

print(f"Loaded session: {session['expdir']}")
print(f"Temporal source: {session['temporal_source']}")
print(f"Images shape (T, Y, X): {images.shape}")
print(f"Time range: {times[0]:.3f}s to {times[-1]:.3f}s")

# Save a 10-image preview locally in this project.
preview_idx = evenly_spaced_indices(images.shape[0], 10)
fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
for ax, idx in zip(axes.ravel(), preview_idx):
    ax.imshow(images[idx], cmap="gray")
    ax.set_title(f"t={times[idx]:.1f}s")
    ax.axis("off")

project_root = Path(__file__).resolve().parents[2]
out_dir = project_root / "tp_python" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"timeseries_preview_{mn}_{td}_{en}.png"
fig.suptitle(f"Widefield Time Series Preview ({mn} {td} exp {en})", fontsize=12)
fig.savefig(out_path, dpi=150)
print(f"Saved preview to: {out_path}")
