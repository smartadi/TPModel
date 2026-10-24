from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

mn = 'AL_0046'; td = '2026-02-22'; en = '1'
n_comps = 50

expdir = Path(r'\\sahale.biostr.washington.edu\data\Subjects') / mn / td / en
if not expdir.exists():
	raise FileNotFoundError(
		f"Experiment directory not found: {expdir}. "
		"Update mn/td/en or base path to a valid dataset location."
	)

corr_temp = expdir / 'corr/svdTemporalComponents_corr.npy'
corr_time = expdir / 'corr/svdTemporalComponents_corr.timestamps.npy'
blue_temp = expdir / 'blue/svdTemporalComponents.npy'
blue_time = expdir / 'blue/svdTemporalComponents.timestamps.npy'

if corr_temp.exists() and corr_time.exists():
	svdTemp = np.squeeze(np.load(corr_temp))
	svdTime = np.squeeze(np.load(corr_time))
	temp_source = 'corr'
elif blue_temp.exists() and blue_time.exists():
	svdTemp = np.squeeze(np.load(blue_temp))
	svdTime = np.squeeze(np.load(blue_time))
	temp_source = 'blue'
else:
	raise FileNotFoundError(
		'Could not find temporal SVD files in either corr/ or blue/ under ' + str(expdir)
	)

svdSpat = np.squeeze(np.load(expdir / 'blue/svdSpatialComponents.npy'))[:, :, :n_comps]
meanImage = np.squeeze(np.load(expdir / 'blue/meanImage.npy'))

# there is a bug that sometimes there is an extra timestamp entry
minTs = np.min((len(svdTime), svdTemp.shape[0]))
t_to_svd = scipy.interpolate.interp1d(svdTime[:minTs], svdTemp[:minTs, :n_comps], axis=0)

# example - get image at time t
t_sec = 300
im = svdSpat @ t_to_svd(t_sec)
print(f"Loaded SVD data from: {expdir}")
print(f"Temporal source: {temp_source}")
print(f"Reconstructed image shape: {im.shape}")

# Plot 10 reconstructed images at evenly spaced timestamps.
n_plot = 10
frame_idx = np.linspace(0, minTs - 1, n_plot).astype(int)
fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)

for ax, idx in zip(axes.ravel(), frame_idx):
	t_curr = float(svdTime[idx])
	im_curr = svdSpat @ t_to_svd(t_curr)
	ax.imshow(im_curr + meanImage, cmap='gray')
	ax.set_title(f"t={t_curr:.1f}s")
	ax.axis('off')

project_root = Path(__file__).resolve().parents[2]
local_out_dir = project_root / 'tp_python' / 'plots'
local_out_dir.mkdir(parents=True, exist_ok=True)
out_path = local_out_dir / f'widefield_10_evenly_spaced_{mn}_{td}_{en}.png'
fig.suptitle(f"Widefield reconstruction ({mn} {td} exp {en})", fontsize=12)
fig.savefig(out_path, dpi=150)
print(f"Saved 10-image montage to: {out_path}")

