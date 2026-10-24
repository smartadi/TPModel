"""
ARX identification + TP model decomposition for OSPulse ECoG sessions.

Protocol
--------
Each trial has three phases (defined by the condition's stim parameters):
  1s silence  →  5s pulse train (freq, pulse-width from condition)  →  silence to end

State space model (ARX)
-----------------------
  y[k+1] = A y[k] + B u[k]

  y[k] ∈ R^{n_ch}   — ECoG across channels at time k
  u[k] ∈ R^1        — stim pulse (0 or 1)
  A    ∈ R^{n_ch × n_ch}
  B    ∈ R^{n_ch × 1}

Fit: per-condition ARX — 80 % of trials within a single condition are used to
fit a condition-specific (A_c, B_c).  Currently only condition 0 is processed.

TP model
--------
Applied to A_c (n_ch × n_ch) with hull type snnn.

Usage
-----
Set the parameters below, then run:
    python tp_python/examples/fit_tp_ospulse.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tp_python.utils.ospulse_loader import load_ospulse_session, FS_RAW
from tp_python.tptoolbox.model import ConvexTPStructure

# =============================================================================
# USER PARAMETERS
# =============================================================================

monkey    = "Jalapeno"
date      = "091114"
site      = "M1"
base_path = r"Z:\Yazdan Lab Members\Grad\brianam2\ParameterStudy\Data"

ds_factor  = 3        # decimation; fs = FS_RAW / ds_factor ≈ 1017 Hz
train_frac = 0.80     # fraction of trials used for fitting

# Stim protocol timing (seconds)
PRE_S  = 1.0          # silence before stim train
STIM_S = 5.0          # duration of pulse-train window
# remaining samples after PRE_S + STIM_S are silence (post period)

# stim_pw in the .mat file is assumed to be in seconds
# (matches the user description "1 sec pulse at 0.1 Hz" for cond 0)
# adjust pw_unit_scale if your data stores it in ms (1e-3) or µs (1e-6)
pw_unit_scale = 1.0   # 1.0 = seconds, 1e-3 = ms, 1e-6 = µs

hull_types = ["snnn"]

# HOSVD truncation tolerance; raise (e.g. 1e-4) to keep fewer vertices
hosvd_tol  = 1e-4

# =============================================================================
# HELPERS
# =============================================================================

def make_stim_input(
    n_samples: int,
    fs: float,
    stim_freq: float,   # Hz
    stim_pw: float,     # seconds (after pw_unit_scale applied)
) -> np.ndarray:
    """
    Build a binary stim input signal of length n_samples.

    Layout:
        [0, PRE_S * fs)         → 0  (silence)
        [PRE_S * fs, end_stim)  → pulse train
        [end_stim, n_samples)   → 0  (silence)

    The pulse train repeats at stim_freq Hz with each pulse lasting stim_pw s.
    """
    u = np.zeros(n_samples, dtype=np.float32)

    pre_n   = int(round(PRE_S * fs))
    stim_n  = min(int(round(STIM_S * fs)), n_samples - pre_n)

    if stim_freq <= 0 or stim_pw <= 0 or stim_n <= 0:
        return u

    period_n = int(round(fs / stim_freq))   # samples per cycle
    pw_n     = max(1, int(round(stim_pw * fs)))  # pulse width in samples

    for onset in range(0, stim_n, period_n):
        off = min(onset + pw_n, stim_n)
        u[pre_n + onset : pre_n + off] = 1.0

    return u


# =============================================================================
# LOAD SESSION
# =============================================================================

print(f"Loading {monkey} {date} {site} ...")
_bad_ch_path = PROJECT_ROOT / "tp_python" / "bad_channels" / f"{monkey}_{date}.pkl"

session = load_ospulse_session(
    monkey          = monkey,
    date            = date,
    site            = site,
    base_path       = base_path,
    sep_stim_params = "1",
    bad_ch_file     = _bad_ch_path if _bad_ch_path.exists() else None,
    ds_factor       = ds_factor,
    remove_dc       = True,
    verbose         = True,
)

fs           = session["fs"]
n_conditions = session["n_conditions"]
n_trials     = session["n_trials"]
n_channels   = session["n_channels"]
n_samples    = session["n_samples"]
conditions   = session["conditions"]

n_train = int(train_frac * n_trials)
n_test  = n_trials - n_train

print(f"\nfs={fs:.2f} Hz   n_conditions={n_conditions}   "
      f"n_trials={n_trials}   n_channels={n_channels}   n_samples={n_samples}")
print(f"Bad channels removed : {session['bad_channels']}")
print(f"Train trials: {n_train}   Test trials: {n_test}")

# =============================================================================
# BUILD STIM INPUT SIGNALS
# =============================================================================
# inputs[c] : (n_trials, n_samples)  float32  — stim signal for condition c

inputs = []
for c, cond in enumerate(conditions):
    freq = cond["stim_freq"]                        # Hz
    pw   = cond["stim_pw"] * pw_unit_scale          # → seconds
    u_template = make_stim_input(n_samples, fs, freq, pw)
    inputs.append(np.tile(u_template, (n_trials, 1)))  # (n_trials, n_samples)

    print(f"  cond {c:2d}: freq={freq:.2f} Hz  pw={pw:.4f} s  "
          f"pulse_samples={int(round(pw*fs))}  "
          f"stim_on_fraction={u_template.mean():.3f}")

# =============================================================================
# TRAIN / TEST SPLIT  (first 80 % of trials = train, last 20 % = test)
# =============================================================================

train_idx = np.arange(n_train)
test_idx  = np.arange(n_train, n_trials)

# =============================================================================
# FIT ARX PER CONDITION — currently only condition 0
# =============================================================================
# For each condition c:
#   1. Build training pairs from its trials only
#   2. Solve  [Y_prev | U_prev] @ M ≈ Y_next  via least squares
#   3. Extract A_c, B_c

FIT_CONDITIONS = [0]   # expand to range(n_conditions) to fit all 12

arx_results: dict[int, dict] = {}   # c → {"A", "B", "fit_err", "spec_rad"}

for c in FIT_CONDITIONS:
    cond    = conditions[c]
    ecog_c  = cond["ecog"]    # (n_trials, n_channels, n_samples)
    input_c = inputs[c]       # (n_trials, n_samples)

    Y_prev_list: list[np.ndarray] = []
    U_prev_list: list[np.ndarray] = []
    Y_next_list: list[np.ndarray] = []

    for t in train_idx:
        y = ecog_c[t].T.astype(np.float64)          # (n_samples, n_channels)
        u = input_c[t, :, None].astype(np.float64)  # (n_samples, 1)
        Y_prev_list.append(y[:-1])
        U_prev_list.append(u[:-1])
        Y_next_list.append(y[1:])

    Y_prev = np.vstack(Y_prev_list)   # (N, n_channels)
    U_prev = np.vstack(U_prev_list)   # (N, 1)
    Y_next = np.vstack(Y_next_list)   # (N, n_channels)
    N = Y_prev.shape[0]

    Z = np.hstack([Y_prev, U_prev])
    print(f"\n[cond {c}] Solving ARX: Z {Z.shape} → Y_next {Y_next.shape}  "
          f"({n_train} trials × {n_samples-1} steps = {N} pairs)")

    M, _, _, _ = np.linalg.lstsq(Z, Y_next, rcond=None)

    A_c = M[:n_channels, :].T    # (n_channels, n_channels)
    B_c = M[n_channels:, :].T    # (n_channels, 1)

    fit_err  = np.sqrt(np.mean((Y_next - Z @ M) ** 2))
    spec_rad = np.max(np.abs(np.linalg.eigvals(A_c)))
    print(f"  ARX fit RMS error : {fit_err:.6f}")
    print(f"  A spectral radius : {spec_rad:.4f}")
    print(f"  A shape: {A_c.shape}   B shape: {B_c.shape}")

    arx_results[c] = {"A": A_c, "B": B_c, "fit_err": fit_err, "spec_rad": spec_rad}

# convenience alias for the single-condition case currently being run
c_fit = FIT_CONDITIONS[0]
A = arx_results[c_fit]["A"]
B = arx_results[c_fit]["B"]

# =============================================================================
# TP MODEL TRANSFORM ON A
# =============================================================================

out_dir = PROJECT_ROOT / "tp_python" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
tag = f"{monkey}_{date}_{site}_cond{c_fit}"

results: dict[str, tuple] = {}   # ht -> (Sco, Uco, sv)

for ht in hull_types:
    print(f"\n--- Hull: {ht} ---")
    try:
        Sco, Uco, sv, _, _ = ConvexTPStructure(A, [ht, ht], tol=hosvd_tol)
        results[ht] = (Sco, Uco, sv)
        print(f"  Core shape : {Sco.shape}")
        for dim_idx, (w, s) in enumerate(zip(Uco, sv)):
            if w is None:
                continue
            print(f"  dim {dim_idx+1} : {w.shape[1]} vertices  "
                  f"top-5 sv = {s[:5].round(4)}")
    except Exception as exc:
        print(f"  FAILED: {exc}")

if not results:
    raise SystemExit("All hull types failed.")

# =============================================================================
# PLOT 1 — TP Singular Values
# =============================================================================

fig_sv, axes_sv = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
fig_sv.suptitle(f"TP Singular Values of A — {monkey} {date} {site}  cond {c_fit}", fontsize=11)

for ax, dim_idx, dlabel in zip(axes_sv, [0, 1],
                                ["dim 1 (output ch)", "dim 2 (input ch)"]):
    for ht, (_, Uco, sv) in results.items():
        if sv[dim_idx] is not None:
            ax.semilogy(sv[dim_idx], "o-", linewidth=2, label=ht)
    ax.axhline(hosvd_tol, color="gray", linestyle=":", linewidth=1,
               label=f"tol={hosvd_tol}")
    ax.set_title(dlabel)
    ax.set_xlabel("index")
    ax.set_ylabel("singular value (log)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

path = out_dir / f"tp_sv_{tag}.png"
fig_sv.savefig(path, dpi=150)
print(f"\nSaved singular values → {path}")

# =============================================================================
# PLOT 2 — Vertex activations on test trials (3 random 5-min segments)
# =============================================================================
# For each test trial (any condition), the TP weighting functions are:
#   alpha[k] = y[k] @ W0   (n_ch,) @ (n_ch, r0) → (r0,)
#   beta[k]  = y[k] @ W1   (n_ch,) @ (n_ch, r1) → (r1,)
# Marginal activations integrated over r1/r0:
#   act_r[k] = sum_s Sco[r,s] * alpha[k,r] * beta[k,s]
#   act_s[k] = sum_r Sco[r,s] * alpha[k,r] * beta[k,s]

ht_style = {"snnn": "-"}
cmap_v   = plt.cm.tab10

# show test trials only from the fitted condition(s)
sample_conditions = FIT_CONDITIONS

for dim_key, act_key, dim_label in [
    ("r0", "act_r", "dim 1 — output channels"),
    ("r1", "act_s", "dim 2 — input channels"),
]:
    n_panels = len(sample_conditions)
    fig_act, axes_act = plt.subplots(
        1, n_panels, figsize=(6 * n_panels, 4),
        sharey=True, constrained_layout=True,
        squeeze=False,
    )
    axes_act = axes_act[0]   # shape (n_panels,)
    fig_act.suptitle(
        f"Vertex activations on test trials — {dim_label}\n"
        f"hull = snnn   ({monkey} {date} {site})",
        fontsize=11,
    )

    for col, c_idx in enumerate(sample_conditions):
        cond      = conditions[c_idx]
        test_trial = test_idx[0]                      # first held-out trial
        y_test = cond["ecog"][test_trial].T           # (n_samples, n_channels)
        u_test = inputs[c_idx][test_trial]            # (n_samples,)
        t_s    = np.arange(n_samples) / fs            # time in seconds

        ax = axes_act[col]
        ax.set_title(
            f"cond {c_idx}  {cond['stim_freq']:.1f} Hz / "
            f"{cond['stim_pw']*pw_unit_scale:.3f} s",
            fontsize=9,
        )

        # shade stim-on periods
        stim_on = u_test.astype(bool)
        if stim_on.any():
            ax.fill_between(t_s, 0, 1, where=stim_on,
                            transform=ax.get_xaxis_transform(),
                            color="orange", alpha=0.25, label="stim on")

        max_verts = 0
        for ht, (Sco, Uco, sv) in results.items():
            W0 = Uco[0]
            W1 = Uco[1]
            if W0 is None or W1 is None:
                continue

            alpha   = y_test @ W0                               # (T, r0)
            beta    = y_test @ W1                               # (T, r1)
            contrib = np.einsum("rs,tr,ts->rst", Sco, alpha, beta)  # (r0, r1, T)
            act_r   = contrib.sum(axis=1).T                    # (T, r0)
            act_s   = contrib.sum(axis=0).T                    # (T, r1)

            act = act_r if act_key == "act_r" else act_s
            n_v = act.shape[1]
            max_verts = max(max_verts, n_v)
            ls  = ht_style.get(ht, "-")

            for v in range(n_v):
                label = f"{ht} v{v+1}" if col == 0 else None
                ax.plot(t_s, act[:, v],
                        linestyle=ls, linewidth=0.8,
                        color=cmap_v(v % 10), label=label, alpha=0.85)

        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.2)
        if col == 0:
            ax.set_ylabel("vertex activation")
            ax.legend(fontsize=7, ncol=max(1, max_verts))

    dim_tag = "0" if act_key == "act_r" else "1"
    path_act = out_dir / f"tp_vertex_test_{tag}_dim{dim_tag}.png"
    fig_act.savefig(path_act, dpi=150)
    print(f"Saved vertex activations dim{dim_tag} → {path_act}")

plt.show()
print("\nDone.")
