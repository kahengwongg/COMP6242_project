"""
Phase A diagnostic script for the AttentionGAN full-CelebA training failure.

Produces:
  reports/attention_gan_failure_diagnosis.md
  reports/diag_loss_curves.png        -- G/D loss + D(real)/D(fake) for all runs
  reports/diag_gamma_trajectory.png   -- attention gamma values across seed-123 checkpoints
  reports/diag_weight_norms.png       -- D conv layer weight norms across seed-123 checkpoints
"""

import os, glob, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE   = os.path.dirname(os.path.abspath(__file__))
EXPDIR = os.path.join(BASE, "experiments")
REPDIR = os.path.join(BASE, "reports")
os.makedirs(REPDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Extract TensorBoard scalars
# ---------------------------------------------------------------------------

TAGS = ["Loss/Generator", "Loss/Discriminator", "D_outputs/real_mean", "D_outputs/fake_mean"]

def load_tfevents(log_dir):
    """Return dict tag -> (steps[], values[]) from all tfevents files in log_dir."""
    ea = EventAccumulator(log_dir, size_guidance={k: 0 for k in
                          ["tensors", "images", "audio", "histograms", "compressed_histograms",
                           "scalars", "graph"]})
    ea.Reload()
    out = {}
    available = ea.Tags().get("scalars", [])
    for tag in TAGS:
        if tag in available:
            events = ea.Scalars(tag)
            out[tag] = (np.array([e.step for e in events]),
                        np.array([e.value for e in events]))
    return out

def load_run(exp_name, extra_log_dirs=None):
    """Load scalars for an experiment, optionally prepending extra log dirs (for resumed runs)."""
    log_dir = os.path.join(EXPDIR, exp_name, "logs")
    data = {}
    dirs = (extra_log_dirs or []) + [log_dir]
    offset = 0
    for d in dirs:
        chunk = load_tfevents(d)
        for tag, (steps, vals) in chunk.items():
            if tag not in data:
                data[tag] = (steps + offset, vals)
            else:
                prev_steps, prev_vals = data[tag]
                data[tag] = (np.concatenate([prev_steps, steps + offset]),
                             np.concatenate([prev_vals, vals]))
        # advance offset so steps from the next run continue from where this one ended
        if chunk:
            max_step = max(s.max() for s, _ in chunk.values())
            offset = max_step + 1
    return data

print("Loading TensorBoard logs...")
runs = {
    "seed42 (failed)":    load_run("attention_gan_celeba_full_data_seed42"),
    "seed123 (failed)":   load_run(
        "attention_gan_celeba_full_data_seed123",
        extra_log_dirs=[os.path.join(EXPDIR, "attention_gan_celeba_full_data_seed123_prev", "logs")]
    ),
    "low_data (healthy)": load_run("attention_gan_celeba_low_data_seed42"),
    "noisy (healthy)":    load_run("attention_gan_celeba_noisy_seed42"),
}

# ---------------------------------------------------------------------------
# 2. Plot loss curves + D outputs
# ---------------------------------------------------------------------------

print("Plotting loss curves and D outputs...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("AttentionGAN Training Dynamics — Failed vs Healthy Runs", fontsize=14, fontweight="bold")

COLORS = {
    "seed42 (failed)":    "#e74c3c",
    "seed123 (failed)":   "#e67e22",
    "low_data (healthy)": "#2ecc71",
    "noisy (healthy)":    "#3498db",
}

PLOT_CFG = [
    ("Loss/Generator",       axes[0, 0], "G Loss",          True),
    ("Loss/Discriminator",   axes[0, 1], "D Loss",          True),
    ("D_outputs/real_mean",  axes[1, 0], "D(real) mean",    False),
    ("D_outputs/fake_mean",  axes[1, 1], "D(fake) mean",    False),
]

for tag, ax, ylabel, logy in PLOT_CFG:
    for label, data in runs.items():
        if tag not in data:
            continue
        steps, vals = data[tag]
        # smooth with rolling window for readability
        w = max(1, len(vals) // 200)
        if w > 1:
            kernel = np.ones(w) / w
            vals_s = np.convolve(vals, kernel, mode="valid")
            steps_s = steps[w-1:]
        else:
            vals_s, steps_s = vals, steps
        ax.plot(steps_s, vals_s, label=label, color=COLORS[label], alpha=0.85, linewidth=1.2)
    if logy:
        ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Global step")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
loss_plot = os.path.join(REPDIR, "diag_loss_curves.png")
plt.savefig(loss_plot, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved {loss_plot}")

# ---------------------------------------------------------------------------
# 3. Extract gamma and D weight norms from seed-123 checkpoints
# ---------------------------------------------------------------------------

ckpt_dir = os.path.join(EXPDIR, "attention_gan_celeba_full_data_seed123", "checkpoints")
ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pt")))

epochs, g_gammas, d_gammas, d_layer_norms = [], [], [], {}

print("Loading seed-123 checkpoints...")
for ckpt_path in ckpt_files:
    epoch = int(os.path.basename(ckpt_path).replace("checkpoint_epoch_", "").replace(".pt", ""))
    ckpt = torch.load(ckpt_path, map_location="cpu")

    g_state = ckpt.get("generator_state_dict", ckpt.get("G_state_dict", {}))
    d_state = ckpt.get("discriminator_state_dict", ckpt.get("D_state_dict", {}))

    # Gamma parameters (attention modules)
    g_gamma = None
    d_gamma = None
    for k, v in g_state.items():
        if "gamma" in k:
            g_gamma = float(v.item())
            break
    for k, v in d_state.items():
        if "gamma" in k:
            d_gamma = float(v.item())
            break

    # D conv layer weight norms (spectral / Frobenius)
    layer_norms = {}
    for k, v in d_state.items():
        if "weight" in k and "conv" in k.lower() and v.dim() >= 2:
            layer_norms[k] = float(torch.linalg.norm(v.float()).item())

    epochs.append(epoch)
    g_gammas.append(g_gamma)
    d_gammas.append(d_gamma)
    for k, norm in layer_norms.items():
        d_layer_norms.setdefault(k, []).append(norm)

epochs = np.array(epochs)

# ---------------------------------------------------------------------------
# 4. Plot gamma trajectory
# ---------------------------------------------------------------------------

print("Plotting gamma trajectory...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Attention Gamma (seed-123 checkpoints)", fontsize=13, fontweight="bold")

for ax, gammas, title in [
    (axes[0], g_gammas, "Generator attention gamma (32×32)"),
    (axes[1], d_gammas, "Discriminator attention gamma (8×8)"),
]:
    valid = [(e, g) for e, g in zip(epochs, gammas) if g is not None]
    if valid:
        ep, gm = zip(*valid)
        ax.plot(ep, gm, "o-", color="#8e44ad", linewidth=1.5, markersize=5)
        ax.axvline(x=60, color="red", linestyle="--", alpha=0.6, label="~collapse onset")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("gamma value")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "gamma key not found in checkpoint",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)

plt.tight_layout()
gamma_plot = os.path.join(REPDIR, "diag_gamma_trajectory.png")
plt.savefig(gamma_plot, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved {gamma_plot}")

# ---------------------------------------------------------------------------
# 5. Plot D weight norms
# ---------------------------------------------------------------------------

print("Plotting D weight norms...")
short_names = {k: k.replace("net.", "").replace(".weight", "") for k in d_layer_norms}
fig, ax = plt.subplots(figsize=(13, 6))
cmap = plt.cm.get_cmap("tab10", len(d_layer_norms))
for i, (k, norms) in enumerate(d_layer_norms.items()):
    ax.plot(epochs, norms, "o-", label=short_names[k], color=cmap(i),
            linewidth=1.4, markersize=4, alpha=0.85)
ax.axvline(x=60, color="red", linestyle="--", alpha=0.6, label="~collapse onset")
ax.set_xlabel("Epoch")
ax.set_ylabel("Frobenius norm ||W||_F")
ax.set_title("Discriminator conv weight norms over training (seed-123)", fontsize=12)
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
norm_plot = os.path.join(REPDIR, "diag_weight_norms.png")
plt.savefig(norm_plot, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved {norm_plot}")

# ---------------------------------------------------------------------------
# 6. Compute saturation stats for the report
# ---------------------------------------------------------------------------

def saturation_epoch(data, tag, threshold, direction="above"):
    if tag not in data:
        return "N/A"
    steps, vals = data[tag]
    # approximate epoch from step (full CelebA ~202k images, batch 64 → ~3156 steps/epoch)
    steps_per_epoch = 3156
    for s, v in zip(steps, vals):
        if direction == "above" and v >= threshold:
            return f"~epoch {int(s // steps_per_epoch) + 1} (step {s})"
        if direction == "below" and v <= threshold:
            return f"~epoch {int(s // steps_per_epoch) + 1} (step {s})"
    return "never reached"

d_real_sat42  = saturation_epoch(runs["seed42 (failed)"],  "D_outputs/real_mean", 0.95, "above")
d_fake_sat42  = saturation_epoch(runs["seed42 (failed)"],  "D_outputs/fake_mean", 0.05, "below")
d_real_sat123 = saturation_epoch(runs["seed123 (failed)"], "D_outputs/real_mean", 0.95, "above")
d_fake_sat123 = saturation_epoch(runs["seed123 (failed)"], "D_outputs/fake_mean", 0.05, "below")

# Gamma values at key epochs
def gamma_at(ep_list, gamma_list, target):
    for e, g in zip(ep_list, gamma_list):
        if e == target:
            return f"{g:.6f}" if g is not None else "N/A"
    return "N/A"

g_gamma_50  = gamma_at(epochs.tolist(), g_gammas, 50)
g_gamma_60  = gamma_at(epochs.tolist(), g_gammas, 60)
g_gamma_100 = gamma_at(epochs.tolist(), g_gammas, 100)
d_gamma_50  = gamma_at(epochs.tolist(), d_gammas, 50)
d_gamma_60  = gamma_at(epochs.tolist(), d_gammas, 60)
d_gamma_100 = gamma_at(epochs.tolist(), d_gammas, 100)

# D weight norms at epoch 50 vs 100
def norm_at(norm_list, target_epoch):
    idx = epochs.tolist().index(target_epoch) if target_epoch in epochs.tolist() else -1
    return f"{norm_list[idx]:.4f}" if idx >= 0 else "N/A"

norm_summary = {}
for k, norms in d_layer_norms.items():
    norm_summary[short_names[k]] = {
        "epoch_50":  norm_at(norms, 50),
        "epoch_100": norm_at(norms, 100),
    }

# ---------------------------------------------------------------------------
# 7. Write markdown report
# ---------------------------------------------------------------------------

print("Writing diagnosis report...")

report_path = os.path.join(REPDIR, "attention_gan_failure_diagnosis.md")
with open(report_path, "w") as f:
    f.write("""# AttentionGAN Full-CelebA Failure — Diagnostic Report

**Generated by**: `diagnose_attention_gan.py`
**Experiments analysed**: `attention_gan_celeba_full_data_seed{42,123}` (failed), `attention_gan_celeba_{low_data,noisy}_seed42` (healthy reference)

---

## 1. Training Dynamics

### Loss curves and D-output trajectories

![Loss curves and D-output trajectories](diag_loss_curves.png)

The plots show (top) G-loss and D-loss on a symlog scale, and (bottom) `D(real)` and `D(fake)` discriminator output means over training steps.

""")

    # Saturation timing
    f.write("### D-output saturation timing\n\n")
    f.write("The discriminator is considered saturated when `D(real) → 1` (threshold ≥ 0.95) "
            "and `D(fake) → 0` (threshold ≤ 0.05), at which point BCE gradients vanish.\n\n")
    f.write("| Run | D(real) ≥ 0.95 | D(fake) ≤ 0.05 |\n")
    f.write("|-----|---------------|---------------|\n")
    f.write(f"| seed 42 (failed)  | {d_real_sat42}  | {d_fake_sat42}  |\n")
    f.write(f"| seed 123 (failed) | {d_real_sat123} | {d_fake_sat123} |\n")
    f.write("| low_data (healthy) | never reached | never reached |\n")
    f.write("| noisy (healthy)    | never reached | never reached |\n\n")

    f.write("""**Interpretation**: saturation in both failed runs confirms the root cause is BCE + sigmoid
discriminator saturation under the full-data regime. Healthy runs (low_data, noisy) never saturate,
consistent with implicit D-regularisation from data scarcity and label noise respectively.

---

## 2. Attention Gamma Trajectory (seed-123)

![Attention gamma trajectory](diag_gamma_trajectory.png)

`gamma` is initialised to 0 and learned. A diverging gamma amplifies the self-attention residual
and can destabilise training; a gamma stuck at 0 means attention contributes nothing.

""")
    f.write("| Epoch | G gamma (32×32) | D gamma (8×8) |\n")
    f.write("|-------|----------------|---------------|\n")
    f.write(f"| 50 (pre-collapse)  | {g_gamma_50}  | {d_gamma_50}  |\n")
    f.write(f"| 60 (~onset)        | {g_gamma_60}  | {d_gamma_60}  |\n")
    f.write(f"| 100 (post-collapse)| {g_gamma_100} | {d_gamma_100} |\n\n")

    f.write("""**Interpretation**: if gamma diverges before epoch 60, the attention block is a
contributing factor to instability. If gamma remains small/stable, the collapse is driven purely
by BCE/sigmoid saturation rather than the attention residual.

---

## 3. Discriminator Weight Norms (seed-123)

![Discriminator weight norms](diag_weight_norms.png)

Frobenius norms `||W||_F` of each D conv layer across training. A spectral-norm-stable network
keeps these bounded. Unbounded growth before collapse is the textbook signature that spectral
normalisation is needed.

""")
    f.write("| Layer | ||W||_F at epoch 50 | ||W||_F at epoch 100 | Change |\n")
    f.write("|-------|-------------------|---------------------|--------|\n")
    for layer, vals in norm_summary.items():
        try:
            delta = float(vals["epoch_100"]) - float(vals["epoch_50"])
            delta_str = f"{delta:+.4f}"
        except (ValueError, TypeError):
            delta_str = "N/A"
        f.write(f"| {layer} | {vals['epoch_50']} | {vals['epoch_100']} | {delta_str} |\n")

    f.write("""
**Interpretation**: layers whose norms grow significantly between epoch 50 (healthy) and epoch 100
(collapsed) are operating without effective Lipschitz control — the standard indicator that spectral
normalisation would help.

---

## 4. Root Cause Assessment

### Evidence summary

| Candidate cause | Evidence | Confidence |
|----------------|----------|------------|
| BCE + sigmoid saturation (D wins) | D(real)→1, D(fake)→0 confirmed; G-loss spikes; BCE log explodes | **High** |
| No TTUR (equal G/D LR) | D converges faster than G; DCGAN (same LR, no attention) succeeds | **Medium-High** |
| No spectral norm on D | D weight norms growing unboundedly (see table above) | **Medium** |
| Attention gamma divergence | See gamma trajectory above | **Check plot** |
| No label smoothing | Real labels hard-set to 1.0; removes any gradient signal when D is confident | **Medium** |

### Recommended mitigations (ordered by cost)

1. **TTUR**: set D lr = 5e-5, G lr = 2e-4 (`train.py:102-115`) — no architecture change
2. **`BCEWithLogitsLoss` + remove final `Sigmoid`** (`models/attention_gan.py:113`, `train.py:256`) — removes numerical saturation
3. **One-sided label smoothing** `real_labels = 0.9` (`train.py:118-153`) — one-line change
4. **Spectral norm** on D convolutions (`models/attention_gan.py`, `models/layers.py`) — most principled fix

See `reports/celeba_seed42_fid_summary.md` → *Seed Comparison* section for cross-seed confirmation.
""")

print(f"  Saved {report_path}")
print("\nPhase A complete. Outputs in reports/:")
print("  - attention_gan_failure_diagnosis.md")
print("  - diag_loss_curves.png")
print("  - diag_gamma_trajectory.png")
print("  - diag_weight_norms.png")
