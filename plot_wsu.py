"""Plot the WSU empirical-validation results: ROC curves (drone vs ambient) and
classical-baseline azimuth MAE bars on real GPS-labeled field recordings.

Reads the per-frame CSVs produced by wsu_eval.py and writes wsu_roc.png.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def roc_curve(pos, neg, n=200):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None, None
    thr = np.linspace(np.min(np.concatenate([pos, neg])),
                      np.max(np.concatenate([pos, neg])), n)
    pfa = np.array([np.mean(neg > t) for t in thr])
    pd_ = np.array([np.mean(pos > t) for t in thr])
    return pfa, pd_


def auc(pfa, pd_):
    if pfa is None:
        return float("nan")
    o = np.argsort(pfa)
    return float(np.trapezoid(pd_[o], pfa[o]))


t1 = pd.read_csv(os.path.join(HERE, "wsu_perframe_test1.csv"))
t2d = pd.read_csv(os.path.join(HERE, "wsu_perframe_test2_drone.csv"))
t2n = pd.read_csv(os.path.join(HERE, "wsu_perframe_test2_nodrone.csv"))

d1 = t1[t1.label == 1]; a1 = t1[t1.label == 0]
d2 = t2d[t2d.label == 1]; a2 = t2n

# ROC curves
roc_comb_e = roc_curve(np.concatenate([d1.energy, d2.energy]),
                       np.concatenate([a1.energy, a2.energy]))
roc_t1_e = roc_curve(d1.energy, a1.energy)
roc_t2_e = roc_curve(d2.energy, a2.energy)
roc_comb_s = roc_curve(np.concatenate([d1.srp_power, d2.srp_power]),
                       np.concatenate([a1.srp_power, a2.srp_power]))
roc_t1_s = roc_curve(d1.srp_power, a1.srp_power)
roc_t2_s = roc_curve(d2.srp_power, a2.srp_power)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

ax1.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
if roc_comb_e[0] is not None:
    ax1.plot(roc_comb_e[0], roc_comb_e[1], "b-", lw=2,
             label=f"combined (energy) AUC={auc(*roc_comb_e):.2f}")
if roc_t1_e[0] is not None:
    ax1.plot(roc_t1_e[0], roc_t1_e[1], "g-", lw=2,
             label=f"test1 only (energy) AUC={auc(*roc_t1_e):.2f}")
if roc_t2_e[0] is not None:
    ax1.plot(roc_t2_e[0], roc_t2_e[1], "r-", lw=2,
             label=f"test2 only (energy) AUC={auc(*roc_t2_e):.2f}")
if roc_comb_s[0] is not None:
    ax1.plot(roc_comb_s[0], roc_comb_s[1], "b:", lw=1.5,
             label=f"combined (SRP-power) AUC={auc(*roc_comb_s):.2f}")
if roc_t1_s[0] is not None:
    ax1.plot(roc_t1_s[0], roc_t1_s[1], "g:", lw=1.2,
             label=f"test1 only (SRP-power) AUC={auc(*roc_t1_s):.2f}")
if roc_t2_s[0] is not None:
    ax1.plot(roc_t2_s[0], roc_t2_s[1], "r:", lw=1.2,
             label=f"test2 only (SRP-power) AUC={auc(*roc_t2_s):.2f}")
ax1.set_xlabel("False-alarm rate $P_{fa}$"); ax1.set_ylabel("Detection rate $P_d$")
ax1.set_title("WSU UAV detection ROC (drone vs ambient)")
ax1.legend(fontsize=7, loc="lower right"); ax1.grid(alpha=0.3)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)

# Azimuth MAE bars (folded, paper convention)
jobs = ["test1\n(site 2)", "test2_drone\n(site 1)"]
gcc = [d1.gcc_err.mean(), d2.gcc_err.mean()]
srp = [d1.srp_err.mean(), d2.srp_err.mean()]
x = np.arange(len(jobs)); w = 0.35
ax2.bar(x - w/2, gcc, w, label="GCC-PHAT", color="#4c72b0")
ax2.bar(x + w/2, srp, w, label="SRP-PHAT", color="#dd8452")
for i, (g, s) in enumerate(zip(gcc, srp)):
    ax2.text(i - w/2, g + 1, f"{g:.1f}", ha="center", fontsize=8)
    ax2.text(i + w/2, s + 1, f"{s:.1f}", ha="center", fontsize=8)
ax2.axhline(45, color="gray", ls="--", lw=1, label="Table 2 synthetic baseline ≈45°")
ax2.set_xticks(x); ax2.set_xticklabels(jobs, fontsize=9)
ax2.set_ylabel("Azimuth MAE (deg, folded to [0,180))")
ax2.set_title("Classical DOA baselines on real GPS-labeled field data")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
ax2.set_ylim(0, max(max(gcc), max(srp)) * 1.25)

fig.tight_layout()
out = os.path.join(HERE, "wsu_roc.png")
# MDPI requires >=300 dpi for raster figures. figsize=(11, 4.2) in -> 3300x1260 px.
fig.savefig(out, dpi=300)
print("wrote", out)
