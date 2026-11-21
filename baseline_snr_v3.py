"""Paired V3 benchmark: proposed network vs GCC-PHAT / SRP-PHAT (optional MUSIC)
on the SAME Monte Carlo trials, using the SAME fractional-delay signal model as
drone_acoustic_joint_v3.py.

New file only -- v1 baseline_snr.py, drone_acoustic_joint.py,
drone_acoustic_joint_v3.py and all existing CSVs are left untouched.

Outputs (all *_v3, written next to this script):
  paired_benchmark_v3.csv : snr, trial, gt_azi, err_proposed, err_gcc, err_srp[, err_music]
  paired_ttest_v3.csv     : per-SNR paired t-tests (proposed vs each baseline)
  baseline_snr_v3.csv     : Fig-2 classical curves (snr, gcc_mae, gcc_std, srp_mae, srp_std)
  paired_summary_v3.csv   : per-SNR proposed / MUSIC MAE+-std (for paired Fig-2 overlay)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drone_acoustic_joint_v3 as v3

HERE = os.path.dirname(os.path.abspath(__file__))
FS, C, ULA_D, ULA_M = v3.FS, v3.C, v3.ULA_D, v3.ULA_M
SNR_LIST = v3.SNR_LIST
N_FFT_TDOA = 2048                     # full-window GCC/SRP PHAT (same length as network input)
MAX_LAG = int(round(ULA_D / C * FS))  # +-3.73 samples for the adjacent ULA pair
SEPARATION = np.array([j - i for i in range(ULA_M) for j in range(i + 1, ULA_M)],
                      dtype=np.float64)   # (15,) pair separations in units of d


# ---------------- fractional-delay classical estimators ----------------
def gcc_phat_frac_azi(arr, ref_pair=(0, 1)):
    """GCC-PHAT with 3-point parabolic sub-sample interpolation of the correlation peak.
    Same PHAT cross-spectrum convention as v3.gcc_phat_ula_azi, but delay is continuous."""
    i, j = ref_pair
    n = arr.shape[1]
    Fi = np.fft.rfft(arr[i], n=n)
    Fj = np.fft.rfft(arr[j], n=n)
    G = Fi * np.conj(Fj)
    G /= (np.abs(G) + 1e-10)
    cc = np.fft.irfft(G, n=n)
    lo, hi = n // 2 - MAX_LAG, n // 2 + MAX_LAG + 1
    k = lo + int(np.argmax(np.abs(cc[lo:hi])))
    delta = 0.0
    if 0 < k < n - 1:
        y0, y1, y2 = abs(cc[k - 1]), abs(cc[k]), abs(cc[k + 1])
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-15:
            delta = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    delay = (k + delta - n // 2) / FS
    cos_t = np.clip(delay * C / ULA_D, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_t)))


def srp_phat_frac_azi(arr, n_fft=N_FFT_TDOA, n_theta=181):
    """SRP-PHAT with linear sub-sample interpolation of each pair's PHAT correlation.
    Steering lag = separation*d*cos(theta)/C (continuous), wrapped modulo n_fft;
    theta grid is 1 deg (v1 used 2 deg) with parabolic peak interpolation."""
    spec = np.fft.rfft(arr, n=n_fft, axis=1)                  # (6, n_fft/2+1)
    cross = []
    for i in range(ULA_M):
        for j in range(i + 1, ULA_M):
            G = spec[i] * np.conj(spec[j])
            G /= (np.abs(G) + 1e-10)
            cross.append(np.fft.irfft(G, n=n_fft))
    cc = np.ascontiguousarray(cross)                          # (15, n_fft)
    thetas = np.linspace(0.0, 180.0, n_theta)
    cos_t = np.cos(np.deg2rad(thetas))                        # (n_theta,)
    idx = (SEPARATION[:, None] * ULA_D / C * FS) * cos_t[None, :]
    idx = np.mod(idx, n_fft)
    i0 = np.floor(idx).astype(np.int64)
    frac = idx - i0
    rows = np.arange(len(SEPARATION))[:, None]
    vals = cc[rows, i0] * (1.0 - frac) + cc[rows, (i0 + 1) % n_fft] * frac
    power = np.abs(vals.sum(axis=0))                          # (n_theta,)
    k = int(np.argmax(power))
    est = float(thetas[k])
    if 0 < k < n_theta - 1:
        y0, y1, y2 = power[k - 1], power[k], power[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-15:
            d = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
            est = float(thetas[k] + d * (thetas[1] - thetas[0]))
    return est % 180.0


def music_ula_azi(arr, n_fft=N_FFT_TDOA):
    """Narrowband MUSIC at the ~800 Hz carrier bin (diagnostic classical subspace baseline).
    Steering a_m(f,theta) = exp(-j 2 pi f (m*d*cos(theta)/C)), m = 0..M-1 from mic0."""
    X = np.fft.rfft(arr, n=n_fft, axis=1)
    k0 = int(round(800.0 * n_fft / FS))
    k0 = min(max(k0, 1), X.shape[1] - 1)
    f0 = k0 * FS / n_fft
    x = X[:, k0]
    R = np.outer(x, np.conj(x))
    _, V = np.linalg.eigh(R)                                  # ascending eigenvalues
    En = V[:, :ULA_M - 1]                                     # noise subspace (1 source)
    thetas = np.linspace(0.0, 180.0, 361)
    tau = np.outer(np.arange(ULA_M) * ULA_D / C, np.cos(np.deg2rad(thetas)))
    A = np.exp(-2j * np.pi * f0 * tau)                        # (M, n_theta)
    proj = np.sum(np.abs(A.conj().T @ En) ** 2, axis=1)
    spectrum = 1.0 / np.maximum(proj, 1e-12)
    return float(thetas[int(np.argmax(spectrum))])


# ---------------- paired evaluation ----------------
def make_trials(snr_db, n_trials, seed):
    """Same (azi, signal, noise) realization for all three/four estimators.
    Azimuth from a local RandomState; per-mic noise scaled exactly as v3.add_noise."""
    rng = np.random.RandomState(seed)
    clean = v3.generate_drone_signal_ula(2048, FS)
    rows = []
    for t in range(n_trials):
        azi = float(rng.uniform(0.0, 180.0))
        delayed = v3.add_time_delay(clean, azi, 90.0, v3.ULA_POS, C, FS)
        noise = rng.randn(ULA_M, 2048)
        scale = np.sqrt(np.var(delayed, axis=1, keepdims=True) / (10 ** (snr_db / 10.0)))
        arr = delayed + noise * scale
        rows.append((t, azi, arr.astype(np.float32)))
    return rows


def predict_proposed(model, rows, batch_size=64):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), batch_size):
            chunk = rows[s:s + batch_size]
            x = torch.from_numpy(np.stack([r[2] for r in chunk])).float().to(v3.DEVICE)
            ang = model(x)[1]
            out.extend(ang.cpu().numpy().tolist())
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-music", action="store_true")
    args = ap.parse_args()

    print("=== Training proposed (V3 protocol, seed=42, identical to --mode simv3) ===", flush=True)
    model, _ = v3.train_synthetic_ula(mode="proposed")

    bench_rows, summ_rows, tt_rows = [], [], []
    for snr in SNR_LIST:
        rows = make_trials(snr, args.n_trials, args.seed)
        gt = np.array([r[1] for r in rows])
        pred = predict_proposed(model, rows)
        e_p = np.array([v3._circular180_err(pred[i], gt[i]) for i in range(len(rows))])
        e_g = np.array([v3._circular180_err(gcc_phat_frac_azi(r[2]), r[1]) for r in rows])
        e_s = np.array([v3._circular180_err(srp_phat_frac_azi(r[2]), r[1]) for r in rows])
        if args.no_music:
            e_m = np.full(len(rows), np.nan)
        else:
            e_m = np.array([v3._circular180_err(music_ula_azi(r[2]), r[1]) for r in rows])
        for i, r in enumerate(rows):
            d = dict(snr=snr, trial=r[0], gt_azi=r[1], err_proposed=e_p[i],
                     err_gcc=e_g[i], err_srp=e_s[i])
            if not args.no_music:
                d['err_music'] = e_m[i]
            bench_rows.append(d)
        summ_rows.append(dict(snr=snr, n=len(rows),
                              proposed_mae=float(e_p.mean()), proposed_std=float(e_p.std()),
                              gcc_mae=float(e_g.mean()), gcc_std=float(e_g.std()),
                              srp_mae=float(e_s.mean()), srp_std=float(e_s.std()),
                              music_mae=float(np.nanmean(e_m)) if not args.no_music else np.nan,
                              music_std=float(np.nanstd(e_m)) if not args.no_music else np.nan))
        for name, e_b in [("gcc", e_g), ("srp", e_s)]:
            diff = e_p - e_b
            t, p = stats.ttest_rel(e_p, e_b)
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            crit = stats.t.ppf(0.975, len(diff) - 1)
            tt_rows.append(dict(snr=snr, comparison=f"proposed_minus_{name}",
                                mean_diff=float(diff.mean()), t=float(t), p=float(p),
                                ci_low=float(diff.mean() - crit * se),
                                ci_high=float(diff.mean() + crit * se), n=len(diff)))
        print(f"[paired] SNR={snr:>3}dB  proposed={e_p.mean():7.3f}  gcc={e_g.mean():7.3f}  "
              f"srp={e_s.mean():7.3f}  music={np.nanmean(e_m):7.3f}" if not args.no_music else
              f"[paired] SNR={snr:>3}dB  proposed={e_p.mean():7.3f}  gcc={e_g.mean():7.3f}  srp={e_s.mean():7.3f}",
              flush=True)

    bench = pd.DataFrame(bench_rows)
    bench.to_csv(os.path.join(HERE, "paired_benchmark_v3.csv"), index=False)
    tt = pd.DataFrame(tt_rows)
    tt.to_csv(os.path.join(HERE, "paired_ttest_v3.csv"), index=False)
    summary = pd.DataFrame(summ_rows)
    summary.to_csv(os.path.join(HERE, "paired_summary_v3.csv"), index=False)
    # Fig-2 经典曲线：与 v1 baseline_snr.csv 完全相同的列名
    summary[["snr", "gcc_mae", "gcc_std", "srp_mae", "srp_std"]].to_csv(
        os.path.join(HERE, "baseline_snr_v3.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n=== paired t-tests (proposed - baseline, negative = proposed better) ===", flush=True)
    print(tt.to_string(index=False), flush=True)
    print("\n=== Fig-2 curves (paired samples) ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nSaved: paired_benchmark_v3.csv, paired_ttest_v3.csv, "
          "paired_summary_v3.csv, baseline_snr_v3.csv", flush=True)


if __name__ == "__main__":
    main()
