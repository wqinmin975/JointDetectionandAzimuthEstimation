"""Faithful numpy-only replica of the classical TDOA baselines (GCC-PHAT / SRP-PHAT)
used in drone_acoustic_joint.py, evaluated per-SNR on the SAME synthetic signal model
(6-element ULA, d=0.08 m, eq.(2) rotor signature) so the MAE can be plotted against SNR
and compared with the proposed model.

Important protocol note (data--paper consistency): the classical curves produced here are
an INDEPENDENT Monte Carlo draw with respect to the network test set of
drone_acoustic_joint.py. The two share the signal model, array geometry, delay application
and estimator code, but they do NOT share per-sample noise realizations, and the sample
counts differ (600 trials per SNR here vs. the mixed-SNR 60-trial draw used for the
classical rows of Table 1). The manuscript therefore reports these curves as an
independent Monte Carlo evaluation rather than as a paired/matched comparison.

Torch is NOT required: the baselines are classical methods. The signal model, array
geometry, delay application (np.roll with the corrected 3-D projection), and noise scaling
are copied verbatim from drone_acoustic_joint.py so the numbers are consistent with the
paper's simulation.

[Reproducibility] All randomness is drawn from local RandomState instances seeded
deterministically (see eval_baselines_snr), never from the global np.random stream.
Two independent processes running this script therefore produce bit-identical
baseline_snr.csv.

Output: baseline_snr.csv  (snr, gcc_mae, gcc_std, srp_mae, srp_std)
"""
import numpy as np

# ---- constants copied from drone_acoustic_joint.py ----
FS = 16000
C = 343.0
ULA_D = 0.08
ULA_POS = np.array([[(i - (6 - 1) / 2) * ULA_D, 0.0, 0.0] for i in range(6)], dtype=np.float64)
ULA_M = 6
SNR_LIST = [-10, -5, 0, 5, 10, 15]


def generate_drone_signal_ula(n_samples, fs):
    t = np.arange(n_samples) / fs
    s = np.sin(2 * np.pi * 800.0 * t) * (0.6 * (1.0 + 0.4 * np.sin(2 * np.pi * 120.0 * t)))
    s = s / np.std(s)
    return s.astype(np.float64)


def _src_vec_3d(azi_deg, ele_deg):
    a = np.deg2rad(azi_deg)
    e = np.deg2rad(ele_deg)
    return np.array([np.sin(e) * np.cos(a), np.sin(e) * np.sin(a), np.cos(e)], dtype=np.float64)


def add_time_delay(signal, azi_deg, ele_deg=90.0, mic_pos=None, c=C, fs=FS, ref_pos=None):
    if mic_pos is None:
        mic_pos = ULA_POS
    if ref_pos is None:
        ref_pos = mic_pos[0]
    src = _src_vec_3d(azi_deg, ele_deg)
    tau = np.dot(mic_pos - ref_pos, src) / c
    delay_samples = np.round(tau * fs).astype(int)
    out = np.zeros((mic_pos.shape[0], np.asarray(signal).shape[0]), dtype=np.float64)
    sig = np.asarray(signal, dtype=np.float64)
    for m in range(mic_pos.shape[0]):
        out[m] = np.roll(sig, int(delay_samples[m]))
    return out


def add_noise(signal, snr_db, rng=None):
    """加白高斯噪声到给定 SNR。rng 为 None 时退回全局 np.random（仅供兼容调用）；
    正式评测一律传入独立 RandomState，使结果不依赖全局随机流状态。"""
    noise = (rng if rng is not None else np.random).randn(*signal.shape)
    signal_power = np.var(signal)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(noise_power)
    return signal + noise


def gcc_phat(s1, s2, fs):
    n = len(s1)
    f1 = np.fft.rfft(s1)
    f2 = np.fft.rfft(s2)
    cross_spec = f1 * np.conj(f2)
    gcc = np.fft.irfft(cross_spec / (np.abs(cross_spec) + 1e-8))
    return gcc


def gcc_phat_ula_azi(arr, ref_pair=(0, 1)):
    i, j = ref_pair
    gcc = gcc_phat(arr[i], arr[j], FS)
    n = len(gcc)
    max_tau = ULA_D / C
    max_s = int(round(max_tau * FS))
    lo = max(0, n // 2 - max_s)
    hi = min(n, n // 2 + max_s)
    seg = np.abs(gcc[lo:hi])
    peak = lo + int(np.argmax(seg))
    delay = (peak - n // 2) / FS
    cos_t = delay * C / ULA_D
    cos_t = np.clip(cos_t, -1, 1)
    return np.rad2deg(np.arccos(cos_t))


def srp_phat_ula_azi(arr, n_fft=512, n_theta=91):
    spec = [np.fft.rfft(arr[k], n=n_fft) for k in range(ULA_M)]
    thetas = np.linspace(0, 180, n_theta)
    best = -1e18
    best_th = 90.0
    for th in thetas:
        th_r = np.deg2rad(th)
        acc = 0.0
        for i in range(ULA_M):
            for j in range(i + 1, ULA_M):
                tau = (ULA_POS[j, 0] - ULA_POS[i, 0]) * np.cos(th_r) / C
                d = int(round(tau * FS)) % n_fft
                G = spec[i] * np.conj(spec[j])
                G /= (np.abs(G) + 1e-10)
                cc = np.fft.irfft(G, n=n_fft)
                acc += cc[d]
        if acc > best:
            best = acc
            best_th = th
    return best_th


def _circular180_err(pred, gt):
    e = np.abs(np.asarray(pred) - np.asarray(gt)) % 180.0
    return np.minimum(e, 180.0 - e)


def eval_baselines_snr(snr_db, n_test=600, seed=2):
    """在固定 SNR 上评测经典基线（n_test 次独立蒙特卡洛抽样）。

    [可复现性] 方位角与噪声各用一条独立的 RandomState(seed) 派生流：方位角用 rng，
    噪声用 noise_rng（种子由 seed 确定性派生）。噪声不再取自全局 np.random 随机流，
    因此本函数的结果与调用顺序、全局随机流状态完全无关——两次独立进程调用得到
    逐位相同的结果（此前噪声走全局流，导致 Fig.2 的经典曲线不可复现）。
    """
    rng = np.random.RandomState(seed)
    noise_rng = np.random.RandomState(seed + 10_000)
    gcc_err = []
    srp_err = []
    for _ in range(n_test):
        azi = rng.uniform(0.0, 180.0)
        s = generate_drone_signal_ula(2048, FS)
        arr = add_time_delay(s, azi, 90.0, ULA_POS, C, FS)
        arr = np.stack([add_noise(arr[m], snr_db, noise_rng) for m in range(ULA_M)])
        g = gcc_phat_ula_azi(arr)
        if not np.isnan(g):
            gcc_err.append(_circular180_err(g, azi))
        srp_err.append(_circular180_err(srp_phat_ula_azi(arr), azi))
    gcc_mean = float(np.mean(gcc_err))
    gcc_std = float(np.std(gcc_err))
    srp_mean = float(np.mean(srp_err))
    srp_std = float(np.std(srp_err))
    return gcc_mean, gcc_std, srp_mean, srp_std


def main():
    rows = []
    print(f"{'SNR':>4} | {'GCC MAE':>9} | {'SRP MAE':>9}")
    print("-" * 34)
    for snr in SNR_LIST:
        gmean, gstd, smean, sstd = eval_baselines_snr(snr)
        rows.append((snr, gmean, gstd, smean, sstd))
        print(f"{snr:>4} | {gmean:>7.2f}±{gstd:<4.2f} | {smean:>7.2f}±{sstd:<4.2f}")
    with open("baseline_snr.csv", "w") as f:
        f.write("snr,gcc_mae,gcc_std,srp_mae,srp_std\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n")
    print("Saved baseline_snr.csv")


if __name__ == "__main__":
    main()
