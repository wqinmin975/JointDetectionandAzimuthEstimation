"""Empirical validation of the classical TDOA baselines (GCC-PHAT, SRP-PHAT) and an
energy-test detector on the Western Sydney University UAV Acoustic Localization Dataset
(24-microphone 3D array, 48 kHz, GPS-synced ground truth).

This is a TORCH-FREE classical evaluation: it does NOT use the paper's trained network
(which would require retraining on this array). It validates the SAME classical baselines
the paper compares against, on genuine field recordings, and produces a REAL receiver
operating characteristic (drone vs. ambient) -- something the IDMT corpus could not provide
because it contains no background-only negative samples.

Ground-truth azimuth is reproduced exactly from the dataset's own pipeline
(code/dataset_creation/create_dataset.py): UTM-projected GPS -> local azimuth, with the
session's `initial_azimuth_deg` alignment. The UTM projection constant cancels in the
azimuth (az_offset = initial_azimuth - azimuth(ref->first point)), so a local ENU gives
identical azimuths without pyproj.

Outputs (written next to this script):
  wsu_perframe_<job>.csv   per-frame: energy, gcc_az, srp_az, csv_az, errs, dist
  wsu_roc.csv              combined energy-based Pd/Pfa sweep + AUC
  wsu_summary.csv          aggregate azimuth MAE (GCC / SRP) +- std and ROC AUC
"""
import os, json, argparse
import numpy as np
import pandas as pd
from wave import open as wave_open
from scipy.signal import butter, filtfilt

R_EARTH = 6378137.0
C = 343.0
SR = 48000
CHUNK = int(0.1 * SR)          # 4800 samples = 0.1 s, matches dataset frame
N_FFT = 2048
LOWCUT, HIGHCUT = 200.0, 4000.0   # spatial-aliasing-limited band for this array


def enu(lat, lon, lat0, lon0):
    """Local east-north meters (projection-constant cancels in azimuth)."""
    lat0r, lon0r = np.radians(lat0), np.radians(lon0)
    x = np.radians(lon - lon0) * np.cos(lat0r) * R_EARTH
    y = np.radians(lat - lat0) * R_EARTH
    return x, y


def wrap(a):
    return ((a + 180.0) % 360.0) - 180.0


def circ180_err(a, b):
    a = ((a % 180.0) + 180.0) % 180.0
    b = ((b % 180.0) + 180.0) % 180.0
    e = abs(a - b)
    return min(e, 180.0 - e)


def load_cfg(path, segment):
    d = os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        p = json.load(f)
    def res(x):
        if x is None or os.path.isabs(x):
            return x
        cand = os.path.normpath(os.path.join(d, x))
        if os.path.exists(cand):
            return cand
        # fallback: some files were stored with URL-encoded brackets [ ] -> %5b %5d
        enc = x.replace("[", "%5b").replace("]", "%5d")
        cand2 = os.path.normpath(os.path.join(d, enc))
        if os.path.exists(cand2):
            return cand2
        return cand
    cfg = dict(sample_rate_hz=p.get("sample_rate_hz", SR),
               corrections_samples=p.get("corrections_samples", [0, 0, 0, 0]),
               wavs_nosync=[res(w) for w in p.get("wavs_nosync", [])],
               ref_csv=res(p.get("ref_csv")), mic_array=p.get("mic_array"),
               session=p.get("session", ""))
    if "segments" in p:
        seg = {s["name"]: s for s in p["segments"]}[segment]
        cfg.update(dict(segment_name=segment, skip_seconds=seg["skip_seconds"],
                        start_index_csv=seg.get("start_index_csv", 0),
                        initial_azimuth=seg.get("initial_azimuth_deg", 0.0),
                        initial_elevation=seg.get("initial_elevation_deg", 0.0),
                        flight_csv=res(seg.get("flight_csv")),
                        use_csv=seg.get("flight_csv") is not None,
                        skip_seconds_end=seg.get("skip_seconds_end")))
    else:
        cfg.update(dict(segment_name=p.get("flight_id", "flight"), skip_seconds=p.get("skip_seconds", 0.0),
                        start_index_csv=p.get("start_index_csv", 0),
                        initial_azimuth=p.get("initial_azimuth_deg", 0.0),
                        initial_elevation=p.get("initial_elevation_deg", 0.0),
                        flight_csv=res(p.get("flight_csv")), use_csv=p.get("flight_csv") is not None))
    return cfg


def get_mic_positions(mic_array):
    a = mic_array["a_"]; a2 = mic_array["a2_"]; h = mic_array["h_"]; r = mic_array["r_"]
    # NOTE: order must match io_utils.initialize_beamforming_params exactly:
    # mics 1-3 r[0]h[0]a ; 4-6 r[1]h[1]a ; 7-9 r[2]h[2]a ; 10-12 r[3]h[3]a ;
    # 13-15 r[4]h[4]a ; 16-18 r[5]h[5]a ; 19-24 r[2]h[2]a2
    pos = []
    for k in range(6):
        for ang in a:
            pos.append([r[k] * np.cos(np.radians(ang)), r[k] * np.sin(np.radians(ang)), h[k]])
    for ang in a2:
        pos.append([r[2] * np.cos(np.radians(ang)), r[2] * np.sin(np.radians(ang)), h[2]])
    return np.array(pos, dtype=np.float64)


def ground_truth_az(cfg):
    """Return (csv_az_array, csv_el_array, n_frames_with_gt)."""
    if not cfg["use_csv"]:
        return np.array([]), np.array([]), 0
    ref = pd.read_csv(cfg["ref_csv"], low_memory=False)
    ref_lat = ref["latitude"].dropna().astype(float).mean()
    ref_lon = ref["longitude"].dropna().astype(float).mean()
    cols = ["latitude", "longitude", "altitude_above_seaLevel(feet)",
            "time(millisecond)", " xSpeed(mph)", " ySpeed(mph)", " zSpeed(mph)"]
    f = pd.read_csv(cfg["flight_csv"], low_memory=False)
    f = f.iloc[cfg["start_index_csv"]:].reset_index(drop=True)
    f = f[cols].dropna()
    f["alt_m"] = f["altitude_above_seaLevel(feet)"] * 0.3048
    init_alt = ref["altitude_above_seaLevel(feet)"].iloc[0] * 0.3048
    rx, ry = enu(ref_lat, ref_lon, ref_lat, ref_lon)
    Xm, Ym = enu(f["longitude"].values, f["latitude"].values, ref_lat, ref_lon)
    init_az = np.degrees(np.arctan2(Ym[0] - ry, Xm[0] - rx))
    az_off = cfg["initial_azimuth"] - init_az
    el_off = cfg["initial_elevation"]
    az, el = [], []
    for k in range(len(f)):
        x, y = Xm[k], Ym[k]
        a = -wrap(np.degrees(np.arctan2(y - ry, x - rx)) + az_off)
        horiz = np.hypot(x - rx, y - ry)
        e = np.degrees(np.arctan2(f["alt_m"].iloc[k] - init_alt, horiz)) + el_off
        az.append(a); el.append(e)
    return np.array(az), np.array(el), len(f)


def process(job, nframes_cap=None):
    cfg = load_cfg(job["config"], job.get("segment"))
    mics = get_mic_positions(cfg["mic_array"])
    M = mics.shape[0]
    pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
    P = len(pairs)
    mi = np.array([mics[i] for i, _ in pairs])      # (P,3)
    mj = np.array([mics[j] for _, j in pairs])
    mdiff = mi - mj                                  # (P,3)
    csv_az, csv_el, n_gt = ground_truth_az(cfg)
    skip = int(cfg["skip_seconds"] * SR)
    corr = cfg["corrections_samples"]
    wavs = [wave_open(w, "rb") for w in cfg["wavs_nosync"]]
    for i, w in enumerate(wavs):
        w.setpos(skip + (corr[i] if i < len(corr) else 0))
    # SRP direction grid (azimuth x elevation)
    az_grid = np.arange(0, 360, 2.0)
    el_grid = np.arange(0, 62, 4.0)
    dirs = np.array([[np.cos(np.radians(e)) * np.cos(np.radians(a)),
                      np.cos(np.radians(e)) * np.sin(np.radians(a)),
                      np.sin(np.radians(e))]
                     for a in az_grid for e in el_grid], dtype=np.float64)   # (Ndir,3)
    Ndir = dirs.shape[0]
    # precompute SRP lag indices (P, Ndir)
    dots = mdiff @ dirs.T                          # (P, Ndir)
    delays = np.round(dots / C * SR).astype(np.int64) % N_FFT   # (P, Ndir)
    b_a, b_h = butter(4, [LOWCUT / (0.5 * SR), HIGHCUT / (0.5 * SR)], btype="band")
    rows = []
    fidx = 0
    while True:
        if nframes_cap is not None and fidx >= nframes_cap:
            break
        bufs = []
        ok = True
        for w in wavs:
            data = w.readframes(CHUNK)
            if len(data) < CHUNK * 4 * 6:   # 6 ch * 4 bytes
                ok = False
                break
            bufs.append(np.frombuffer(data, dtype=np.int32).reshape(-1, 6).astype(np.float64))
        if not ok or len(bufs) < len(wavs):
            break
        x = np.hstack(bufs)                     # (CHUNK, 24)
        x = filtfilt(b_a, b_h, x, axis=0)
        x = x - x.mean(0, keepdims=True)
        # PHAT cross-spectra
        X = np.fft.rfft(x, n=N_FFT, axis=0)     # (N_FFT/2+1, 24)
        Xi = X[:, [p[0] for p in pairs]]
        Xj = X[:, [p[1] for p in pairs]]
        G = Xi * np.conj(Xj)                     # (F, P)
        G = G / (np.abs(G) + 1e-10)
        G = np.ascontiguousarray(G.T)            # (P, F)
        # GCC-PHAT cross-correlation (circular, N_FFT lags)
        gcc = np.fft.irfft(G, n=N_FFT, axis=1)   # (P, N_FFT)
        # SRP-PHAT: steer each pair's cross-correlation to candidate direction d
        gv = gcc[np.arange(P)[:, None], delays]  # (P, Ndir), lag-indexed by `delays`
        srp = np.abs(gv.sum(0))                  # (Ndir,)
        bi = int(np.argmax(srp))
        est_az = az_grid[bi // len(el_grid)]
        est_el = el_grid[bi % len(el_grid)]
        # GCC-PHAT: per-pair TDOA peak -> LSQ direction
        peak = np.argmax(np.abs(gcc), axis=1)    # (P,)
        lag = np.where(peak > N_FFT // 2, peak - N_FFT, peak)
        tau = lag.astype(np.float64) / SR        # (P,)
        A = mdiff; bvec = C * tau                # (P,3),(P,)
        d, *_ = np.linalg.lstsq(A, bvec, rcond=None)
        nrm = np.linalg.norm(d)
        if nrm > 1e-9:
            d = d / nrm
        gcc_az = np.degrees(np.arctan2(d[1], d[0]))
        # energy statistic
        energy = float(np.sum(x ** 2))
        has_gt = fidx < n_gt
        ca = csv_az[fidx] if has_gt else np.nan
        ce = csv_el[fidx] if has_gt else np.nan
        gerr = circ180_err(gcc_az, ca) if has_gt else np.nan
        serr = circ180_err(est_az, ca) if has_gt else np.nan
        rows.append(dict(frame=fidx, audio_time_s=cfg["skip_seconds"] + fidx * 0.1,
                         label=1 if has_gt else 0, energy=energy,
                         srp_power=float(srp.max()),
                         gcc_az=gcc_az, srp_az=est_az, srp_el=est_el,
                         csv_az=ca, csv_el=ce, gcc_err=gerr, srp_err=serr))
        fidx += 1
    for w in wavs:
        w.close()
    return pd.DataFrame(rows), cfg


def combine(outdir):
    """Build the REAL ROC (drone vs. ambient) and aggregate azimuth MAE from the
    three per-frame CSVs produced by --job test1 / test2_drone / test2_nodrone."""
    jobs = ["test1", "test2_drone", "test2_nodrone"]
    dfs = {}
    for j in jobs:
        p = os.path.join(outdir, f"wsu_perframe_{j}.csv")
        if not os.path.exists(p):
            print(f"MISSING {p} -- run --job {j} first")
            return
        dfs[j] = pd.read_csv(p)
    # Azimuth MAE (real field data, frames with GPS ground truth)
    summ_rows = []
    for j in jobs:
        d = dfs[j]
        drone = d[d.label == 1]
        row = dict(job=j, n_frames=len(d), n_drone=len(drone),
                   gcc_mae=np.nan, gcc_std=np.nan, srp_mae=np.nan, srp_std=np.nan)
        if len(drone):
            row.update(gcc_mae=float(drone.gcc_err.mean()), gcc_std=float(drone.gcc_err.std()),
                       srp_mae=float(drone.srp_err.mean()), srp_std=float(drone.srp_err.std()))
        summ_rows.append(row)
    summary = pd.DataFrame(summ_rows)
    # Combined ROC: drone = test1(drone) + test2_drone ; ambient = test1(tail) + test2_nodrone
    drone_parts = [dfs["test1"][dfs["test1"].label == 1], dfs["test2_drone"][dfs["test2_drone"].label == 1]]
    amb_parts = [dfs["test1"][dfs["test1"].label == 0], dfs["test2_nodrone"]]
    drone_df = pd.concat(drone_parts, ignore_index=True)
    amb_df = pd.concat(amb_parts, ignore_index=True)
    roc_energy, auc_e = roc_from_energy(drone_df["energy"].values, amb_df["energy"].values)
    roc_srp, auc_s = roc_from_energy(drone_df["srp_power"].values, amb_df["srp_power"].values)
    roc = pd.DataFrame(dict(
        threshold_energy=roc_energy["threshold"], Pd_energy=roc_energy["Pd"], Pfa_energy=roc_energy["Pfa"],
        threshold_srp=roc_srp["threshold"], Pd_srp=roc_srp["Pd"], Pfa_srp=roc_srp["Pfa"]))
    out_roc = os.path.join(outdir, "wsu_roc.csv")
    out_sum = os.path.join(outdir, "wsu_summary.csv")
    out_rocs = os.path.join(outdir, "wsu_roc_summary.csv")
    roc.to_csv(out_roc, index=False)
    summary.to_csv(out_sum, index=False)
    # Per-scope ROC AUCs (honest, sites separated)
    roc_rows = []
    # combined
    roc_rows.append(dict(scope="combined", n_drone=len(drone_df), n_ambient=len(amb_df),
                         energy_AUC=auc_e, srp_power_AUC=auc_s))
    # test1 only: drone(test1) vs tail(test1)
    d1 = dfs["test1"][dfs["test1"].label == 1]
    a1 = dfs["test1"][dfs["test1"].label == 0]
    if len(d1) and len(a1):
        re, ae = roc_from_energy(d1["energy"].values, a1["energy"].values)
        rs, as_ = roc_from_energy(d1["srp_power"].values, a1["srp_power"].values)
        roc_rows.append(dict(scope="test1_only", n_drone=len(d1), n_ambient=len(a1),
                             energy_AUC=ae, srp_power_AUC=as_))
    # test2 only: drone(test2) vs nodrone(test2)
    d2 = dfs["test2_drone"][dfs["test2_drone"].label == 1]
    a2 = dfs["test2_nodrone"]
    if len(d2) and len(a2):
        re, ae = roc_from_energy(d2["energy"].values, a2["energy"].values)
        rs, as_ = roc_from_energy(d2["srp_power"].values, a2["srp_power"].values)
        roc_rows.append(dict(scope="test2_only", n_drone=len(d2), n_ambient=len(a2),
                             energy_AUC=ae, srp_power_AUC=as_))
    pd.DataFrame(roc_rows).to_csv(out_rocs, index=False)
    print("=== Azimuth MAE (real field recordings, GPS ground truth) ===")
    for _, r in summary.iterrows():
        g = f"{r.gcc_mae:.2f}±{r.gcc_std:.2f}" if len(dfs[r.job][dfs[r.job].label == 1]) else "n/a"
        s = f"{r.srp_mae:.2f}±{r.srp_std:.2f}" if len(dfs[r.job][dfs[r.job].label == 1]) else "n/a"
        print(f"  {r.job:12s} n_drone={int(r.n_drone):5d}  GCC={g:>12s}  SRP={s:>12s}")
    print(f"=== ROC (drone vs ambient) ===")
    for rr in roc_rows:
        print(f"  {rr['scope']:12s} n_drone={int(rr['n_drone']):5d} n_amb={int(rr['n_ambient']):5d} "
              f"energy_AUC={rr['energy_AUC']:.4f}  srp_power_AUC={rr['srp_power_AUC']:.4f}")
    print("wrote", out_roc, out_sum, out_rocs)


def roc_from_energy(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None, float("nan")
    thr = np.linspace(np.min(np.concatenate([pos, neg])),
                      np.max(np.concatenate([pos, neg])), 200)
    pfa = np.array([np.mean(neg > t) for t in thr])
    pd_ = np.array([np.mean(pos > t) for t in thr])
    order = np.argsort(pfa)
    auc = float(np.trapezoid(pd_[order], pfa[order]))
    return pd.DataFrame(dict(threshold=thr, Pd=pd_, Pfa=pfa)), auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=["test1", "test2_drone", "test2_nodrone", "combine"])
    ap.add_argument("--nframes", type=int, default=None)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    if a.job == "combine":
        combine(a.outdir)
        return
    base = r"E:/Datasets/WSU_UAV_Acoustic_Localization/dataset"
    if a.job == "test1":
        job = dict(config=os.path.join(base, "test_1/Oct_11_2024/flight_5/alignment_params.json"),
                   segment=None)
    elif a.job == "test2_drone":
        job = dict(config=os.path.join(base, "test_2/Mar_18_2025/flight_2/alignment_params.json"),
                   segment="drone")
    else:
        job = dict(config=os.path.join(base, "test_2/Mar_18_2025/flight_2/alignment_params.json"),
                   segment="no_drone_2")
    df, cfg = process(job, a.nframes)
    out = os.path.join(a.outdir, f"wsu_perframe_{a.job}.csv")
    df.to_csv(out, index=False)
    # summary for this job
    drone = df[df.label == 1]
    summ = dict(job=a.job, n_frames=len(df), n_drone=len(drone),
               gcc_mae=np.nan, gcc_std=np.nan, srp_mae=np.nan, srp_std=np.nan)
    if len(drone):
        summ.update(gcc_mae=float(drone.gcc_err.mean()), gcc_std=float(drone.gcc_err.std()),
                     srp_mae=float(drone.srp_err.mean()), srp_std=float(drone.srp_err.std()))
    print(f"[{a.job}] frames={len(df)} drone={len(drone)} "
          f"GCC_MAE={summ['gcc_mae']:.2f}±{summ['gcc_std']:.2f} "
          f"SRP_MAE={summ['srp_mae']:.2f}±{summ['srp_std']:.2f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
