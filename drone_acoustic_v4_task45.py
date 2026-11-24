"""V4 Task 4 (IDMT 360-degree full-circle re-finetune) + Task 5
(IDMT detection on real background + WSU 360-degree unfold).

NEW files only: real_data_results_360.csv, real_folded_v3repro_v4.csv,
idmt_det_real_v4.csv, wsu_unfold_v4.csv, plus v4 logs/weights.
"""
import os
import copy
import json
import time
import wave
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.signal as sig
from sklearn.metrics import roc_auc_score

import drone_acoustic_v4_common as c

OUT = c.OUTDIR
v3 = c.v3
ZIP = v3.REAL_ZIP_DEFAULT


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def circular_l1_360(pred, gt):
    d = torch.abs(pred - gt) % 360.0
    return torch.minimum(d, 360.0 - d).mean()


def train_real(model, segs, epochs=8, lr=1e-4, full360=False, tag="ft", seed=42):
    """v3.real_finetune clone (L999-1038) with optional 360-degree labels/loss.
    v3 uses Adam(model.parameters()) only -> log_var frozen at zero -> 0.5/0.5.
    Kept identical for both arms so the ONLY difference is the label/loss metric."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0; cnt = 0
        for r in segs:
            f = r["feats"]
            gt = r["gt_win"].copy()
            if np.all(np.isnan(gt)):
                continue
            gt_seg = float(np.nanmean(gt))
            ya = np.array([(g if not np.isnan(g) else gt_seg) for g in gt],
                          dtype=np.float32)
            if not full360:
                ya = ya % 180.0
            wt = torch.from_numpy(f).float().to(c.DEVICE)
            yd = torch.ones(len(f), device=c.DEVICE)
            ya_t = torch.tensor(ya, dtype=torch.float32, device=c.DEVICE)
            pd_, ang_ = model(wt)
            lc = bce(pd_, yd)
            lr_ = (circular_l1_360(ang_, ya_t) if full360
                   else v3.masked_circular_l1(ang_, ya_t, yd))
            loss = 0.5 * lc + 0.5 * lr_     # == v3 DynamicMultiTaskLoss at log_var=0
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(lc.item()); cnt += 1
        print(f"[{tag}] ep{ep+1}/{epochs} Lcls={tot/max(cnt,1):.4f} "
              f"t={time.time()-t0:.1f}s", flush=True)
    return model, time.time() - t0


def eval_real(model, segs, tag=""):
    model.eval()
    rows = []
    allp, allgt = [], []
    with torch.no_grad():
        for r in segs:
            f, gt = r["feats"], r["gt_win"]
            pd_, ang_ = c.predict_batches(model, f)
            ok = ~np.isnan(gt)
            a = np.asarray(ang_, dtype=np.float64)
            e360 = c.circ360(a[ok], gt[ok])
            e180 = c.circ180(a[ok], gt[ok])
            g = float(np.nanmean(gt))
            rows.append(dict(segment=r["segment"], gt_azi=g,
                             gt_azi_mod=g % 180.0,
                             pred_azi=float(a.mean()),
                             pred_min=float(a.min()), pred_max=float(a.max()),
                             det_prob=float(np.mean(pd_)),
                             err360=float(e360.mean()),
                             err_fold180=float(e180.mean()),
                             err360_std=float(e360.std()),
                             n_windows=int(ok.sum()), gt_gt180=bool(g > 180.0)))
            allp.append(pd_); allgt.append(gt)
    df = pd.DataFrame(rows)
    print(f"[{tag}] n_seg={len(df)} n_win={df.n_windows.sum()} "
          f"MAE360={df.err360.mean():.3f} MAEfold={df.err_fold180.mean():.3f} "
          f"gt>180={(df.gt_gt180).sum()}", flush=True)
    return df


# ---------------------------------------------------------------------------
# TASK 4
# ---------------------------------------------------------------------------
def task4():
    T = time.time()
    print("=== TASK 4: IDMT full-circle 360-degree re-finetune ===", flush=True)
    gain = v3.estimate_gain(ZIP, n=20)
    print(f"[task4] global gain={gain:.4f}", flush=True)
    cp = os.path.join(OUT, "real_gcc_cache_v4.npz")
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        segs = [dict(segment=str(s), feats=f, gt_win=g)
                for s, f, g in zip(z["seg"], z["feat"], z["gt"])]
        print(f"[task4] loaded cached real GCC segments={len(segs)}", flush=True)
    else:
        segs = c.cache_real_segments(ZIP, gain, n_segments=None)
        np.savez_compressed(cp,
                            seg=np.asarray([r["segment"] for r in segs], dtype=object),
                            feat=np.asarray([r["feats"] for r in segs], dtype=object),
                            gt=np.asarray([r["gt_win"] for r in segs], dtype=object))
    # pretrained proposed model from Task 1 (identical to v3.real pretrain protocol)
    sd = os.path.join(OUT, "geom_proposed_v4.pt")
    base = c.make_proposed(seed=42)
    if os.path.exists(sd):
        base.load_state_dict(torch.load(sd, map_location=c.DEVICE, weights_only=True))
        print("[task4] loaded geom_proposed_v4.pt", flush=True)
    else:
        ds = c.CachedCubicDataset(n_total=8000, seed=42)
        base, _ = c.train_cubic(base, ds, epochs=20, dyn_logvar=False, tag="T4-pretrain")

    zs = eval_real(base, segs, "T4-zero-shot")
    zs.to_csv(os.path.join(OUT, "real_zeroshot_v3repro_v4.csv"), index=False)

    # Arm A: folded 180-degree fine-tune (v3.real_finetune reproduced on cached frontend)
    m_fold, t_fold = train_real(copy.deepcopy(base).eval(), segs, full360=False, tag="T4-folded")
    df_fold = eval_real(m_fold, segs, "T4-folded-ft")
    df_fold.to_csv(os.path.join(OUT, "real_folded_v3repro_v4.csv"), index=False)
    torch.save(m_fold.state_dict(), os.path.join(OUT, "real_folded_v4.pt"))

    # Arm B: full 360-degree fine-tune from the SAME pretrained initialization
    m360, t360 = train_real(copy.deepcopy(base).eval(), segs, full360=True, tag="T4-360")
    df360 = eval_real(m360, segs, "T4-360-ft")
    df360.to_csv(os.path.join(OUT, "real_data_results_360.csv"), index=False)
    torch.save(m360.state_dict(), os.path.join(OUT, "real_360_v4.pt"))

    gt180 = df360.gt_gt180
    summ = pd.DataFrame([
        dict(arm="zero_shot", n_seg=len(zs), mae360=zs.err360.mean(),
             mae_fold180=zs.err_fold180.mean(),
             mae360_gt180=zs[gt180].err360.mean(),
             mae_fold180_gt180=zs[gt180].err_fold180.mean(),
             n_gt180=int(gt180.sum()), sec=0.0),
        dict(arm="finetune_folded180", n_seg=len(df_fold), mae360=df_fold.err360.mean(),
             mae_fold180=df_fold.err_fold180.mean(),
             mae360_gt180=df_fold[gt180].err360.mean(),
             mae_fold180_gt180=df_fold[gt180].err_fold180.mean(),
             n_gt180=int(gt180.sum()), sec=t_fold),
        dict(arm="finetune_full360", n_seg=len(df360), mae360=df360.err360.mean(),
             mae_fold180=df360.err_fold180.mean(),
             mae360_gt180=df360[gt180].err360.mean(),
             mae_fold180_gt180=df360[gt180].err_fold180.mean(),
             n_gt180=int(gt180.sum()), sec=t360),
    ])
    summ.to_csv(os.path.join(OUT, "real_360_summary_v4.csv"), index=False)
    print(summ.to_string(index=False), flush=True)
    with open(os.path.join(OUT, "real360_notes_v4.json"), "w") as f:
        json.dump(dict(gain=gain, n_segments=len(segs),
                       n_windows=int(sum(r["gt_win"].size for r in segs)),
                       n_gt180=int(gt180.sum()),
                       total_sec=time.time() - T,
                       v3_real_gt_note=("interp_azi_at returns [0,360) (v3 L920-930); "
                                        "v3.real_finetune L1027 folds with %180; "
                                        "v3.real_eval L992 folds GT for reporting."),
                       folded_v1_baseline_mae=float(
                           pd.read_csv(os.path.join(OUT, "real_data_results_ft.csv")).ang_err.mean()),
                       ), f, indent=2)
    print(f"[task4] elapsed {time.time()-T:.1f}s", flush=True)
    return df_fold, df360, segs, gain, m_fold


# ---------------------------------------------------------------------------
# TASK 5a: real-background detection (IDMT positives vs WSU ambient negatives)
# ---------------------------------------------------------------------------
def _read_wsu_ambient(limit_windows=1200, seed=0):
    """WSU test_2 no_drone_2: 4 devices x 6ch (24-bit int), 48k -> 16k,
    use 8 channels (dev1 all 6 + dev2 first 2), 1-s windows."""
    base = r"E:/Datasets/WSU_UAV_Acoustic_Localization/dataset/test_2/Mar_18_2025/flight_2"
    cfg = json.load(open(os.path.join(base, "alignment_params.json")))
    wavs = [os.path.normpath(os.path.join(base, w)) for w in cfg["wavs_nosync"]]
    seg = [s for s in cfg["segments"] if s["name"] == "no_drone_2"][0]
    start = int(seg["skip_seconds"] * 48000)
    n_samp = int(min(seg["skip_seconds_end"] - seg["skip_seconds"], 287.0) * 48000)
    handles = []
    for w in wavs:
        h = wave.open(w, "rb")
        assert h.getsampwidth() == 4 and h.getnchannels() == 6, (w, h.getparams())
        h.setpos(start)
        handles.append(h)
    rng = np.random.RandomState(seed)
    total = n_samp // 16000
    pick = np.sort(rng.choice(total, size=min(limit_windows, total), replace=False))
    feats = []
    for wi in pick:
        blocks = []
        for h in handles:
            h.setpos(start + int(wi) * 16000)
            raw = h.readframes(16000)
            b = np.frombuffer(raw, dtype="<i4").reshape(-1, 6).astype(np.float64) / 2 ** 23  # 24-bit PCM in 32-bit container
            blocks.append(b)
        x48 = np.hstack(blocks)[:, :8]                  # (16000,8)
        x16 = sig.resample_poly(x48, 1, 3, axis=0).T    # (8,5333)
        feats.append(v3.wav_to_gcc_phat(x16))
    for h in handles:
        h.close()
    return np.stack(feats).astype(np.float32), pick


def task5a(m_fold, segs):
    T = time.time()
    print("=== TASK 5a: detection on real background ===", flush=True)
    # positives: per-window det_prob from IDMT drone segments (folded fine-tuned head)
    pos = []
    m_fold.eval()
    with torch.no_grad():
        for r in segs:
            p, _ = c.predict_batches(m_fold, r["feats"])
            pos.append(p)
    pos = np.concatenate(pos)
    # negatives: WSU no-drone ambient (CROSS-DOMAIN, exploratory)
    neg_f, picks = _read_wsu_ambient(limit_windows=1200, seed=0)
    neg, _ = c.predict_batches(m_fold, neg_f)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    auc = float(roc_auc_score(y, s))
    rows = [
        dict(dataset_pos="IDMT MicArrayData30s (8ch cubic)",
             dataset_neg="WSU test_2 no_drone_2 (24ch array, 8ch subset)",
             domain="CROSS-DOMAIN EXPLORATORY", n_pos=len(pos), n_neg=len(neg),
             det_th=0.5, Pd=float((pos > 0.5).mean()), Pfa=float((neg > 0.5).mean()),
             roc_auc=auc, pos_score_mean=float(pos.mean()),
             neg_score_mean=float(neg.mean()),
             neg_score_p95=float(np.percentile(neg, 95)),
             sec=time.time() - T, seed_neg_windows=0),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "idmt_det_real_v4.csv"), index=False)
    print(df.to_string(index=False), flush=True)
    # save scores for auditing (new file)
    pd.DataFrame(dict(label=y, score=s, source=np.r_[np.full(len(pos), "idmt_drone"),
                                                      np.full(len(neg), "wsu_ambient")])
                 ).to_csv(os.path.join(OUT, "idmt_det_scores_v4.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# TASK 5b: WSU 360-degree unfold
# ---------------------------------------------------------------------------
def task5b():
    T = time.time()
    rows = []
    for job in ("test1", "test2_drone", "test2_nodrone"):
        p = os.path.join(OUT, f"wsu_perframe_{job}.csv")
        d = pd.read_csv(p)
        if job == "test2_nodrone" or len(d) == 0:
            rows.append(dict(job=job, n_frames=len(d), n_valid=0,
                             note="no ground truth (ambient-only segment)"))
            continue
        g = d["csv_az"].to_numpy(dtype=float)
        ok = np.isfinite(g)
        for est_col in ("srp_az", "gcc_az"):
            e = d[est_col].to_numpy(dtype=float)
            v = ok & np.isfinite(e)
            if not v.any():
                continue
            gt = g[v] % 360.0
            pr = e[v] % 360.0
            d180 = np.abs(pr % 180.0 - gt % 180.0)
            err180 = np.minimum(d180, 180.0 - d180)
            d360 = np.abs(pr - gt)
            err360 = np.minimum(d360, 360.0 - d360)
            rows.append(dict(job=job, n_frames=len(d), n_valid=int(v.sum()),
                             estimator=est_col,
                             mae_fold180=float(err180.mean()),
                             mae_full360=float(err360.mean()),
                             delta_360_minus_180=float(err360.mean() - err180.mean()),
                             std_full360=float(err360.std()),
                             pct_worse_gt1deg=float(100 * np.mean(err360 - err180 > 1.0)),
                             sec=time.time() - T))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "wsu_unfold_v4.csv"), index=False)
    print(df.to_string(index=False), flush=True)
    return df


if __name__ == "__main__":
    T = time.time()
    _, _, segs, gain, m_fold = task4()
    task5a(m_fold, segs)
    task5b()
    print(f"=== tasks45 total elapsed {time.time()-T:.1f}s ===", flush=True)
