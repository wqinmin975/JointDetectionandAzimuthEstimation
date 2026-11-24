"""V4 Task 1 (geometry-MLP ablation) + Task 2 (MEMS impairment robustness).

All outputs are NEW *_v4 files. v3 source and every pre-existing CSV are untouched.
Seed policy: seed=42 everywhere for training (same as v3); paired evaluation on
identically generated test signals. Every condition reports n, seed, seconds.
"""
import os
import time
import json
import numpy as np
import pandas as pd
import torch

import drone_acoustic_v4_common as c

OUT = c.OUTDIR
v3 = c.v3


def make_cubic_testset(n=600, snr_db=-5.0, seed=20240517):
    """Paired 8-mic cubic @ fixed SNR. Same generator/geometry as v3 sim, GT=a%180."""
    np.random.seed(seed)
    feats, gt = [], []
    for _ in range(n):
        azi = np.random.uniform(*v3.REAL_AZI_RANGE)
        s = v3.generate_drone_signal(2048, v3.FS)
        d = v3.add_time_delay(s, azi, 90.0, v3.MIC_POS, v3.C, v3.FS)
        arr = np.stack([v3.add_noise(d[m], snr_db) for m in range(v3.MIC_NUM)])
        feats.append(v3.wav_to_gcc_phat(arr))
        gt.append(azi % 180.0)
    return np.stack(feats).astype(np.float32), np.asarray(gt)


def make_ula_testset(n=600, snr_db=-5.0, seed=0):
    """v3-identical ULA test generation at fixed SNR (mirrors _eval_ula_model's RNG
    call order with is_drone always True)."""
    np.random.seed(seed)   # generate_drone_signal_ula/add_noise use global RNG
    rng = np.random.RandomState(seed)
    feats_raw, gt = [], []
    for _ in range(n):
        azi = rng.uniform(0.0, 180.0)
        s = v3.generate_drone_signal_ula(2048, v3.FS)
        arr = v3.add_time_delay(s, azi, 90.0, v3.ULA_POS, v3.C, v3.FS)
        arr = np.stack([v3.add_noise(arr[m], snr_db) for m in range(v3.ULA_M)])
        feats_raw.append(arr.astype(np.float32))
        gt.append(azi)
    return np.stack(feats_raw), np.asarray(gt)


def eval_ula_fixed(model, arrs, gt):
    model.eval()
    ang = []
    with torch.no_grad():
        for i in range(0, len(arrs), 32):
            x = torch.from_numpy(arrs[i:i + 32]).to(c.DEVICE)
            _, a = model(x)
            ang.append(a.cpu().numpy())
    ang = np.concatenate(ang)
    err = c.circ180(ang, gt)
    return float(err.mean()), float(err.std()), ang, err


# ---------------------------------------------------------------------------
# TASK 1
# ---------------------------------------------------------------------------
def task1():
    t0 = time.time()
    rows, notes = [], []
    print("=== TASK 1: geometry-MLP ablation ===", flush=True)

    # --- (A) 8-mic GCC branch: the ONLY network that contains geom_enc/fuse ---
    ds = c.CachedCubicDataset(n_total=8000, seed=42)
    base = c.make_proposed(seed=42)
    nog = c.make_nogeom_from(base)
    # Confirm bit-identical init on the shared modules
    same = all(np.array_equal(a.detach().cpu().numpy(), b.detach().cpu().numpy())
               for a, b in zip(base.backbone.parameters(), nog.backbone.parameters()))
    same &= all(np.array_equal(a.detach().cpu().numpy(), b.detach().cpu().numpy())
                for pair in [("det_head", "det_head"), ("ang_head", "ang_head")]
                for a, b in zip(getattr(base, pair[0]).parameters(),
                                getattr(nog, pair[1]).parameters()))
    print(f"[task1] shared-module init bit-identical = {same}", flush=True)

    # v3.pretrain_cubic_for_real L599: Adam(model.parameters()) ONLY -> log_var frozen.
    # Replicate exactly (dyn_logvar=False) for fidelity to the v3 real baseline.
    prop, ip = c.train_cubic(base, ds, epochs=20, lr=1e-3, dyn_logvar=False,
                             seed=42, tag="T1-proposed")
    no, inn = c.train_cubic(nog, ds, epochs=20, lr=1e-3, dyn_logvar=False,
                            seed=42, tag="T1-no-geom")
    torch.save(prop.state_dict(), os.path.join(OUT, "geom_proposed_v4.pt"))
    torch.save(no.state_dict(), os.path.join(OUT, "geom_nogeom_v4.pt"))

    xt, gt = make_cubic_testset(n=600, snr_db=-5.0, seed=20240517)
    rp = c.eval_cubic(prop, xt, gt)
    rn = c.eval_cubic(no, xt, gt)
    paired = rn["err"] - rp["err"]
    rows.append(dict(experiment="geom_ablation_8mic_GCC", variant="proposed_with_geom",
                     n=rp["n"], snr_db=-5, seed_train=42, seed_test=20240517,
                     mae_deg=rp["mae"], mae_std=rp["std"],
                     pair_diff_deg=float("nan"), sec=ip["sec"]))
    rows.append(dict(experiment="geom_ablation_8mic_GCC", variant="no_geom",
                     n=rn["n"], snr_db=-5, seed_train=42, seed_test=20240517,
                     mae_deg=rn["mae"], mae_std=rn["std"],
                     pair_diff_deg=float(paired.mean()), sec=inn["sec"]))
    print(f"[task1] 8mic @-5dB proposed={rp['mae']:.3f}±{rp['std']:.3f}  "
          f"no-geom={rn['mae']:.3f}±{rn['std']:.3f}  "
          f"Delta(no-prop)={paired.mean():+.3f} deg", flush=True)
    notes.append(f"8mic GCC: paired n={len(paired)} 95%CI=("
                 f"{paired.mean()-1.96*paired.std()/np.sqrt(len(paired)):+.3f},"
                 f"{paired.mean()+1.96*paired.std()/np.sqrt(len(paired)):+.3f}) deg")

    # --- (B) 6-mic ULA branch: has NO geom encoder at all ---
    # Reproduce v3 proposed @-5dB (7.499±10.385) and show the ablation is a no-op.
    np.random.seed(42); torch.manual_seed(42)
    ds_ula = v3.DroneAcousticDatasetULA(n_total=4000)
    torch.manual_seed(42)
    loader = torch.utils.data.DataLoader(ds_ula, batch_size=32, shuffle=True)
    model_u = v3.JointDetDOANetRaw().to(c.DEVICE)
    dyn = v3.DynamicMultiTaskLoss()
    opt = torch.optim.Adam(list(model_u.parameters()) + list(dyn.parameters()), lr=1e-3)
    tu = time.time()
    for ep in range(20):
        model_u.train()
        for x, yd, ya in loader:
            x, yd, ya = x.to(c.DEVICE), yd.to(c.DEVICE), ya.to(c.DEVICE)
            pd_, an_ = model_u(x)
            loss, _, _ = dyn(pd_, yd, an_, ya, 0.5)
            opt.zero_grad(); loss.backward(); opt.step()
    sec_u = time.time() - tu
    torch.save(model_u.state_dict(), os.path.join(OUT, "ula_proposed_v4.pt"))
    au, gu = make_ula_testset(n=600, snr_db=-5.0, seed=0)
    mu, su, _, _ = eval_ula_fixed(model_u, au, gu)
    # "no-geom" ULA is structurally identical (JointDetDOANetRaw has no geom path)
    model_u2 = v3.JointDetDOANetRaw().to(c.DEVICE)
    model_u2.load_state_dict(model_u.state_dict())
    mu2, su2, _, _ = eval_ula_fixed(model_u2, au, gu)
    rows.append(dict(experiment="geom_ablation_6mic_ULA_structural",
                     variant="proposed_v3_repro", n=600, snr_db=-5, seed_train=42,
                     seed_test=0, mae_deg=mu, mae_std=su,
                     pair_diff_deg=float(mu2 - mu), sec=sec_u))
    rows.append(dict(experiment="geom_ablation_6mic_ULA_structural",
                     variant="no_geom_identical_network", n=600, snr_db=-5,
                     seed_train=42, seed_test=0, mae_deg=mu2, mae_std=su2,
                     pair_diff_deg=0.0, sec=0.0))
    print(f"[task1] ULA @-5dB repro={mu:.3f}±{su:.3f} (v3 file: 7.499±10.385) "
          f"no-geom delta={mu2-mu:+.6f}", flush=True)
    notes.append("ULA branch (JointDetDOANetRaw, v3 L191-206) contains NO geom_enc/fuse; "
                 "the 7.50±10.38 baseline cannot exhibit a geometry ablation. "
                 "The informative ablation is the 8-mic GCC branch above.")

    df = pd.DataFrame(rows)
    fp = os.path.join(OUT, "geom_ablation_v4.csv")
    df.to_csv(fp, index=False)
    with open(os.path.join(OUT, "geom_ablation_notes_v4.json"), "w") as f:
        json.dump(dict(notes=notes, total_sec=time.time() - t0,
                       v3_real_branch_logvar_note=(
                           "v3.pretrain_cubic_for_real L599 passes only model.parameters() "
                           "to Adam; log_var_det/ang stay 0 -> 0.5/0.5 weights. "
                           "Replicated exactly here. Only ULA proposed (v3 L690-691) "
                           "optimizes log_var.")), f, indent=2)
    print(f"[task1] saved {fp} rows={len(df)} elapsed={time.time()-t0:.1f}s", flush=True)
    return prop, no, df


# ---------------------------------------------------------------------------
# TASK 2
# ---------------------------------------------------------------------------
def _base_signals(n, seed=20240517):
    """Generate (clean_delayed, noise_draws, azi) with v3-identical RNG order so
    every impairment is applied to the SAME underlying realisation."""
    np.random.seed(seed)
    out = []
    for _ in range(n):
        azi = np.random.uniform(*v3.REAL_AZI_RANGE)
        s = v3.generate_drone_signal(2048, v3.FS)
        d = v3.add_time_delay(s, azi, 90.0, v3.MIC_POS, v3.C, v3.FS)
        noise = np.random.randn(v3.MIC_NUM, 2048)   # v3.add_noise draws per channel
        out.append((s, azi, noise))
    return out


def _with_noise(delayed, noise, snr_db):
    return np.stack([delayed[m] + noise[m] * np.sqrt(np.var(delayed[m]) /
                                                     10 ** (snr_db / 10.0))
                     for m in range(v3.MIC_NUM)])


def task2(model):
    t0 = time.time()
    print("=== TASK 2: MEMS impairment robustness ===", flush=True)
    n_test, seed = 600, 20240517
    base = _base_signals(n_test, seed)
    snrs = np.random.RandomState(seed + 1).choice(v3.SNR_LIST, size=n_test)
    zs = np.random.RandomState(seed + 2)   # defect draws (fixed across all levels)

    gt = np.asarray([a % 180.0 for (_, a, _) in base])
    clean = np.stack([_with_noise(v3.add_time_delay(s, a, 90.0, v3.MIC_POS,
                                                    v3.C, v3.FS), nz, snr)
                      for (s, a, nz), snr in zip(base, snrs)])
    conditions = {}

    def add(name, arrs):
        t = time.time()
        f = np.stack([v3.wav_to_gcc_phat(a) for a in arrs]).astype(np.float32)
        r = c.eval_cubic(model, f, gt)
        r.update(dict(sec_gcc_eval=time.time() - t))
        conditions[name] = r

    add("baseline", clean)

    # (a) channel gain mismatch g_m ~ N(1, sigma_g): same z across sigma levels
    zg = zs.randn(n_test, v3.MIC_NUM)
    for sg in (0.05, 0.10, 0.20):
        g = 1.0 + sg * zg
        add(f"gain_mismatch_sg{sg:.2f}", clean * g[:, :, None])

    # (b) one dead channel: sweep 8 positions (v3 MIC_NUM=8; paper asked >=6)
    for k in range(v3.MIC_NUM):
        a = clean.copy(); a[:, k, :] = 0.0
        add(f"dead_channel_mic{k+1}", a)

    # (c) geometry error: MIC_POS coords += N(0, sigma_p), same z across levels
    zp = zs.randn(n_test, v3.MIC_NUM, 3)
    for sp in (0.002, 0.005, 0.010):
        arrs = []
        for (s, a, nz), snr, dp in zip(base, snrs, zp):
            pos = v3.MIC_POS + sp * dp
            d = v3.add_time_delay(s, a, 90.0, pos, v3.C, v3.FS)
            arrs.append(_with_noise(d, nz, snr))
        add(f"geometry_error_sp{sp:.3f}m", np.stack(arrs))

    # (d) independent device self-noise at relative SNR 20/10 dB
    zn = zs.randn(n_test, v3.MIC_NUM, 2048)
    for rsnr in (20, 10):
        arrs = []
        for i, a in enumerate(clean):
            p = np.mean(a ** 2)
            arrs.append(a + zn[i] * np.sqrt(p / 10 ** (rsnr / 10.0)))
        add(f"device_selfnoise_{rsnr}dB", np.stack(arrs))

    b = conditions["baseline"]
    rows = []
    for name, r in conditions.items():
        rows.append(dict(experiment="mems_robustness", condition=name,
                         n=r["n"], seed_train=42, seed_test=seed,
                         mae_deg=r["mae"], mae_std=r["std"],
                         delta_mae_deg=r["mae"] - b["mae"],
                         rel_degradation_pct=100.0 * (r["mae"] - b["mae"]) / b["mae"],
                         sec=r["sec_gcc_eval"]))
    df = pd.DataFrame(rows)
    fp = os.path.join(OUT, "mems_robustness_v4.csv")
    df.to_csv(fp, index=False)
    print(df[["condition", "mae_deg", "delta_mae_deg"]].to_string(index=False), flush=True)
    print(f"[task2] saved {fp} rows={len(df)} elapsed={time.time()-t0:.1f}s", flush=True)
    return df


if __name__ == "__main__":
    ta = time.time()
    prop, no, d1 = task1()
    d2 = task2(prop)
    print(f"=== tasks12 total elapsed {time.time()-ta:.1f}s ===", flush=True)
