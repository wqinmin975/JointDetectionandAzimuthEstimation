"""V4 Task 3: dynamic-weighting mechanism, measured (not assumed).

Four instrumented reproductions of v3's --mode sim proposed/fixed protocols:
  proposed 20ep, proposed 30ep, fixed lambda=1 20ep, fixed lambda=1 30ep.
Records per-epoch and per-10-batch log_var, precision weights, task losses and
backbone/head gradient norms. All numbers come from actual backward passes.
"""
import os
import time
import json
import numpy as np
import pandas as pd
import torch

import drone_acoustic_v4_common as c
from drone_acoustic_v4_tasks12 import make_ula_testset, eval_ula_fixed

OUT = c.OUTDIR
v3 = c.v3


def run(mode, epochs, seed=42, n_total=4000, lam=1.0, xt=None, gt=None,
        every=10, ckpt=(5, 10, 15, 20, 25, 30)):
    """Instrumented clone of v3.train_synthetic_ula (same RNG order, same data)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    ds = v3.DroneAcousticDatasetULA(n_total=n_total)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    model = v3.JointDetDOANetRaw().to(c.DEVICE)
    bce = torch.nn.BCELoss()
    dyn = v3.DynamicMultiTaskLoss()
    params = list(model.parameters())
    if mode == "proposed":
        params += list(dyn.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    ep_rows, batch_rows = [], []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        cls_l, reg_l, gb, gd, ga, g_all = [], [], [], [], [], []
        for bi, (x, yd, ya) in enumerate(loader):
            x, yd, ya = x.to(c.DEVICE), yd.to(c.DEVICE), ya.to(c.DEVICE)
            pd_, an_ = model(x)
            if mode == "fixed":
                lc = bce(pd_, yd)
                lr_ = v3.masked_circular_l1(an_, ya, yd)
                total = lc + lam * lr_
            else:
                total, lc, lr_ = dyn(pd_, yd, an_, ya, lam)
            opt.zero_grad()
            total.backward()
            # --- gradient norms, measured before optimizer step ---
            def gn(mods):
                sq = 0.0
                for m in mods:
                    for p in m.parameters():
                        if p.grad is not None:
                            sq += float(p.grad.detach().pow(2).sum().item())
                return sq ** 0.5
            b = gn([model.backbone]); d = gn([model.det_head]); a = gn([model.ang_head])
            g = gn([model])
            gb.append(b); gd.append(d); ga.append(a); g_all.append(g)
            cls_l.append(float(lc.item())); reg_l.append(float(lr_.item()))
            if (bi + 1) % every == 0 and mode == "proposed":
                batch_rows.append(dict(mode=mode, epoch=ep + 1, batch=bi + 1,
                                       log_var_det=float(dyn.log_var_det.item()),
                                       log_var_ang=float(dyn.log_var_ang.item()),
                                       w_det=float(torch.exp(-dyn.log_var_det).item()),
                                       w_ang=float(torch.exp(-dyn.log_var_ang).item()),
                                       loss_cls=float(lc.item()),
                                       loss_reg=float(lr_.item()),
                                       grad_backbone=b))
            opt.step()
            if (bi + 1) % every == 0 and mode == "fixed":
                batch_rows.append(dict(mode=mode, epoch=ep + 1, batch=bi + 1,
                                       log_var_det=float("nan"),
                                       log_var_ang=float("nan"),
                                       w_det=float("nan"), w_ang=float("nan"),
                                       loss_cls=float(lc.item()),
                                       loss_reg=float(lr_.item()),
                                       grad_backbone=b))
        lvd = float(dyn.log_var_det.item()) if mode == "proposed" else float("nan")
        lva = float(dyn.log_var_ang.item()) if mode == "proposed" else float("nan")
        wd = float(torch.exp(-dyn.log_var_det).item()) if mode == "proposed" else float("nan")
        wa = float(torch.exp(-dyn.log_var_ang).item()) if mode == "proposed" else float("nan")
        mae = mstd = float("nan")
        if (ep + 1) in ckpt or ep == epochs - 1:
            mae, mstd, _, _ = eval_ula_fixed(model, xt, gt)
        ep_rows.append(dict(mode=mode, epochs=epochs, epoch=ep + 1,
                            log_var_det=lvd, log_var_ang=lva,
                            w_det=wd, w_ang=wa,
                            loss_cls=float(np.mean(cls_l)),
                            loss_reg=float(np.mean(reg_l)),
                            reg_over_cls=float(np.mean(reg_l) / max(np.mean(cls_l), 1e-12)),
                            grad_backbone=float(np.mean(gb)),
                            grad_det_head=float(np.mean(gd)),
                            grad_ang_head=float(np.mean(ga)),
                            grad_all=float(np.mean(g_all)),
                            mae_ula_5dB=mae, mae_std_ula_5dB=mstd,
                            sec=time.time() - t0))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[{mode}-ep{epochs}] e{ep+1:02d} lvd={lvd:+.3f} lva={lva:+.3f} "
                  f"w_det={wd:.3f} w_ang={wa:.3f} "
                  f"grad_bb={np.mean(gb):.4f} Lr/Lc={np.mean(reg_l)/max(np.mean(cls_l),1e-12):.2f} "
                  f"MAE5={mae:.3f} t={time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(),
               os.path.join(OUT, f"dyn_{mode}_ep{epochs}_v4.pt"))
    return ep_rows, batch_rows


def main():
    T = time.time()
    xt, gt = make_ula_testset(n=600, snr_db=-5.0, seed=0)
    print(f"[task3] fixed paired ULA testset n={len(gt)} @-5dB seeded 0", flush=True)
    all_ep, all_b = [], []
    for mode in ("proposed", "fixed"):
        for ep in (20, 30):
            r, b = run(mode, ep, seed=42, lam=1.0, xt=xt, gt=gt)
            all_ep += r; all_b += b
    df_ep = pd.DataFrame(all_ep)
    df_b = pd.DataFrame(all_b)
    df_ep.to_csv(os.path.join(OUT, "dyn_mechanism_v4.csv"), index=False)
    df_b.to_csv(os.path.join(OUT, "dyn_logvar_batch_v4.csv"), index=False)
    print("\n=== epoch-end summary ===", flush=True)
    cols = ["mode", "epochs", "epoch", "log_var_det", "log_var_ang",
            "w_det", "w_ang", "grad_backbone", "grad_ang_head",
            "reg_over_cls", "mae_ula_5dB"]
    print(df_ep[df_ep.epoch.isin([1, 5, 10, 20, 30])][cols].to_string(index=False),
          flush=True)

    # Final MAE on full 600-sample paired testset for every config
    finals = []
    for mode in ("proposed", "fixed"):
        for ep in (20, 30):
            sub = df_ep[(df_ep["mode"] == mode) & (df_ep["epochs"] == ep)].iloc[-1]
            finals.append(dict(mode=mode, epochs=ep,
                               mae=float(sub.mae_ula_5dB),
                               mae_std=float(sub.mae_std_ula_5dB),
                               log_var_det=float(sub.log_var_det),
                               log_var_ang=float(sub.log_var_ang),
                               w_det=float(sub.w_det), w_ang=float(sub.w_ang)))
    fd = pd.DataFrame(finals)
    fd.to_csv(os.path.join(OUT, "dyn_epoch_compare_v4.csv"), index=False)
    print("\n=== 20ep vs 30ep ===", flush=True)
    print(fd.to_string(index=False), flush=True)

    p20 = fd[(fd["mode"] == "proposed") & (fd.epochs == 20)].mae.iloc[0]
    p30 = fd[(fd["mode"] == "proposed") & (fd.epochs == 30)].mae.iloc[0]
    f20 = fd[(fd["mode"] == "fixed") & (fd.epochs == 20)].mae.iloc[0]
    f30 = fd[(fd["mode"] == "fixed") & (fd.epochs == 30)].mae.iloc[0]
    with open(os.path.join(OUT, "dyn_mechanism_notes_v4.json"), "w") as f:
        json.dump(dict(
            proposed_20ep_mae=p20, proposed_30ep_mae=p30,
            fixed_lam1_20ep_mae=f20, fixed_lam1_30ep_mae=f30,
            proposed_gain_20_to_30=float(p20 - p30),
            fixed_gain_20_to_30=float(f20 - f30),
            total_sec=time.time() - T,
            method="v3.train_synthetic_ula cloned line-for-line with grad-norm "
                   "hooks; seed=42 dataset+shuffle; paired 600-sample -5dB testset; "
                   "backbone grad norm = L2 over backbone parameters, batch-averaged.",
        ), f, indent=2)
    print(f"[task3] total elapsed {time.time()-T:.1f}s", flush=True)


if __name__ == "__main__":
    main()
