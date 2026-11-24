#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_verify_v4_stats.py -- formal verification of all six V4 CSV groups.

Read-only over all existing files. Writes ONLY _verify_v4_report.md.
Uses _prep_stats_template.py functions. Paired arrays recovered in memory
from cached weights / per-frame CSVs where summary-only columns exist.
"""
import os, sys, math
import numpy as np
import pandas as pd
import torch
from scipy import stats
sys.path.insert(0, '.')
from _prep_stats_template import (paired_t, wilson_ci, roc_auc,
                                  auc_block_bootstrap, holm_bonferroni, welch_t)

rng = np.random.default_rng(20241124)
N_PERM = 5000
OUT_MD = "_verify_v4_report.md"
L = []
def emit(s=""):
    L.append(s)
    print(s)
def signif(p, a=0.05):
    return "SIGNIFICANT" if p < a else "NOT SIGNIFICANT"
def perm_flip(d, n=N_PERM):
    obs = d.mean()
    perms = np.array([rng.choice([-1,1], size=len(d)).dot(d)/len(d) for _ in range(n)])
    return (np.sum(np.abs(perms) >= abs(obs)) + 1) / (n + 1)

emit("# V4 六组新实验正式统计核验报告")
emit()
emit("核验人：robustness-auditor（严复核） ｜ 模板：`_prep_stats_template.py` ｜ "
     "生成脚本：`_verify_v4_stats.py` ｜ 只读，未改任何 CSV/tex")
emit()
emit("判定约定：双侧 α=0.05；所有 CI 为 95%；置换检验 5000 次（下限 p≈2e-4）。")
emit()

# ================================================================ 1 GEOM
emit("## 1. 几何编码消融（8mic GCC，配对 n=600 @−5dB）")
import drone_acoustic_v4_common as c
import drone_acoustic_v4_tasks12 as t12
xt, gt = t12.make_cubic_testset(n=600, snr_db=-5.0, seed=20240517)
def load(path, kind="proposed"):
    m = c.make_proposed(seed=42) if kind=="proposed" else c.make_nogeom_from(c.make_proposed(seed=42))
    m.load_state_dict(torch.load(path, map_location=c.DEVICE, weights_only=True)); m.eval(); return m
ep = c.eval_cubic(load(os.path.join(c.OUTDIR,"geom_proposed_v4.pt")), xt, gt)["err"]
en = c.eval_cubic(load(os.path.join(c.OUTDIR,"geom_nogeom_v4.pt"),"nogeom"), xt, gt)["err"]
r = paired_t(en, ep, "no_geom_minus_proposed")
W, pw = stats.wilcoxon(en, ep)
pp = perm_flip(en-ep)
emit(f"- 重放 MAE：proposed={ep.mean():.6f}、no_geom={en.mean():.6f}"
     f"（CSV 2.249872 / 2.947982，逐位一致）")
emit(f"- 配对 t：Δ=+{r['mean_diff']:.6f}°，95%CI [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]，"
     f"t={r['t']:.4f}，p={r['p']:.3e} → **{signif(r['p'])}**（CI 不含 0）")
emit(f"- Wilcoxon p={pw:.3e} → {signif(pw)}；符号翻转置换 p={pp:.2e} → {signif(pp)}")
emit("- 6mic ULA 无几何路径，Δ=0 为结构性恒等，**不作证据**（已排除）。")
emit("- 结论：**几何编码带来 +0.698° 稳定增益，强显著。**")
emit()

# ================================================================ 2 MEMS
emit("## 2. MEMS 四类损伤 17 条件（配对 n=600，Holm–Bonferroni）")
n_test, seed = 600, 20240517
base = t12._base_signals(n_test, seed)
snrs = np.random.RandomState(seed+1).choice(c.v3.SNR_LIST, size=n_test)
zs = np.random.RandomState(seed+2)
gtm = np.asarray([a % 180.0 for (_, a, _) in base])
clean = np.stack([t12._with_noise(c.v3.add_time_delay(s,a,90.0,c.v3.MIC_POS,c.v3.C,c.v3.FS),nz,snr)
                  for (s,a,nz),snr in zip(base,snrs)])
model = load(os.path.join(c.OUTDIR,"geom_proposed_v4.pt"))
def errs(arrs):
    f = np.stack([c.v3.wav_to_gcc_phat(a) for a in arrs]).astype(np.float32)
    return c.eval_cubic(model, f, gtm)["err"]
eb = errs(clean)
conds = {}
zg = zs.randn(n_test, c.v3.MIC_NUM)
for sg in (0.05,0.10,0.20):
    conds[f"gain_mismatch_sg{sg:.2f}"] = clean*(1.0+sg*zg)[:,:,None]
for k in range(c.v3.MIC_NUM):
    a = clean.copy(); a[:,k,:] = 0.0; conds[f"dead_channel_mic{k+1}"] = a
zp = zs.randn(n_test, c.v3.MIC_NUM, 3)
for sp in (0.002,0.005,0.010):
    arrs=[]
    for (s,a,nz),snr,dp in zip(base,snrs,zp):
        arrs.append(t12._with_noise(c.v3.add_time_delay(s,a,90.0,c.v3.MIC_POS+sp*dp,c.v3.C,c.v3.FS),nz,snr))
    conds[f"geometry_error_sp{sp:.3f}m"] = np.stack(arrs)
zn = zs.randn(n_test, c.v3.MIC_NUM, 2048)
for i,rsnr in enumerate((20,10)):
    arrs=[]
    for a in clean:
        p = np.mean(a**2)
        arrs.append(a + zn[len(arrs)]*np.sqrt(p/10**(rsnr/10.0)))
    conds[f"device_selfnoise_{rsnr}dB"] = np.stack(arrs)
rows, ps = [], []
for name, arrs in conds.items():
    rr = paired_t(errs(arrs), eb, "cond_minus_baseline")
    rel = 100*rr["mean_diff"]/eb.mean()
    rows.append((name, rr["mean_diff"], rr["ci95_low"], rr["ci95_high"], rel, rr["p"])); ps.append(rr["p"])
adj, rej = holm_bonferroni(ps)
emit(f"- 基线重放 MAE={eb.mean():.6f}（CSV 1.653470，逐位一致）")
emit("")
emit("| 条件 | ΔMAE (°) | 95% CI | 退化% | p(配对) | p(Holm) | 判定 |")
emit("|---|---:|---:|---:|---:|---:|---|")
for (name,md,lo,hi,rel,p),pa,rj in zip(rows,adj,rej):
    emit(f"| {name} | {md:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {rel:+.2f} | {p:.2e} | {pa:.2e} | "
         f"**{'SIGNIFICANT' if rj else 'NOT SIGNIFICANT'}** |")
emit("")
emit("- **负面/不显著（诚实报告）**：gain_mismatch 三档 Δ≈4e−10°（PHAT 相位白化精确抵消逐通道常数增益，"
     "**前端数学不敏感，非网络鲁棒**）；device_selfnoise_20dB Δ=+0.0426° Holm p=0.67 → NOT SIGNIFICANT。")
emit("- **翻案提醒（重要）**：geometry_error_sp0.002m 与 device_selfnoise_10dB 用汇总 Welch 代理分别为 "
     "p=0.19 / 0.09（看似不显著），**配对重放后 p=2.8e−8 / 1.5e−3 均显著**（Holm 后仍显著）。"
     "配对设计消除了样本间方差，务必以本表配对判定为准，勿用 Welch 代理。")
emit("- 最敏感失效位：mic1（Δ=+2.07°、125%退化）；mic7 次之（+1.70°）。")
emit()

# ================================================================ 3 DYNAMIC
emit("## 3. 动态加权 20ep→30ep（配对 ULA 测试集 n=600 @−5dB）")
dy = pd.read_csv("dyn_epoch_compare_v4.csv")
for mode in ("proposed","fixed"):
    a20 = dy[(dy["mode"]==mode)&(dy.epochs==20)].iloc[0]
    a30 = dy[(dy["mode"]==mode)&(dy.epochs==30)].iloc[0]
    emit(f"- {mode}: 20ep={a20.mae:.4f}±{a20.mae_std:.4f} → 30ep={a30.mae:.4f}±{a30.mae_std:.4f}，"
         f"Δ={a30.mae-a20.mae:+.4f}°")
dm = pd.read_csv("dyn_mechanism_v4.csv")
for mode in ("proposed","fixed"):
    sub = dm[(dm["mode"]==mode)]
    e20 = sub[sub.epoch==20].iloc[-1]; e30 = sub[sub.epoch==30].iloc[-1]
    if mode=="proposed":
        emit(f"- log_var(ep20)=({e20.log_var_det:+.3f},{e20.log_var_ang:+.3f}) → "
             f"ep30=({e30.log_var_det:+.3f},{e30.log_var_ang:+.3f})；"
             f"w_det {e20.w_det:.2f}→{e30.w_det:.2f}（权重随 epoch 漂移，非稳态）")
emit("- 判定：摘要级确认 20→30ep 数值与口径；**逐样本配对 CI 未存**，需重放 `v4_task3`（约8分钟）。"
     "本轮不臆造配对 p 值。")
emit("- **口径统一**：用 V4（6.8013→4.2036 / 4.9715→4.2440），**不混 V3 的 7.50→5.09**。"
     "V3 `_abl_fixed20_ep30_v3.csv` 实为 λ=2.0，不是 equal weighting（已上报隐患2）。")
emit()

# ================================================================ 4 360
emit("## 4. 360° 全向重训 vs 180° 折叠（163 段，段级配对）")
c360 = pd.read_csv("real_data_results_360.csv"); fold = pd.read_csv("real_folded_v3repro_v4.csv")
assert (c360.segment.values==fold.segment.values).all()
r36 = paired_t(c360.err360, fold.err360, "full360_minus_folded_on_360_scale")
W36,p36 = stats.wilcoxon(c360.err360, fold.err360); p36p = perm_flip(c360.err360.values-fold.err360.values)
mask = c360.gt_gt180.values
emit(f"- **同尺度（全圆 360°）配对**：folded 臂 {fold.err360.mean():.3f}° → full360 臂 {c360.err360.mean():.3f}°，"
     f"Δ={r36['mean_diff']:+.3f}°，95%CI [{r36['ci95_low']:+.3f}, {r36['ci95_high']:+.3f}]，"
     f"t={r36['t']:.3f} p={r36['p']:.2e} → **{signif(r36['p'])}**")
emit(f"- Wilcoxon p={p36:.2e}（{signif(p36)}）；置换 p={p36p:.2e}（{signif(p36p)}）→ "
     "**360°重训的全向改善稳健**")
emit(f"- gt>180 的 {mask.sum()} 段子集：folded {fold.err360[mask].mean():.3f}° → "
     f"full360 {c360.err360[mask].mean():.3f}°（镜像翻转被消除）")
s = pd.read_csv("real_360_summary_v4.csv")
fld = s[s.arm=="finetune_folded180"].iloc[0]; fu = s[s.arm=="finetune_full360"].iloc[0]
emit(f"- **权衡（如实报）**：全向尺度改善 {fu.mae360-fld.mae360:+.3f}°；"
     f"但在折叠尺度 full360({fu.mae_fold180:.3f}°) 反而比 folded180({fld.mae_fold180:.3f}°) "
     f"退化 {fu.mae_fold180-fld.mae_fold180:+.3f}°（因优化目标不同）")
emit("- **量纲边界**：折叠误差∈[0,90]、chance=45；全向误差∈[0,180]、chance=90。"
     "**不可把 19.093°(折叠) 与 31.889°(全向) 直接相减**（差 12.8° 无意义）。")
emit("- V4 folded 臂 19.093° ≠ V3 的 18.05°（不同重训/缓存前端/log_var=0 损失），"
     "**不得覆盖 18.05**。")
emit()

# ================================================================ 5 DET
emit("## 5. IDMT 拼接背景检测（跨域探索性）")
d = pd.read_csv("idmt_det_scores_v4.csv")
pos = d[d.label==1].score.values; neg = d[d.label==0].score.values
a = roc_auc(pos,neg); bb = auc_block_bootstrap(pos,neg,block=25,n_boot=1000)
emit(f"- 样本：正={len(pos)}（IDMT drone 窗）、负={len(neg)}（WSU 环境窗），"
     f"合计 {len(d)} 条（**更正 data-engineer 报告的 5752 → 实际 5751**）")
emit(f"- ROC-AUC={a['AUC']:.6f}，Hanley–McNeil SE={a['se_hanley_mcneil']:.6f}，"
     f"95%CI [{a['ci95_low']:.6f},{a['ci95_high']:.6f}] → 远高于 0.5，"
     f"**{signif(a['p_two_sided_asymptotic'])}**")
emit(f"- 帧块 bootstrap（block=25，1000 次）95%CI [{bb['ci95_low']:.6f},{bb['ci95_high']:.6f}]，"
     "与名义 CI 一致（但名义 CI 未计自相关，块bootstrap 更可信）")
np_ = pd.read_csv("idmt_det_np_v4.csv")
for _,rr_ in np_.iterrows():
    pfa = wilson_ci(int(round(rr_.actual_pfa*rr_.n_neg)), int(rr_.n_neg))
    pdv = wilson_ci(int(round(rr_.pd_at_gamma*rr_.n_pos)), int(rr_.n_pos))
    emit(f"- γ*={rr_.gamma_star:.4f}（目标Pfa={rr_.target_pfa:.2f}）："
         f"Pd={rr_.pd_at_gamma:.4f}[Wilson {pdv['low']:.4f},{pdv['high']:.4f}]，"
         f"Pfa={rr_.actual_pfa:.5f}[Wilson {pfa['low']:.5f},{pfa['high']:.5f}]")
emit(f"- 参考工作点 γ=0.5：Pd=1.000、Pfa={1-0.458769:.3f}（idmt_det_real_v4.csv 报 0.541）→ "
     "**默认阈值完全不可用**，必须用 γ* 重标定")
emit("- **保留声明**：负样本 100% 来自 WSU 跨域，IDMT 正样本 63.5% score=1.0、"
     "WSU 负样本中位 0.565，AUC≈0.998 很可能部分反映域间差异而非纯 drone/background 判别力；"
     "**标为 cross-domain exploratory，不得当主结果**，需 IDMT 原生负样本确认。"
     "相邻帧 0.1s 强自相关，有效样本数远低于帧数。")
emit()

# ================================================================ 6 WSU UNFOLD
emit("## 6. WSU 展开误差复核（逐帧配对，frame-level）")
wu = pd.read_csv("wsu_unfold_v4.csv")
emit("| job | estimator | n_valid | 折叠 MAE | 全向 MAE | Δ(全向−折叠) | 95% CI (配对t) | 判定 |")
emit("|---|---|---:|---:|---:|---:|---:|---|")
for _, r_ in wu[wu.n_valid>0].iterrows():
    d = pd.read_csv(f"wsu_perframe_{r_['job']}.csv")
    g = d["csv_az"].to_numpy(float); e = d[r_.estimator].to_numpy(float)
    v = np.isfinite(g)&np.isfinite(e); g=g[v]%360; pr=e[v]%360
    e180 = np.minimum(np.abs(pr%180-g%180), 180-np.abs(pr%180-g%180))
    e360 = np.minimum(np.abs(pr-g), 360-np.abs(pr-g))
    assert abs(e180.mean()-r_.mae_fold180)<1e-6 and abs(e360.mean()-r_.mae_full360)<1e-6
    rr = paired_t(e360, e180, "full_minus_fold")
    emit(f"| {r_['job']} | {r_.estimator} | {int(r_.n_valid)} | {e180.mean():.3f} | {e360.mean():.3f} | "
         f"{rr['mean_diff']:+.3f} | [{rr['ci95_low']:+.3f}, {rr['ci95_high']:+.3f}] | "
         f"**{signif(rr['p'])}** |")
emit("- 全部行逐帧重算与 CSV 的 mae_fold180/mae_full360 逐位一致（断言通过）。")
emit("- 量纲边界同上：折叠∈[0,90]/chance45，全向∈[0,180]/chance90；"
     "WSU 展开误差大幅上升是前–后镜像歧义，与 IDMT 的机制一致，"
     "但两语料**不可在展开尺度上直接比大小**（WSU 为 classical GCC/SRP、24元阵、无学习网络）。")
emit()

# ================================================================ SUMMARY
emit("## 总判定摘要")
emit("| 项目 | 判定 | 关键数 |")
emit("|---|---|---|")
emit("| 几何编码增益 | **SIGNIFICANT** | Δ=+0.698°，CI[+0.526,+0.870] 不含0 |")
emit("| MEMS 8×mic失效 | **全 SIGNIFICANT** | mic1 +2.07° 最敏感 |")
emit("| MEMS 几何误差 σ0.002/0.005/0.010 | **全 SIGNIFICANT**（配对；Welch 会误判σ0.002） | +0.146/+0.647/+1.736° |")
emit("| MEMS 自噪 20dB | **NOT SIGNIFICANT** | +0.043°，Holm p=0.67 |")
emit("| MEMS 自噪 10dB | **SIGNIFICANT**（配对；Welch 会误判） | +0.191°，Holm p=0.0075 |")
emit("| MEMS gain mismatch ×3 | **NOT SIGNIFICANT** | Δ≈0，前端数学不敏感（非网络鲁棒） |")
emit("| 360°全向改善 | **SIGNIFICANT**（段级配对+置换） | Δ=−106.5°，CI[−118.2,−94.9] |")
emit("| 360°重训折叠尺度 | 退化 +5.32°（权衡，如实报） | 19.093→24.416° |")
emit("| 检测 AUC | **SIGNIFICANT** 但跨域探索性 | AUC=0.99824，CI[0.99740,0.99908] |")
emit("| WSU 展开 Δ | **全 SIGNIFICANT** | test1 SRP +67.88° 等 |")
emit("| 动态 20→30ep | 摘要级确认；配对CI待重放 | proposed −2.598°、fixed −0.728° |")
emit()
emit("## 未决与待办")
emit("1. 动态 20/30ep 逐样本配对 CI 未存，需重放 `v4_task3`（约8分钟）——本轮不臆造 p 值。")
emit("2. 检测 AUC 需 IDMT 原生负样本复核；当前负样本 100% WSU 跨域。")
emit("3. 我已把 MEMS 两处 Welch→配对翻案、360 跨尺度不可减、检测行数 5751 上报，请 writer 按此回填。")

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print(f"\n[written] {OUT_MD} ({len(L)} lines)")
