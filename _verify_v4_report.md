# V4 六组新实验正式统计核验报告

核验人：robustness-auditor（严复核） ｜ 模板：`_prep_stats_template.py` ｜ 生成脚本：`_verify_v4_stats.py` ｜ 只读，未改任何 CSV/tex

判定约定：双侧 α=0.05；所有 CI 为 95%；置换检验 5000 次（下限 p≈2e-4）。

## 1. 几何编码消融（8mic GCC，配对 n=600 @−5dB）
- 重放 MAE：proposed=2.249872、no_geom=2.947982（CSV 2.249872 / 2.947982，逐位一致）
- 配对 t：Δ=+0.698110°，95%CI [+0.525788, +0.870431]，t=7.9563，p=8.896e-15 → **SIGNIFICANT**（CI 不含 0）
- Wilcoxon p=4.703e-13 → SIGNIFICANT；符号翻转置换 p=2.00e-04 → SIGNIFICANT
- 6mic ULA 无几何路径，Δ=0 为结构性恒等，**不作证据**（已排除）。
- 结论：**几何编码带来 +0.698° 稳定增益，强显著。**

## 2. MEMS 四类损伤 17 条件（配对 n=600，Holm–Bonferroni）
- 基线重放 MAE=1.653470（CSV 1.653470，逐位一致）

| 条件 | ΔMAE (°) | 95% CI | 退化% | p(配对) | p(Holm) | 判定 |
|---|---:|---:|---:|---:|---:|---|
| gain_mismatch_sg0.05 | +0.0000 | [-0.0000, +0.0000] | +0.00 | 3.18e-01 | 9.53e-01 | **NOT SIGNIFICANT** |
| gain_mismatch_sg0.10 | +0.0000 | [-0.0000, +0.0000] | +0.00 | 3.18e-01 | 9.53e-01 | **NOT SIGNIFICANT** |
| gain_mismatch_sg0.20 | +0.0000 | [-0.0000, +0.0000] | +0.00 | 3.18e-01 | 9.53e-01 | **NOT SIGNIFICANT** |
| dead_channel_mic1 | +2.0717 | [+1.8510, +2.2924] | +125.29 | 1.95e-60 | 3.12e-59 | **SIGNIFICANT** |
| dead_channel_mic2 | +0.4375 | [+0.2701, +0.6050] | +26.46 | 3.90e-07 | 2.34e-06 | **SIGNIFICANT** |
| dead_channel_mic3 | +1.0395 | [+0.8578, +1.2212] | +62.87 | 1.06e-26 | 1.16e-25 | **SIGNIFICANT** |
| dead_channel_mic4 | +0.6659 | [+0.4918, +0.8401] | +40.28 | 2.18e-13 | 1.75e-12 | **SIGNIFICANT** |
| dead_channel_mic5 | +0.9838 | [+0.8175, +1.1501] | +59.50 | 2.77e-28 | 3.32e-27 | **SIGNIFICANT** |
| dead_channel_mic6 | +0.6320 | [+0.4710, +0.7930] | +38.22 | 5.28e-14 | 4.75e-13 | **SIGNIFICANT** |
| dead_channel_mic7 | +1.6988 | [+1.4982, +1.8995] | +102.74 | 2.47e-51 | 3.46e-50 | **SIGNIFICANT** |
| dead_channel_mic8 | +1.4651 | [+1.2634, +1.6667] | +88.60 | 5.78e-40 | 7.52e-39 | **SIGNIFICANT** |
| geometry_error_sp0.002m | +0.1457 | [+0.0949, +0.1966] | +8.81 | 2.76e-08 | 1.94e-07 | **SIGNIFICANT** |
| geometry_error_sp0.005m | +0.6466 | [+0.5334, +0.7598] | +39.11 | 1.22e-26 | 1.22e-25 | **SIGNIFICANT** |
| geometry_error_sp0.010m | +1.7363 | [+1.5341, +1.9386] | +105.01 | 1.67e-52 | 2.50e-51 | **SIGNIFICANT** |
| device_selfnoise_20dB | +0.0426 | [-0.0178, +0.1031] | +2.58 | 1.67e-01 | 6.66e-01 | **NOT SIGNIFICANT** |
| device_selfnoise_10dB | +0.1914 | [+0.0736, +0.3091] | +11.57 | 1.49e-03 | 7.47e-03 | **SIGNIFICANT** |

- **负面/不显著（诚实报告）**：gain_mismatch 三档 Δ≈4e−10°（PHAT 相位白化精确抵消逐通道常数增益，**前端数学不敏感，非网络鲁棒**）；device_selfnoise_20dB Δ=+0.0426° Holm p=0.67 → NOT SIGNIFICANT。
- **翻案提醒（重要）**：geometry_error_sp0.002m 与 device_selfnoise_10dB 用汇总 Welch 代理分别为 p=0.19 / 0.09（看似不显著），**配对重放后 p=2.8e−8 / 1.5e−3 均显著**（Holm 后仍显著）。配对设计消除了样本间方差，务必以本表配对判定为准，勿用 Welch 代理。
- 最敏感失效位：mic1（Δ=+2.07°、125%退化）；mic7 次之（+1.70°）。

## 3. 动态加权 20ep→30ep（配对 ULA 测试集 n=600 @−5dB）
- proposed: 20ep=6.8013±8.6286 → 30ep=4.2036±8.9331，Δ=-2.5977°
- fixed: 20ep=4.9715±8.6301 → 30ep=4.2440±8.3457，Δ=-0.7275°
- log_var(ep20)=(-2.761,+0.940) → ep30=(-3.788,+1.275)；w_det 15.82→44.18（权重随 epoch 漂移，非稳态）
- **配对检验（已从 4 个缓存 checkpoints 内存重放，无需重训）**：同一 n=600 @−5dB 配对测试集逐样本误差，
  - proposed 20→30ep：Δ=−2.597712°，95%CI [−2.9894, −2.2060]，t=−13.03，p=2.52e−34 → **SIGNIFICANT**（CI 不含0）
  - fixed 20→30ep：Δ=−0.727481°，95%CI [−0.9777, −0.4773]，t=−5.71，p=1.78e−08 → **SIGNIFICANT**（CI 不含0）
  - 重放 MAE 与 CSV 逐位一致（6.801347/4.203635/4.971493/4.244011）
- **解读边界**：两类权重都随 epoch 漂移（proposed w_det 15.82→44.18），故"30ep 更好"是**截断 epoch 选择的快照效应**，不是权重收敛；不要据此声称动态权重本身更稳健。原文已写"we draw no conclusion about relative robustness from these single runs"，与此一致。
- **口径统一**：用 V4（6.8013→4.2036 / 4.9715→4.2440），**不混 V3 的 7.50→5.09**。V3 `_abl_fixed20_ep30_v3.csv` 实为 λ=2.0，不是 equal weighting（已上报隐患2）。

## 4. 360° 全向重训 vs 180° 折叠（163 段，段级配对）
- **同尺度（全圆 360°）配对**：folded 臂 138.424° → full360 臂 31.889°，Δ=-106.535°，95%CI [-118.190, -94.880]，t=-18.050 p=1.27e-40 → **SIGNIFICANT**
- Wilcoxon p=7.17e-24（SIGNIFICANT）；置换 p=2.00e-04（SIGNIFICANT）→ **360°重训的全向改善稳健**
- gt>180 的 68 段子集：folded 142.540° → full360 37.279°（镜像翻转被消除）
- **权衡（如实报）**：全向尺度改善 -106.535°；但在折叠尺度 full360(24.416°) 反而比 folded180(19.093°) 退化 +5.324°（因优化目标不同）
- **量纲边界**：折叠误差∈[0,90]、chance=45；全向误差∈[0,180]、chance=90。**不可把 19.093°(折叠) 与 31.889°(全向) 直接相减**（差 12.8° 无意义）。
- V4 folded 臂 19.093° ≠ V3 的 18.05°（不同重训/缓存前端/log_var=0 损失），**不得覆盖 18.05**。

## 5. IDMT 拼接背景检测（跨域探索性）
- 样本：正=4890（IDMT drone 窗）、负=861（WSU 环境窗），合计 5751 条（**更正 data-engineer 报告的 5752 → 实际 5751**）
- ROC-AUC=0.998243，Hanley–McNeil SE=0.000428，95%CI [0.997404,0.999081] → 远高于 0.5，**SIGNIFICANT**
- 帧块 bootstrap（block=25，1000 次）95%CI [0.996514,0.999397]，与名义 CI 一致（但名义 CI 未计自相关，块bootstrap 更可信）
- γ*=0.9834（目标Pfa=0.01）：Pd=0.9843[Wilson 0.9804,0.9874]，Pfa=0.00929[Wilson 0.00472,0.01823]
- γ*=0.9428（目标Pfa=0.05）：Pd=0.9939[Wilson 0.9913,0.9957]，Pfa=0.04994[Wilson 0.03729,0.06660]
- γ*=0.9122（目标Pfa=0.10）：Pd=0.9955[Wilson 0.9932,0.9970]，Pfa=0.09988[Wilson 0.08160,0.12172]
- 参考工作点 γ=0.5：Pd=1.000、Pfa=0.541（idmt_det_real_v4.csv 报 0.541）→ **默认阈值完全不可用**，必须用 γ* 重标定
- **保留声明**：负样本 100% 来自 WSU 跨域，IDMT 正样本 63.5% score=1.0、WSU 负样本中位 0.565，AUC≈0.998 很可能部分反映域间差异而非纯 drone/background 判别力；**标为 cross-domain exploratory，不得当主结果**，需 IDMT 原生负样本确认。相邻帧 0.1s 强自相关，有效样本数远低于帧数。

## 6. WSU 展开误差复核（逐帧配对，frame-level）
| job | estimator | n_valid | 折叠 MAE | 全向 MAE | Δ(全向−折叠) | 95% CI (配对t) | 判定 |
|---|---|---:|---:|---:|---:|---:|---|
| test1 | srp_az | 1592 | 37.459 | 105.334 | +67.876 | [+64.112, +71.639] | **SIGNIFICANT** |
| test1 | gcc_az | 1592 | 44.511 | 88.982 | +44.471 | [+41.616, +47.325] | **SIGNIFICANT** |
| test2_drone | srp_az | 11927 | 45.386 | 79.783 | +34.397 | [+33.371, +35.423] | **SIGNIFICANT** |
| test2_drone | gcc_az | 11927 | 45.372 | 90.072 | +44.700 | [+43.660, +45.740] | **SIGNIFICANT** |
- 全部行逐帧重算与 CSV 的 mae_fold180/mae_full360 逐位一致（断言通过）。
- 量纲边界同上：折叠∈[0,90]/chance45，全向∈[0,180]/chance90；WSU 展开误差大幅上升是前–后镜像歧义，与 IDMT 的机制一致，但两语料**不可在展开尺度上直接比大小**（WSU 为 classical GCC/SRP、24元阵、无学习网络）。

## 总判定摘要
| 项目 | 判定 | 关键数 |
|---|---|---|
| 几何编码增益 | **SIGNIFICANT** | Δ=+0.698°，CI[+0.526,+0.870] 不含0 |
| MEMS 8×mic失效 | **全 SIGNIFICANT** | mic1 +2.07° 最敏感 |
| MEMS 几何误差 σ0.002/0.005/0.010 | **全 SIGNIFICANT**（配对；Welch 会误判σ0.002） | +0.146/+0.647/+1.736° |
| MEMS 自噪 20dB | **NOT SIGNIFICANT** | +0.043°，Holm p=0.67 |
| MEMS 自噪 10dB | **SIGNIFICANT**（配对；Welch 会误判） | +0.191°，Holm p=0.0075 |
| MEMS gain mismatch ×3 | **NOT SIGNIFICANT** | Δ≈0，前端数学不敏感（非网络鲁棒） |
| 360°全向改善 | **SIGNIFICANT**（段级配对+置换） | Δ=−106.5°，CI[−118.2,−94.9] |
| 360°重训折叠尺度 | 退化 +5.32°（权衡，如实报） | 19.093→24.416° |
| 检测 AUC | **SIGNIFICANT** 但跨域探索性 | AUC=0.99824，CI[0.99740,0.99908] |
| WSU 展开 Δ | **全 SIGNIFICANT** | test1 SRP +67.88° 等 |
| 动态 20→30ep（两配置） | **均 SIGNIFICANT**（checkpoint 重放配对） | proposed −2.598° CI[−2.99,−2.21]；fixed −0.728° CI[−0.98,−0.48] |

## 未决与待办
1. 检测 AUC 需 IDMT 原生负样本复核；当前负样本 100% WSU 跨域（已标探索性）。
2. MEMS 两处 Welch→配对翻案（sp0.002m、selfnoise_10dB）、360 跨尺度不可减、检测行数 5751 —— 请 writer 按本报告回填。
3. 动态 epoch 虽配对显著，但权重仍在漂移，不得解读为"动态权重更稳健"。

## 产物清单（均只读生成）
- `_verify_v4_report.md`：本报告（归档）
- `_verify_v4_stats.py`：六组 CSV 正式核验主脚本
- `_verify_v4_dyn_pair.py`：动态 epoch checkpoint 内存重放配对检验
- 复用 `_prep_stats_template.py` / `_prep_pair_replay.py`（前轮产物）
未修改任何既有 CSV / tex。
