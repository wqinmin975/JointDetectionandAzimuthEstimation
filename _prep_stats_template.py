#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_prep_stats_template.py -- reusable statistical-test templates for R3 new experiments.

STATUS: PREPARATION ONLY (read-only w.r.t. all existing CSV / TeX deliverables).
        This script defines functions; running it prints a self-test / dry demo.
        It never writes to an existing file. Pass --out <new _prep*.csv> to dump
        results to a NEW file only.

Author: robustness-auditor (Yan Fuku)  |  Team: drone-paper-round3
Python: C:/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe
Deps  : numpy, pandas, scipy (statsmodels NOT installed -- everything below is
        implemented with numpy/scipy only).

Covers the four new-experiment test plans requested by team-lead:
  (B1) Geometry-MLP ablation: proposed vs no-geom -- paired t (if identical
       samples) else Welch t; report dMAE +/- 95% CI.
  (B2) MEMS four damage families vs no-damage baseline: paired test + relative
       degradation % CI; multi-comparison correction; honest null reporting.
  (B3) IDMT 360-degree full-circle vs 180-degree folded: paired-by-segment
       comparison + EXPLICIT scale boundary (folded <=90 deg, full <=180 deg).
  (B4) Detection Pd / Pfa / ROC-AUC: Wilson CI for proportions, Hanley-McNeil
       (DeLong-free) AUC SE/CI matching wsu_auc_ci.csv, plus a frame-block
       bootstrap option because 0.1-s frames are strongly autocorrelated.

Every function returns a plain dict so callers can assemble a results table.
Significance convention: two-sided alpha=0.05 unless overridden; every result
carries an explicit verdict string ("PASS"/"FAIL"/"NOT SIGNIFICANT") so that
negative findings are not silently dropped.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _as_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    return a


def wilson_ci(k: int, n: int, alpha: float = ALPHA) -> dict:
    """Wilson score interval for a proportion (better than Wald for small n,
    e.g. the n_ambient=54 test1 background tail in wsu_auc_ci.csv)."""
    if n <= 0:
        return dict(p=float("nan"), low=float("nan"), high=float("nan"), n=n)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return dict(p=p, low=max(0.0, ctr - half), high=min(1.0, ctr + half), n=n)


def paired_t(x, y, delta_name: str = "diff", alpha: float = ALPHA) -> dict:
    """Paired t-test on d_i = x_i - y_i.

    Use ONLY when x and y are measured on the SAME units (same segments /
    same Monte-Carlo sample index / same seed_test). n is the number of
    finite paired differences after alignment, NOT the number of raw rows.
    """
    x, y = _as_array(x), _as_array(y)
    if x.size != y.size:
        raise ValueError(f"paired_t needs equal lengths, got {x.size} vs {y.size}")
    d = x - y
    n = d.size
    if n < 2:
        raise ValueError("paired_t needs n>=2")
    md, sd = d.mean(), d.std(ddof=1)
    se = sd / math.sqrt(n)
    t = md / se if se > 0 else float("nan")
    p = 2 * stats.t.sf(abs(t), df=n - 1) if se > 0 else 1.0
    crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return dict(
        test="paired_t", delta_name=delta_name, n=n,
        mean_x=float(x.mean()), mean_y=float(y.mean()),
        mean_diff=float(md), sd_diff=float(sd), se_diff=float(se),
        t=float(t), df=n - 1, p=float(p),
        ci95_low=float(md - crit * se), ci95_high=float(md + crit * se),
        verdict="SIGNIFICANT" if p < alpha else "NOT SIGNIFICANT",
    )


def welch_t(m1, s1, n1, m2, s2, n2, delta_name: str = "m1-m2", alpha=ALPHA) -> dict:
    """Non-paired Welch t from summary statistics.

    This is the test used for the V3 GCC/SRP classical-vs-network comparison
    in welch_test.csv (n_classical=60, n_proposed=422), and is the correct
    fallback for geometry ablation if the two arms use independent draws.
    """
    v1, v2 = s1 * s1 / n1, s2 * s2 / n2
    se = math.sqrt(v1 + v2)
    diff = m1 - m2
    df = (v1 + v2) ** 2 / (v1 * v1 / (n1 - 1) + v2 * v2 / (n2 - 1))
    t = diff / se
    p = 2 * stats.t.sf(abs(t), df=df)
    crit = stats.t.ppf(1 - alpha / 2, df=df)
    return dict(
        test="welch_t", delta_name=delta_name,
        n1=int(n1), n2=int(n2), mean1=float(m1), mean2=float(m2),
        mean_diff=float(diff), se_diff=float(se),
        t=float(t), df=float(df), p=float(p),
        ci95_low=float(diff - crit * se), ci95_high=float(diff + crit * se),
        verdict="SIGNIFICANT" if p < alpha else "NOT SIGNIFICANT",
    )


def holm_bonferroni(pvals: Iterable[float], alpha: float = ALPHA):
    """Holm step-down adjustment for the 16 MEMS conditions tested against
    one shared baseline. Returns (adjusted_p, reject_flags)."""
    p = np.asarray(list(pvals), dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj, adj < alpha


# ---------------------------------------------------------------------------
# B1 -- geometry-MLP ablation
# ---------------------------------------------------------------------------
def geom_ablation(proposed_err, nogeom_err, *, paired: bool,
                  summary=None, alpha=ALPHA) -> dict:
    """Delta MAE (no-geom MINUS proposed; positive = geometry helps).

    Parameters
    ----------
    proposed_err, nogeom_err : per-sample error arrays (REQUIRED if paired).
    paired : True only when both arms were scored on the identical samples
             (same seed_test / same sample IDs). See geom_ablation_notes_v4.json:
             the 8-mic GCC branch IS paired (n=600, seed_test=20240517).
    summary : optional dict(m1,s1,n1,m2,s2,n2) for Welch when only aggregate
              columns exist (geom_ablation_v4.csv stores only MAE/std/n).

    NOTE: the 6-mic ULA branch has NO geometry encoder (see notes json), so a
    'no_geom' delta there is structurally exactly 0 and is NOT informative.
    """
    if paired:
        r = paired_t(nogeom_err, proposed_err, "no_geom_minus_proposed", alpha)
    else:
        if summary is None:
            raise ValueError("Welch needs summary=dict(m1,s1,n1,m2,s2,n2)")
        # m1 = no-geom arm, m2 = proposed arm -> positive delta = geometry helps
        r = welch_t(summary["m1"], summary["s1"], summary["n1"],
                    summary["m2"], summary["s2"], summary["n2"],
                    "no_geom_minus_proposed", alpha)
    r["experiment"] = "geom_ablation"
    r["direction"] = "positive delta means geometry encoding reduces MAE"
    return r


# ---------------------------------------------------------------------------
# B2 -- MEMS impairment robustness
# ---------------------------------------------------------------------------
def mems_condition(base_err, cond_err, *, name: str, paired: bool = True,
                   n_boot: int = 2000, seed: int = 0, alpha=ALPHA) -> dict:
    """One impairment condition vs the shared no-damage baseline.

    Reports (i) paired absolute delta MAE with t-based CI, and (ii) relative
    degradation % = (MAE_cond - MAE_base)/MAE_base*100 with a BOOTSTRAP CI
    over paired units (ratio CI via delta method is unreliable when the
    baseline MAE is small, as it is here ~1.65 deg).

    Honest-null rule: gain-mismatch arms in mems_robustness_v4.csv show
    delta ~= 4e-10 deg; they MUST be reported as NOT SIGNIFICANT, not omitted.
    """
    base_err, cond_err = _as_array(base_err), _as_array(cond_err)
    if paired and base_err.size != cond_err.size:
        raise ValueError("paired MEMS comparison needs equal-length error arrays")
    r = paired_t(cond_err, base_err, "cond_minus_baseline", alpha)
    mae_b, mae_c = base_err.mean(), cond_err.mean()
    rel = (mae_c - mae_b) / mae_b * 100.0 if mae_b > 0 else float("nan")

    rng = np.random.default_rng(seed)
    n = base_err.size
    if n > 0:
        idx = rng.integers(0, n, size=(n_boot, n))
        boot = (cond_err[idx].mean(axis=1) - base_err[idx].mean(axis=1)) / mae_b * 100.0
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    else:
        lo = hi = float("nan")
    r.update(experiment="mems_robustness", condition=name,
             mae_baseline=float(mae_b), mae_condition=float(mae_c),
             rel_degradation_pct=float(rel),
             rel_ci95_low=float(lo), rel_ci95_high=float(hi))
    return r


# ---------------------------------------------------------------------------
# B3 -- 360-degree full circle vs 180-degree folded
# ---------------------------------------------------------------------------
def full_vs_folded(folded_err, full_err, *, segment_ids=None,
                   n_boot=2000, seed=0, alpha=ALPHA) -> dict:
    """Paired-by-segment comparison of two DIFFERENT error definitions.

    >>> SCALE BOUNDARY (must be stated in the manuscript) <<<
      folded 180-deg circular error: max 90 deg, chance level 45 deg.
      full 360-deg circular error  : max 180 deg, chance level 90 deg.
    The two are NOT the same measurement scale. A paired t-test is valid as a
    description of the within-segment shift but the magnitude difference is
    LARGELY DETERMINISTIC: every segment with GT in (180,360] is mirrored by
    a folded-only output, so the increase is a label-space consequence, not a
    stochastic training effect. Report the mirrored-segment count/fraction
    alongside the paired CI, and do NOT phrase the result as 'one method is
    significantly worse' in the usual causal sense.
    """
    folded_err, full_err = _as_array(folded_err), _as_array(full_err)
    if folded_err.size != full_err.size:
        raise ValueError("full vs folded must be paired segment-by-segment")
    r = paired_t(full_err, folded_err, "full_minus_folded", alpha)
    rng = np.random.default_rng(seed)
    n = folded_err.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = (full_err[idx] - folded_err[idx]).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    r.update(experiment="idmt_360_vs_180", n_segments=n,
             folded_max_possible_deg=90.0, full_max_possible_deg=180.0,
             folded_chance_deg=45.0, full_chance_deg=90.0,
             boot_ci95_low=float(lo), boot_ci95_high=float(hi),
             scale_note="different units; paired CI is descriptive, "
                        "mirrored-segment fraction is the mechanism",
             segment_ids=list(segment_ids) if segment_ids is not None else None)
    return r


# ---------------------------------------------------------------------------
# B4 -- detection Pd / Pfa / ROC-AUC
# ---------------------------------------------------------------------------
def roc_auc(scores_pos, scores_neg, *, higher_is_positive=True) -> dict:
    """AUC by the Mann-Whitney U statistic (== trapezoidal ROC AUC), with
    Hanley-McNeil SE/CI exactly as in wsu_auc_ci.csv:
        Q1 = A/(2-A), Q2 = 2A^2/(1+A)
        SE = sqrt( [A(1-A) + (n1-1)(Q1-A^2) + (n0-1)(Q2-A^2)] / (n1*n0) )
    Verified to reproduce all six rows of wsu_auc_ci.csv to 1e-6.
    """
    sp, sn = _as_array(scores_pos), _as_array(scores_neg)
    n1, n0 = sp.size, sn.size
    if n1 == 0 or n0 == 0:
        raise ValueError("need both positive and negative scores")
    # U = # of (pos,neg) pairs correctly ordered; ties count 0.5
    if higher_is_positive:
        gt = (sp[:, None] > sn[None, :]).sum()
        eq = (sp[:, None] == sn[None, :]).sum()
    else:
        gt = (sp[:, None] < sn[None, :]).sum()
        eq = (sp[:, None] == sn[None, :]).sum()
    A = (gt + 0.5 * eq) / (n1 * n0)
    Q1 = A / (2 - A)
    Q2 = 2 * A * A / (1 + A)
    se = math.sqrt(max(0.0, (A * (1 - A) + (n1 - 1) * (Q1 - A * A)
                              + (n0 - 1) * (Q2 - A * A)) / (n1 * n0)))
    z = (A - 0.5) / se if se > 0 else float("nan")
    p = 2 * stats.norm.sf(abs(z)) if se > 0 else 1.0
    return dict(AUC=float(A), se_hanley_mcneil=float(se),
                ci95_low=float(max(0.0, A - 1.96 * se)),
                ci95_high=float(min(1.0, A + 1.96 * se)),
                z_vs_0p5=float(z), p_two_sided_asymptotic=float(p),
                n_drone=n1, n_ambient=n0,
                verdict="ABOVE CHANCE" if (A - 1.96 * se) > 0.5 else
                        ("BELOW CHANCE" if (A + 1.96 * se) < 0.5 else "NOT DISTINGUISHABLE FROM CHANCE"))


def auc_block_bootstrap(scores_pos, scores_neg, *, block=25, n_boot=1000,
                        seed=0, alpha=ALPHA, higher_is_positive=True) -> dict:
    """Autocorrelation-aware AUC CI: resample CONTIGUOUS frame blocks rather
    than i.i.d. frames. Consecutive WSU frames are 0.1 s apart and strongly
    correlated; the nominal Hanley-McNeil CI treats them as independent.

    This is the quantitative companion to the manuscript's retained caveat:
    'test1 contributes only 54 background frames from one 5.3-s segment, so
    the effective sample size is far below the frame count.'
    """
    sp, sn = _as_array(scores_pos), _as_array(scores_neg)
    rng = np.random.default_rng(seed)

    def _resample(a):
        nb = max(1, math.ceil(a.size / block))
        starts = rng.integers(0, max(1, a.size - block + 1), size=nb)
        parts = [a[s:s + block] for s in starts]
        return np.concatenate(parts) if parts else a

    boots = []
    for _ in range(n_boot):
        b1, b0 = _resample(sp), _resample(sn)
        if b1.size == 0 or b0.size == 0:
            continue
        gt = (b1[:, None] > b0[None, :]).sum() if higher_is_positive else \
             (b1[:, None] < b0[None, :]).sum()
        eq = (b1[:, None] == b0[None, :]).sum()
        boots.append((gt + 0.5 * eq) / (b1.size * b0.size))
    arr = np.asarray(boots)
    return dict(method="frame_block_bootstrap", block_frames=block,
                n_boot=int(arr.size),
                ci95_low=float(np.percentile(arr, 100 * alpha / 2)),
                ci95_high=float(np.percentile(arr, 100 * (1 - alpha / 2))),
                note="block bootstrap still assumes blocks are independent; "
                     "report alongside nominal Hanley-McNeil CI, not instead of it")


def pd_pfa_ci(n_hit, n_pos, n_fa, n_neg, alpha=ALPHA) -> dict:
    """Pd / Pfa point estimates with Wilson CIs at a fixed operating point."""
    return dict(Pd=wilson_ci(n_hit, n_pos, alpha),
                Pfa=wilson_ci(n_fa, n_neg, alpha),
                note="IDMT has NO background-only negatives: Pfa/Pd there are "
                     "undefined; use WSU / stitched-background set instead")


# ---------------------------------------------------------------------------
# Dry self-test / demonstration (no files written)
# ---------------------------------------------------------------------------
def _selftest() -> None:
    print("=" * 68)
    print("PREP-STATS SELF-TEST (no files written)")
    print("=" * 68)

    # Hanley-McNeil must reproduce wsu_auc_ci.csv
    ref = pd.read_csv("wsu_auc_ci.csv")
    ok = True
    # analytic check of the SE formula used inside roc_auc()
    for _, r in ref.iterrows():
        A, n1, n0 = r.AUC, int(r.n_drone), int(r.n_ambient)
        Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
        se = math.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A * A)
                        + (n0 - 1) * (Q2 - A * A)) / (n1 * n0))
        match = abs(se - r.se_hanley_mcneil) < 2e-6
        ok &= match
        print(f"  HM SE {r.scope:10s}/{r.detector:10s} "
              f"{se:.6f} vs {r.se_hanley_mcneil:.6f}  {'OK' if match else 'MISMATCH'}")
    print(f"  Hanley-McNeil reproduction: {'PASS' if ok else 'FAIL'}")

    # Welch must reproduce welch_test.csv
    w = pd.read_csv("welch_test.csv")
    for _, r in w.iterrows():
        out = welch_t(r.mean_classical, r.std_classical, int(r.n_classical),
                      r.mean_proposed, r.std_proposed, int(r.n_proposed))
        print(f"  Welch {r.comparison[:28]:28s} diff={out['mean_diff']:.4f} "
              f"CI=[{out['ci95_low']:.4f},{out['ci95_high']:.4f}] "
              f"p={out['p']:.2e} -- file "
              f"[{r.ci95_low:.4f},{r.ci95_high:.4f}]")

    # Paired demo with the already-produced V4 geometry summary
    g = pd.read_csv("geom_ablation_v4.csv")
    row = g[g.variant == "no_geom"].iloc[0]
    prop = g[g.variant == "proposed_with_geom"].iloc[0]
    demo = geom_ablation(None, None, paired=False,
                         summary=dict(m1=row.mae_deg, s1=row.mae_std, n1=int(row.n),
                                      m2=prop.mae_deg, s2=prop.mae_std, n2=int(prop.n)))
    print("  Geometry ablation (Welch from summary, 8-mic GCC):")
    print(f"    dMAE={demo['mean_diff']:.4f} deg CI=[{demo['ci95_low']:.4f},"
          f"{demo['ci95_high']:.4f}] p={demo['p']:.3g} -> {demo['verdict']}")
    print("  NOTE: true 8-mic arm is PAIRED (n=600); geom_ablation_notes_v4.json "
          "reports paired CI=(+0.526,+0.870) deg. Prefer paired_t() on raw errors "
          "when the per-sample file is available.")
    print("=" * 68)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="reproduce welch_test.csv / wsu_auc_ci.csv from the formulas")
    ap.add_argument("--out", default=None,
                    help="NEW _prep*.csv path; never overwrite an existing file")
    args = ap.parse_args()
    if args.selftest or args.out is None:
        _selftest()
    if args.out:
        import os
        if os.path.exists(args.out):
            raise SystemExit(f"Refusing to overwrite existing file: {args.out}")
        if not os.path.basename(args.out).startswith("_prep"):
            raise SystemExit("Output filename must start with '_prep'")
        print(f"(no tabular results assembled yet; template only) --out={args.out}")
