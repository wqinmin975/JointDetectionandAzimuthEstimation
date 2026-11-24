#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_verify_v4_dyn_pair.py -- recover paired ULA error arrays from the four
cached epoch checkpoints (no training, no file writes) and test 20ep vs 30ep.
"""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0,'.')
import drone_acoustic_joint_v3 as v3
from drone_acoustic_v4_tasks12 import make_ula_testset, eval_ula_fixed
from _prep_stats_template import paired_t

xt, gt = make_ula_testset(n=600, snr_db=-5.0, seed=0)
def load(fp):
    m = v3.JointDetDOANetRaw().to('cpu')
    m.load_state_dict(torch.load(fp, map_location='cpu', weights_only=True)); m.eval(); return m
E = {}
for mode in ('proposed','fixed'):
    for ep in (20,30):
        m = load(os.path.join(os.getcwd(), f'dyn_{mode}_ep{ep}_v4.pt'))
        mae, mstd, ang, err = eval_ula_fixed(m, xt, gt)
        E[(mode,ep)] = err
        print(f'{mode:9s} ep{ep}: MAE={mae:.6f} std={mstd:.6f} (CSV compare)')
print()
for mode in ('proposed','fixed'):
    r = paired_t(E[(mode,30)], E[(mode,20)], 'ep30_minus_ep20')
    print(f'{mode:9s} 20->30ep paired: delta={r["mean_diff"]:+.6f} '
          f'95%CI=[{r["ci95_low"]:+.6f},{r["ci95_high"]:+.6f}] '
          f't={r["t"]:.4f} p={r["p"]:.3e} -> '
          f'{"SIGNIFICANT" if r["p"]<0.05 else "NOT SIGNIFICANT"}')
