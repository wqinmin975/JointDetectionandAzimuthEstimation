"""Neyman-Pearson operating points for the cross-domain detector (read-only on
idmt_det_scores_v4.csv; writes a NEW idmt_det_np_v4.csv)."""
import os
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(OUT, "idmt_det_scores_v4.csv"))
pos = d[d.label == 1].score.to_numpy()
neg = d[d.label == 0].score.to_numpy()
grid = np.linspace(0.0, 1.0, 10001)
pfa = np.array([(neg > g).mean() for g in grid])
rows = []
for target in (0.01, 0.05, 0.10):
    ok = np.where(pfa <= target)[0]
    g = float(grid[ok[0]]) if len(ok) else float("nan")
    rows.append(dict(target_pfa=target, gamma_star=g,
                     actual_pfa=float(pfa[ok[0]]) if len(ok) else float("nan"),
                     pd_at_gamma=float((pos > g).mean()) if len(ok) else float("nan"),
                     n_pos=len(pos), n_neg=len(neg)))
df = pd.DataFrame(rows)
p = os.path.join(OUT, "idmt_det_np_v4.csv")
df.to_csv(p, index=False)
print(df.to_string(index=False))
print("saved", p)
