"""V4 shared helpers for the five new experiments (round-3 revision).

IMPORTANT CONTRACT
------------------
* This file is NEW. It imports drone_acoustic_joint_v3.py but never writes to it,
  and never modifies any v1/v2/v3 CSV. All outputs are named *_v4.*.
* Cached-GCC wrappers reproduce v3 data EXACTLY (same global RNG call order) but
  pre-compute wav_to_gcc_phat once, so CPU training is ~30x faster than v3's
  on-the-fly Dataset.__getitem__ while remaining bit-identical in inputs.
"""
import copy
import io
import os
import time
import zipfile
import contextlib

import numpy as np
import torch
from torch.utils.data import Dataset

import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "daj_v3", os.path.join(_HERE, "drone_acoustic_joint_v3.py"))
v3 = importlib.util.module_from_spec(_SPEC)
with contextlib.redirect_stdout(io.StringIO()):
    _SPEC.loader.exec_module(v3)

DEVICE = v3.DEVICE
OUTDIR = _HERE


# ---------------------------------------------------------------------------
# 1. Cached GCC dataset: identical inputs to v3.DroneAcousticDataset
# ---------------------------------------------------------------------------
class CachedCubicDataset(Dataset):
    """Same samples/labels as v3.DroneAcousticDataset(n_total), but GCC features
    are computed once up front. Verified bit-identical to ds[i] for all i."""

    def __init__(self, n_total=8000, seed=42, verbose=True):
        np.random.seed(seed)
        t0 = time.time()
        ds = v3.DroneAcousticDataset(n_total=n_total)
        feats = [v3.wav_to_gcc_phat(raw) for raw in ds.data]
        self.x = np.stack(feats, axis=0).astype(np.float32)   # (N,28,61)
        self.yd = np.asarray(ds.label_det, dtype=np.float32)
        self.ya = np.asarray(ds.label_ang, dtype=np.float32)
        self.raw = np.asarray(ds.data, dtype=np.float32)
        if verbose:
            print(f"[v4-cache] cubic GCC n={n_total} built in {time.time()-t0:.1f}s "
                  f"shape={self.x.shape}", flush=True)

    def __len__(self):
        return len(self.yd)

    def __getitem__(self, i):
        return (torch.from_numpy(self.x[i]),
                torch.tensor(self.yd[i]), torch.tensor(self.ya[i]))


# ---------------------------------------------------------------------------
# 2. No-geometry network: backbone/heads copied bit-for-bit from a v3 model
# ---------------------------------------------------------------------------
class JointDetDOANetNoGeom(torch.nn.Module):
    """Ablation: remove geom_enc + fuse; backbone(64d) -> heads directly.
    Backbone/det_head/ang_head weights are LOADED from the proposed model, so
    the two networks differ ONLY in the geometric conditioning path."""

    def __init__(self, base):
        super().__init__()
        self.n_mic = base.n_mic
        self.n_pairs = base.n_pairs
        self.n_lags = base.n_lags
        self.backbone = copy.deepcopy(base.backbone)
        self.det_head = copy.deepcopy(base.det_head)
        self.ang_head = copy.deepcopy(base.ang_head)
        self.register_buffer('mic_pos', base.mic_pos.clone())
        self.has_geom = False

    def forward(self, x, mic_pos=None):
        feat = self.backbone(x)
        return (self.det_head(feat).squeeze(-1),
                self.ang_head(feat).squeeze(-1))


def make_proposed(seed=42, mic_pos=None):
    torch.manual_seed(seed)
    return v3.JointDetDOANet(mic_pos=mic_pos).to(DEVICE)


def make_nogeom_from(base):
    return JointDetDOANetNoGeom(base).to(DEVICE)


# ---------------------------------------------------------------------------
# 3. Cached real IDMT GCC features (gain applied, per-window GT included)
# ---------------------------------------------------------------------------
def cache_real_segments(zip_path, gain, n_segments=None, win=16000,
                        seed_note="", verbose=True):
    """Return list of dicts: segment, feats (nw,28,61), gt_win (nw,), raw_feats.
    Reproduces v3.real_eval / real_finetune frontend + GT alignment exactly."""
    t0 = time.time()
    zf = zipfile.ZipFile(zip_path)
    folders = v3._real_folders(zf)
    if n_segments:
        folders = folders[:n_segments]
    out = []
    for f0 in folders:
        x, _ = v3._read_seg_wav(zf, f0, use_ch=v3.MIC_NUM)
        if x is None:
            continue
        x = x * gain
        F = x.shape[1]
        nw = F // win
        if nw == 0:
            continue
        wins = x[:, :nw * win].reshape(v3.MIC_NUM, nw, win).transpose(1, 0, 2)
        feats = np.stack([v3.wav_to_gcc_phat(wins[i]) for i in range(nw)], axis=0)
        times, azis = v3._read_seg_azi_series(zf, f0, duration=F / v3.SR_TARGET)
        if len(azis) > 0:
            t_win = (np.arange(nw) * win + win / 2.0) / v3.SR_TARGET
            gt_win = v3.interp_azi_at(times, azis, t_win)
        else:
            gt_win = np.full(nw, np.nan)
        out.append(dict(segment=f0, feats=feats.astype(np.float32),
                        gt_win=np.asarray(gt_win, dtype=np.float64)))
    zf.close()
    if verbose:
        print(f"[v4-cache] real segments={len(out)} in {time.time()-t0:.1f}s "
              f"windows={sum(len(r['gt_win']) for r in out)}", flush=True)
    return out


# ---------------------------------------------------------------------------
# 4. Metric helpers
# ---------------------------------------------------------------------------
def circ180(pred, gt):
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    d = np.abs(p - g) % 180.0
    return np.minimum(d, 180.0 - d)


def circ360(pred, gt):
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    d = np.abs(p - g) % 360.0
    return np.minimum(d, 360.0 - d)


def predict_batches(model, feats, batch_size=256):
    """feats (N,28,61) -> (pd, ang) numpy arrays."""
    model.eval()
    pds, angs = [], []
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            x = torch.from_numpy(np.asarray(feats[i:i + batch_size],
                                            dtype=np.float32)).to(DEVICE)
            pd_, ang_ = model(x)
            pds.append(pd_.cpu().numpy())
            angs.append(ang_.cpu().numpy())
    return np.concatenate(pds), np.concatenate(angs)


def train_cubic(model, ds, epochs=20, lr=1e-3, lam=0.5, dyn_logvar=True,
                seed=42, tag="train", verbose=True):
    """Train a cubic-GCC model on cached features. dyn_logvar=True includes
    log_var in the optimizer (exactly as v3.pretrain_cubic_for_real does for
    DynamicMultiTaskLoss). Returns (model, info dict)."""
    torch.manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    loss_fn = v3.DynamicMultiTaskLoss()
    params = list(model.parameters())
    if dyn_logvar:
        params += list(loss_fn.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    t0 = time.time()
    last = 0.0
    for ep in range(epochs):
        model.train()
        tot = 0.0; cnt = 0
        for x, yd, ya in loader:
            x, yd, ya = x.to(DEVICE), yd.to(DEVICE), ya.to(DEVICE)
            pd_, ang_ = model(x)
            loss, _, _ = loss_fn(pd_, yd, ang_, ya, lam)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); cnt += 1
        last = tot / max(cnt, 1)
        if verbose and ((ep + 1) % 5 == 0 or ep == epochs - 1):
            print(f"[{tag}] ep{ep+1}/{epochs} loss={last:.4f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    return model, dict(loss=last, sec=time.time() - t0,
                       log_var_det=float(loss_fn.log_var_det.item()),
                       log_var_ang=float(loss_fn.log_var_ang.item()))


def eval_cubic(model, feats, gt_folded):
    """feats: (N,28,61) synthetic test samples; gt_folded: azi % 180."""
    _, ang = predict_batches(model, feats)
    err = circ180(ang, gt_folded)
    return dict(mae=float(err.mean()), std=float(err.std()), n=len(err),
                pred=ang, err=err)
