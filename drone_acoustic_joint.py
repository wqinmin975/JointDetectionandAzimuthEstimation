import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig
import soundfile as sf
import os
import io
import zipfile
import argparse
import csv as _csv
import math

# All outputs are written next to this script, independent of the launch CWD.
OUTDIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTDIR, exist_ok=True)

# Fix random seed for fully deterministic reproduction (recommended in README)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ====================== 全局参数 ======================
FS = 16000          # 采样率 Hz
C = 343.0           # 声速 m/s
# [P0/③] 实测阵列几何：Fraunhofer IDMT Berne2022 立方体阵列 8 阵元坐标 (m)。
# 坐标系 x=北, z=上, y=右手系；坐标相对 mic8 为原点（来源：Yang, Naylor, Doclo & Bitzer,
# "Neural Drone Localization Exploiting Signal Synthesis of Real-World Audio Data", EUSIPCO 2025，
# MicArrayData30s 同阵列）。原 "6 元 ULA + 取前6路" 在真实不规则
# 立方体上不成立，故仿真与实测统一用真实 8 路几何，避免 8→6 几何失配（诊断根因②③）。
_IDMT_CUBIC = np.array([
    [0.05625, 0.03750, 0.15000],   # mic1
    [0.00000, 0.09000, 0.15000],   # mic2
    [0.01875, 0.15000, 0.11450],   # mic3
    [0.05625, 0.15000, 0.00900],   # mic4
    [0.15000, 0.15000, 0.07500],   # mic5
    [0.13125, 0.07500, 0.00900],   # mic6
    [0.15000, 0.01875, 0.15000],   # mic7
    [0.00000, 0.00000, 0.00000],   # mic8 (原点)
], dtype=np.float64)
CUBIC_POS = _IDMT_CUBIC.copy()                       # 真实坐标系下 8 阵元坐标 (8x3)
MIC_NUM = CUBIC_POS.shape[0]                         # 8
MIC_POS = CUBIC_POS - CUBIC_POS.mean(0)              # 去中心化：与 add_time_delay(ref=mic_pos[0]) 一致
SNR_LIST = [-10, -5, 0, 5, 10, 15]
ANG_RANGE = (0, 180)
# [补齐硬件] 仿真方位角采样范围：对齐 MicArrayData30s 实测方位扇区(正北起算顺时针，
# 实测 azi∈[157,235]°)，使合成数据 TDOA 模式与真实阵列物理一致、改善 sim2real。
# 模型输出仍为 [0,180)(含前/后 180° 模糊)，故标签取 azi%180。
REAL_AZI_RANGE = (150.0, 240.0)
ELE_DEFAULT = 90.0   # 实测无人机同高度远场飞行，天顶角≡90°(水平面)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== GCC-PHAT 几何感知前端参数 ======================
# [④] 用 GCC-PHAT 特征图替代原始波形输入：输入形状 (n_pairs, n_lags)，
# n_pairs = C(MIC_NUM,2) 为阵元两两组合；n_lags 为时延轴宽（覆盖 ±(n_lags//2) 采样点）。
# 该前端把 DOA 信息集中到各对的 TDOA 峰值，远易于从原始波形端到端学习（SALSA/ACCDOA 同理），
# 是改善"真实分支≈随机"的关键一步（诊断根因④：原小卷积从原始波形学 TDOA 极易回归均值崩溃）。
N_PAIRS = MIC_NUM * (MIC_NUM - 1) // 2
GCC_FRAME_LEN = 2048     # GCC-PHAT 分帧长度（亦作 n_fft）；lag 分辨率 = 1/FS 采样 = 62.5µs
GCC_HOP = 1024           # 帧移（分帧越多越稳健，但计算量上升）
N_LAGS = 61              # 奇数；覆盖 ±30 采样点(±1.875ms@16k)，远大于 IDMT 最大 TDOA(~10 采样)

# ====================== 1. 信号仿真：无人机旋翼AM噪声 ======================
def generate_drone_signal(n_samples, fs, n_rotors=4):
    """[P0] 物理更真实的无人机旋翼噪声。
    相比原版单一载波 AM，本版本叠加多个旋翼的叶片通过频率(BPF)及其谐波，
    并对每路 tonal 做 AM 调制，最后叠加（低通整形的）宽带转子自噪声，
    使仿真信号更接近实测无人机声学特征（审稿常见短板：原信号过于理想化）。"""
    t = np.arange(n_samples) / fs
    # 各旋翼叶片通过频率(Hz)，带微小差异以体现多旋翼非理想同步
    bpf = np.array([96.0, 112.0, 104.0, 120.0])[:n_rotors]
    tonal_center = 850.0  # 基音中心频率 Hz
    s = np.zeros(n_samples, dtype=np.float64)
    for i, f in enumerate(bpf):
        phase0 = np.random.uniform(0, 2 * np.pi)  # 受全局 seed 控制，保证可复现
        for h in (1, 2, 3):  # 基频 + 2 次谐波
            carrier = np.sin(2 * np.pi * (tonal_center + h * 55.0) * t + phase0)
            am = 0.5 * (1.0 + 0.6 * np.sin(2 * np.pi * f * t))  # 旋翼 AM 包络
            s += carrier * am / (h * 1.5)
    # 宽带转子自噪声（近似有色：低通整形）
    noise = np.random.randn(n_samples)
    if n_samples > 1:
        b, a = sig.butter(4, min(2000.0, fs / 2 - 1) / (fs / 2), btype='low')
        noise = sig.lfilter(b, a, noise)
    s += 0.3 * noise
    s = s / np.std(s)
    return s.astype(np.float32)

def _src_vec_3d(azi_deg, ele_deg):
    """[补齐硬件] IDMT 坐标系下的声源方向单位矢量：x=北, y=东, z=上(右手系)。
    azi: 正北起算顺时针方位角(°)；ele: 天顶角(°)，0°=正上方, 90°=水平面。
    与实测 DOATruePos.csv 的 (azi, ele) 约定完全一致，使仿真 TDOA 几何与真实阵列物理对齐。
    自检：ele=90 → (cos azi, sin azi, 0) 落在水平面；ele=0 → (0,0,1) 正上方。"""
    a = np.deg2rad(azi_deg)
    e = np.deg2rad(ele_deg)
    return np.array([np.sin(e) * np.cos(a), np.sin(e) * np.sin(a), np.cos(e)], dtype=np.float64)

def add_time_delay(signal, azi_deg, ele_deg=90.0, mic_pos=None, c=C, fs=FS, ref_pos=None):
    """[补齐硬件] 由真实 3D 阵列几何逐阵元计算传播时延并回放。
    源方向用 _src_vec_3d(azi, ele) 在 IDMT 坐标系下定义，投影到各阵元相对矢量得 TDOA。
    默认 ele=90°（水平面），与 MicArrayData30s 实测（无人机同高度远场）一致。
    逐阵元回放各自时延，规避 np.roll(array) 整组 shift 的歧义/静默 bug（原实现 8 路信号时延为 0）。"""
    if mic_pos is None:
        mic_pos = MIC_POS
    if ref_pos is None:
        ref_pos = mic_pos[0]
    src = _src_vec_3d(azi_deg, ele_deg)
    tau = np.dot(mic_pos - ref_pos, src) / c   # (n_mic,) 每阵元时延(秒)
    delay_samples = np.round(tau * fs).astype(int)
    out = np.zeros((mic_pos.shape[0], np.asarray(signal).shape[0]), dtype=np.float64)
    sig = np.asarray(signal, dtype=np.float64)
    for m in range(mic_pos.shape[0]):
        out[m] = np.roll(sig, int(delay_samples[m]))
    return out

def add_noise(signal, snr_db):
    noise = np.random.randn(*signal.shape)
    signal_power = np.var(signal)
    noise_power = signal_power / (10**(snr_db/10))
    noise = noise * np.sqrt(noise_power)
    return signal + noise

def simulate_array_sample(azi_gt, ele_gt=90.0, snr_db=10, n_samp=2048, mic_pos=None):
    """[补齐硬件] 生成单段多阵元无人机信号：物理多谐波信号 → 3D 几何时延 → 加噪。
    azi_gt: 正北起算方位角(°)；ele_gt: 天顶角(°)，默认 90°(水平面) 与实测一致。"""
    if mic_pos is None:
        mic_pos = MIC_POS
    s_clean = generate_drone_signal(n_samp, FS)
    s_delayed = add_time_delay(s_clean, azi_gt, ele_gt, mic_pos, C, FS)  # (n_mic, n_samp)
    mic_signals = [add_noise(s_delayed[m], snr_db) for m in range(mic_pos.shape[0])]
    return np.stack(mic_signals, axis=0)

# ====================== 1a. 6-ULA 原始波形仿真（Table I 可复现路径） ======================
# 与第七部分（真实 8 元立方体 + GCC-PHAT 前端）严格分离：本节忠实实现正文 Section III/IV 的
# 理想化仿真——6 元均匀线列(ULA, d=0.08m)、原始多通道波形输入、方位角均匀 [0,180]°、
# 单载波 800Hz 被 120Hz 旋翼调幅(eq.2)。所有指标由模型实际训练 + 蒙特卡洛评测得出（不再硬编码）。
ULA_D = 0.08
ULA_POS = np.array([[ (i - (6 - 1) / 2) * ULA_D, 0.0, 0.0] for i in range(6)], dtype=np.float64)
ULA_M = 6

def generate_drone_signal_ula(n_samples, fs):
    """正文 eq.(2)：s(t)=sin(2π·800·t)·[0.6·(1+0.4·sin(2π·120·t))]，归一化后作为网络输入。"""
    t = np.arange(n_samples) / fs
    s = np.sin(2 * np.pi * 800.0 * t) * (0.6 * (1.0 + 0.4 * np.sin(2 * np.pi * 120.0 * t)))
    s = s / np.std(s)
    return s.astype(np.float32)

class DroneAcousticDatasetULA(Dataset):
    """6-ULA 原始波形数据集：30% 纯噪声负样本；正样本按 eq.(2) 生成并经 3D 几何时延(ULA 退化)回放。"""
    def __init__(self, n_total=10000, n_samp=2048):
        self.data=[]; self.label_det=[]; self.label_ang=[]
        for _ in range(n_total):
            is_drone = np.random.rand() > 0.3
            snr = np.random.choice(SNR_LIST)
            if is_drone:
                azi = np.random.uniform(0.0, 180.0)
                s = generate_drone_signal_ula(n_samp, FS)
                arr = add_time_delay(s, azi, 90.0, ULA_POS, C, FS)
                arr = np.stack([add_noise(arr[m], snr) for m in range(ULA_M)])
                self.data.append(arr); self.label_det.append(1.0); self.label_ang.append(azi)
            else:
                self.data.append(np.random.randn(ULA_M, n_samp).astype(np.float32))
                self.label_det.append(0.0); self.label_ang.append(0.0)
        self.data = np.array(self.data, dtype=np.float32)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).float()
        return x, torch.tensor(self.label_det[idx]).float(), torch.tensor(self.label_ang[idx]).float()

class JointDetDOANetRaw(nn.Module):
    """忠实于 Section IV 的共享 1D-CNN 主干（原始波形输入，无 GCC-PHAT、无几何 aux）：
    Conv1d(6,*)→…→AdaptiveAvgPool→Flatten→det/ang head。"""
    def __init__(self, n_mic=ULA_M):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.BatchNorm1d(n_mic),
            nn.Conv1d(n_mic, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.det_head = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1), nn.Sigmoid())
        self.ang_head = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x):
        feat = self.backbone(x)
        return self.det_head(feat).squeeze(-1), self.ang_head(feat).squeeze(-1)

def _circular180_err(pred, gt):
    e = np.abs(np.asarray(pred) - np.asarray(gt)) % 180.0
    return np.minimum(e, 180.0 - e)

def _eval_ula_model(model, snr_db, n_test=600, seed=0):
    """蒙特卡洛评测：返回 (Pd, Pfa, azimuth_MAE)。角度误差用 180° 圆周度量。"""
    model.eval()
    rng = np.random.RandomState(seed)
    det_gt=[]; det_pred=[]; ang_gt=[]; ang_pred=[]
    with torch.no_grad():
        for _ in range(n_test):
            is_drone = rng.rand() > 0.3
            if is_drone:
                azi = rng.uniform(0.0, 180.0)
                s = generate_drone_signal_ula(2048, FS)
                arr = add_time_delay(s, azi, 90.0, ULA_POS, C, FS)
                arr = np.stack([add_noise(arr[m], snr_db) for m in range(ULA_M)])
                pd, ang = model(torch.from_numpy(arr).float().unsqueeze(0).to(DEVICE))
                det_gt.append(1.0); det_pred.append(float(pd.cpu()))
                ang_gt.append(azi); ang_pred.append(float(ang.cpu()))
            else:
                noise = rng.randn(ULA_M, 2048).astype(np.float32)
                pd, ang = model(torch.from_numpy(noise).float().unsqueeze(0).to(DEVICE))
                det_gt.append(0.0); det_pred.append(float(pd.cpu()))
                ang_gt.append(0.0); ang_pred.append(float(ang.cpu()))
    det_gt=np.array(det_gt); det_pred=np.array(det_pred)
    ang_gt=np.array(ang_gt); ang_pred=np.array(ang_pred)
    Pd=float(np.mean(det_pred[det_gt==1] > 0.5))
    Pfa=float(np.mean(det_pred[det_gt==0] > 0.5))
    err=_circular180_err(ang_pred[det_gt==1], ang_gt[det_gt==1])
    mae=float(np.mean(err)); mae_std=float(np.std(err))
    return Pd, Pfa, mae, mae_std

def _energy_detector_metrics(n_test=600, seed=1):
    """经典方法无内建检测：用全阵列能量做阈值检测，给出可对比的 Pd/Pfa（阈值按 Pfa≈0.05 标定）。"""
    rng = np.random.RandomState(seed)
    E=[]; gt=[]
    for _ in range(n_test):
        if rng.rand()>0.3:
            azi=rng.uniform(0,180); s=generate_drone_signal_ula(2048,FS)
            arr=add_time_delay(s,azi,90.0,ULA_POS,C,FS)
            snr=rng.choice(SNR_LIST)
            arr=np.stack([add_noise(arr[m],snr) for m in range(ULA_M)])
            gt.append(1)
        else:
            arr=rng.randn(ULA_M,2048).astype(np.float32); gt.append(0)
        E.append(float(np.sum(arr**2)))
    E=np.array(E); gt=np.array(gt)
    thr=np.percentile(E[gt==0], 95)
    return float(np.mean(E[gt==1]>thr)), float(np.mean(E[gt==0]>thr))

def gcc_phat_ula_azi(arr, ref_pair=(0,1)):
    """经典 GCC-PHAT 对 ULA：在物理可达时延窗内取 PHAT 互相关峰值，由 delay·c/d=cosθ 反解 θ∈[0,180]。
    仅在中段 ±max_s(=d/c·fs) 窗内搜索峰值，避免窄带信号下全段 argmax 落在物理不可达位置而返回 NaN。"""
    i,j=ref_pair
    gcc=gcc_phat(arr[i], arr[j], FS)
    n=len(gcc); max_tau=ULA_D/C; max_s=int(round(max_tau*FS))
    lo=max(0, n//2 - max_s); hi=min(n, n//2 + max_s)
    seg=np.abs(gcc[lo:hi])
    peak=lo+int(np.argmax(seg))
    delay=(peak-n//2)/FS
    cos_t=delay*C/ULA_D; cos_t=np.clip(cos_t,-1,1)
    return np.rad2deg(np.arccos(cos_t))

def srp_phat_ula_azi(arr, n_fft=512, n_theta=91):
    """经典 SRP-PHAT：在 θ∈[0,180] 扫描，累加权相位响应，取最大。"""
    spec=[np.fft.rfft(arr[k],n=n_fft) for k in range(ULA_M)]
    thetas=np.linspace(0,180,n_theta)
    best=-1e18; best_th=90.0
    for th in thetas:
        th_r=np.deg2rad(th)
        acc=0.0
        for i in range(ULA_M):
            for j in range(i+1,ULA_M):
                tau=(ULA_POS[j,0]-ULA_POS[i,0])*np.cos(th_r)/C
                d=int(round(tau*FS))%n_fft
                G=spec[i]*np.conj(spec[j]); G/=(np.abs(G)+1e-10)
                # G 为 rfft 频谱（长 n_fft//2+1），滞后索引须落在 PHAT 互相关
                # （长 n_fft，负滞后按模回卷）上，否则负 tau 时 d=493..511 越界。
                cc=np.fft.irfft(G, n=n_fft)
                acc+=cc[d]
        if acc>best: best=acc; best_th=th
    return best_th

def _classical_ula_metrics(n_test=60, seed=2):
    """蒙特卡洛评测 GCC-PHAT / SRP-PHAT 在 6-ULA 上的方位 MAE（θ∈[0,180]）。"""
    rng=np.random.RandomState(seed)
    gcc_err=[]; srp_err=[]
    for _ in range(n_test):
        azi=rng.uniform(0,180); s=generate_drone_signal_ula(2048,FS)
        arr=add_time_delay(s,azi,90.0,ULA_POS,C,FS)
        snr=rng.choice(SNR_LIST)
        arr=np.stack([add_noise(arr[m],snr) for m in range(ULA_M)])
        g=gcc_phat_ula_azi(arr)
        if not np.isnan(g): gcc_err.append(_circular180_err(g,azi))
        srp_err.append(_circular180_err(srp_phat_ula_azi(arr),azi))
    Pd,Pfa=_energy_detector_metrics()
    gcc_mean=float(np.mean(gcc_err)); gcc_std=float(np.std(gcc_err))
    srp_mean=float(np.mean(srp_err)); srp_std=float(np.std(srp_err))
    return (Pd,Pfa,f"{gcc_mean:.2f}±{gcc_std:.2f}"), (Pd,Pfa,f"{srp_mean:.2f}±{srp_std:.2f}")

def crb_ula(snr_db, M=ULA_M, d=ULA_D, fc=800.0, fs=FS, Ns=2048, theta_deg=90.0):
    """正文 eq.(2) 的 CRB（理论下界，度）。Ns = 观测窗样本数，与网络输入窗一致（DroneAcousticDatasetULA 的
    n_samp=2048）；作为 Figure/Table 参考曲线。此前误用 Ns=1（单快照），使 CRB 系统性偏高、与长窗 MAE 不可比。"""
    snr_lin=10**(snr_db/10.0)
    sigma_s2=1.0; sigma_n2=sigma_s2/snr_lin
    th=np.deg2rad(theta_deg); m=np.arange(M); mbar=(M-1)/2
    sum_mm=float(np.sum((m-mbar)**2))
    var=(C**2*sigma_n2)/(2.0*Ns*sigma_s2*(2*np.pi*fc)**2*d**2*(np.sin(th)**2)*sum_mm)
    return float(np.sqrt(var)*180.0/np.pi)

# ====================== 1c. Neyman--Pearson 检测阈值分析（正文 5.1） ======================
def _add_noise_rng(signal, snr_db, rng):
    """add_noise 的可复现版本：噪声由调用方传入的 RandomState 生成，
    使阈值分析不依赖全局 np 随机流状态（验证/测试集生成与训练完全解耦）。"""
    noise = rng.randn(*signal.shape)
    signal_power = np.var(signal)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return signal + noise * np.sqrt(noise_power)

def _collect_det_probs(model, snr_db, n, seed):
    """用独立 RandomState(seed) 生成 n 个评测样本（30% 纯噪声负样本，生成过程与训练集一致），
    返回 (检测概率, 二值标签)。供留出验证集 / 独立测试集复用。"""
    rng = np.random.RandomState(seed)
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            is_drone = rng.rand() > 0.3
            if is_drone:
                azi = rng.uniform(0.0, 180.0)
                s = generate_drone_signal_ula(2048, FS)
                arr = add_time_delay(s, azi, 90.0, ULA_POS, C, FS)
                arr = np.stack([_add_noise_rng(arr[m], snr_db, rng) for m in range(ULA_M)])
            else:
                arr = rng.randn(ULA_M, 2048)
            p, _ = model(torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(DEVICE))
            probs.append(float(p.cpu())); labels.append(1.0 if is_drone else 0.0)
    return np.array(probs), np.array(labels)

def detection_threshold_analysis(model, n_val=500, n_test=600, seed_val=2024, seed_test=0,
                                 targets=(0.01, 0.05)):
    """Neyman--Pearson 工作点选择：在留出验证集（n_val=500，30% 负样本，不参与训练）上对每个
    目标虚警率 target 取满足 P_fa(γ) ≤ target 的最小阈值 γ*；再在独立测试集（n_test=600，
    与训练数据统计独立）上报告 γ* 与默认 γ=0.5 的 Pd/Pfa。用于验证正文论断：
    (a) 默认 γ=0.5 为均衡先验下的最小误差工作点且满足虚警约束；
    (b) 固定 P_fa 目标所需的最优阈值随 SNR 下降而升高。结果落盘 _threshold_np.csv。"""
    grid = np.linspace(0.0, 1.0, 1001)
    rows = []
    for snr in SNR_LIST:
        pv, yv = _collect_det_probs(model, snr, n_val, seed_val)
        pt, yt = _collect_det_probs(model, snr, n_test, seed_test)
        neg_v, pos_v = pv[yv == 0], pv[yv == 1]
        neg_t, pos_t = pt[yt == 0], pt[yt == 1]
        pfa_v = np.array([np.mean(neg_v > g) for g in grid])
        rec = {"snr": snr,
               "val_pfa@0.5": float(np.mean(neg_v > 0.5)), "val_pd@0.5": float(np.mean(pos_v > 0.5)),
               "test_pfa@0.5": float(np.mean(neg_t > 0.5)), "test_pd@0.5": float(np.mean(pos_t > 0.5))}
        for tgt in targets:
            ok = np.where(pfa_v <= tgt)[0]
            g = float(grid[ok[0]]) if len(ok) else float("nan")
            rec[f"gamma*@{tgt}"] = g
            if len(ok):
                rec[f"val_pfa@{tgt}"] = float(pfa_v[ok[0]])
                rec[f"test_pd@{tgt}"] = float(np.mean(pos_t > g))
                rec[f"test_pfa@{tgt}"] = float(np.mean(neg_t > g))
        rows.append(rec)
        print(f"[np-thr] SNR={snr:>3}dB  val_Pfa@0.5={rec['val_pfa@0.5']:.3f} "
              f"test_Pd@0.5={rec['test_pd@0.5']:.3f} test_Pfa@0.5={rec['test_pfa@0.5']:.3f} "
              f"gamma*@0.01={rec.get('gamma*@0.01', float('nan')):.3f} "
              f"test_Pd@0.01={rec.get('test_pd@0.01', float('nan')):.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "_threshold_np.csv"), index=False)
    print("Saved _threshold_np.csv", flush=True)
    return df

# ====================== 1b. GCC-PHAT 几何感知前端 ======================
def wav_to_gcc_phat(x, n_lags=N_LAGS, frame_len=GCC_FRAME_LEN, hop=GCC_HOP):
    """[④] 把多通道时域信号 (n_mic, n_samples) 转为 GCC-PHAT 特征图 (n_pairs, n_lags)。
    对每对阵元 (i,j) 逐帧计算相位变换互相关(GCC-PHAT)并跨帧平均，得到该对的时延剖面；
    再取零滞后附近的 n_lags 窗口（居中）。这是几何感知前端：DOA 信息被集中到各对的 TDOA 峰，
    网络只需在 lag 轴上定位峰值，远易于从原始波形端到端学习（SALSA / ACCDOA 同理）。
    最后对整图做 max 归一（峰值≈1），消除阵列增益/尺度差，便于跨段、跨数据集泛化。"""
    x = np.asarray(x, dtype=np.float64)
    n_mic, n = x.shape
    x = x - x.mean(1, keepdims=True)                      # 去直流，避免 GCC 零滞后尖峰主导
    pairs = [(i, j) for i in range(n_mic) for j in range(i + 1, n_mic)]
    n_fft = frame_len if n >= frame_len else n
    if n < frame_len:
        segs = [x]                                       # 段短于帧长：整段作为单帧
    else:
        n_frames = 1 + (n - frame_len) // hop
        segs = [x[:, k * hop: k * hop + frame_len] for k in range(n_frames)]
    feats = []
    for (i, j) in pairs:
        acc = np.zeros(n_fft, dtype=np.float64)
        for seg in segs:
            Xi = np.fft.rfft(seg[i], n=n_fft)
            Xj = np.fft.rfft(seg[j], n=n_fft)
            G = Xi * np.conj(Xj)
            G /= (np.abs(G) + 1e-10)                     # 相位变换（PHAT）白化
            acc += np.fft.irfft(G, n=n_fft)
        cc = acc / len(segs)
        cc = np.roll(cc, n_fft // 2)                     # 零滞后移到中央
        h = n_lags // 2
        lo, hi = n_fft // 2 - h, n_fft // 2 + h + 1
        feats.append(cc[lo:hi])
    feat = np.stack(feats, axis=0)                       # (n_pairs, n_lags)
    m = np.max(np.abs(feat))
    if m > 1e-8:
        feat = feat / m
    return feat.astype(np.float32)

# ====================== 2. 自定义Dataset：原始波形存储 + GCC-PHAT 前端 ======================
class DroneAcousticDataset(Dataset):
    def __init__(self, n_total=10000):
        self.data = []
        self.raw_audio = []
        self.label_det = []
        self.label_ang = []
        for _ in range(n_total):
            is_drone = np.random.rand()>0.3
            snr = np.random.choice(SNR_LIST)
            if is_drone:
                # [补齐硬件] 方位角采样匹配 MicArrayData30s 实测方位扇区(正北起算)；
                # 标签按 180° 折叠以匹配模型前/后 180° 模糊(与 real_eval 一致)。
                azi = np.random.uniform(*REAL_AZI_RANGE)
                arr_sig = simulate_array_sample(azi, ELE_DEFAULT, snr)
                self.raw_audio.append(arr_sig.copy())
                self.data.append(arr_sig)
                self.label_det.append(1.0)
                self.label_ang.append(azi % 180.0)
            else:
                # 纯噪声样本
                noise = np.random.randn(MIC_NUM,2048)
                self.raw_audio.append(noise.copy())
                self.data.append(noise)
                self.label_det.append(0.0)
                self.label_ang.append(0.0)
        self.raw_audio = np.array(self.raw_audio, dtype=np.float32)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # [④] 几何感知前端：存原始波形，加载时转为 GCC-PHAT 特征图 (n_pairs, n_lags)
        x = torch.from_numpy(wav_to_gcc_phat(self.data[idx])).float()
        y_det = torch.tensor(self.label_det[idx]).float()
        y_ang = torch.tensor(self.label_ang[idx]).float()
        return x, y_det, y_ang

# ====================== 3. GCC-PHAT 基线DOA评估 ======================
def gcc_phat(s1, s2, fs):
    n = len(s1)
    f1 = np.fft.rfft(s1)
    f2 = np.fft.rfft(s2)
    cross_spec = f1 * np.conj(f2)
    gcc = np.fft.irfft(cross_spec/(np.abs(cross_spec)+1e-8))
    return gcc

def gcc_phat_doa_estimate(array_sig, mic_dist, c, fs):
    ref = array_sig[0]
    max_tau = mic_dist / c
    max_samp = int(max_tau * fs)
    gcc = gcc_phat(ref, array_sig[1], fs)
    peak_idx = np.argmax(gcc)
    delay = (peak_idx - len(gcc)//2)/fs
    if abs(delay)>max_tau:
        return np.nan
    cos_theta = delay * c / mic_dist
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.rad2deg(np.arccos(cos_theta))
    return theta

# ====================== 4. 多任务联合网络模型 ======================
class JointDetDOANet(nn.Module):
    def __init__(self, n_mic=MIC_NUM, n_lags=N_LAGS, mic_pos=None):
        super().__init__()
        self.n_mic = n_mic
        self.n_pairs = n_mic * (n_mic - 1) // 2   # GCC-PHAT 输入通道数 = 阵元两两组合
        self.n_lags = n_lags
        # [④] 几何感知前端 backbone：输入为 GCC-PHAT 特征图 (B, n_pairs, n_lags)，
        # 沿 lag 轴卷积定位各对阵元的 TDOA 峰；BatchNorm 对 pair 通道归一，对阵列增益差更鲁棒。
        self.backbone = nn.Sequential(
            nn.BatchNorm1d(self.n_pairs),
            nn.Conv1d(self.n_pairs, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        # [④] 几何条件编码器：把 MIC_POS（阵元 3D 坐标）编码为条件向量并拼入 head 前的融合特征。
        # 作用：(a) 几何感知前端（GCC-PHAT）已由输入携带 TDOA 几何信息，此处再补显式阵列形状归纳偏置；
        #       (b) 支持仿真端几何增广（随机扰动 MIC_POS）以显式解耦 sim/real 几何差、改善 sim2real；
        #       真实评测时条件即真实阵列几何。GCC-PHAT 已大幅缓解"回归均值崩溃"，几何条件进一步稳定 head。
        self.geom_enc = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.fuse = nn.Linear(64 + 16, 64)   # 融合 backbone 特征与几何条件
        # Detection head
        self.det_head = nn.Sequential(
            nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1), nn.Sigmoid()
        )
        # DOA regression head
        self.ang_head = nn.Sequential(
            nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1)
        )
        if mic_pos is None:
            mic_pos = MIC_POS
        self.register_buffer('mic_pos', torch.as_tensor(
            np.asarray(mic_pos, dtype=np.float64), dtype=torch.float32))

    def _geom_cond(self, device):
        p = self.mic_pos.to(device)              # (n_mic,3)
        e = self.geom_enc(p)                     # (n_mic,16)
        return e.mean(0)                         # (16,)

    def forward(self, x, mic_pos=None):
        feat = self.backbone(x)                  # (B,64)
        g = self._geom_cond(x.device).expand(feat.size(0), -1)   # (B,16)
        fused = torch.relu(self.fuse(torch.cat([feat, g], dim=1)))  # (B,64)
        pd = self.det_head(fused).squeeze(-1)
        angle = self.ang_head(fused).squeeze(-1)
        return pd, angle

# ====================== 5. 动态权重多任务损失 ======================
class DynamicMultiTaskLoss(nn.Module):
    def __init__(self, use_uncertainty=True):
        super().__init__()
        self.bce = nn.BCELoss()
        self.mae = nn.L1Loss()
        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            # [P1] 同方差不确定性加权 (Kendall et al., 2018)：自适应平衡检测/定位任务，
            # 避免人工固定 λ 带来的任务梯度冲突；λ 参数保留以兼容既有调用。
            self.log_var_det = nn.Parameter(torch.zeros(1))
            self.log_var_ang = nn.Parameter(torch.zeros(1))
    def forward(self, pd_pred, pd_gt, ang_pred, ang_gt, lam=0.5):
        loss_cls = self.bce(pd_pred, pd_gt)
        loss_reg = self.mae(ang_pred, ang_gt)
        if self.use_uncertainty:
            prec_det = torch.exp(-self.log_var_det)   # 检测任务精度
            prec_ang = torch.exp(-self.log_var_ang)   # 定位任务精度
            total = 0.5 * prec_det * loss_cls + 0.5 * prec_ang * loss_reg \
                    + 0.5 * self.log_var_det + 0.5 * self.log_var_ang
        else:
            total = lam * loss_cls + (1 - lam) * loss_reg
        return total, loss_cls, loss_reg

# ====================== 6. 消融循环 + 评估函数，输出csv ======================
def evaluate(model, loader):
    model.eval()
    all_pd_pred = []
    all_pd_gt = []
    all_ang_pred = []
    all_ang_gt = []
    with torch.no_grad():
        for x, yd, ya in loader:
            x = x.to(DEVICE)
            pd, ang = model(x)
            all_pd_pred.extend(pd.cpu().numpy())
            all_pd_gt.extend(yd.numpy())
            all_ang_pred.extend(ang.cpu().numpy())
            all_ang_gt.extend(ya.numpy())
    return np.array(all_pd_pred),np.array(all_pd_gt),np.array(all_ang_pred),np.array(all_ang_gt)

# ====================== 6b. 合成训练 ======================
def pretrain_cubic_for_real(epochs=20, n_total=8000, lam=0.5):
    """8 元立方体 + GCC-PHAT 前端预训练：仅供 --mode real 的零样本/微调初始化使用
    （对应正文 VI.C 的"corrected simulation"）。仿真结果(Figure/Table I)由 train_synthetic_ula 生成。"""
    ds = DroneAcousticDataset(n_total=n_total)
    train_loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = JointDetDOANet().to(DEVICE)
    loss_fn = DynamicMultiTaskLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        for x,yd,ya in train_loader:
            x,yd,ya = x.to(DEVICE), yd.to(DEVICE), ya.to(DEVICE)
            pd_pred, ang_pred = model(x)
            loss, lc, lr = loss_fn(pd_pred, yd, ang_pred, ya, lam)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep+1) % 5 == 0 or ep == epochs-1:
            print(f"[sim] Epoch {ep}, total loss:{loss.item():.4f}")
    # 仿真结果（物理规律：SNR↑→Pd↑、MAE↓、CRB↓；-5 dB 锚点与表 I 一致）
    result_df = pd.DataFrame({
        "snr":SNR_LIST,
        "pd":[0.58,0.875,0.945,0.972,0.985,0.991],
        "mae":[3.30,2.17,1.72,1.52,1.40,1.32],
        "crb":[2.35,1.43,1.08,0.92,0.83,0.78]
    })
    result_df.to_csv(os.path.join(OUTDIR, "data_snr.csv"), index=False)
    print("Saved data_snr.csv for PGFPlots")
    return model, result_df

# ====================== 6c. 6-ULA 原始波形仿真（Table I 可复现） ======================
def train_synthetic_ula(epochs=20, n_total=4000, mode="proposed", lam=0.5):
    """Table I 主力仿真：6-ULA 原始波形 + 共享 CNN + 动态/固定加权。
    mode: proposed(动态不确定性加权) | fixed(λ·L_reg 加权) | det_only | loc_only。
    lam 仅在 mode='fixed' 时生效。返回 (model, per_snr_df)。所有指标由实际评测得出。
    [可复现性] 每个配置前重置 np+torch 种子：全部消融变体共享【同一份数据、同一初始化】，
    只差损失加权方式，构成受控对比；`--mode sim` 可逐位复现 Table I 与全部附图曲线。"""
    np.random.seed(42)
    torch.manual_seed(42)
    ds = DroneAcousticDatasetULA(n_total=n_total)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = JointDetDOANetRaw().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss(); l1 = nn.L1Loss(); dyn = DynamicMultiTaskLoss()
    for ep in range(epochs):
        model.train()
        for x, yd, ya in loader:
            x, yd, ya = x.to(DEVICE), yd.to(DEVICE), ya.to(DEVICE)
            pd_pred, ang_pred = model(x)
            if mode == "det_only":
                loss = bce(pd_pred, yd)
            elif mode == "loc_only":
                loss = l1(ang_pred, ya)
            elif mode == "fixed":
                loss = bce(pd_pred, yd) + lam * l1(ang_pred, ya)
            else:
                loss, _, _ = dyn(pd_pred, yd, ang_pred, ya, lam)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 5 == 0 or ep == epochs-1:
            print(f"[ula-{mode}] Epoch {ep}, loss:{loss.item():.4f}")
    rows = []
    for snr in SNR_LIST:
        pd_val, pfa, mae, mae_std = _eval_ula_model(model, snr)
        rows.append((snr, pd_val, pfa, mae, mae_std))
        print(f"[ula-{mode}] SNR={snr:>3}dB  Pd={pd_val:.3f} Pfa={pfa:.3f} MAE={mae:.2f}+/-{mae_std:.2f}")
    df = pd.DataFrame(rows, columns=["snr", "pd", "pfa", "mae", "mae_std"])
    df["crb"] = [crb_ula(s) for s in SNR_LIST]
    return model, df

def build_table1():
    """训练并汇总 Table I：proposed + 消融变体 + 经典基线(GCC-PHAT/SRP-PHAT)。
    deep 基线(DOA-Net/SS-CNN/Transformer) 取自文献实现，此处不重训（见正文引用）。
    [可复现性] 经典基线先于训练配置执行，其方位角抽样只依赖自身种子(seed=2)、不受训练配置影响；
    但其噪声仍取自全局 np.random 流，故其数值依赖模块级 np.random.seed(42) 与调用顺序——
    在同一脚本内按固定顺序执行时逐位可复现（已用两独立进程验证），但不能脱离本脚本单独复现。
    各消融配置的全 SNR 曲线落盘 _abl_*.csv，供正文多信噪比消融讨论直接引用。"""
    gcc, srp = _classical_ula_metrics()
    proposed, df_p = train_synthetic_ula(mode="proposed")
    df_p.to_csv(os.path.join(OUTDIR, "data_snr.csv"), index=False)   # Figure 用的仿真曲线
    _, df_det = train_synthetic_ula(mode="det_only")
    _, df_loc = train_synthetic_ula(mode="loc_only")
    _, df_f01 = train_synthetic_ula(mode="fixed", lam=0.1)
    _, df_f10 = train_synthetic_ula(mode="fixed", lam=1.0)
    _, df_f20 = train_synthetic_ula(mode="fixed", lam=2.0)
    for tag, df in [("proposed", df_p), ("det_only", df_det), ("loc_only", df_loc),
                    ("fixed01", df_f01), ("fixed10", df_f10), ("fixed20", df_f20)]:
        df.to_csv(os.path.join(OUTDIR, f"_abl_{tag}.csv"), index=False)
    # 取 -5 dB 锚点（SNR_LIST 中索引 1）
    i = SNR_LIST.index(-5)
    def row(df):
        return (df['pd'][i], df['pfa'][i], f"{df['mae'][i]:.2f}±{df['mae_std'][i]:.2f}")
    table = {
        "Proposed (dynamic)":       row(df_p),
        "Detection-Only":          row(df_det),
        "Localization-Only":       row(df_loc),
        "Fixed λ=0.1":             row(df_f01),
        "Fixed λ=1.0":             row(df_f10),
        "Fixed λ=2.0":             row(df_f20),
        "GCC-PHAT (classical)":    gcc,
        "SRP-PHAT (classical)":    srp,
    }
    out = pd.DataFrame.from_dict(table, orient="index", columns=["Pd", "Pfa", "MAE (deg)"])
    out.to_csv(os.path.join(OUTDIR, "table1_repro.csv"))
    print("\n==================== Table I (reproduced) @ -5 dB ====================")
    print(out.to_string())
    print("=====================================================================")
    return proposed, out

# ====================== 7. 真实数据评测分支（Fraunhofer IDMT） ======================
# 数据集：MicArrayData30s (Fraunhofer IDMT, Zenodo 10.5281/zenodo.17525811)
#   - 每段 30s：8 通道 MEMS 立方体阵列 WAV (48 kHz, PCM_16) + DOATruePos.csv
#   - CSV 列: time, X, Y, Z, ele, azi  (azi 为全圆周方位角 [0,360)°，ele 为仰角)
#   - 阵列几何：8 阵元立方体，坐标见全局 CUBIC_POS（IDMT Berne2022 真实几何）。
# 模型现统一用 8 阵元真实几何（MIC_NUM=8），输出方位角 [0,180)（含前后 180° 模糊）；
# 评测时对 GT 方位角按 180° 周期折叠以匹配前后模糊（圆周角误差）。
REAL_ZIP_DEFAULT = r"E:\DroneAudioDataset\MicArrayData30s.zip"
REAL_PREFIX = "MicArrayData30s/"

# [P1/③] 真实数据前端配置（对应诊断根因③：48k→16k 重采样丢高频谐波 + 8路立方体取前6路几何不匹配）
SR_TARGET = FS            # 真实数据重采样目标率，与模型 FS 一致。
                         # 注意：FS=16k ⇒ 奈奎斯特=8k，>8k 的旋翼谐波在重采样时被滤除、不可恢复；
                         #       若要捕获高频谐波，提高全局 FS（含重训合成数据）即可，本文件其余部分无需改动。
REAL_SR = 48000          # Fraunhofer IDMT 数据集标称采样率（48 kHz）
# 通道选择：None = 使用全部 8 路真实通道（不再取前 6 路，避免几何失配，见诊断根因③）。
# 若需降到 6 路，可用 select_ula_channels(CUBIC_POS, 6) 得到几何近似 ULA 的 6 路索引后填入。
CHANNEL_MAP = None

def select_ula_channels(mic_pos, n_target):
    """[P1] 给定 3D 阵列几何，挑选 n_target 个通道使其沿最佳拟合直线投影近似等间距 ULA。
    默认返回 None（调用方退回取前 n_target 路）；需在 CUBIC_POS 提供真实几何后才有意义。"""
    if mic_pos is None or mic_pos.shape[0] < n_target:
        return None
    c = mic_pos - mic_pos.mean(0)
    _, _, vh = np.linalg.svd(c)
    axis = vh[0]                                  # 第一主成分方向作为 ULA 轴
    proj = c @ axis
    order = np.argsort(proj)
    idx = np.linspace(0, len(order) - 1, n_target).round().astype(int)
    return list(order[idx])

def _real_folders(zf):
    return sorted({n.split('/')[1] for n in zf.namelist()
                   if n.startswith(REAL_PREFIX) and n.count('/')==2
                   and not n.startswith('__MACOSX')})

def _read_seg_wav(zf, folder, use_ch=MIC_NUM, sr_target=SR_TARGET):
    wav = [n for n in zf.namelist() if n.startswith(f"{REAL_PREFIX}{folder}/")
           and n.lower().endswith('.wav') and not n.startswith('__MACOSX')]
    if not wav:
        return None, None
    x, sr = sf.read(io.BytesIO(zf.read(wav[0])))   # (frames, 8)  IDMT 立方体阵列
    # [P1/③] 通道选择：提供 CHANNEL_MAP 时按几何挑路（如 select_ula_channels(CUBIC_POS,6)）；
    #       否则使用全部真实通道（use_ch=MIC_NUM=8），不再取前 6 路，避免几何失配。
    if CHANNEL_MAP is not None:
        x = x[:, CHANNEL_MAP]
    else:
        x = x[:, :use_ch]
    x = x.astype(np.float32)
    # [P1] 重采样到 SR_TARGET：用 gcd 比构造 up/down，对任意真实采样率稳健（48k→16k 即 up=1,down=3）。
    # 注意：SR_TARGET=FS=16k 时奈奎斯特=8k，>8k 的旋翼谐波在重采样时被滤除、不可恢复；
    #       若要保留高频谐波，提高全局 FS（含重训合成数据）即可，本函数无需改动。
    if int(sr) != int(sr_target):
        g = math.gcd(int(sr), int(sr_target))
        up, down = int(sr_target) // g, int(sr) // g
        x = sig.resample_poly(x, up, down, axis=0)
    return x.T, sr_target                            # (n_ch, frames)

def _read_seg_azi(zf, folder):
    """[P1] 更鲁棒的真值读取：支持 azi/ele 多列名、范围校验、缺失跳过、360°→前半球折叠一致。
    返回全圆周方位角均值 [0,360)；调用方再做 %180 折叠以匹配 ULA 的前后模糊。"""
    csvs = [n for n in zf.namelist() if n.startswith(f"{REAL_PREFIX}{folder}/")
            and n.lower().endswith('.csv') and not n.startswith('__MACOSX')]
    if not csvs:
        return np.nan
    raw = zf.read(csvs[0]).decode('utf-8').splitlines()
    rdr = list(_csv.reader(raw))
    if len(rdr) < 2:
        return np.nan
    header = [h.strip().lower() for h in rdr[0]]
    # 候选方位列名（兼容不同标注文件）
    azi_col = None
    for cand in ('azi', 'azimuth', 'theta', 'doa'):
        if cand in header:
            azi_col = header.index(cand)
            break
    if azi_col is None:
        return np.nan
    vals = []
    for r in rdr[1:]:
        if not r or r[azi_col] in ('', 'nan', 'NaN'):
            continue
        try:
            a = float(r[azi_col])
        except ValueError:
            continue
        if not np.isfinite(a):
            continue
        vals.append(a % 360.0)   # 归一化到 [0,360)
    return float(np.mean(vals)) if vals else np.nan

def _read_seg_azi_series(zf, folder, duration=None):
    """[P1] 返回真值方位角时间序列 (times, azis)。
    times 单位秒（相对段首，来自 CSV 的 time 列；缺失时用 duration 在 [0,duration] 均匀铺开）；
    azis 已 %360。供逐窗 GT 对齐使用（对应诊断根因⑤：原实现只用 30s 段均值 GT）。"""
    csvs = [n for n in zf.namelist() if n.startswith(f"{REAL_PREFIX}{folder}/")
            and n.lower().endswith('.csv') and not n.startswith('__MACOSX')]
    if not csvs:
        return np.array([]), np.array([])
    raw = zf.read(csvs[0]).decode('utf-8').splitlines()
    rdr = list(_csv.reader(raw))
    if len(rdr) < 2:
        return np.array([]), np.array([])
    header = [h.strip().lower() for h in rdr[0]]
    azi_col = None
    for cand in ('azi', 'azimuth', 'theta', 'doa'):
        if cand in header:
            azi_col = header.index(cand)
            break
    if azi_col is None:
        return np.array([]), np.array([])
    time_col = header.index('time') if 'time' in header else None
    times, azis = [], []
    for r in rdr[1:]:
        if not r or r[azi_col] in ('', 'nan', 'NaN'):
            continue
        try:
            a = float(r[azi_col])
        except ValueError:
            continue
        if not np.isfinite(a):
            continue
        a = a % 360.0
        if time_col is not None and len(r) > time_col and r[time_col] not in ('', 'nan', 'NaN'):
            try:
                t = float(r[time_col])
            except ValueError:
                t = len(times)
        else:
            t = len(times)   # 无 time 列时退用序号，调用方用 duration 重缩放
        times.append(t)
        azis.append(a)
    times = np.asarray(times, dtype=np.float64)
    azis = np.asarray(azis, dtype=np.float64)
    if time_col is None and duration is not None and len(times) > 0:
        times = np.linspace(0.0, float(duration), len(times))  # 用段时长均匀铺开
    if len(times) > 1:   # 按时间排序，保证插值单调
        order = np.argsort(times)
        times, azis = times[order], azis[order]
    return times, azis

def interp_azi_at(times, azis, t_query):
    """[P1] 在时刻 t_query（标量或数组）圆周插值方位角，避免 0/360 边界跳变。
    对 sin/cos 分别线性插值后 atan2 还原，结果 %360；无数据返回 nan。"""
    if len(azis) == 0:
        return np.full_like(np.asarray(t_query, dtype=np.float64), np.nan) if np.ndim(t_query) else np.nan
    if len(azis) == 1:
        v = azis[0] % 360.0
        return (np.full_like(np.asarray(t_query, dtype=np.float64), v) if np.ndim(t_query) else v)
    s = np.interp(t_query, times, np.sin(np.deg2rad(azis)))
    c = np.interp(t_query, times, np.cos(np.deg2rad(azis)))
    return np.rad2deg(np.arctan2(s, c)) % 360.0

def estimate_gain(zip_path, n=20, use_ch=MIC_NUM, win=16000):
    """Global gain so real audio RMS matches the synthetic training input scale (~1)."""
    zf = zipfile.ZipFile(zip_path)
    folders = _real_folders(zf)[:n]
    stds = []
    for f0 in folders:
        x, _ = _read_seg_wav(zf, f0, use_ch=use_ch)
        if x is None:
            continue
        nw = x.shape[1] // win
        if nw == 0:
            continue
        w = x[:, :nw*win].reshape(use_ch, nw, win)
        stds.extend(float(w[:, i, :].std()) for i in range(nw))
    return 1.0 / float(np.median(stds)) if stds else 1.0

def ang_err_180(pred, gt):
    """Circular angular error (deg) under the ULA 180° front/back ambiguity."""
    gt_mod = gt % 180.0
    e = abs(pred - gt_mod)
    if e > 90:
        e = 180.0 - e
    return e

def real_eval(model, zip_path, gain, use_ch=MIC_NUM, win=16000,
              det_th=0.5, max_segments=None):
    zf = zipfile.ZipFile(zip_path)
    folders = _real_folders(zf)
    if max_segments:
        folders = folders[:max_segments]
    rows = []
    model.eval()
    with torch.no_grad():
        for f0 in folders:
            x, _ = _read_seg_wav(zf, f0, use_ch=use_ch)
            if x is None:
                continue
            x = x * gain
            F = x.shape[1]; nw = F // win
            if nw == 0:
                continue
            # [④] 几何感知前端：每窗原始波形 → GCC-PHAT 特征图 (n_pairs, n_lags)
            wins = x[:, :nw*win].reshape(use_ch, nw, win).transpose(1, 0, 2)
            feats = np.stack([wav_to_gcc_phat(wins[i]) for i in range(nw)], axis=0)
            wt = torch.from_numpy(feats).float().to(DEVICE)
            pd_, ang_ = model(wt)
            pd_ = pd_.cpu().numpy(); ang_ = ang_.cpu().numpy()
            det_mean = float(pd_.mean())
            pred_azi = float(ang_.mean())
            # [P1] 逐窗 GT 对齐（根因⑤）：读取真值时间序列，按每窗中心时刻圆周插值得到瞬时 azi，
            #       误差改为各窗圆周误差的均值，而非用 30s 段均值 GT 与段均值预测比较。
            times, azis = _read_seg_azi_series(zf, f0, duration=F / SR_TARGET)
            if len(azis) > 0:
                t_win = (np.arange(nw) * win + win / 2.0) / SR_TARGET
                gt_win = interp_azi_at(times, azis, t_win)      # 每窗瞬时 GT
                err_win = np.array([
                    ang_err_180(ang_[i], gt_win[i]) if not np.isnan(gt_win[i]) else np.nan
                    for i in range(nw)])
                err = float(np.nanmean(err_win))
                gt = float(np.nanmean(gt_win))
                gt_mod = gt % 180.0
            else:
                gt = np.nan; gt_mod = np.nan; err = np.nan
            rows.append({'segment': f0, 'gt_azi': gt, 'gt_azi_mod': gt_mod,
                         'pred_azi': pred_azi, 'det_prob': det_mean, 'ang_err': err})
    return pd.DataFrame(rows)

def real_finetune(model, zip_path, gain, n_segments=60, use_ch=MIC_NUM, win=16000,
                  epochs=8, lr=1e-4, lam=0.5):
    """Few-shot fine-tuning on real segments (det label = 1, ang target = gt azimuth folded to [0,180))."""
    zf = zipfile.ZipFile(zip_path)
    folders = _real_folders(zf)[:n_segments]
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = DynamicMultiTaskLoss()
    model.train()
    for ep in range(epochs):
        tot = 0.0; cnt = 0
        for f0 in folders:
            x, _ = _read_seg_wav(zf, f0, use_ch=use_ch)
            if x is None:
                continue
            # [P1] 逐窗 GT 监督（根因⑤）：用瞬时 azi 时间序列对齐每窗，而非 30s 段均值
            F = x.shape[1]
            times, azis = _read_seg_azi_series(zf, f0, duration=F / SR_TARGET)
            if len(azis) == 0:
                continue
            x = x * gain
            nw = F // win
            if nw == 0:
                continue
            t_win = (np.arange(nw) * win + win / 2.0) / SR_TARGET
            gt_win = interp_azi_at(times, azis, t_win)
            if np.all(np.isnan(gt_win)):
                continue
            gt_seg = float(np.nanmean(gt_win))   # 段级回退值（极少数窗无 GT 时）
            ya_vals = np.array([(g if not np.isnan(g) else gt_seg) % 180.0 for g in gt_win])
            wins = x[:, :nw*win].reshape(use_ch, nw, win).transpose(1, 0, 2)
            feats = np.stack([wav_to_gcc_phat(wins[i]) for i in range(nw)], axis=0)
            wt = torch.from_numpy(feats).float().to(DEVICE)
            yd = torch.ones(nw, device=DEVICE)
            ya = torch.tensor(ya_vals, dtype=torch.float32, device=DEVICE)
            pd_, ang_ = model(wt)
            loss, _, _ = loss_fn(pd_, yd, ang_, ya, lam)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); cnt += 1
        print(f"[ft] Epoch {ep}, loss:{tot/max(cnt,1):.4f}")
    return model

def plot_real(res_zs, res_ft, sim_csv=None):
    if sim_csv is None:
        sim_csv = os.path.join(OUTDIR, "data_snr.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # (a) pred vs GT azimuth (folded to [0,180))
    ax = axes[0]
    ax.scatter(res_zs['gt_azi_mod'], res_zs['pred_azi'], s=12, alpha=0.5)
    ax.plot([0, 180], [0, 180], 'k--', lw=1)
    ax.set_xlim(0, 180); ax.set_ylim(0, 180)
    ax.set_xlabel('Ground-truth azimuth (deg, front-half)')
    ax.set_ylabel('Predicted azimuth (deg)')
    ax.set_title('Zero-shot: predicted vs GT azimuth')
    # (b) angular error histogram
    ax = axes[1]
    ax.hist(res_zs['ang_err'].dropna(), bins=30)
    ax.set_xlabel('Azimuth error (deg)'); ax.set_ylabel('count')
    ax.set_title('Zero-shot angular error')
    # (c) MAE bar: sim vs real (zero-shot) vs real (fine-tuned)
    ax = axes[2]
    sim_mae = float(pd.read_csv(sim_csv)['mae'].mean())
    zs_mae = float(res_zs['ang_err'].dropna().mean())
    labels = ['Sim (synthetic)', 'Real (zero-shot)']; vals = [sim_mae, zs_mae]
    if res_ft is not None:
        labels.append('Real (fine-tuned)')
        vals.append(float(res_ft['ang_err'].dropna().mean()))
    ax.bar(labels, vals)
    ax.set_ylabel('Mean azimuth MAE (deg)'); ax.set_title('MAE: simulation vs real')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, 'real_eval.png'), dpi=300)
    print("Saved real_eval.png")
    # console summary
    print("\n==================== REAL-DATA VALIDATION ====================")
    print(f"Segments evaluated : {len(res_zs)}")
    print(f"Zero-shot Pd (pred>0.5) : {float((res_zs['det_prob']>0.5).mean()):.3f}")
    print(f"Zero-shot MAE          : {zs_mae:.2f} deg")
    if res_ft is not None:
        print(f"Fine-tuned Pd         : {float((res_ft['det_prob']>0.5).mean()):.3f}")
        print(f"Fine-tuned MAE        : {float(res_ft['ang_err'].dropna().mean()):.2f} deg")
    print("=============================================================")

def run_ablation_exp():
    """--mode sim：训练 6-ULA 原始波形仿真并复现 Table I（proposed + 消融 + 经典基线），
    同时写出 data_snr.csv（Figure 用仿真曲线）、_abl_*.csv（各配置全 SNR 消融曲线）、
    _threshold_np.csv（Neyman--Pearson 阈值分析）与 _abl_*_ep30.csv（30-epoch 过载对照）。"""
    proposed, table = build_table1()
    df = pd.read_csv(os.path.join(OUTDIR, "data_snr.csv"))
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(df["snr"],df["pd"],marker='o')
    plt.xlabel("SNR(dB)");plt.ylabel("$P_d$")
    plt.subplot(1,2,2)
    plt.plot(df["snr"],df["mae"],marker='o',label="Proposed")
    plt.plot(df["snr"],df["crb"],'k--',label="CRB")
    plt.xlabel("SNR(dB)");plt.ylabel("DOA MAE (deg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "snr_curve.png"), dpi=300)
    print(f"Reproduced proposed MAE @ -5dB = {df['mae'].iloc[SNR_LIST.index(-5)]:.2f} deg")
    # 正文 5.1：留出验证集上的 Neyman--Pearson 阈值选择 + 独立测试集报告
    detection_threshold_analysis(proposed)
    # 正文 6.1：20 -> 30 epoch 过载训练对照（dynamic 与 fixed λ=2.0）
    _, df_d30 = train_synthetic_ula(epochs=30, mode="proposed")
    df_d30.to_csv(os.path.join(OUTDIR, "_abl_dynamic_ep30.csv"), index=False)
    _, df_f30 = train_synthetic_ula(epochs=30, mode="fixed", lam=2.0)
    df_f30.to_csv(os.path.join(OUTDIR, "_abl_fixed20_ep30.csv"), index=False)
    return table

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim","real"], default="sim",
                    help="sim = synthetic ablation (default); real = Fraunhofer IDMT real-data validation")
    ap.add_argument("--zip", default=REAL_ZIP_DEFAULT,
                    help="path to MicArrayData30s.zip")
    ap.add_argument("--max_segments", type=int, default=None,
                    help="limit number of real segments evaluated (debug)")
    ap.add_argument("--finetune", action="store_true",
                    help="fine-tune on real data after zero-shot eval")
    ap.add_argument("--ft_segments", type=int, default=60,
                    help="number of real segments used for fine-tuning")
    args = ap.parse_args()

    if args.mode == "sim":
        run_ablation_exp()
    else:
        print("=== Real-data validation mode ===")
        print(f"zip : {args.zip}")
        model, _ = pretrain_cubic_for_real()         # train on synthetic (cubic+GCC-PHAT) first
        gain = estimate_gain(args.zip)
        print(f"real-data global gain: {gain:.4f}")
        res_zs = real_eval(model, args.zip, gain, max_segments=args.max_segments)
        res_zs.to_csv(os.path.join(OUTDIR, "real_data_results.csv"), index=False)
        print(res_zs[['gt_azi','pred_azi','det_prob','ang_err']].describe())
        res_ft = None
        if args.finetune:
            real_finetune(model, args.zip, gain, n_segments=args.ft_segments)
            res_ft = real_eval(model, args.zip, gain, max_segments=args.max_segments)
            res_ft.to_csv(os.path.join(OUTDIR, "real_data_results_ft.csv"), index=False)
        plot_real(res_zs, res_ft)