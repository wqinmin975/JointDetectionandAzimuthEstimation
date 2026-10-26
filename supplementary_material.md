# Supplementary Material — Reproducibility Package

This package contains the code that regenerates **every table and figure** in the
manuscript *"Joint Detection and Azimuth Estimation of UAVs Using MEMS Microphone
Arrays and Multi-Task Deep Learning: From Simulation to Real-World Recordings"* (MDPI *Sensors*).

## What is included

| File | Purpose |
| --- | --- |
| `drone_acoustic_joint.py` | Main simulation + multi-task network. `--mode sim` reproduces the synthetic study and exports `data_snr.csv` (Fig. 2) and `table1_repro.csv` (Table 2). |
| `baseline_snr.py` | Classical GCC-PHAT / SRP-PHAT baselines on the synthetic 6-ULA; exports `baseline_snr.csv` (Fig. 2 classical curves). |
| `wsu_eval.py` | WSU classical-baseline azimuth MAE + detection ROC (Table 4, Table 5, Fig. 4). Requires the WSU UAV Acoustic Localization Dataset (DOI 10.26183/07nm-4p35). |
| `plot_wsu.py` | Plots `wsu_roc.png` (Fig. 4) from `wsu_eval.py` output. |
| `data_snr.csv`, `baseline_snr.csv`, `table1_repro.csv` | Exported result tables consumed by the figures. |
| `real_eval.png`, `wsu_roc.png` | Rendered validation figures (Fig. 3, Fig. 4). |
| `paper_mdpi.tex`, `Definitions/` | The MDPI *Sensors* LaTeX source and class files used to typeset the manuscript. |

## How to reproduce

```bash
# 1) Synthetic study -> Table 2 + Fig. 2 (data_snr.csv, table1_repro.csv)
python drone_acoustic_joint.py --mode sim

# 2) Classical baselines on synthetic data -> baseline_snr.csv (Fig. 2 classical curves)
python baseline_snr.py

# 3) WSU classical baselines + ROC -> Table 4, Table 5, Fig. 4
#    (download the WSU dataset first; see wsu_eval.py for the path)
python wsu_eval.py
python plot_wsu.py
```

All random seeds are fixed (`np.random.seed(42)` + `torch.manual_seed(42)`) so the
ablation variants share identical data and initialisation; results are bit-for-bit
reproducible with the accompanying CSVs. `baseline_snr.py` draws its randomness from
locally seeded `RandomState` instances rather than the global NumPy stream, so the
classical curves of Fig. 2 are reproduced byte-for-byte across independent processes.

Note on the classical baselines: the classical rows of Table 2 and the classical
curves of Fig. 2 come from an *independent* Monte Carlo draw with respect to the
network test set. They share the signal model, array geometry, delay model and
estimator code with the network evaluation, but they do not reuse per-sample noise
realizations, and the sample counts differ (60 mixed-SNR trials for the Table 2 rows,
600 trials per SNR for the Fig. 2 curves). The manuscript therefore reports them as an
independent evaluation rather than as a paired comparison; because the classical
azimuth error is nearly flat in SNR, the mixed-SNR mean is directly comparable to the
$-5$ dB network value.

## Notes on data and scope

- **Fraunhofer IDMT** corpus (Zenodo 10.5281/zenodo.17525811) is used for network
  transfer (zero-shot / fine-tuned) in Table 3 and Fig. 3.
- **WSU** corpus (DOI 10.26183/07nm-4p35) is used for the classical baselines and the
  real-world detection ROC in Table 4, Table 5 and Fig. 4. No in-house MEMS hardware
  prototype is required to reproduce any result in the paper; the hardware field test is
  declared as future work.
- The learned network is **not** evaluated on WSU (different array geometry, no domain
  adaptation in this work); WSU concerns the classical baselines only. This is stated
  explicitly in the manuscript (Scope paragraph, Section 6.3.2).

## Environment

Python 3 with `numpy`, `torch`, `scipy`, `pandas`, `matplotlib`, `soundfile`. LaTeX
typesetting uses the MDPI `mdpi.cls` (2023-03-25) bundled in `Definitions/`.

## Typesetting the manuscript

```bash
python _make_mdpi.py                      # paper_v8.tex -> paper_mdpi.tex
pdflatex paper_mdpi && bibtex paper_mdpi && pdflatex paper_mdpi && pdflatex paper_mdpi
```

The MDPI class enables marginal line numbering in `submit` status (used for reviewer
proofs). For a clean typeset PDF the generated source neutralises that switch with
`\renewcommand{\linenumbers}{}`, so no line numbers appear in the margin. The
bibliography is produced by `mdpi.bst` from `refs.bib` (22 entries).

