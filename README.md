# Joint Detection and Azimuth Estimation of Drones via MEMS Microphone Arrays and Multi-Task Deep Learning

Code and simulation release accompanying the manuscript submitted to **MDPI *Sensors***
("End-to-End Joint Detection and Azimuth Estimation of Drones Using MEMS Microphone
Arrays and Multi-Task Deep Learning").

> All experiments in the core pipeline are **pure numerical simulation** (no hardware
> measurements). The code generates synthetic drone-rotor acoustic signals, trains a
> multi-task joint detection + DOA regression network, evaluates classical baselines
> (GCC-PHAT / SRP-PHAT), runs ablation studies, and exports CSV result tables that are
> consumed directly by the paper's PGFPlots figures. A separate branch validates on the
> Fraunhofer IDMT `MicArrayData30s` real-world recordings.

## Repository layout

| File | Purpose |
|------|---------|
| `drone_acoustic_joint.py` | Main script: signal simulation, dataset construction, multi-task network, training, GCC-PHAT baseline, Monte-Carlo evaluation, CSV export. `python drone_acoustic_joint.py --mode sim` reproduces Table I and all simulation curves bit-for-bit. |
| `baseline_snr.py` | Independent Monte-Carlo evaluation of the classical baselines (GCC-PHAT, SRP-PHAT) across SNR. Sole source of the classical curves in Fig. 2. Uses a local `np.random.RandomState`, so two independent processes produce a byte-identical `baseline_snr.csv`. |
| `wsu_eval.py` | Evaluation on the Fraunhofer IDMT `MicArrayData30s` real-world drone recordings (zero-shot and fine-tuned). |
| `plot_wsu.py` | ROC / scatter plots for the real-data evaluation (Fig. 4). |
| `_make_mdpi.py` | Builds the MDPI *Sensors* LaTeX source (`paper_mdpi.tex`) from the elsarticle master. |
| `_make_coverletter.py`, `_verify_coverletter.py` | Cover-letter generation and post-flight consistency validation. |
| `paper_mdpi.tex` | MDPI *Sensors* single-column manuscript (target journal). |
| `paper_v8.tex` | elsarticle 5p two-column manuscript (lineage / IEEE-Measurement variant). |
| `refs.bib` | Bibliography. |
| `Definitions/` | MDPI document class and supporting style files. |
| `supplementary_code.zip` | Self-contained supplementary bundle (code + result CSVs + figures) referenced by the paper. |
| `README.md`, `cover_letter.md`, `supplementary_material.md` | Documentation. |
| `*.csv` (non-underscore) | Result exports behind every table / figure: `data_snr.csv`, `table1_repro.csv`, `baseline_snr.csv`, `wsu_*.csv`, `real_data_results*.csv`, etc. |

## Environment

Python 3.9+ (tested with 3.13). Install dependencies:

```bash
pip install numpy torch scipy pandas matplotlib
```

LaTeX (to compile the manuscript): MiKTeX / TeX Live with `mdpi`, `booktabs`,
`pgfplots`, `siunitx`, `subcaption`, `graphicx`, `hyperref`.

## How to reproduce

1. **Main simulation — Table I and simulation curves:**
   ```bash
   python drone_acoustic_joint.py --mode sim
   ```
   Exports `data_snr.csv`, `table1_repro.csv`, and the ablation CSVs.

2. **Classical baselines — Fig. 2 classical curves (independently reproducible):**
   ```bash
   python baseline_snr.py
   ```
   Exports `baseline_snr.csv`.

3. **Real-data evaluation** (requires the Fraunhofer IDMT `MicArrayData30s` dataset,
   Zenodo `10.5281/zenodo.17525811`):
   ```bash
   python wsu_eval.py
   ```

4. **Compile the manuscript** with your LaTeX editor on `paper_mdpi.tex` (or run the
   MDPI build chain from `Definitions/`).

## Reproducibility notes

- Every training configuration resets `np.random.seed(42)` + `torch.manual_seed(42)`;
  all ablated variants share the same data and initialization and differ *only* in loss
  weighting. `drone_acoustic_joint.py --mode sim` reproduces Table I and all curves
  bit-for-bit.
- Classical baselines in `baseline_snr.py` use a local `np.random.RandomState(seed)`,
  so two independent processes produce a byte-identical `baseline_snr.csv`.
- **Honesty note (per the manuscript's reporting policy):** dynamic loss weighting is
  *not* globally optimal — at several SNRs the fixed-weight variant marginally
  outperforms. The paper reports the full SNR map rather than a single favorable point,
  and the IDMT real-data results are presented as an in-distribution upper bound after
  fine-tuning.

## Authors & contact

Qinmin Wang, Zixin Wang, Zhenzhen Hu, Hongyuan Wang, Liwei Tian (corresponding author,
tianliwei@gdust.edu.cn). Funding: National Natural Science Foundation of China under
Grant No. 62101076.

## License

Released for academic reproducibility. The MDPI *Sensors* manuscript and its
supplementary material govern the use of the reported results.
