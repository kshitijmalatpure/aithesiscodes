# Automated Essay Scoring (AES): ML & LLM Pipelines

This repository contains the full experimental stack for a thesis on **Automated Essay Scoring (AES)** for **L2 Dutch**. It combines Machine Learning (ML) pipelines (Classic, Binary, FabOF, and regression–classification hybrids) and Large Language Model (LLM) trait scorers. Everything is organized for replication and extension on your own data.

> **Important:** Many scripts define constants like `base_dir` and output roots. You **must change these directories/paths to suit your environment** (see **Paths & Configuration** below).

---

## Repository Map

```
.
├── LLM/
│   ├── Trait calls/
│   │   ├── AESWO.py       # Word Order (WO)
│   │   ├── AESGR.py       # Grammar (GR)
│   │   ├── AESCO.py       # Cohesion (CO)
│   │   └── AESTT.py       # Text Structure (TT)
│   ├── JSON_files.py      # JSON → CSV conversion (see workflow below)
│   ├── Analysis.py        # Aggregates & evaluates LLM predictions
│   └── LLM sampler.py     # Prompting / sampling utilities
│
├── ML/
│   ├── Classic/           # Non-FabOF multiclass (1–4 scale)
│   │   ├── FullSET.py
│   │   ├── FullSET_regression.py
│   │   └── FullSET_regression_v2.py
│   │
│   ├── Binary/            # Non-FabOF binary counterparts
│   │   ├── FullSET_binary.py
│   │   └── FullSET_binary_regression.py
│   │
│   ├── FabOF/             # FabOF ordinal-forest pipelines (4-class & binary)
│   │   ├── FullSETFabOF.py
│   │   ├── FullSETFabOF_tuning.py
│   │   ├── FullSETFabOF_tuned.py
│   │   ├── FullSETFabOF_binary.py
│   │   ├── FullSETFabOF_binary_tuning.py
│   │   ├── FullSETFabOF_binary_tuned.py
│   │   ├── FullSETFabOF_ERRORONLY.py
│   │   └── FullSETFabOF_sanserror.py
│   │
│   ├── Appendix_HOL/
│   │   ├── FullSETFabOF_HOL.py
│   │   ├── FullSETFabOF_HOL_tuning.py
│   │   └── FullSETFabOF_HOL_tuned.py
│   │
│   └── Preprocessing/
│       ├── K-fold and logging.py
│       ├── LT.py                  # LanguageTool-based error features
│       └── to_remove_0&na_only.py # Cleaning helpers
│
├── requirements.txt
└── README.md
```

---

## Environment & Installation

```bash
# (Optional) Create & activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Key packages:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, optional: `xgboost`, `imbalanced-learn`, `statsmodels`, `shap`, `language_tool_python`, `tqdm`.

---

## Paths & Configuration (read this!)

Most scripts declare paths near the top (e.g., `base_dir = Path(...)`) and output roots like `.../comprehensive_*`. Before running, **change these directories to match your machine**:

*   **Folds root** (examples used in scripts):
    *   Multiclass: `Main/10-fold-final/fold_{n}/fold{n}tr.csv`, `fold{n}te.csv`
    *   Binary: `Main/10-fold-final/fold_{n}/fold{n}trb.csv`, `fold{n}teb.csv`
*   **Output roots**: e.g., `comprehensive_fabOF/`, `comprehensive_fabOF_binary/`
*   **LLM I/O**: where JSON predictions are written and CSVs are read for `Analysis.py`

> Tip: On Windows, prefer raw strings (e.g., `r"C:\path\to\data"`). On macOS/Linux, use POSIX paths (e.g., `"/home/user/data"`).

---

## Data & Features

*   **Columns**:
    *   ID: `Inputfile`
    *   Targets (multiclass): `A_WO`, `A_GR`, `A_CO`, `A_TT` (1–4)
    *   Targets (binary): `A_WOB`, `A_GRB`, `A_COB`, `A_TTB` (0/1)
    *   Features: all other columns
*   **Feature extraction**:
    *   **T-Scan**: extracted online at **[https://tscan.hum.uu.nl/tscan/](https://tscan.hum.uu.nl/tscan/)**
    *   **LanguageTool**: see `ML/Preprocessing/LT.py`

> **Datasets are not included** in this repository.

---

## LLM Workflow (JSON → CSV → evaluation)

1.  **Generate predictions (per trait)**
    Run any script in `LLM/Trait calls/` (e.g., `AESWO.py`, `AESGR.py`). Outputs are **JSON**.

2.  **Convert JSON → CSV**
    Run `LLM/JSON_files.py`. It creates:
    ```
    filename,ypred
    essay1.txt,3
    essay2.txt,2
    ...
    ```

3.  **Manually add ground truth**
    Insert `ytrue` as the **second column** and shift `ypred` to the **third**:
    ```
    filename,ytrue,ypred
    essay1.txt,3,3
    essay2.txt,2,2
    ...
    ```

4.  **Evaluate**
    Run `LLM/Analysis.py` to compute Accuracy, F1 (macro), Precision/Recall (macro), QWK.

> **Remember:** You may need to change input/output directories in these scripts to your own paths.

---

## ML Pipelines Overview

All families share a **preprocess → feature-select → model → evaluate** pattern.

### Classic (multiclass 1–4)

*   `ML/Classic/FullSET.py`: Multi-model classifiers (RF, SVM, XGB), optional SMOTE.
*   `ML/Classic/FullSET_regression.py`, `FullSET_regression_v2.py`: Regression–classification hybrids (continuous → rounded).

### Binary (0/1)

*   `ML/Binary/FullSET_binary.py`, `FullSET_binary_regression.py`: Binary analogues (including hybrid).

### FabOF (ordinal forest)

*   `ML/FabOF/FullSETFabOF*.py`: RandomForestRegressor + **data-driven boundaries per trait** from OOB predictions.
    *   4-class: learns **3 thresholds**
    *   Binary: **1 threshold**
*   Variants:
    *   **base** (exploratory), **tuning** (`*_tuning.py` → search & save best), **tuned** (`*_tuned.py` → fixed best; SHAP visuals), plus error-focused versions.

### Appendix\_HOL

*   HOL-specific FabOF experiments (appendix only).

### Preprocessing

*   `K-fold and logging.py`: fold creation & logging
*   `LT.py`: LanguageTool feature pipeline
*   `to_remove_0&na_only.py`: cleaning helpers

---

## Feature Selection (FS)

*   `baseline` (all features)
*   `kbest_50`, `kbest_100`, `kbest_150`, `kbest_200` (ANOVA f\_regression)
*   `lasso` (sparse linear selection)

> In tuned scripts, FS and hyperparameters are fixed from tuning outputs. Paths to tuned configs/results may also need to be **changed for your setup**.

---

## Metrics & Outputs

*   **Classification**: Accuracy, F1 (macro), Precision/Recall (macro), QWK
*   **Regression/Hybrids**: RMSE, MAE, R² (plus classification metrics after rounding)
*   **Visuals**:
    *   Per-fold & aggregated confusion matrices
    *   In tuned FabOF scripts: **SHAP** (beeswarm, bars, dependence, waterfalls)
*   **Output directories** (change as needed):
    Each script writes to a structured folder (e.g., `.../comprehensive_fabOF/...`) including:
    *   `logs/`, `aggregated_results/aggregated_results.csv`, `best_selector_confusion_matrices/*.png`, and (optionally) SHAP plots.
---

## Citation

If this work helps your research, please cite:

> Kshitij Malatpure (2025). *Automated Essay Scoring for L2 Dutch: Comparing Machine Learning and LLM Approaches*. Master’s Thesis, KU Leuven.

---

*Note:* **T-Scan features are extracted online** at **[https://tscan.hum.uu.nl/tscan/](https://tscan.hum.uu.nl/tscan/)**.
