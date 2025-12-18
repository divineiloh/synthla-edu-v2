# SYNTHLA-EDU V2: Quick Start for GitHub Users

## For Non-Technical Users: Run Everything with One Command

### Prerequisites (One-time setup)

1. **Download and install Python 3.10 or later**
   - Go to https://www.python.org/downloads/
   - Download Python 3.10+ (Windows/Mac/Linux)
   - During installation, **check the box: "Add Python to PATH"**
   - Verify: Open a terminal and type `python --version` (should show 3.10+)

2. **Download and install Git** (optional but recommended)
   - Go to https://git-scm.com/downloads
   - Install with default settings
   - Or just download SYNTHLA-EDU V2 as ZIP from GitHub

### Step 1: Get the Code

**Option A: Using Git (Recommended)**
```bash
git clone https://github.com/your-repo/synthla-edu-v2.git
cd synthla-edu-v2
```

**Option B: Download ZIP**
- Click green "Code" button on GitHub
- Click "Download ZIP"
- Extract to a folder
- Open terminal in that folder

### Step 2: Download the Data

The project requires two educational datasets. Download them once and place in the data folder:

#### Dataset 1: OULAD
1. Go to https://analyse.kmi.open.ac.uk/open_dataset
2. Download the **complete OULAD dataset**
3. Extract the 7 CSV files to: `data/raw/oulad/`
   - studentInfo.csv
   - studentRegistration.csv
   - studentVle.csv
   - studentAssessment.csv
   - assessments.csv
   - courses.csv
   - vle.csv

#### Dataset 2: ASSISTments
1. Go to https://www.kaggle.com/datasets/identify/assistments-org-data
2. Download **assistments_2009_2010.csv**
3. Place in: `data/raw/assistments/`

### Step 3: Install Dependencies (One-time)

```bash
pip install -r requirements.txt
```

This installs all necessary libraries (SDV, scikit-learn, pandas, etc.)

### Step 4: Run the Benchmark

#### Quick Run (~5-10 minutes, both datasets, 2 synthesizers)
```bash
set PYTHONPATH=src
python -m synthla_edu_v2.run --config configs/quick.yaml
```

#### Full Run (~30-45 minutes, both datasets, all 3 synthesizers including diffusion)
```bash
set PYTHONPATH=src
python -m synthla_edu_v2.run --config configs/full.yaml
```

#### Minimal Run (~2 minutes, OULAD only, baseline only)
```bash
set PYTHONPATH=src
python -m synthla_edu_v2.run --config configs/minimal.yaml
```

### Step 5: View Results

Results are saved to `runs/` folder:

```
runs/
├── v2_quick/              (or v2_full/minimal/ depending on which you ran)
│   ├── oulad/
│   │   ├── synthetic_train___gaussian_copula.parquet    ← Synthetic data
│   │   ├── quality_gaussian_copula.json                 ← Quality score (SDMetrics)
│   │   ├── c2st_gaussian_copula.json                    ← Realism score (C2ST)
│   │   ├── mia_gaussian_copula.json                     ← Privacy score (MIA)
│   │   └── utility_gaussian_copula.json                 ← Utility score (TSTR)
│   │
│   ├── assistments/        (same structure as oulad)
│   ├── results.csv         ← Summary table of all metrics
│   ├── results.json        ← Same as CSV but JSON format
│   └── run.log             ← Complete execution log
```

### Understanding the Scores

| Metric | Range | Meaning |
|--------|-------|---------|
| **Quality (SDMetrics)** | 0-100% | How well synthetic matches real data (higher = better) |
| **Realism (C2ST AUC)** | 0.5-1.0 | Classifier's ability to detect synthetic (0.5 = undetectable/realistic, 1.0 = easily detected) |
| **Privacy (MIA AUC)** | 0.5-1.0 | Privacy leakage via membership inference (0.5 = no leakage/private, 1.0 = complete leakage) |
| **Utility (TSTR AUC)** | 0-1.0 | Classification accuracy on real test data when trained on synthetic (higher = better) |

### View Results in Detail

Open the CSV or JSON file with any text editor or spreadsheet program:
- `results.csv` - Easy to view in Excel or Google Sheets
- `results.json` - Detailed metrics in JSON format

Example CSV contents:
```
dataset,synthesizer,quality_score,c2st_mean,mia_auc,tstr_auc,tstr_ci_low,tstr_ci_high
oulad,gaussian_copula,73.38,0.9999,0.5039,0.72,0.68,0.76
oulad,ctgan,70.12,0.9889,0.5124,0.68,0.64,0.72
assistments,gaussian_copula,4.13,1.00,0.8444,0.55,0.48,0.62
```

### Troubleshooting

**"Python is not recognized"**
- Install Python 3.10+ again and **check "Add to PATH"** during installation
- Restart your terminal after installation

**"ModuleNotFoundError: No module named 'synthla_edu_v2'"**
- Make sure `PYTHONPATH=src` is set before running
- On Windows: `set PYTHONPATH=src`
- On Mac/Linux: `export PYTHONPATH=src`

**"No such file or directory: data/raw/oulad"**
- Download the OULAD dataset and place CSV files in `data/raw/oulad/`
- Check folder names are exactly: `data/raw/oulad/` and `data/raw/assistments/`

**"Out of memory" or "Process killed"**
- This is a resource limitation on very large datasets (22K+ students)
- Use `configs/quick.yaml` or `configs/minimal.yaml` instead of `full.yaml`
- Or run on a machine with 16GB+ RAM

**Synthetic data has quality 0% or NaN**
- This happens with very small datasets or single-class data
- ASSISTments has intentionally low quality due to only 1,000 samples (expected behavior)
- OULAD (32K samples) has high quality (73%+)

### Next Steps

1. **Read the Results**
   - Open `runs/v2_quick/results.csv` in Excel or Google Sheets
   - Understand what each metric means (see table above)

2. **Explore Synthetic Data**
   - Open `runs/v2_quick/oulad/synthetic_train___gaussian_copula.parquet` with pandas:
     ```python
     import pandas as pd
     df = pd.read_parquet('runs/v2_quick/oulad/synthetic_train___gaussian_copula.parquet')
     print(df.head())  # First 5 rows
     print(df.describe())  # Statistics
     ```

3. **Run Your Own Experiments**
   - Modify `configs/quick.yaml` to change hyperparameters
   - Change dataset, number of synthesizers, or evaluation settings
   - Re-run with `python -m synthla_edu_v2.run --config configs/quick.yaml`

4. **Use Synthetic Data in Your Research**
   - Synthetic data is available as Parquet files (Python: pandas)
   - Available as CSV via: `parquet_file.to_csv('output.csv')`
   - Share without privacy concerns (validated via MIA privacy audit)

### Testing (For Developers)

```bash
# Run all tests (6 tests, should all pass)
set PYTHONPATH=src
pytest tests/ -v
```

### More Information

- **Full Documentation**: See [README_COMPREHENSIVE.md](README_COMPREHENSIVE.md)
- **Usage Guide**: See [USAGE.md](USAGE.md)
- **Quick Reference**: See [QUICKREF.md](QUICKREF.md)
- **Deployment**: See [DEPLOYMENT.md](DEPLOYMENT.md)

---

## One Command to Rule Them All

For the absolute simplest experience, just run:

**Windows:**
```bash
set PYTHONPATH=src && python -m synthla_edu_v2.run --config configs/quick.yaml
```

**Mac/Linux:**
```bash
export PYTHONPATH=src && python -m synthla_edu_v2.run --config configs/quick.yaml
```

That's it! Everything runs end-to-end. Results saved to `runs/v2_quick/`.

---

## Docker (Optional: Easier for Some Users)

If you have Docker installed, you can skip Python setup entirely:

```bash
# Build once
docker build -t synthla-edu-v2 .

# Run anytime (assuming data in local data/ folder)
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  synthla-edu-v2 \
  python -m synthla_edu_v2.run --config configs/quick.yaml
```

---

**Happy synthesizing! 🚀**

For questions or issues, please open an issue on GitHub.
