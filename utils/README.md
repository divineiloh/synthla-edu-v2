# Utils Module

Supporting utilities for SYNTHLA-EDU V2 that extend functionality without modifying core pipeline logic.

## Modules

### timing.py - Timing Instrumentation
Captures actual training and sampling times for synthesizers.

**Classes:**
- `TimingMonitor`: Context manager for timing operations
- `SynthesizerTimingWrapper`: Automatic timing for synthesizer fit/sample

**Example:**
```python
from utils.timing import TimingMonitor, SynthesizerTimingWrapper
from synthla_edu_v2 import CTGANSynth

monitor = TimingMonitor()

# Manual timing
with monitor.time("data_loading"):
    df, schema = build_dataset("oulad", "data/raw")

# Automatic synthesizer timing
synth = CTGANSynth(schema, epochs=300)
timed_synth = SynthesizerTimingWrapper(synth, monitor, "ctgan")

timed_synth.fit(df)  # Automatically timed as "ctgan_fit"
synthetic = timed_synth.sample(n=10000)  # Timed as "ctgan_sample"

# Save results
monitor.save("runs/oulad/timing.json")
monitor.print_summary()
```

**Output Format (timing.json):**
```json
{
  "metadata": {
    "start_time": "2025-01-15T10:30:00",
    "end_time": "2025-01-15T11:45:30",
    "version": "2.0.0"
  },
  "raw_timings": {
    "ctgan_fit": [3456.2],
    "ctgan_sample": [45.8]
  },
  "summary": {
    "ctgan_fit": {
      "mean_seconds": 3456.2,
      "mean_minutes": 57.6
    }
  }
}
```

---

### effect_size.py - Effect Size Calculations
Quantifies practical significance of differences between real and synthetic data.

**Functions:**
- `cohens_d(group1, group2)`: Cohen's d effect size
- `interpret_cohens_d(d)`: Interpretation (negligible/small/medium/large)
- `cohens_d_for_features(real_df, synth_df, cols)`: Batch calculation
- `hedges_g(group1, group2)`: Bias-corrected Cohen's d
- `cliff_delta(group1, group2)`: Non-parametric effect size

**Example:**
```python
from utils.effect_size import cohens_d_for_features, effect_size_summary

# Calculate Cohen's d for all numeric features
results = cohens_d_for_features(
    real_df, synth_df,
    feature_cols=['age', 'total_clicks', 'avg_assessment_score']
)

for feature, (d, interpretation) in results.items():
    print(f"{feature}: d={d:.3f} ({interpretation})")

# Output:
# age: d=0.05 (negligible)
# total_clicks: d=0.15 (negligible)
# avg_assessment_score: d=0.32 (small)

# Generate comprehensive summary
summary_df = effect_size_summary(real_df, synth_df, numeric_cols)
print(summary_df.to_string())
```

**Effect Size Interpretation:**

| Measure | Negligible | Small | Medium | Large |
|---------|------------|-------|--------|-------|
| Cohen's d | < 0.2 | 0.2-0.5 | 0.5-0.8 | ≥ 0.8 |
| Cliff's Delta | < 0.147 | 0.147-0.33 | 0.33-0.474 | ≥ 0.474 |

---

### logging_config.py - Structured Logging
Replaces print statements with structured, configurable logging.

**Classes:**
- `ColoredFormatter`: Color-coded console output
- `ProgressLogger`: Percentage-based progress tracking
- `LogContext`: Hierarchical logging sections

**Functions:**
- `setup_logger(name, log_file, level)`: Configure logger

**Example:**
```python
from utils.logging_config import setup_logger, LogContext, ProgressLogger

# Setup logger (console + file)
logger = setup_logger(
    name="experiment",
    log_file="runs/oulad/experiment.log",
    level="INFO",          # Console level
    file_level="DEBUG"     # File level (more detailed)
)

# Basic logging
logger.info("Starting experiment...")
logger.debug("This only appears in log file")
logger.warning("Warning message")
logger.error("Error occurred")

# Structured sections
with LogContext(logger, "Data Loading"):
    logger.info("Loading OULAD...")
    df, schema = build_dataset("oulad", "data/raw")
    logger.info(f"Loaded {len(df)} rows")

# Progress tracking
with LogContext(logger, "Training CTGAN"):
    progress = ProgressLogger(logger, total_steps=300, prefix="Epoch")
    
    for epoch in range(300):
        loss = train_epoch()
        progress.update(1, extra_msg=f"Loss: {loss:.4f}")
    
    progress.finish()
```

**Console Output:**
```
INFO | Starting experiment...
============================================================
Data Loading
============================================================
INFO | Loading OULAD...
INFO | Loaded 32593 rows
------------------------------------------------------------
Data Loading complete | Time: 12.3s

============================================================
Training CTGAN
============================================================
INFO | Epoch: 10.0% (30/300) | Loss: 0.8234 | Elapsed: 45.2s
INFO | Epoch: 20.0% (60/300) | Loss: 0.7123 | Elapsed: 89.1s
...
INFO | Epoch: Complete | Total time: 432.5s
------------------------------------------------------------
Training CTGAN complete | Time: 432.5s
```

---

## Integration with Main Pipeline

These utilities are designed to be **optional additions** that don't require modifying `synthla_edu_v2.py`.

### Future Integration Strategy

1. **Timing**: Wrap synthesizers at instantiation
2. **Effect Sizes**: Add to results JSON under new "effect_sizes" key
3. **Logging**: Replace print statements with logger calls

**Example Minimal Integration:**
```python
# In synthla_edu_v2.py (minimal changes)
from utils.timing import TimingMonitor
from utils.effect_size import cohens_d_for_features
from utils.logging_config import setup_logger

def run_all(datasets, synthesizers, output_base="runs", logger=None, timing_monitor=None):
    """Run experiments with optional timing and logging."""
    
    if logger is None:
        # Fallback to print statements
        logger = type('obj', (object,), {'info': print, 'warning': print, 'error': print})()
    
    if timing_monitor is None:
        timing_monitor = TimingMonitor()
    
    # Rest of pipeline remains unchanged...
```

---

## Standalone Usage

All utility modules can be run standalone for testing:

```bash
# Test timing utilities
python utils/timing.py

# Test effect size calculations
python utils/effect_size.py

# Test logging configuration
python utils/logging_config.py
```

---

## Dependencies

All utilities use only standard library or already-required packages:
- `numpy` (already required)
- `pandas` (already required)
- `logging` (standard library)
- `time` (standard library)
- `contextlib` (standard library)

No additional installations needed.

---

## Use Cases

### Timing Analysis
- Compare actual training times across synthesizers
- Validate computational efficiency claims
- Estimate production deployment costs
- Benchmark hardware configurations

### Effect Size Analysis
- Assess practical significance beyond statistical tests
- Identify features with largest distributional shifts
- Support result interpretation in paper
- Compare synthesizers on feature fidelity

### Structured Logging
- Reproduce experiments from log files
- Debug long-running experiments
- Monitor progress without GUI
- Archive experiment configurations

---

## Citation

If using these utilities in published research, cite:

```bibtex
@software{synthla_edu_v2_utils,
  title={SYNTHLA-EDU V2: Supporting Utilities},
  author={Divine Iloh},
  year={2025},
  version={2.0.0},
  url={https://github.com/yourusername/synthla-edu-v2}
}
```
