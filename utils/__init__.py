"""
Utility modules for SYNTHLA-EDU V2.

This package provides supporting utilities that can be integrated into
the main pipeline without modifying core logic:

- timing.py: Timing instrumentation for synthesizers
- effect_size.py: Cohen's d and other effect size calculations
- logging_config.py: Structured logging configuration

Example integration (future enhancement):
    from utils.timing import TimingMonitor, SynthesizerTimingWrapper
    from utils.effect_size import cohens_d_for_features
    from utils.logging_config import setup_logger
    
    logger = setup_logger("experiment", log_file="runs/experiment.log")
    monitor = TimingMonitor()
    
    # Wrap synthesizer for automatic timing
    synth = CTGANSynth(schema, epochs=300)
    timed_synth = SynthesizerTimingWrapper(synth, monitor, "ctgan")
    
    logger.info("Training CTGAN...")
    timed_synth.fit(train_data)
    
    logger.info("Generating synthetic data...")
    synthetic = timed_synth.sample(n=10000)
    
    # Compute effect sizes
    effect_sizes = cohens_d_for_features(
        real_data, synthetic,
        numeric_cols=['age', 'total_clicks', 'avg_score']
    )
    
    for feature, (d, interp) in effect_sizes.items():
        logger.info(f"{feature}: Cohen's d = {d:.3f} ({interp})")
    
    # Save timing results
    monitor.save("runs/timing.json")
    monitor.print_summary()
"""

__version__ = "2.0.0"

# Import key functions for easy access
try:
    from .timing import TimingMonitor, SynthesizerTimingWrapper, timing_decorator
except ImportError:
    pass

try:
    from .effect_size import (
        cohens_d,
        cohens_d_from_means,
        interpret_cohens_d,
        cohens_d_for_features,
        hedges_g,
        cliff_delta,
        interpret_cliff_delta,
        effect_size_summary
    )
except ImportError:
    pass

try:
    from .logging_config import (
        setup_logger,
        get_logger,
        ProgressLogger,
        LogContext
    )
except ImportError:
    pass

__all__ = [
    # Timing
    "TimingMonitor",
    "SynthesizerTimingWrapper",
    "timing_decorator",
    # Effect size
    "cohens_d",
    "cohens_d_from_means",
    "interpret_cohens_d",
    "cohens_d_for_features",
    "hedges_g",
    "cliff_delta",
    "interpret_cliff_delta",
    "effect_size_summary",
    # Logging
    "setup_logger",
    "get_logger",
    "ProgressLogger",
    "LogContext",
]
