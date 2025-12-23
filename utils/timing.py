"""
Timing instrumentation utilities for SYNTHLA-EDU V2.

Provides decorators and context managers for capturing actual training
and sampling times of synthesizers. Can be integrated into the main pipeline
without modifying core logic.

Usage:
    from utils.timing import TimingMonitor
    
    monitor = TimingMonitor()
    
    with monitor.time("gaussian_copula_fit"):
        synthesizer.fit(data)
    
    with monitor.time("gaussian_copula_sample"):
        synthetic = synthesizer.sample(n=1000)
    
    monitor.save("runs/oulad/timing.json")
"""

import time
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class TimingMonitor:
    """Monitor and record timing information for operations."""
    
    def __init__(self):
        """Initialize timing monitor."""
        self.timings: Dict[str, List[float]] = {}
        self.metadata: Dict[str, any] = {
            "start_time": datetime.now().isoformat(),
            "version": "2.0.0"
        }
    
    @contextmanager
    def time(self, operation_name: str):
        """
        Context manager to time an operation.
        
        Args:
            operation_name: Unique identifier for the operation
        
        Example:
            with monitor.time("model_training"):
                model.fit(X, y)
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self._record(operation_name, elapsed)
    
    def _record(self, operation_name: str, elapsed: float):
        """Record a timing measurement."""
        if operation_name not in self.timings:
            self.timings[operation_name] = []
        self.timings[operation_name].append(elapsed)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all timed operations.
        
        Returns:
            Dictionary with mean, median, min, max, total for each operation
        """
        import statistics
        
        summary = {}
        for op_name, times in self.timings.items():
            summary[op_name] = {
                "count": len(times),
                "mean_seconds": statistics.mean(times),
                "median_seconds": statistics.median(times),
                "min_seconds": min(times),
                "max_seconds": max(times),
                "total_seconds": sum(times),
                "mean_minutes": statistics.mean(times) / 60,
                "total_minutes": sum(times) / 60
            }
        return summary
    
    def save(self, filepath: str):
        """
        Save timing results to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "metadata": {
                **self.metadata,
                "end_time": datetime.now().isoformat()
            },
            "raw_timings": self.timings,
            "summary": self.get_summary()
        }
        
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
    
    def load(self, filepath: str):
        """
        Load timing results from JSON file.
        
        Args:
            filepath: Path to input JSON file
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        
        self.timings = data.get("raw_timings", {})
        self.metadata = data.get("metadata", {})
    
    def print_summary(self):
        """Print formatted timing summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("TIMING SUMMARY")
        print("="*80)
        
        for op_name, stats in sorted(summary.items()):
            print(f"\n{op_name}:")
            print(f"  Count:       {stats['count']}")
            print(f"  Mean:        {stats['mean_minutes']:.2f} min ({stats['mean_seconds']:.2f} sec)")
            print(f"  Median:      {stats['median_seconds']:.2f} sec")
            print(f"  Range:       {stats['min_seconds']:.2f} - {stats['max_seconds']:.2f} sec")
            print(f"  Total:       {stats['total_minutes']:.2f} min")
        
        print("\n" + "="*80)


def timing_decorator(operation_name: str, monitor: Optional[TimingMonitor] = None):
    """
    Decorator to automatically time a function.
    
    Args:
        operation_name: Name for the operation
        monitor: TimingMonitor instance (optional)
    
    Example:
        monitor = TimingMonitor()
        
        @timing_decorator("data_loading", monitor)
        def load_data():
            return pd.read_csv("data.csv")
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if monitor is None:
                # No monitor provided, just run function
                return func(*args, **kwargs)
            
            with monitor.time(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class SynthesizerTimingWrapper:
    """
    Wrapper for synthesizers to automatically time fit() and sample() calls.
    
    Example:
        from synthla_edu_v2 import GaussianCopulaSynth
        
        synth = GaussianCopulaSynth(schema)
        timed_synth = SynthesizerTimingWrapper(synth, monitor, "gaussian_copula")
        
        timed_synth.fit(data)  # Automatically timed
        synthetic = timed_synth.sample(n=1000)  # Automatically timed
    """
    
    def __init__(self, synthesizer, monitor: TimingMonitor, synth_name: str):
        """
        Initialize timing wrapper.
        
        Args:
            synthesizer: Synthesizer instance to wrap
            monitor: TimingMonitor instance
            synth_name: Name prefix for timing operations
        """
        self.synthesizer = synthesizer
        self.monitor = monitor
        self.synth_name = synth_name
    
    def fit(self, data, *args, **kwargs):
        """Fit synthesizer with timing."""
        with self.monitor.time(f"{self.synth_name}_fit"):
            return self.synthesizer.fit(data, *args, **kwargs)
    
    def sample(self, n, *args, **kwargs):
        """Sample from synthesizer with timing."""
        with self.monitor.time(f"{self.synth_name}_sample"):
            return self.synthesizer.sample(n, *args, **kwargs)
    
    def __getattr__(self, name):
        """Forward all other attributes to wrapped synthesizer."""
        return getattr(self.synthesizer, name)


# Example usage script
if __name__ == "__main__":
    # Demonstration of timing utilities
    monitor = TimingMonitor()
    
    # Example: Time a simulated operation
    import time
    import random
    
    print("Running timing demonstration...")
    
    # Simulate multiple operations
    for i in range(3):
        with monitor.time("data_loading"):
            time.sleep(random.uniform(0.1, 0.3))
        
        with monitor.time("model_training"):
            time.sleep(random.uniform(0.5, 1.0))
        
        with monitor.time("inference"):
            time.sleep(random.uniform(0.05, 0.15))
    
    # Print summary
    monitor.print_summary()
    
    # Save to file
    monitor.save("demo_timing.json")
    print("\nTiming results saved to demo_timing.json")
