"""
Effect size calculations for SYNTHLA-EDU V2.

Provides functions to compute Cohen's d and other effect size metrics
for comparing real vs. synthetic data distributions and model performance.

Effect sizes provide interpretable measures of practical significance
beyond statistical significance (p-values).

Usage:
    from utils.effect_size import cohens_d, interpret_cohens_d
    
    # Compare feature distributions
    d = cohens_d(real_data['age'], synthetic_data['age'])
    print(f"Cohen's d: {d:.3f} ({interpret_cohens_d(d)})")
    
    # Compare model performance
    d_utility = cohens_d_from_means(r2_real, r2_synth, std_real, std_synth, n_real, n_synth)
"""

import numpy as np
from typing import Union, Tuple, Dict, List
import pandas as pd


def cohens_d(group1: Union[np.ndarray, pd.Series], 
             group2: Union[np.ndarray, pd.Series],
             pooled: bool = True) -> float:
    """
    Calculate Cohen's d effect size between two groups.
    
    Cohen's d measures the standardized difference between two means:
    - Small effect: |d| = 0.2
    - Medium effect: |d| = 0.5
    - Large effect: |d| = 0.8
    
    Args:
        group1: First group (e.g., real data)
        group2: Second group (e.g., synthetic data)
        pooled: If True, use pooled standard deviation. If False, use group1's SD.
    
    Returns:
        Cohen's d effect size
    
    Reference:
        Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.).
    """
    # Convert to numpy arrays
    g1 = np.asarray(group1)
    g2 = np.asarray(group2)
    
    # Remove NaN values
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    
    if len(g1) == 0 or len(g2) == 0:
        return np.nan
    
    # Calculate means
    mean1 = np.mean(g1)
    mean2 = np.mean(g2)
    
    # Calculate standard deviations
    std1 = np.std(g1, ddof=1)
    std2 = np.std(g2, ddof=1)
    
    # Calculate pooled or control-group standard deviation
    if pooled:
        n1, n2 = len(g1), len(g2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        denominator = pooled_std
    else:
        denominator = std1
    
    if denominator == 0:
        return np.nan
    
    # Cohen's d = (mean1 - mean2) / SD
    d = (mean1 - mean2) / denominator
    
    return d


def cohens_d_from_means(mean1: float, mean2: float,
                       std1: float, std2: float,
                       n1: int, n2: int) -> float:
    """
    Calculate Cohen's d from summary statistics.
    
    Useful when you only have means, SDs, and sample sizes
    (e.g., from published results or aggregated metrics).
    
    Args:
        mean1: Mean of group 1
        mean2: Mean of group 2
        std1: Standard deviation of group 1
        std2: Standard deviation of group 2
        n1: Sample size of group 1
        n2: Sample size of group 2
    
    Returns:
        Cohen's d effect size
    """
    if std1 == 0 and std2 == 0:
        return np.nan
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return np.nan
    
    d = (mean1 - mean2) / pooled_std
    return d


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size magnitude.
    
    Args:
        d: Cohen's d value
    
    Returns:
        Interpretation string
    
    Reference:
        Cohen, J. (1988). Conventional effect size thresholds.
    """
    if np.isnan(d):
        return "undefined"
    
    abs_d = abs(d)
    
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def cohens_d_for_features(real_df: pd.DataFrame,
                          synth_df: pd.DataFrame,
                          feature_cols: List[str],
                          pooled: bool = True) -> Dict[str, Tuple[float, str]]:
    """
    Calculate Cohen's d for multiple features.
    
    Args:
        real_df: Real data DataFrame
        synth_df: Synthetic data DataFrame
        feature_cols: List of numeric feature columns to compare
        pooled: Use pooled standard deviation
    
    Returns:
        Dictionary mapping feature names to (cohen's d, interpretation)
    
    Example:
        results = cohens_d_for_features(
            real_df, synth_df,
            ['age', 'total_clicks', 'avg_score']
        )
        
        for feature, (d, interp) in results.items():
            print(f"{feature}: d={d:.3f} ({interp})")
    """
    results = {}
    
    for col in feature_cols:
        if col not in real_df.columns or col not in synth_df.columns:
            results[col] = (np.nan, "missing")
            continue
        
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(real_df[col]):
            results[col] = (np.nan, "non-numeric")
            continue
        
        d = cohens_d(real_df[col], synth_df[col], pooled=pooled)
        interpretation = interpret_cohens_d(d)
        results[col] = (d, interpretation)
    
    return results


def glass_delta(group1: Union[np.ndarray, pd.Series],
               group2: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Glass's Δ (delta) effect size.
    
    Similar to Cohen's d, but uses only the control group's (group1) SD.
    Useful when group2 has artificially reduced variance.
    
    Args:
        group1: Control group (e.g., real data)
        group2: Treatment group (e.g., synthetic data)
    
    Returns:
        Glass's delta
    """
    return cohens_d(group1, group2, pooled=False)


def hedges_g(group1: Union[np.ndarray, pd.Series],
            group2: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Hedges' g effect size (bias-corrected Cohen's d).
    
    Provides better estimate for small sample sizes (n < 20).
    
    Args:
        group1: First group
        group2: Second group
    
    Returns:
        Hedges' g effect size
    
    Reference:
        Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect size.
    """
    g1 = np.asarray(group1)
    g2 = np.asarray(group2)
    
    # Remove NaN
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    
    n1, n2 = len(g1), len(g2)
    
    if n1 == 0 or n2 == 0:
        return np.nan
    
    # Calculate Cohen's d
    d = cohens_d(g1, g2, pooled=True)
    
    if np.isnan(d):
        return np.nan
    
    # Correction factor J
    df = n1 + n2 - 2
    J = 1 - (3 / (4 * df - 1))
    
    # Hedges' g = J * d
    g = J * d
    
    return g


def cliff_delta(group1: Union[np.ndarray, pd.Series],
               group2: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Cliff's delta (non-parametric effect size).
    
    Measures the degree to which values in one group are larger than in another.
    Range: [-1, 1]
    - delta = 1: All values in group1 > all values in group2
    - delta = 0: Complete overlap
    - delta = -1: All values in group1 < all values in group2
    
    Interpretation:
    - |delta| < 0.147: negligible
    - |delta| < 0.33: small
    - |delta| < 0.474: medium
    - |delta| >= 0.474: large
    
    Args:
        group1: First group
        group2: Second group
    
    Returns:
        Cliff's delta
    
    Reference:
        Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions.
    """
    g1 = np.asarray(group1)
    g2 = np.asarray(group2)
    
    # Remove NaN
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    
    n1, n2 = len(g1), len(g2)
    
    if n1 == 0 or n2 == 0:
        return np.nan
    
    # Count dominance
    greater = sum(x > y for x in g1 for y in g2)
    less = sum(x < y for x in g1 for y in g2)
    
    # Cliff's delta
    delta = (greater - less) / (n1 * n2)
    
    return delta


def interpret_cliff_delta(delta: float) -> str:
    """
    Interpret Cliff's delta magnitude.
    
    Args:
        delta: Cliff's delta value
    
    Returns:
        Interpretation string
    """
    if np.isnan(delta):
        return "undefined"
    
    abs_delta = abs(delta)
    
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"


def effect_size_summary(real_df: pd.DataFrame,
                       synth_df: pd.DataFrame,
                       numeric_cols: List[str]) -> pd.DataFrame:
    """
    Generate comprehensive effect size summary for all numeric features.
    
    Args:
        real_df: Real data
        synth_df: Synthetic data
        numeric_cols: List of numeric columns to analyze
    
    Returns:
        DataFrame with effect sizes and interpretations
    
    Example:
        summary = effect_size_summary(real_df, synth_df, ['age', 'score', 'clicks'])
        print(summary.to_string())
    """
    results = []
    
    for col in numeric_cols:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        
        if not pd.api.types.is_numeric_dtype(real_df[col]):
            continue
        
        # Calculate multiple effect sizes
        d = cohens_d(real_df[col], synth_df[col])
        g = hedges_g(real_df[col], synth_df[col])
        delta = cliff_delta(real_df[col], synth_df[col])
        
        results.append({
            "Feature": col,
            "Cohen's d": d,
            "d Interpretation": interpret_cohens_d(d),
            "Hedges' g": g,
            "Cliff's Delta": delta,
            "Delta Interpretation": interpret_cliff_delta(delta)
        })
    
    return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # Demonstration of effect size calculations
    np.random.seed(42)
    
    # Simulate real and synthetic data
    print("Effect Size Demonstration\n" + "="*60)
    
    # Example 1: Similar distributions (small effect)
    real_similar = np.random.normal(100, 15, 1000)
    synth_similar = np.random.normal(102, 15, 1000)
    
    d_similar = cohens_d(real_similar, synth_similar)
    print(f"\nExample 1: Similar distributions")
    print(f"Cohen's d = {d_similar:.3f} ({interpret_cohens_d(d_similar)})")
    
    # Example 2: Different distributions (large effect)
    real_different = np.random.normal(100, 15, 1000)
    synth_different = np.random.normal(120, 15, 1000)
    
    d_different = cohens_d(real_different, synth_different)
    print(f"\nExample 2: Different distributions")
    print(f"Cohen's d = {d_different:.3f} ({interpret_cohens_d(d_different)})")
    
    # Example 3: Non-parametric alternative
    delta_different = cliff_delta(real_different, synth_different)
    print(f"Cliff's Delta = {delta_different:.3f} ({interpret_cliff_delta(delta_different)})")
    
    print("\n" + "="*60)
    print("\nEffect Size Interpretation Guide:")
    print("Cohen's d / Hedges' g:")
    print("  0.0 - 0.2: negligible")
    print("  0.2 - 0.5: small")
    print("  0.5 - 0.8: medium")
    print("  0.8+     : large")
    print("\nCliff's Delta:")
    print("  0.0 - 0.147: negligible")
    print("  0.147 - 0.33: small")
    print("  0.33 - 0.474: medium")
    print("  0.474+      : large")
