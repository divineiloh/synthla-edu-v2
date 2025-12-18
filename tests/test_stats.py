import numpy as np

from synthla_edu_v2.eval.stats import bootstrap_ci, paired_permutation_test


def test_bootstrap_ci_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    p = rng.random(100)
    res = bootstrap_ci(y, p, metric="auc", n_boot=100, seed=0, alpha=0.05)
    assert res["metric"] == "auc"
    assert res["n_boot"] == 100
    assert res["ci_low"] <= res["ci_high"]


def test_paired_perm_test_auc():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=100)
    a = rng.random(100)
    b = rng.random(100)
    res = paired_permutation_test(y, a, b, metric="auc", n_perm=200, seed=0)
    assert res["metric"] == "auc"
    assert "p_value_two_sided" in res
    assert 0.0 <= res["p_value_two_sided"] <= 1.0
