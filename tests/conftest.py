import pandas as pd


class _DummySynth:
    def __init__(self, params=None):
        self.params = params or {}
        self.name = self.__class__.__name__.lower()

    def fit(self, df):
        # no-op
        self._df = df.copy()

    def sample(self, n):
        # return first n rows (or repeat) as a simple synthetic set
        if len(self._df) == 0:
            return pd.DataFrame({})
        if len(self._df) >= n:
            return self._df.head(n).copy()
        # else repeat rows
        reps = (n // len(self._df)) + 1
        return pd.concat([self._df.copy()] * reps, ignore_index=True).head(n)


def pytest_configure(config):
    # Monkeypatch imports in the package to avoid heavy dependencies during CI/smoke runs
    import synthla_edu_v2.synth.sdv_wrappers as svd
    import synthla_edu_v2.synth.tabddpm_wrappers as tdd

    svd.GaussianCopulaSynth = _DummySynth
    svd.CTGANSynth = _DummySynth
    tdd.TabDDPMSynth = _DummySynth
