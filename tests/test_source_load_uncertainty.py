import numpy as np
import pandas as pd

from scripts.run_paper_reference_case33 import sample_source_load_sequences


def _sequence(n=48):
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "load_kw": np.full(n, 100.0),
        "wind_kw": np.full(n, 30.0),
        "pv_kw": np.full(n, 20.0),
        "random_sequence_id": "base",
        "sequence_weight": 1.0,
    })


def test_reference_source_path_is_preserved_and_samples_are_correlated():
    base = _sequence()
    paths = sample_source_load_sequences(base, seed=42, sample_count=4)
    assert len(paths) == 4
    pd.testing.assert_frame_equal(paths[0], base)
    assert all(path.sequence_weight.iloc[0] == 0.25 for path in paths[1:])
    assert not np.allclose(paths[1].load_kw, base.load_kw)
    assert not np.allclose(paths[1].wind_kw, base.wind_kw)
    assert (paths[1][["load_kw", "wind_kw", "pv_kw"]] >= 0).all().all()


def test_source_uncertainty_is_reproducible():
    left = sample_source_load_sequences(_sequence(), seed=7, sample_count=3)
    right = sample_source_load_sequences(_sequence(), seed=7, sample_count=3)
    for a, b in zip(left, right):
        pd.testing.assert_frame_equal(a, b)
