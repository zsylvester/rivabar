"""Signal-processing utilities: variance detection, pair analysis, classification."""
import numpy as np
import pandas as pd
import pytest

from rivabar.utils import (detect_high_variance_segments, _analyze_pair_core,
                           classify_pairs, compute_s_distance,
                           compute_migration_distances)


def test_moving_variance_matches_reference_loop():
    """The cumulative-sum vectorization must reproduce the original
    per-window np.var loop exactly."""
    rng = np.random.default_rng(2)
    data = rng.normal(0, 1, 1000)
    data[300:350] += rng.normal(0, 30, 50)
    window = 50

    segments, var_profile, var_smooth = detect_high_variance_segments(data, window_size=window)

    reference = np.zeros(len(data))
    for i in range(len(data)):
        reference[i] = np.var(data[max(0, i - window // 2):min(len(data), i + window // 2)])
    assert np.allclose(var_profile, reference)
    # the injected high-variance zone is detected, roughly in place
    assert len(segments) == 1
    start, end = segments[0]
    assert start < 350 and end > 300


def _sine_info(shift, idx, date, n=2000, spacing=25.0, wavelength=8000.0):
    s_axis = np.arange(n) * spacing
    x = s_axis.copy()
    y = 150 * np.sin(2 * np.pi * (s_axis - shift) / wavelength)
    return {'x': x, 'y': y, 's': compute_s_distance(x, y),
            'width': np.full(n, 200.0),
            'curvature': 1e-3 * np.sin(2 * np.pi * (s_axis - shift) / wavelength),
            'original_index': idx, 'date': pd.Timestamp(date),
            'year': pd.Timestamp(date).year}


def test_analyze_pair_core_structure():
    pair = _analyze_pair_core(_sine_info(0, 0, '2000-06-01'),
                              _sine_info(200, 1, '2003-06-01'),
                              time_gap_days=1096, time_gap_years=3.0,
                              variance_threshold=100.0, min_segment_length=50)
    assert set(pair) == {'cost', 'lags', 'curvature', 'distances', 's',
                         'centerline_coords', 'pair_info'}
    n = len(pair['curvature'])
    assert len(pair['distances']) == n
    assert len(pair['s']) == n
    info = pair['pair_info']
    assert info['time_gap_years'] == 3.0
    assert info['n_stable_segments'] >= 1
    assert 0 < info['stable_data_ratio'] <= 1


def test_compute_migration_distances_sign_convention():
    """A centerline shifted in +y relative to a +x-oriented reference gives
    positive (leftward) migration distances."""
    n = 200
    x = np.arange(n) * 10.0
    y1 = np.zeros(n)
    y2 = np.full(n, 50.0)
    p = np.arange(n)
    q = np.arange(n)
    distances, valid = compute_migration_distances(x, y1, x, y2, p, q)
    assert np.nanmedian(distances[valid]) == pytest.approx(50.0)


def test_classify_pairs_categories():
    r = [0.6, 0.6, 0.05, np.nan, 0.6]
    r2 = [0.4, 0.4, 0.0, np.nan, 0.4]
    ws = [1.0, 1.5, 1.0, np.nan, 1.0]
    lag = [1.0, 1.0, 1.0, np.nan, 10.0]
    classes = classify_pairs(r, r2, ws, lag)
    assert list(classes) == ['good', 'stage_mismatch', 'weak', 'insufficient',
                             'reversed']
