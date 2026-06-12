"""Howard & Knutson migration model: invariants and calibration recovery."""
import numpy as np
import pytest

from rivabar.prediction import (predicted_migration_rate, calibrate_from_curvature,
                                calibrate_local_kl, _migrate_forward,
                                _migrate_forward_local, _stable_segment_mask)


N = 2000
S = np.arange(N) * 25.0  # 25 m spacing, 50 km reach
W = 200.0


def test_constant_curvature_gives_exactly_1p5_R0():
    """For constant R0, R1 = (Omega + Gamma) * R0 = 1.5 * R0 everywhere —
    including at the upstream boundary (regression test for the kernel
    normalization bug)."""
    R1 = predicted_migration_rate(np.ones(N), S, D=5.4, Cf=0.005)
    assert np.max(np.abs(R1 - 1.5)) < 1e-9


def test_masked_calibration_recovers_parameters():
    """Calibration statistics restricted to stable segments must not be
    distorted by a corrupted (high-variance) zone, and the prediction must
    be computed on the full contiguous arrays."""
    rng = np.random.default_rng(0)
    curv = 1e-3 * np.sin(2 * np.pi * S / 5000.0)
    kl_true = 5.0
    obs = -1.5 * kl_true * W * np.roll(curv, 30) + rng.normal(0, 0.05, N)
    obs[800:1000] += rng.normal(0, 5.0, 200)  # corrupted zone
    mask = np.ones(N, dtype=bool)
    mask[800:1000] = False

    cal = calibrate_from_curvature(curv, obs, S, W, mask=mask)
    assert len(cal['R1']) == N  # full-extent prediction
    assert cal['r_predicted'] > 0.95
    assert abs(cal['kl'] - kl_true) / kl_true < 0.25

    # without the mask the corrupted zone destroys the correlation
    cal_unmasked = calibrate_from_curvature(curv, obs, S, W)
    assert cal_unmasked['r_predicted'] < cal['r_predicted']


def test_local_kl_recovers_step_change():
    rng = np.random.default_rng(1)
    n = 3000
    s = np.arange(n) * 25.0
    curv = 1e-3 * np.sin(2 * np.pi * s / 5000.0)
    kl_true = np.where(s < s[n // 2], 3.0, 8.0)
    obs = -1.5 * kl_true * W * np.roll(curv, 30) + rng.normal(0, 0.1, n)
    obs[1400:1600] += rng.normal(0, 20.0, 200)
    mask = np.ones(n, dtype=bool)
    mask[1400:1600] = False

    cal = calibrate_local_kl(curv, obs, s, W, mask=mask)
    assert len(cal['kl_local']) == n
    assert abs(np.median(cal['kl_local'][:1000]) - 3.0) < 1.0
    assert abs(np.median(cal['kl_local'][-1000:]) - 8.0) < 1.5


def test_migrate_forward_wrapper_equals_local_with_constant_kl():
    x0 = S.copy()
    y0 = 100 * np.sin(2 * np.pi * S / 5000.0)
    xa, ya = _migrate_forward(x0.copy(), y0.copy(), W, 5.4, 0.005, 5.0,
                              -1.0, 2.5, 10.0, 25.0)
    xb, yb = _migrate_forward_local(x0.copy(), y0.copy(), W, 5.4, 0.005,
                                    np.array([5.0, 5.0]), np.array([0.0, 1.0]),
                                    -1.0, 2.5, 10.0, 25.0)
    assert np.allclose(xa, xb) and np.allclose(ya, yb)
    # the channel actually moved
    assert np.max(np.abs(ya - y0[:len(ya)] if len(ya) == len(y0) else ya)) > 0


def test_stable_segment_mask():
    pair_info = {'segment_results': [{'start_idx': 2, 'end_idx': 4},
                                     {'start_idx': 8, 'end_idx': 9}]}
    mask = _stable_segment_mask(pair_info, 12)
    assert list(np.where(mask)[0]) == [2, 3, 4, 8, 9]
    assert _stable_segment_mask({'segment_results': []}, 12) is None
    assert _stable_segment_mask({}, 12) is None


def test_predict_forward_local_accepts_non_windowed_calibration():
    """Regression test for the eagerly-evaluated dict.get fallback that
    raised KeyError for calibrations without a kl_local_series."""
    from rivabar.prediction import predict_forward_local
    from tests.conftest import make_synthetic_river

    river = make_synthetic_river(n=1000, amplitude=200.0, wavelength=8000.0,
                                 with_dataset=True)
    s_ref = np.arange(1000) * 100.0
    local_calibration = {
        'kl_local_median': np.full(1000, 5.0),
        's_reference': s_ref,
        'Cf': 0.005,  # temporal_cross_validate schema uses 'Cf', not 'Cf_global'
        'x_reference': s_ref,
        'y_reference': 200 * np.sin(2 * np.pi * s_ref / 8000.0),
    }
    result = predict_forward_local(river, local_calibration, delta_s=100)
    assert np.isfinite(result['predicted_mr']).any()
