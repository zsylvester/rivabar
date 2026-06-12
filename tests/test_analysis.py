"""Channel morphology basics: curvature and zero crossings on known shapes."""
import numpy as np
import pytest

from rivabar.analysis import compute_curvature, find_zero_crossings


def test_curvature_of_circle():
    R = 500.0
    theta = np.linspace(0, 2 * np.pi, 2000)
    curv, s = compute_curvature(R * np.cos(theta), R * np.sin(theta))
    # interior points: |curvature| = 1/R, s spans the circumference
    interior = curv[10:-10]
    assert np.allclose(np.abs(interior), 1.0 / R, rtol=1e-3)
    assert s[-1] == pytest.approx(2 * np.pi * R, rel=1e-3)


def test_curvature_sign_flips_with_orientation():
    R = 500.0
    theta = np.linspace(0, 2 * np.pi, 2000)
    curv_ccw, _ = compute_curvature(R * np.cos(theta), R * np.sin(theta))
    curv_cw, _ = compute_curvature(R * np.cos(-theta), R * np.sin(-theta))
    assert np.sign(np.median(curv_ccw)) == -np.sign(np.median(curv_cw))


def test_find_zero_crossings_on_sine():
    # phase offset avoids a degenerate zero-length first segment at s=0
    s = np.linspace(0.5, 0.5 + 4 * 2 * np.pi, 4000)  # 4 full periods
    curve = np.sin(s)
    loc_zero, loc_max = find_zero_crossings(curve)
    # 8 true zero crossings, plus the two synthetic endpoints
    assert len(loc_zero) == 8 + 2
    # the interior bend apexes have |curve| close to 1 (the two boundary
    # segments are partial periods, so their maxima are not full apexes)
    apex_values = np.abs(curve[loc_max[1:-1]])
    assert np.all(apex_values > 0.97)
