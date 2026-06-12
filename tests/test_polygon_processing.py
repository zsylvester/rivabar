"""Smoothing functions: window clamps and main-bankline smoothing."""
import numpy as np
from shapely.geometry import Polygon, LineString

from rivabar.polygon_processing import (smooth_polygon, smooth_line, extend_line,
                                        smooth_main_bankline)


def test_smooth_polygon_handles_tiny_polygons():
    # 4-vertex square: too small for any meaningful savgol window — must not
    # raise (regression test for window_length > len(x) crash)
    tiny = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    out = smooth_polygon(tiny)
    assert out.is_valid
    # a triangle is below the minimum window and is returned unchanged
    tri = Polygon([(0, 0), (10, 0), (5, 8)])
    assert smooth_polygon(tri).equals(tri)


def test_smooth_polygon_smooths_large_polygons():
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    rng = np.random.default_rng(0)
    r = 1000 + rng.normal(0, 20, 200)
    noisy = Polygon(zip(r * np.cos(theta), r * np.sin(theta)))
    smooth = smooth_polygon(noisy)
    assert smooth.is_valid
    assert len(smooth.exterior.coords) < len(noisy.exterior.coords)


def test_smooth_line_handles_short_lines():
    x, y = smooth_line([0, 1, 2, 3], [0, 1, 0, 1])
    assert len(x) == 4  # returned unchanged, no crash


def test_extend_line_short_and_long():
    # short line (would have raised IndexError with the old x[10] anchors)
    xs = np.linspace(0, 100, 8)
    ext = extend_line(xs, np.zeros(8), 5)
    assert ext.length > 100
    # long line
    xs = np.linspace(0, 10000, 200)
    ext = extend_line(xs, np.zeros(200), 10)
    assert ext.length > 10000


def test_smooth_main_bankline_replaces_staircase():
    """A staircase bank polygon (pixel-contour style) gets a smoothed
    replacement that splits the image boundary correctly."""
    # Image boundary: 100 km x 50 km rectangle
    im_boundary = Polygon([(0, 0), (100000, 0), (100000, 50000), (0, 50000)])
    # Bank = area below a staircase line wiggling around y = 25000
    xs = np.arange(0, 100001, 30)
    ys = 25000 + 2000 * np.sin(2 * np.pi * xs / 20000)
    ys = np.round(ys / 30) * 30  # quantize to 30 m "pixels"
    coords = list(zip(xs, ys)) + [(100000, 0), (0, 0)]
    bank = Polygon(coords)
    assert bank.is_valid

    smooth = smooth_main_bankline(bank, im_boundary)
    assert smooth is not None
    # the smoothed bank covers (roughly) the same area
    assert abs(smooth.area - bank.area) / bank.area < 0.05
    # and the boundary is no longer a 30 m staircase
    seg = np.diff(np.array(smooth.exterior.coords), axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    assert np.median(seglen) > 60


def test_smooth_main_bankline_returns_none_on_failure():
    # A bank polygon far inside the boundary produces a short bankline whose
    # extension cannot split the image boundary into two pieces
    im_boundary = Polygon([(0, 0), (100000, 0), (100000, 50000), (0, 50000)])
    blob = Polygon([(45000, 20000), (55000, 20000), (55000, 30000),
                    (45000, 30000)]).buffer(0)
    result = smooth_main_bankline(blob, im_boundary)
    assert result is None or result.is_valid  # must not raise
