"""Multi-scene helpers: river filtering and river-to-image matching."""
import numpy as np
import pytest

from rivabar.temporal_analysis import filter_rivers_by_length, match_rivers_to_images
from tests.conftest import make_synthetic_river


def test_filter_rivers_by_length():
    rivers = [make_synthetic_river('full_1', n=1000),
              make_synthetic_river('full_2', n=980),
              make_synthetic_river('full_3', n=1010),
              make_synthetic_river('truncated', n=200),
              make_synthetic_river('failed', n=500, processed=False)]
    filtered, flens, vidx, lens, widths, thr = filter_rivers_by_length(
        rivers, std_threshold=0.3)
    assert [r.fname for r in filtered] == ['full_1', 'full_2', 'full_3']
    assert vidx == [0, 1, 2]
    assert lens[4] == 0          # failed river counts as zero length
    assert widths[0] == pytest.approx(120.0)  # pixel_size=30 fallback
    assert len(lens) == len(widths) == len(rivers)


def test_match_rivers_to_images(tmp_path):
    # image files named like the Earth Engine false-color downloads
    names = ['false_color_LC08_001069_20200515.tif',
             'false_color_LC08_001069_20200616.tif',
             'false_color_LT05_001069_19990819.tif',
             'no_date_here.tif']
    for name in names:
        (tmp_path / name).touch()

    rivers = [make_synthetic_river('a', acquisition_date='2020-05-15'),
              make_synthetic_river('b', acquisition_date='2020-06-17'),  # 1 day off
              make_synthetic_river('c', acquisition_date='1995-01-01')]  # no image
    matched, files, dates = match_rivers_to_images(rivers, tmp_path,
                                                   tolerance_days=1)
    assert [r.fname for r in matched] == ['c'] or [r.fname for r in matched] == ['a', 'b']
    # ordering follows image dates; river c has no image within tolerance
    assert [r.fname for r in matched] == ['a', 'b']
    assert [d.strftime('%Y%m%d') for d in dates] == ['20200515', '20200616']
    assert len(files) == len(matched) == len(dates)


def test_match_rivers_to_images_unique_pairing(tmp_path):
    # two images close to the same river: the river is used once (closest wins)
    (tmp_path / 'LC08_001069_20200515.tif').touch()
    (tmp_path / 'LC08_001069_20200516.tif').touch()
    rivers = [make_synthetic_river('a', acquisition_date='2020-05-15')]
    matched, files, dates = match_rivers_to_images(rivers, tmp_path,
                                                   tolerance_days=1)
    assert len(matched) == 1
    assert files[0].name == 'LC08_001069_20200515.tif'


def test_normalize_image_edge_cases():
    from rivabar.data_io import normalize_image
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1000, (50, 60, 3)).astype(np.float32)
    img[:, :, 1] = 0.0  # all-zero band must not crash percentile stretch
    img[0, 0, 0] = np.nan
    out = normalize_image(img)
    assert out.shape == img.shape
    assert np.all((out >= 0) & (out <= 1))
    assert np.all(np.isfinite(out))
