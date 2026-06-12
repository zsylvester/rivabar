"""Raster cropping and image-stack preparation on a synthetic GeoTIFF."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from rivabar.data_io import crop_image_to_aoi, prepare_image_stack


@pytest.fixture
def synthetic_tif(tmp_path):
    """A 200x300 3-band GeoTIFF with a 30 m transform."""
    path = tmp_path / 'scene.tif'
    rng = np.random.default_rng(0)
    data = rng.integers(1, 255, (3, 200, 300)).astype(np.uint8)
    transform = Affine(30.0, 0, 500000, 0, -30.0, 8000000)
    with rasterio.open(path, 'w', driver='GTiff', height=200, width=300,
                       count=3, dtype='uint8', transform=transform,
                       crs='EPSG:32619') as dst:
        dst.write(data)
    return path


def test_crop_image_to_aoi(synthetic_tif):
    # a 60x40 pixel AOI inside the raster
    left, top = 500000 + 30 * 100, 8000000 - 30 * 50
    aoi = [left, left + 30 * 60, top - 30 * 40, top]
    cropped = crop_image_to_aoi(synthetic_tif, aoi)
    assert cropped.shape == (40, 60, 3)


def test_prepare_image_stack_keeps_alignment(synthetic_tif, tmp_path):
    bad = tmp_path / 'not_a_raster.tif'
    bad.write_text('this is not a tif')
    aoi = [500000, 500000 + 30 * 60, 8000000 - 30 * 40, 8000000]
    stack = prepare_image_stack([synthetic_tif, bad, synthetic_tif], aoi)
    assert len(stack) == 3
    assert stack[0] is not None and stack[2] is not None
    assert stack[1] is None  # failure keeps its slot, alignment preserved
    assert stack[0].min() >= 0 and stack[0].max() <= 1
