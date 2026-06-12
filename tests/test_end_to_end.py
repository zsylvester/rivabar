"""End-to-end test of the full map_river_banks pipeline on a synthetic
water mask: a sinuous ~480 m wide channel crossing a 30 km x 15 km scene,
with one mid-channel island."""
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

import rivabar as rb

NROWS, NCOLS = 500, 1000
PIXEL = 30.0
HALF_W = 8  # channel half-width in pixels


@pytest.fixture(scope='module')
def pipeline_result(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp('e2e')
    cols = np.arange(NCOLS)
    center_row = 250 + 100 * np.sin(2 * np.pi * cols / 400.0)

    mask = np.zeros((NROWS, NCOLS), dtype='float32')
    for c in range(NCOLS):
        r0 = int(round(center_row[c]))
        mask[max(0, r0 - HALF_W):min(NROWS, r0 + HALF_W + 1), c] = 1.0
    # a mid-channel island (dry strip inside the channel)
    for c in range(480, 540):
        r0 = int(round(center_row[c]))
        mask[r0 - 2:r0 + 3, c] = 0.0

    transform = Affine(PIXEL, 0, 500000, 0, -PIXEL, 8000000)
    fname = 'synthetic_channel.tif'
    with rasterio.open(os.path.join(tmpdir, fname), 'w', driver='GTiff',
                       height=NROWS, width=NCOLS, count=1, dtype='float32',
                       transform=transform, crs='EPSG:32619') as dst:
        dst.write(mask, 1)

    start_x, start_y = transform * (0.5, center_row[0] + 0.5)
    end_x, end_y = transform * (NCOLS - 0.5, center_row[-1] + 0.5)

    out = rb.map_river_banks(
        fname=fname, dirname=str(tmpdir) + '/',
        start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y,
        file_type='water_index', mndwi_threshold=0.5,
        min_g_primal_length=5000, min_main_path_length=500,
        remove_smaller_components=True, solidity_filter=False)
    (D_primal, G_rook, G_primal, mndwi, dataset, left_utm_x, right_utm_x,
     lower_utm_y, upper_utm_y, xs, ys) = out
    return {'D_primal': D_primal, 'G_rook': G_rook, 'G_primal': G_primal,
            'mndwi': mndwi, 'dataset': dataset}


def test_pipeline_succeeds(pipeline_result):
    assert pipeline_result['D_primal'] is not None
    assert pipeline_result['G_rook'] is not None
    assert pipeline_result['G_primal'] is not None


def test_island_is_detected(pipeline_result):
    G_rook = pipeline_result['G_rook']
    assert len(G_rook.nodes) >= 3  # two banks + at least the island
    island_areas = [G_rook.nodes[n]['bank_polygon'].area
                    for n in G_rook if n >= 2
                    if hasattr(G_rook.nodes[n].get('bank_polygon'), 'area')]
    assert len(island_areas) >= 1
    # the island strip is ~60 px x ~5 px = ~270,000 m2; smoothing shrinks it
    assert max(island_areas) > 5e4


def test_main_path_spans_the_channel(pipeline_result):
    D_primal = pipeline_result['D_primal']
    assert 'main_path' in D_primal.graph
    cl = np.asarray(D_primal.graph['main_channel_cl_coords'])
    x_span = cl[:, 0].max() - cl[:, 0].min()
    assert x_span > 0.95 * (NCOLS * PIXEL - 2 * PIXEL)


def test_channel_widths_are_right(pipeline_result):
    D_primal = pipeline_result['D_primal']
    xl, yl, w1l, w2l, w, s = rb.get_channel_widths_along_path(
        D_primal, D_primal.graph['main_path'])
    widths_m = np.array(w) * PIXEL
    true_width = (2 * HALF_W + 1) * PIXEL  # ~510 m
    median_width = np.nanmedian(widths_m)
    assert true_width * 0.6 < median_width < true_width * 1.4


def test_banklines_are_smoothed(pipeline_result):
    G_rook = pipeline_result['G_rook']
    for node in (0, 1):
        xy = np.array(G_rook.nodes[node]['bank_polygon'].exterior.coords)
        seg = np.diff(xy, axis=0)
        axis_aligned = np.mean((np.abs(seg[:, 0]) < 1e-9) |
                               (np.abs(seg[:, 1]) < 1e-9))
        # raw pixel contours are > 40% axis-aligned; smoothed ones are not
        assert axis_aligned < 0.2
