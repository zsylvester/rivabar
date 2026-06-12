"""Shared fixtures: synthetic rivers and datasets (no external data needed)."""
import matplotlib
matplotlib.use('Agg')  # noqa: E402 — must run before any pyplot import

import numpy as np
import networkx as nx
import pytest
from shapely.geometry import LineString
from rasterio.transform import Affine

from rivabar.river import River


class FakeDataset:
    """Stands in for a rasterio dataset (crs, transform, shape)."""
    def __init__(self, pixel_size=30.0, crs='EPSG:32619', shape=(1000, 1000)):
        self.crs = crs
        self.transform = Affine(pixel_size, 0, 500000, 0, -pixel_size, 8000000)
        self.shape = shape


def make_synthetic_river(name='synthetic', n=500, spacing=100.0, amplitude=0.0,
                         wavelength=10000.0, half_width_px=2.0, with_dataset=False,
                         processed=True, acquisition_date=None, jitter_seed=None):
    """
    Build a minimal processed River: a single-edge main path along x with an
    optional sinusoidal y component, constant half-widths, and (optionally) a
    fake dataset with a 30 m transform.
    """
    xs = np.arange(n) * spacing
    if amplitude > 0:
        ys = amplitude * np.sin(2 * np.pi * xs / wavelength)
    else:
        ys = np.zeros_like(xs)
    if jitter_seed is not None:
        ys = ys + np.random.default_rng(jitter_seed).normal(0, 5, n)

    D = nx.MultiDiGraph()
    D.add_edge(0, 1, key=0,
               geometry=LineString(np.column_stack([xs, ys])),
               half_widths={'L': list(np.full(n, half_width_px)),
                            'R': list(np.full(n, half_width_px))},
               mm_len=float(xs[-1]), width=2 * half_width_px)
    D.graph['main_path'] = [(0, 1, 0)]
    D.graph['main_channel_cl_coords'] = np.column_stack([xs, ys])

    river = River(fname=name, dirname='.', start_x=0.0, start_y=0.0,
                  end_x=float(xs[-1]), end_y=0.0)
    river._D_primal = D
    river._G_rook = nx.Graph()
    river._G_primal = nx.Graph()
    river._is_processed = processed
    river._processing_successful = processed
    if with_dataset:
        river._dataset = FakeDataset()
    if acquisition_date is not None:
        river.acquisition_date = acquisition_date
    return river


@pytest.fixture
def synthetic_river():
    return make_synthetic_river(with_dataset=True)


@pytest.fixture
def fake_dataset():
    return FakeDataset()
