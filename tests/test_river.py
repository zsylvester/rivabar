"""River object: persistence round trips, width conversion, summary statistics."""
import os

import numpy as np
import pytest

from rivabar.river import River, collect_river_stats
from tests.conftest import make_synthetic_river


def test_save_load_round_trip_preserves_dataset_metadata(tmp_path):
    river = make_synthetic_river(with_dataset=True)
    path = os.path.join(tmp_path, 'river.pkl')
    river.save_results(path)
    loaded = River.load_results(path)

    s, w = loaded.get_channel_widths()
    assert w[0] == pytest.approx(4.0 * 30.0)  # half-widths 2+2 px at 30 m
    assert loaded._dataset.crs is not None
    gdfs = loaded.to_geopandas()
    assert gdfs['centerline'].crs is not None
    loaded.summary()  # must not raise on a loaded river


def test_load_without_dataset_needs_pixel_size(tmp_path):
    river = make_synthetic_river(with_dataset=False)
    path = os.path.join(tmp_path, 'river.pkl')
    river.save_results(path)
    loaded = River.load_results(path)

    with pytest.raises(ValueError):
        loaded.get_channel_widths()
    s, w = loaded.get_channel_widths(pixel_size=30.0)
    assert w[0] == pytest.approx(120.0)
    loaded.summary()  # degrades gracefully, no raise


def test_collect_river_stats_handles_degenerate_islands():
    from shapely.geometry import Polygon
    river = make_synthetic_river(with_dataset=True)
    G_rook = river._G_rook
    G_rook.add_node(0, bank_polygon=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
    G_rook.add_node(1, bank_polygon=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
    G_rook.add_node(2, bank_polygon=Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]))
    G_rook.add_node(3, bank_polygon=[])   # empty placeholder
    G_rook.add_node(4)                    # missing attribute
    G_rook.add_edge(0, 2)
    G_rook.add_edge(1, 2)

    stats = collect_river_stats(river)
    assert stats['n_islands'] == 1
    assert stats['width_mean'] == pytest.approx(120.0)
    assert stats['sinuosity'] == pytest.approx(1.0, abs=1e-6)
    assert stats['degree_node0'] == 1

    # method form matches function form
    assert river.collect_stats() == stats


def test_collect_river_stats_missing_bank_nodes():
    river = make_synthetic_river(with_dataset=True)  # empty G_rook
    stats = collect_river_stats(river)
    assert np.isnan(stats['degree_node0'])
    assert np.isnan(stats['degree_node1'])
    assert stats['n_islands'] == 0
