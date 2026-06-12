"""Segment splitting and cross-scene segment matching: alignment invariants."""
import numpy as np

from rivabar.temporal_analysis import match_river_segments
from tests.conftest import make_synthetic_river


def edge_x_extent(river, segment):
    xs = []
    for (s, e, d) in segment:
        xs.extend(river._D_primal[s][e][d]['geometry'].xy[0])
    return min(xs), max(xs)


def test_split_alignment_invariants():
    """segments must always have len(split_points)+1 positionally aligned
    entries, even with out-of-order, coincident, and endpoint-snapping split
    points (regression test for the segment-misalignment bug)."""
    river = make_synthetic_river(n=1500)
    pts = [(90000.0, 0.0),   # input 0 -> along-channel LAST
           (10.0, 0.0),      # input 1 -> snaps to the path start
           (30000.0, 0.0),   # input 2
           (30020.0, 0.0)]   # input 3 -> same vertex as input 2
    segments, split_info = river.split_main_path_at_points(pts)

    assert len(segments) == len(pts) + 1
    assert [info['input_index'] for info in split_info] == [1, 2, 3, 0]
    # along-channel distances are sorted
    dists = [info['along_channel_distance'] for info in split_info]
    assert dists == sorted(dists)
    # segment 0 (before the start-snapping point) and the coincident-split
    # segment are empty; the rest are contiguous
    assert segments[0] == []
    assert segments[2] == []
    assert edge_x_extent(river, segments[1]) == (0.0, 30000.0)
    assert edge_x_extent(river, segments[3]) == (30000.0, 90000.0)
    assert edge_x_extent(river, segments[4])[0] == 90000.0


def test_split_within_single_edge():
    river = make_synthetic_river(n=1000)
    segments, split_info = river.split_main_path_at_points([(50000.0, 0.0)])
    assert len(segments) == 2
    assert edge_x_extent(river, segments[0]) == (0.0, 50000.0)
    assert edge_x_extent(river, segments[1])[0] == 50000.0
    # synthetic edges carry sliced half_widths matching their geometry
    s, e, d = segments[0][0]
    edge = river._D_primal[s][e][d]
    n_coords = len(edge['geometry'].coords)
    assert len(edge['half_widths']['L']) == n_coords


def test_match_river_segments_groups_by_input_index():
    """Group keys must reference common_confluences input indices even when
    the input order differs from the along-channel order, and unreachable
    confluences must be merged across."""
    rivers = [make_synthetic_river(name=f'scene{i}', n=1500, jitter_seed=i)
              for i in range(4)]
    confluences = [
        {'utm_coords': (100000.0, 0.0)},    # input 0, along-channel 2nd
        {'utm_coords': (40000.0, 0.0)},     # input 1, along-channel 1st
        {'utm_coords': (70000.0, 99999.0)}, # input 2, unreachable
    ]
    groups, rejected = match_river_segments(rivers, confluences,
                                            max_snapping_distance=500.0,
                                            min_rivers_per_segment=4)
    keys = {g['segment_index'] for g in groups}
    assert keys == {(None, 1), (1, 0), (0, None)}
    assert all(g['n_rivers'] == 4 for g in groups)
    by_key = {g['segment_index']: g for g in groups}
    assert by_key[(None, 1)]['downstream_confluence'] == (40000.0, 0.0)
    assert by_key[(1, 0)]['upstream_confluence'] == (40000.0, 0.0)
    assert by_key[(1, 0)]['downstream_confluence'] == (100000.0, 0.0)
    # the unreachable confluence bounds no group
    assert not any(2 in k for k in keys)


def test_repeated_split_uses_fresh_synthetic_keys():
    river = make_synthetic_river(n=500)
    seg1, _ = river.split_main_path_at_points([(25000.0, 0.0)])
    seg2, _ = river.split_main_path_at_points([(25000.0, 0.0)])
    keys1 = {d for seg in seg1 for (_, _, d) in seg}
    keys2 = {d for seg in seg2 for (_, _, d) in seg}
    assert keys1.isdisjoint(keys2)
