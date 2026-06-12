# Multi-temporal analysis

Once a set of scenes of the same reach has been processed (see
[batch processing](../getting-started.md#batch-processing)), `rivabar` can split the rivers
at persistent tributary confluences and measure migration between scene pairs.

## Splitting at confluences and matching segments

```python
import rivabar as rb

# 1. Cluster tributary confluences detected across scenes
common_confluences = rb.find_common_confluences(rivers, min_scene_fraction=0.5)

# 2. Split each river's main path at the common confluences and group
#    corresponding segments across scenes
segment_groups, rejected = rb.match_river_segments(rivers, common_confluences)

# 3. Visualize the matched segments
rb.plot_river_segments(rivers, common_confluences, segment_groups=segment_groups)
```

## Curvature–migration analysis

Pairwise DTW-based analysis of migration rates, either on full main paths or per segment
group:

```python
# full reaches
results = rb.analyze_river_pairs_filtered(rivers, min_width_m=100,
                                          min_time_gap_days=365,
                                          max_time_gap_years=5)

# or per segment group, merged into one DataFrame
results_list, df = rb.analyze_all_segment_groups(segment_groups)
```

The output DataFrame contains, per scene pair, the curvature–migration rate correlation
(`r_curv_mr`), the regression slope in physical units (`slope_curv_mr`), width-regression
diagnostics (`width_slope`, a stage indicator), and a `pair_class` label
(`good` / `stage_mismatch` / `reversed` / `weak` / `insufficient`) from `classify_pairs`.

Use `rb.interactive_scatter(df)` to select pairs in a notebook and `rb.plot_pair` to
inspect a single pair with filled channel polygons and centerlines.

## Deposition and erosion maps

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15), sharex=True, sharey=True)
chs, bars, erosions, dates, centerlines = rb.create_and_plot_bars(
    rivers, 0, len(rivers) - 1, ax1=ax1, ax2=ax2)
```
