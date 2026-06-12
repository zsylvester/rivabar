# Getting started

`rivabar` provides an object-oriented API through the `River` class and a functional API
whose main entry point is `map_river_banks` (the original `extract_centerline` is kept as a
backward-compatible alias). The package is imported as `rb` by convention:

```python
import rivabar as rb
```

## Object-oriented API (recommended)

```python
# Create a River object
river = rb.River(
    fname="LC08_L2SP_232060_20140219_20200911_02_T1_SR",
    dirname="../data/Branco/",
    file_type="multiple_tifs"
)

# Interactively select start and end points
river.get_start_end_points_interactive()

# Process the river to extract centerlines and banklines
river.map_river_banks(
    mndwi_threshold=0.0,
    ch_belt_smooth_factor=1e8,
    ch_belt_half_width=2000,
    remove_smaller_components=True,
    small_hole_threshold=64
)

# Access results through properties
centerlines = river.directed_graph
banklines = river.bankline_graph
mndwi_image = river.mndwi

# Analyze channel morphology
s, widths = river.get_channel_widths()   # along-channel distance + widths (m)
stats = river.collect_stats()            # summary statistics dictionary

# Visualize and save
river.plot_overview()
river.save_results("my_river_analysis.pkl")
```

Saved results restore with their georeferencing intact:

```python
river = rb.River.load_results("my_river_analysis.pkl")
river.summary()
gdfs = river.to_geopandas()   # GeoDataFrames for GIS export
```

## Functional API

```python
# UTM coordinates of the channel start and end
start_x, start_y = 675796.2, 98338.8
end_x, end_y = 628190.3, -91886.6

D_primal, G_rook, G_primal, mndwi, dataset, left_utm_x, right_utm_x, \
    lower_utm_y, upper_utm_y, xs, ys = rb.map_river_banks(
        fname="LC08_L2SP_232060_20140219_20200911_02_T1_SR",
        dirname="../data/Branco/",
        start_x=start_x, start_y=start_y,
        end_x=end_x, end_y=end_y,
        file_type='multiple_tifs',
        mndwi_threshold=0.0,
        ch_belt_smooth_factor=1e8,
        ch_belt_half_width=2000,
        remove_smaller_components=True,
        small_hole_threshold=64,
        plot_D_primal=True
    )

# Save the extracted centerlines and banklines as shapefiles
rb.save_shapefiles(dirname="output_directory", fname="output_prefix",
                   G_rook=G_rook, dataset=dataset)
```

## Key data structures

- **`D_primal`** — directed multigraph of centerlines. Edges have `'geometry'`
  (LineString in UTM), `'half_widths'` (per-vertex half-widths for the two banks),
  `'mm_len'`, and `'width'`. The main path is stored as
  `D_primal.graph['main_path']` (a list of `(start, end, key)` edge tuples).
- **`G_rook`** — "rook" adjacency graph of land polygons. Nodes 0 and 1 are the two
  main banks; nodes ≥ 2 are islands/bars, each with a `'bank_polygon'` attribute.
- **`G_primal`** — undirected centerline graph.

Coordinates are in UTM (meters) throughout.

## Batch processing

Multiple Landsat scenes can be downloaded (via Google Earth Engine) and processed in
one call:

```python
rivers = rb.River.batch_process_landsat_scenes(
    path_number=232, row_number=60,
    start_x=675796.2, start_y=98338.8,
    end_x=628190.3, end_y=-91886.6,
    years=range(2020, 2024), max_cloud_cover=10, n_scenes_per_year=3,
    download_false_color=True
)

# later: reload everything that was saved
rivers = rb.River.load_batch_results('river_results')

# drop scenes where only part of the reach was extracted
rivers, *_ = rb.filter_rivers_by_length(rivers, std_threshold=0.3)
```

Earth Engine access requires the optional `earthengine-api` and `geemap` packages and an
authenticated session.

For complete worked examples, see the
[notebooks](https://github.com/zsylvester/rivabar/tree/main/notebooks) directory.
