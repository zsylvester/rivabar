# Changelog

## 0.2.0 (2026-06-12)

### New features

- **Migration prediction module** (`rivabar.prediction`): Howard & Knutson (1984)
  convolution model following Sylvester et al. (2019, *Geology*) — calibration of the
  friction factor Cf and migration rate constant kl (`calibrate_pair`,
  `calibrate_segment`), spatially-varying erodibility (`calibrate_local_kl`,
  `calibrate_segment_local`), forward prediction with uncertainty envelopes
  (`predict_forward`, `predict_forward_local`), neck cutoff risk detection, parameter
  stability tracking, and temporal cross-validation.
- **Tributary detection and segment-based analysis**: `find_tributary_branches`,
  `find_common_confluences`, `match_river_segments`, `River.split_main_path_at_points`,
  and per-segment pairwise DTW analysis (`analyze_segment_group`,
  `analyze_all_segment_groups`) with pair classification (`classify_pairs`).
- **Stable-segment DTW analysis**: high-variance (cutoff-affected) reaches are detected
  and excluded from curvature–migration correlation (`detect_high_variance_segments`,
  `run_dtw_by_stable_segments`).
- **River evolution animation workflow**: `match_rivers_to_images`,
  `prepare_image_stack`, `select_aoi_interactive`, `make_river_evolution_frames`, and
  `assemble_movie` turn batch-processed scenes plus false-color images into a movie.
- **Summary statistics and filtering**: `collect_river_stats` / `River.collect_stats`
  and `filter_rivers_by_length` for cleaning batch-processed scene series.
- Water mask generation from mapping results (`create_water_mask_from_mapping`),
  Earth Engine batch processing with cloud filtering
  (`River.batch_process_landsat_scenes`).

### Bug fixes

- **Corrected the upstream-boundary normalization of the Howard & Knutson convolution**:
  predicted migration rates near the upstream end of a reach were attenuated and
  sign-inverted, biasing Cf calibration. The kernel is now normalized by the mass
  actually available upstream of each point.
- **Stable-segment masks are no longer applied before the convolution** during
  calibration: excised high-variance gaps used to distort the upstream-influence kernel;
  masks now restrict only the calibration statistics.
- **Fixed a 30x inflated width filter** in `analyze_river_pairs_filtered` that
  effectively disabled the minimum-width criterion.
- **Fixed segment/confluence misalignment** in `match_river_segments` when the
  along-channel order of confluences differed from their input order, and when split
  points snapped to coincident or endpoint vertices.
- **Made bankline smoothing robust and repairable**: the main-bank smoothing failure
  path no longer silently keeps raw pixel-staircase polygons without a clear warning;
  Savitzky-Golay windows are clamped to the available points (small islands no longer
  crash); `smooth_main_bankline` is exported so saved rivers can be re-smoothed without
  reprocessing.
- **Pickled rivers are fully usable again**: `save_results`/`load_results` round-trip
  the dataset CRS/transform; `get_channel_widths` accepts a `pixel_size` fallback;
  `summary()` and `to_geopandas()` work on loaded rivers; `MinimalDataset` gained an
  `xy()` method.
- Fixed start/end point insertion when both snap to the same skeleton edge (the
  simplest single-thread case failed), several crash bugs in the core pipeline
  (`_create_primal_graph` failure paths, `insert_node` argument mix-up on the
  bridged-mask retry, a `set_crs` no-op that produced CRS-less shapefiles), an
  off-by-one in `create_and_plot_bars` validation, and a numpy >= 2.3 incompatibility
  in `find_zero_crossings`.
- **Packaging**: `pip install rivabar` produced an unimportable package — `librosa` was
  missing from the requirements and the optional Earth Engine / Jupyter-widget
  dependencies were imported at module level; they are now imported lazily.

### Changed

- `rivabar_legacy.py` was removed. `extract_centerline` remains available as a
  backward-compatible alias for `map_river_banks` (same parameters, same return
  values); note its `remove_smaller_components` default is now True.
- Extensive dead-code removal, deduplication, and vectorization across the package
  (~6,000 lines removed net).
- `matplotlib-scalebar` is an optional dependency (scale bars are skipped with a
  warning when it is not installed).

### Infrastructure

- pytest test suite (40 synthetic tests, including an end-to-end run of the full
  extraction pipeline on a synthetic water mask) with GitHub Actions CI on Python
  3.10 and 3.12.
- Documentation site at https://zsylvester.github.io/rivabar/ (MkDocs Material +
  mkdocstrings), deployed automatically on every push.

## 0.1.2 and earlier

Initial releases: centerline/bankline extraction pipeline, `River` class
(object-oriented API), channel width and wavelength analysis, multi-temporal
deposition/erosion mapping.
