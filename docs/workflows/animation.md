# River evolution animations

The animation workflow turns a batch-processed scene series plus the false-color images
downloaded alongside it (`download_false_color=True`) into a movie of channel migration,
deposition, and erosion.

```python
import rivabar as rb

rivers = rb.River.load_batch_results('river_results')

# 1. Pair each false-color image with the river processed from the same scene
matched_rivers, image_files, dates = rb.match_rivers_to_images(
    rivers, 'false_color_images/')

# 2. Choose an area of interest (interactively, or reuse saved bounds)
# aoi_bounds = rb.select_aoi_interactive(image, extent)
aoi_bounds = [668234.3, 684507.4, -1509737.7, -1479760.9]

# 3. Crop and normalize the background images
image_stack = rb.prepare_image_stack(image_files, aoi_bounds)

# 4. Render the frames: three panels per scene
#    (false color | cumulative deposition | cumulative erosion)
frames = rb.make_river_evolution_frames(matched_rivers, image_stack, dates,
                                        aoi_bounds, output_dir='frames')

# 5. Assemble the movie (requires ffmpeg on the PATH)
rb.assemble_movie(frames, 'river_evolution.mp4', fps=6)
```

`make_river_evolution_frames` accepts a `select_dates` list to render a subset of scenes
(e.g., one per year), styling parameters (colormaps, alpha, dpi), and a `baseline_index`
for the scene that anchors the cumulative deposition/erosion maps. Scale bars are added
when the optional `matplotlib-scalebar` package is installed.
