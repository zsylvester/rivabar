# rivabar

<img src="https://raw.githubusercontent.com/zsylvester/rivabar/main/images/rivabar_logo.png" width="300">

## Description

`rivabar` is a Python package that aims to automatically extract channel centerlines and banklines from water index images of rivers. 
The focus is on getting good representations of the banklines, as centerlines are not physical features and how they are exactly derived 
is subjective. This is achieved in part by viewing channels as boundaries between polygons that correspond to islands or bars, as this 
allows us to take advantage of algorithms developed for spatial analysis. In this view, even a single-thread channel becomes the boundary 
between two land domains.

`rivabar` relies on the following Python packages, among others:
* [networkx](https://networkx.org/) to work with graphs
* [sknw](https://github.com/Image-Py/sknw) for converting the channel skeleton into a graph
* [Python Spatial Analysis Library (libpysal)](https://pysal.org/libpysal/) to get neighborhood relations between banks / islands
* [Urban Morphology Measuring Toolkit (momepy)](http://docs.momepy.org/en/stable/) to create clean centerline graphs

`rivabar` can be used to map single-thread and multithread rivers, in an almost entirely automated fashion. It requires a water mask as input 
and a start (source) and end (sink) points for the channel / channel belt / delta.

The package now features a **object-oriented API** through the `River` class, which provides:
- **Intuitive workflow**: Create a river object, process it, and access results through properties
- **Interactive tools**: Built-in methods for selecting start/end points and visualizing results  
- **State management**: Automatic handling of processing state and data persistence
- **Batch processing**: Class methods for processing multiple scenes efficiently
- **Backward compatibility**: All original functions remain available and unchanged

The images below illustrate how `rivabar` extracts both centerlines and banklines from a Landsat image of the Brahmaputra River, and creates 
an island neighborhood graph in addition to the centerline graph.

<p align="left">
  <img src="https://raw.githubusercontent.com/zsylvester/rivabar/main/images/brahmaputra_1.png" width="400">
  <br>
  <em>Brahmaputra River - water index</em>
</p>

<p align="left">
  <img src="https://raw.githubusercontent.com/zsylvester/rivabar/main/images/brahmaputra_2.png" width="400">
  <br>
  <em>Rook graph and centerline polygons</em>
</p>

<p align="left">
  <img src="https://raw.githubusercontent.com/zsylvester/rivabar/main/images/brahmaputra_3.png" width="400">
  <br>
  <em>Centerline graph and bar polygons</em>
</p>

## Installation

You can install `rivabar` directly from PyPI:

```bash
pip install rivabar
```

Alternatively, you can install from source:

```bash
git clone https://github.com/zsylvester/rivabar.git
cd rivabar
pip install -e .
```

## Getting started

`rivabar` now provides both an **object-oriented API** through the `River` class and the **original functional API** for backward compatibility.

### Option 1: Object-Oriented API (Recommended)

The new `River` class provides a clean, intuitive interface:

```python
import rivabar as rb

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
banklines = river.polygon_graph
mndwi_image = river.mndwi

# Analyze channel morphology
widths = river.get_channel_widths()
wavelength_analysis = river.analyze_wavelength_and_width()

# Visualize results
river.plot_overview()

# Save results
river.save_results("my_river_analysis.pkl")
```

### Option 2: Original Functional API

The original functional interface remains fully supported:

#### Interactively Selecting Start/End Points

```python
import rivabar as rb
import matplotlib.pyplot as plt

# Define input parameters
dirname = "../data/Branco/"  # Adjust path to your data
fname = "LC08_L2SP_232060_20140219_20200911_02_T1_SR" # Adjust filename/folder
file_type = "multiple_tifs" # or 'water_index' if the water mask already exists

# 1. Create the MNDWI water mask image
mndwi, dataset = rb.create_mndwi(
    dirname=dirname,
    fname=fname,
    file_type=file_type,
    mndwi_threshold=0.0, # Adjust threshold as needed
    small_hole_threshold=16,
    remove_smaller_components=True
)

# 2. Display the image
fig, ax = plt.subplots(figsize=(10, 10))
rb.plot_im_and_lines(mndwi, dataset.bounds.left, dataset.bounds.right, 
                     dataset.bounds.bottom, dataset.bounds.top, ax=ax, plot_image=True, plot_lines=False)
plt.title("Click START point, then END point")
plt.show() # Make sure the plot window appears

# 3. Get start and end points using ginput
# Click on the plot: first for the start point, then for the end point.
points = plt.ginput(n=2, timeout=-1) # timeout=-1 waits indefinitely

# Close the plot window automatically if desired
# plt.close(fig)

# Extract coordinates
start_x, start_y = points[0]
end_x, end_y = points[1]

# 4. Now you can use these coordinates in extract_centerline
# D_primal, G_rook, G_primal, ... = rb.extract_centerline(
#     fname=fname,
#     dirname=dirname,
#     start_x=start_x,
#     start_y=start_y,
#     end_x=end_x,
#     end_y=end_y,
#     file_type=file_type,
#     ...
# )
```

#### Centerline Extraction (Functional API)

```python
import rivabar as rb

# Extract channel centerlines and banklines
# Define start and end points of the channel you want to extract (see previous section)
start_x, start_y = 675796.2, 98338.8 # UTM coordinates of the channel start
end_x, end_y = 628190.3, -91886.6 # UTM coordinates of the channel end
fname="LC08_L2SP_232060_20140219_20200911_02_T1_SR" # assumes that the Landsat bands are located in a folder with this name
dirname="../data/Branco/" # parent folder of the 'LC08...' folder

# Extract the channel centerline and related graphs
D_primal, G_rook, G_primal, mndwi, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, xs, ys = rb.extract_centerline(
    fname=fname,
    dirname=dirname,
    start_x=start_x,
    start_y=start_y,
    end_x=end_x,
    end_y=end_y,
    file_type='multiple_tifs',
    flip_outlier_edges=True,
    mndwi_threshold=0.0,
    ch_belt_smooth_factor=1e8,
    ch_belt_half_width=2000,
    remove_smaller_components=True,
    delete_pixels_polys=False,
    small_hole_threshold=64,
    solidity_filter=False,
    plot_D_primal=True
)

# Save the extracted centerlines and banklines as shapefiles
rb.save_shapefiles(
    dirname="output_directory",
    fname="output_prefix",
    G_rook=G_rook,
    dataset=dataset
)
```

### Analyzing Channel Widths and Morphology

#### Using the Object-Oriented API

```python
# Create and process river (assuming you have a processed River object)
river = rb.River(
    fname="LC08_L2SP_232060_20140219_20200911_02_T1_SR",
    dirname="../data/Branco/",
    start_x=675796.2, start_y=98338.8,
    end_x=628190.3, end_y=-91886.6,
    file_type='multiple_tifs'
)

# Process the river
river.map_river_banks()

# Get channel widths along the main path
widths = river.get_channel_widths()

# Analyze width-wavelength relationships
wavelength_analysis = river.analyze_wavelength_and_width(
    delta_s=5,
    smoothing_factor=0.5*1e7,
    min_sinuosity=1.1,
    dx=30
)

# Plot results
river.plot_overview()

# Access individual components
centerlines = river.directed_graph
main_path = river.main_path
```

#### Using the Functional API

```python
import matplotlib.pyplot as plt
import numpy as np

# Get the main path through the channel network (assuming D_primal from previous example)
edge_path = rb.get_main_path(D_primal)

# Analyze channel width - wavelength scaling
df, curv, s, loc_zero_curv, xsmooth, ysmooth = rb.analyze_width_and_wavelength(
    D_primal=D_primal,
    main_path=edge_path,
    ax=None,
    delta_s=5,
    smoothing_factor=0.5*1e7,
    min_sinuosity=1.1,
    dx=30
)

# Extract and plot channel widths along main path
xl, yl, w1l, w2l, w, s = rb.get_channel_widths_along_path(D_primal, D_primal.graph['main_path'])
plt.figure(figsize=(12, 4))
plt.plot(s, np.array(w)*30.0)
plt.xlabel('along-channel distance (m)')
plt.ylabel('channel width (m)')
plt.show()
```

### Additional River Class Features

The `River` class provides many additional methods for advanced analysis:

```python
# Batch processing multiple Landsat scenes
rivers = rb.River.batch_process_landsat_scenes(
    path_number=232, row_number=60,
    start_x=675796.2, start_y=98338.8,
    end_x=628190.3, end_y=-91886.6,
    start_date='2020-01-01', end_date='2023-12-31'
)

# Load and analyze saved results
river = rb.River.load_results("my_river_analysis.pkl")

# Get summary information
river.summary()

# Export to GeoDataFrames for further GIS analysis
gdfs = river.to_geopandas()

# Memory management for large datasets
river.clear_raster_data()  # Remove large raster data while keeping graphs
memory_usage = river.get_memory_usage()
```

For more examples and detailed usage, check out the example notebooks in the [notebooks](https://github.com/zsylvester/rivabar/tree/main/notebooks) directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.