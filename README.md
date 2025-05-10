# rivabar

<img src="https://raw.githubusercontent.com/zsylvester/rivabar/main/images/rivabar_logo.png" width="400">

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

Here are some basic examples of how to use `rivabar`:

### Basic Usage

```python
import rivabar as rb

# Extract channel centerlines and banklines
# Define start and end points of the channel you want to extract
start_x, start_y = 675796.2, 98338.8 # UTM coordinates of the channel start
end_x, end_y = 628190.3, -91886.6 # UTM coordinates of the channel end
fname="LC08_L2SP_232060_20140219_20200911_02_T1_SR", # assumes that the Landsat bands are located in a folder with this name
dirname="../data/Branco/", # parent folder of the 'LC08...' folder


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
    remove_smaller_components=True,\
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

```python
# Get the main path through the channel network
edge_path = rb.get_main_path(D_primal)

# Analyze channel width - wavelength scaling
df, curv, s, loc_zero_curv, xsmooth, ysmooth = rb.analyze_width_and_wavelength(
    D_primal=D_primal,
    main_path=edge_path,
    ax,
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
plt.ylabel('channel width (m)');
```

For more examples and detailed usage, check out the example notebooks in the [notebooks](https://github.com/zsylvester/rivabar/tree/main/notebooks) directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.