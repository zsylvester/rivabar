# rivabar

<img src="https://github.com/zsylvester/rivabar/blob/main/images/rivabar_logo.png" width="400">

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
  <img src="https://github.com/zsylvester/rivabar/blob/main/images/brahmaputra_1.png" width="400">
  <br>
  <em>Brahmaputra River - water index</em>
</p>

<p align="left">
  <img src="https://github.com/zsylvester/rivabar/blob/main/images/brahmaputra_2.png" width="400">
  <br>
  <em>Rook graph and centerline polygons</em>
</p>

<p align="left">
  <img src="https://github.com/zsylvester/rivabar/blob/main/images/brahmaputra_3.png" width="400">
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.