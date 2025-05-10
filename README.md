# rivabar

<img src="https://github.com/zsylvester/rivabar/blob/main/rivabar_logo.png" width="400">

## Description

`rivabar` is a Python package that aims to automatically extract channel centerlines and banklines from water index images of rivers. 
The focus is on getting good representations of the banklines, as centerlines are not physical features and how they are exactly derived 
is subjective.

`rivabar` relies on the following Python packages, among others:
* [networkx](https://networkx.org/) to work with graphs
* [sknw](https://github.com/Image-Py/sknw) for converting the channel skeleton into a graph
* [Python Spatial Analysis Library (libpysal)](https://pysal.org/libpysal/) to get neighborhood relations between banks / islands
* [Urban Morphology Measuring Toolkit (momepy)](http://docs.momepy.org/en/stable/) to create clean centerline graphs

`rivabar` can be used to map single-thread and multithread rivers, in an almost entirely automated fashion. It requires a water mask as input 
and a start (source) and end (sink) points for the channel / channel belt / delta.

Here's an example of `rivabar` in action on a Landsat image of (part of) the Lena Delta:

**Lena Delta - Original Landsat image**
<img src="https://github.com/zsylvester/rivabar/blob/main/images/lena_delta_1.jpeg" width="800">

**Lena Delta - With Rivabar-extracted centerlines and banklines**
<img src="https://github.com/zsylvester/rivabar/blob/main/images/lena_delta_2.jpeg" width="800">


## Installation

## Getting started

## License