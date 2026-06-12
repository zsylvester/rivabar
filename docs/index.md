# rivabar

![tests](https://github.com/zsylvester/rivabar/actions/workflows/tests.yml/badge.svg)

<img src="https://raw.githubusercontent.com/zsylvester/rivabar/main/images/rivabar_logo.png" width="300">

`rivabar` is a Python package that automatically extracts channel centerlines and banklines
from water index images of rivers. The focus is on getting good representations of the
banklines, as centerlines are not physical features and how they are exactly derived is
subjective. This is achieved in part by viewing channels as boundaries between polygons that
correspond to islands or bars, which allows us to take advantage of algorithms developed for
spatial analysis. In this view, even a single-thread channel becomes the boundary between two
land domains.

`rivabar` can be used to map single-thread and multithread rivers, in an almost entirely
automated fashion. It requires a water mask as input and a start (source) and an end (sink)
point for the channel / channel belt / delta.

<img src="https://raw.githubusercontent.com/zsylvester/rivabar/main/images/brahmaputra_2.png" width="500">

## Capabilities

- **Centerline and bankline extraction** from Landsat or PlanetScope imagery or pre-computed
  water masks, for single-thread and multithread rivers — see [Getting started](getting-started.md)
- **Channel morphology**: widths along the channel, curvature, wavelength analysis, sinuosity
- **Multi-temporal analysis**: tributary detection, splitting rivers at persistent confluences,
  matching segments across scenes, deposition/erosion mapping —
  see [Multi-temporal analysis](workflows/temporal.md)
- **Curvature–migration analysis**: DTW-based migration rate measurement between scenes and
  pair classification
- **Migration prediction**: calibration and forward prediction with the Howard & Knutson (1984)
  model, including spatially-varying erodibility —
  see [Migration prediction](workflows/prediction.md)
- **Animations**: river evolution movies from batch-processed scenes —
  see [Animations](workflows/animation.md)

`rivabar` relies on [networkx](https://networkx.org/) for graphs,
[sknw](https://github.com/Image-Py/sknw) for skeleton-to-graph conversion,
[libpysal](https://pysal.org/libpysal/) for neighborhood relations between banks and islands,
and [momepy](http://docs.momepy.org/en/stable/) for clean centerline graphs, among others.

## Installation

```bash
pip install rivabar
```

Or from source:

```bash
git clone https://github.com/zsylvester/rivabar.git
cd rivabar
pip install -e .
```

## License

MIT — see the [LICENSE](https://github.com/zsylvester/rivabar/blob/main/LICENSE) file.
