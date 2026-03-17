import matplotlib.pyplot as plt
import numpy as np
import warnings
from tqdm import tqdm, trange
from scipy.signal import savgol_filter
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import Normalize
from shapely.geometry import Polygon, LineString
import rasterio

# Internal imports
from .geometry_utils import find_closest_point, getExtrapolatedLine, angle_between
from .polygon_processing import smooth_line, remove_endpoints, vertex_density_tolerance, create_main_channel_banks

def plot_im_and_lines(im, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, G_rook, 
                G_primal, D_primal=None, plot_main_banklines=True, plot_lines=True, plot_image=True, 
                smoothing=False, start_x=None, start_y=None, end_x=None, end_y=None,
                bankline_color='tab:blue', bankline_alpha=1.0,
                cmap='Grays', alpha=0.5, ax=None):
    """
    Plots an image with overlaid lines representing bank polygons and centerlines from graphs.

    Parameters
    ----------
    im : ndarray
        The image to be plotted.
    left_utm_x : float
        The left boundary of the image in UTM coordinates.
    right_utm_x : float
        The right boundary of the image in UTM coordinates.
    lower_utm_y : float
        The lower boundary of the image in UTM coordinates.
    upper_utm_y : float
        The upper boundary of the image in UTM coordinates.
    G_rook : networkx.Graph
        A graph where nodes contain 'bank_polygon' attributes representing bank polygons.
    G_primal : networkx.Graph
        A graph where edges contain 'geometry' attributes representing edge geometries.
    D_primal : networkx.DiGraph, optional
        A directed graph containing main channel bank coordinates in graph attributes.
    plot_main_banklines : bool, optional
        If True, plot the main channel banklines (default is True).
    plot_lines : bool, optional
        If True, plot the centerlines from G_primal (default is True).
    plot_image : bool, optional
        If True, plot the image (default is True).
    smoothing : bool, optional
        If True, apply smoothing to the lines (default is False).
    start_x : float, optional
        The x-coordinate of the starting point for smoothing (default is None).
    start_y : float, optional
        The y-coordinate of the starting point for smoothing (default is None).
    end_x : float, optional
        The x-coordinate of the ending point for smoothing (default is None).
    end_y : float, optional
        The y-coordinate of the ending point for smoothing (default is None).
    bankline_color : str, optional
        The color of the banklines (default is 'tab:blue').
    bankline_alpha : float, optional
        The transparency of the banklines (default is 1.0).
    cmap : str, optional
        The colormap to use for the image (default is 'Grays').
    alpha : float, optional
        The transparency of the image (default is 0.5).
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, creates new figure and axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if plot_image:
        if im is None:
            print('No image to plot!')
        else:
            plt.imshow(im, extent = [left_utm_x, right_utm_x, lower_utm_y, upper_utm_y], cmap=cmap, alpha=alpha)
    else:
        plt.axis('equal')
    
    # Plot main channel banklines from nodes 0 and 1
    for i in range(2):
        if type(G_rook.nodes()[i]['bank_polygon']) == Polygon:
            x = np.array(G_rook.nodes()[i]['bank_polygon'].exterior.xy[0])
            y = np.array(G_rook.nodes()[i]['bank_polygon'].exterior.xy[1])
            if smoothing and start_x is not None and end_x is not None:
                ind1 = find_closest_point(start_x, start_y, np.vstack((x, y)).T)
                ind2 = find_closest_point(end_x, end_y, np.vstack((x, y)).T)
                if ind1 < ind2:
                    x = x[ind1:ind2]
                    y = y[ind1:ind2]
                else:
                    x = x[ind2:ind1]
                    y = y[ind2:ind1]                
        else:
            x = np.array(G_rook.nodes()[i]['bank_polygon'].xy[0])
            y = np.array(G_rook.nodes()[i]['bank_polygon'].xy[1])
        if smoothing and len(x) > 31:
            x, y = smooth_line(x, y, spline_ds=100, spline_smoothing=10000, 
                              savgol_window=min(31, len(x)), savgol_poly_order=3)
        if plot_main_banklines:
            plt.plot(x, y, color=bankline_color, alpha=bankline_alpha)
    
    # Plot interior banklines (islands)
    for i in trange(2, len(G_rook.nodes)):
        x_interior = np.array(G_rook.nodes()[i]['bank_polygon'].exterior.xy[0])
        y_interior = np.array(G_rook.nodes()[i]['bank_polygon'].exterior.xy[1])
        if smoothing and len(x_interior) > 21:
            x_interior_sm = savgol_filter(x_interior, 21, 3)
            y_interior_sm = savgol_filter(y_interior, 21, 3)
            interior_lstr = LineString(zip(x_interior_sm, y_interior_sm))
            x_int, y_int = remove_endpoints(interior_lstr, remove_count=1)
            interior_lstr = LineString(zip(x_int, y_int)).simplify(3)
            x_interior = np.array(interior_lstr.xy[0])
            y_interior = np.array(interior_lstr.xy[1])
        plt.plot(x_interior, y_interior, '-', color=bankline_color, alpha=bankline_alpha)
    
    if plot_lines:
        for s,e,d in tqdm(G_primal.edges):
            x = G_primal[s][e][d]['geometry'].xy[0]
            y = G_primal[s][e][d]['geometry'].xy[1]
            plt.plot(x, y, 'k')
    return fig, ax

def plot_colored_line(x, y, cmap='magma', linewidth=2, ax=None):
    """
    Plot a line with (x, y) coordinates colored according to a colormap.

    Parameters
    ----------
    x : array-like, shape (n,)
        The x coordinates of the points.
    y : array-like, shape (n,)
        The y coordinates of the points.
    cmap : str or Colormap, optional, default: 'viridis'
        The colormap to use for coloring the line.
    linewidth : float, optional, default: 2
        The width of the line.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the line. If None, a new figure and axes are created.

    Returns
    -------
    line : matplotlib.collections.LineCollection
        The LineCollection object representing the colored line.
    """
    
    # Create segments of the line
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Normalize the colormap
    norm = Normalize(0, len(segments))
    # Create the LineCollection
    lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=norm, linewidths=linewidth, joinstyle='round', capstyle='round')
    lc.set_array(np.arange(len(segments)))  # Use the x-values for the color mapping
    # Plot the line using the LineCollection
    if not ax:
        fig, ax = plt.subplots()
    line = ax.add_collection(lc)
    return line

def plot_graph_w_colors(D_primal, ax):
    for s,e,d in tqdm(D_primal.edges):
        x = np.array(D_primal[s][e][d]['geometry'].xy[0])
        y = np.array(D_primal[s][e][d]['geometry'].xy[1])
        plot_colored_line(x, y, linewidth=2, cmap='magma', ax=ax)
    sources = [node for node in D_primal.nodes() if D_primal.in_degree(node) == 0]
    sinks = [node for node in D_primal.nodes() if D_primal.out_degree(node) == 0]
    # Plot source nodes as blue circles
    for node in sources:
        x = D_primal.nodes()[node]['geometry'].xy[0][0]
        y = D_primal.nodes()[node]['geometry'].xy[1][0]
        plt.plot(x, y, 'o', color='blue', markersize=5, zorder=10)
    # Plot sink nodes as black circles
    for node in sinks:
        x = D_primal.nodes()[node]['geometry'].xy[0][0]
        y = D_primal.nodes()[node]['geometry'].xy[1][0]
        plt.plot(x, y, 'o', color='black', markersize=5, zorder=10)
    # plt.axis('equal')

def plot_directed_graph(D_primal, mndwi, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, edge_path=False, arrow_width=100, alpha=1.0):
    """
    Plots the 'D_primal' directed graph on a background image.

    The nodes of the graph are plotted as red dots, and the edges are plotted as red lines. If a main path is provided, it is plotted in blue. Arrows are also drawn to indicate the direction of the edges.

    Parameters
    ----------
    D_primal : networkx.classes.digraph.DiGraph
        The directed graph to plot.
    mndwi : numpy.ndarray
        A 2D array representing the background image.
    left_utm_x : float
        The leftmost x-coordinate of the extent of the image.
    right_utm_x : float
        The rightmost x-coordinate of the extent of the image.
    lower_utm_y : float
        The lowest y-coordinate of the extent of the image.
    upper_utm_y : float
        The highest y-coordinate of the extent of the image.
    edge_path : list, optional
        A list of edges defining the main path to plot. Defaults to False.
    arrow_width : int, optional
        The width of the arrows indicating the direction of the edges. Defaults to 100.
    alpha : float, optional
        The transparency of the edges and the main path. Defaults to 1.0.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes of the plot.
    """

    fig, ax = plt.subplots()
    plt.imshow(mndwi, extent=[left_utm_x, right_utm_x, lower_utm_y, upper_utm_y], cmap='gray_r', alpha=0.5)
    # plot nodes:
    for node in D_primal.nodes:
        plt.plot(D_primal.nodes()[node]['geometry'].xy[0], D_primal.nodes()[node]['geometry'].xy[1], 'ro')
        plt.text(D_primal.nodes()[node]['geometry'].xy[0][0]+100, D_primal.nodes()[node]['geometry'].xy[1][0], str(node))
    # plot edges:    
    for s,e,d in D_primal.edges:
        if 'geometry' in D_primal[s][e][d].keys():
            x = D_primal[s][e][d]['geometry'].xy[0]
            y = D_primal[s][e][d]['geometry'].xy[1]
            plt.plot(x, y, 'r', alpha=alpha)
    # plot main path:
    if edge_path:
        for s,e,d in edge_path:
            x = D_primal[s][e][d]['geometry'].xy[0]
            y = D_primal[s][e][d]['geometry'].xy[1]
            plt.plot(x, y, 'b', alpha=alpha)
    # plot arrows:
    for s,e,d in D_primal.edges:
        x1 = D_primal.nodes()[s]['geometry'].xy[0][0]
        y1 = D_primal.nodes()[s]['geometry'].xy[1][0]
        x2 = D_primal.nodes()[e]['geometry'].xy[0][0]
        y2 = D_primal.nodes()[e]['geometry'].xy[1][0]
        plt.arrow(x1, y1, x2-x1, y2-y1, width=arrow_width, length_includes_head=True, linewidth=None)
    return fig, ax

def read_and_plot_im(dirname, fname):
    """
    Reads a raster image from a file and plots it using matplotlib.

    Parameters
    ----------
    dirname : str
        The directory name where the raster file is located.
    fname : str
        The filename of the raster file.

    Returns
    -------
    im : numpy.ndarray
        The raster image data.
    dataset : rasterio.io.DatasetReader
        The dataset object containing metadata and other information about the raster.
    left_utm_x : float
        The UTM x-coordinate of the left edge of the raster.
    right_utm_x : float
        The UTM x-coordinate of the right edge of the raster.
    lower_utm_y : float
        The UTM y-coordinate of the lower edge of the raster.
    upper_utm_y : float
        The UTM y-coordinate of the upper edge of the raster.
    """
    with rasterio.open(dirname+fname) as dataset:
        im = dataset.read(1)
        left_utm_x = dataset.transform[2]
        upper_utm_y = dataset.transform[5]
        delta_x = dataset.transform[0]
        delta_y = dataset.transform[4]
        nxpix = im.shape[1]
        nypix = im.shape[0]
        right_utm_x = left_utm_x + delta_x*nxpix
        lower_utm_y = upper_utm_y + delta_y*nypix  
    plt.figure()
    plt.imshow(im, extent = [left_utm_x, right_utm_x, lower_utm_y, upper_utm_y], cmap='gray')
    return im, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y

def plot_graph_mappings(D_primal_t1, G_rook_t1, D_primal_t2, G_rook_t2, mappings, ax=None,
                        plot_rook_nodes=True, plot_primal_nodes=True, plot_primal_edges=True):
    """
    Plots the results of graph mapping between two time steps.
    Args:
        D_primal_t1, G_rook_t1: Graphs from the first time step.
        D_primal_t2, G_rook_t2: Graphs from the second time step.
        mappings (dict): The dictionary returned by map_graphs_over_time.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, a new figure is created.
        plot_rook_nodes (bool): Whether to plot G_rook node mappings.
        plot_primal_nodes (bool): Whether to plot D_primal node mappings.
        plot_primal_edges (bool): Whether to plot D_primal edge mappings.
    Returns:
        matplotlib.axes.Axes: The axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))

    # Plot graphs from t1
    for n, data in G_rook_t1.nodes(data=True):
        poly = data.get('cl_polygon')
        if poly and not poly.is_empty:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='blue', alpha=0.5, label='t1 Polygons' if 't1 Polygons' not in [l.get_label() for l in ax.get_lines()] else "")
    for u, v, k, data in D_primal_t1.edges(keys=True, data=True):
        line = data['geometry']
        x, y = line.xy
        ax.plot(x, y, color='cyan', alpha=0.7, linewidth=2, label='t1 Centerline' if 't1 Centerline' not in [l.get_label() for l in ax.get_lines()] else "")

    # Plot graphs from t2
    for n, data in G_rook_t2.nodes(data=True):
        poly = data.get('cl_polygon')
        if poly and not poly.is_empty:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='red', alpha=0.5, label='t2 Polygons' if 't2 Polygons' not in [l.get_label() for l in ax.get_lines()] else "")
    for u, v, k, data in D_primal_t2.edges(keys=True, data=True):
        line = data['geometry']
        x, y = line.xy
        ax.plot(x, y, color='magenta', alpha=0.7, linewidth=2, label='t2 Centerline' if 't2 Centerline' not in [l.get_label() for l in ax.get_lines()] else "")

    # Plot mappings
    # G_rook node mappings
    if plot_rook_nodes and 'rook_nodes' in mappings:
        for n1, n2 in mappings['rook_nodes'].items():
            if G_rook_t1.has_node(n1) and G_rook_t2.has_node(n2):
                p1 = G_rook_t1.nodes[n1]['cl_polygon'].centroid
                p2 = G_rook_t2.nodes[n2]['cl_polygon'].centroid
                ax.plot([p1.x, p2.x], [p1.y, p2.y], 'k--', alpha=0.6, linewidth=0.8, label='Rook Node Mapping' if 'Rook Node Mapping' not in [l.get_label() for l in ax.get_lines()] else "")

    # D_primal node mappings
    if plot_primal_nodes and 'primal_nodes' in mappings:
        for n1, n2 in mappings['primal_nodes'].items():
            if D_primal_t1.has_node(n1) and D_primal_t2.has_node(n2):
                p1 = D_primal_t1.nodes[n1]['geometry']
                p2 = D_primal_t2.nodes[n2]['geometry']
                ax.plot([p1.x, p2.x], [p1.y, p2.y], 'g-', alpha=0.7, linewidth=1, label='Primal Node Mapping' if 'Primal Node Mapping' not in [l.get_label() for l in ax.get_lines()] else "")

    # D_primal edge mappings
    if plot_primal_edges and 'primal_edges' in mappings:
        for e1, e2 in mappings['primal_edges'].items():
            if D_primal_t1.has_edge(*e1) and D_primal_t2.has_edge(*e2):
                line1 = D_primal_t1.edges[e1]['geometry']
                line2 = D_primal_t2.edges[e2]['geometry']
                c1 = line1.centroid
                c2 = line2.centroid
                ax.plot([c1.x, c2.x], [c1.y, c2.y], 'y:', alpha=0.8, linewidth=1.2, label='Primal Edge Mapping' if 'Primal Edge Mapping' not in [l.get_label() for l in ax.get_lines()] else "")

    ax.set_title("Graph Mappings Over Time")
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    return ax

def plot_deviation_rose_diagram(deviations, classifications, weights=None, bins=12):
    """
    Plots a rose diagram (polar histogram) of node displacement deviations,
    separated for confluences and splits.

    Args:
        deviations (dict): A dictionary of node IDs to deviation angles (in degrees).
        classifications (dict): A dictionary of node IDs to their classification.
        weights (dict, optional): A dictionary of node IDs to displacement lengths to weight the histogram.
        bins (int): The number of bins for the histogram.

    Returns:
        matplotlib.figure.Figure, array of matplotlib.axes.Axes: The figure and axes objects.
    """
    confluence_deviations = [d for n, d in deviations.items() if classifications.get(n) == 'confluence' and d is not None]
    split_deviations = [d for n, d in deviations.items() if classifications.get(n) == 'split' and d is not None]

    confluence_weights = None
    if weights:
        confluence_weights = [weights.get(n) for n, d in deviations.items() if classifications.get(n) == 'confluence' and d is not None]

    split_weights = None
    if weights:
        split_weights = [weights.get(n) for n, d in deviations.items() if classifications.get(n) == 'split' and d is not None]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), subplot_kw={'projection': 'polar'})

    # Convert degrees to radians
    confluence_rad = np.radians(confluence_deviations)
    split_rad = np.radians(split_deviations)

    # Confluence plot
    if confluence_rad.size > 0:
        counts, bin_edges = np.histogram(confluence_rad, bins=np.linspace(0, 2 * np.pi, bins + 1), weights=confluence_weights)
        widths = np.diff(bin_edges)
        axes[0].bar(bin_edges[:-1], counts, width=widths, edgecolor='k', alpha=0.7, zorder=2)
    axes[0].set_title('Confluence Deviations')
    axes[0].set_theta_zero_location('N')
    axes[0].set_theta_direction(-1)
    axes[0].grid(True, zorder=0)

    # Split plot
    if split_rad.size > 0:
        counts, bin_edges = np.histogram(split_rad, bins=np.linspace(0, 2 * np.pi, bins + 1), weights=split_weights)
        widths = np.diff(bin_edges)
        axes[1].bar(bin_edges[:-1], counts, width=widths, edgecolor='k', alpha=0.7, color='C1', zorder=2)
    axes[1].set_title('Split Deviations')
    axes[1].set_theta_zero_location('N')
    axes[1].set_theta_direction(-1)
    axes[1].grid(True, zorder=0)
    
    fig.tight_layout()
    return fig, axes

def plot_deviation_histogram(deviations, classifications, distances, bins=18):
    """
    Plots a histogram of node displacement deviations, separated for confluences and splits.
    Only includes nodes with a displacement distance greater than zero.

    Args:
        deviations (dict): A dictionary of node IDs to deviation angles (in degrees).
        classifications (dict): A dictionary of node IDs to their classification.
        distances (dict): A dictionary of node IDs to displacement lengths.
        bins (int): The number of bins for the histogram (default is 18 for 10-degree bins).

    Returns:
        matplotlib.figure.Figure, array of matplotlib.axes.Axes: The figure and axes objects.
    """
    confluence_deviations = [
        d for n, d in deviations.items()
        if d is not None and classifications.get(n) == 'confluence' and distances.get(n, 0) > 0
    ]
    split_deviations = [
        d for n, d in deviations.items()
        if d is not None and classifications.get(n) == 'split' and distances.get(n, 0) > 0
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Confluence plot
    axes[0].hist(confluence_deviations, bins=bins, range=(0, 180), edgecolor='k')
    axes[0].set_xlim(0, 180)
    axes[0].set_title('Confluence Deviations')
    axes[0].set_xlabel('Deviation Angle (degrees)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.5)

    # Split plot
    axes[1].hist(split_deviations, bins=bins, range=(0, 180), edgecolor='k', color='C1')
    axes[1].set_xlim(0, 180)
    axes[1].set_title('Split Deviations')
    axes[1].set_xlabel('Deviation Angle (degrees)')
    axes[1].grid(True, alpha=0.5)

    fig.tight_layout()
    return fig, axes


def plot_river_segments(rivers, common_confluences, segment_groups=None,
                        ncols=4, figsize_per_panel=(4, 5),
                        segment_cmap='tab10', rejected_color='0.75',
                        confluence_color='k',
                        confluence_marker='o', confluence_size=40, linewidth=2,
                        show_mndwi=True, mndwi_cmap='Greys', mndwi_alpha=0.4,
                        show_tributaries=True, tributary_color='0.5',
                        tributary_linewidth=1.0):
    """
    Plot main-path segments for multiple rivers in a multi-panel figure.

    Each panel shows one river's main path colored by segment (split at the
    common confluence points), with the confluence locations marked.

    If *segment_groups* (output of :func:`match_river_segments`) is provided,
    only matched segments are colored; rejected segments are drawn in grey.
    Segment colors are keyed to the ``segment_index`` in each group so that
    the same physical reach has the same color across all panels.

    Parameters
    ----------
    rivers : list of River
        Processed River objects.
    common_confluences : list of dict
        Output of :func:`find_common_confluences`. Each dict must have a
        ``'utm_coords'`` key with ``(x, y)`` UTM coordinates.
    segment_groups : list of dict, optional
        Output of :func:`match_river_segments`. When provided, colors are
        assigned by ``segment_index`` and only matched river/segment pairs
        are colored. Unmatched segments are drawn in *rejected_color*.
    ncols : int, optional
        Number of columns in the panel grid (default 4).
    figsize_per_panel : tuple, optional
        ``(width, height)`` in inches per panel (default ``(4, 5)``).
    segment_cmap : str, optional
        Matplotlib colormap name for segment colours (default ``'tab10'``).
    rejected_color : str or tuple, optional
        Color for segments that were not matched (default ``'0.75'``, light
        grey).
    confluence_color : str, optional
        Marker colour for confluence points (default ``'k'``).
    confluence_marker : str, optional
        Marker style for confluence points (default ``'o'``).
    confluence_size : float, optional
        Marker size for confluence points (default 40).
    linewidth : float, optional
        Line width for centerline segments (default 2).
    show_mndwi : bool, optional
        If True, show the MNDWI water mask as a background (default True).
    mndwi_cmap : str, optional
        Colormap for the MNDWI background (default ``'Greys'``).
    mndwi_alpha : float, optional
        Alpha for the MNDWI background (default 0.4).
    show_tributaries : bool, optional
        If True, plot tributary branches from each river for context
        (default True).
    tributary_color : str or tuple, optional
        Color for tributary lines (default ``'0.5'``, medium grey).
    tributary_linewidth : float, optional
        Line width for tributary branches (default 1.0).

    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes
    """
    split_points = [c['utm_coords'] for c in common_confluences]

    # Build a lookup: river id -> {segment_index -> path} for matched segments
    matched_lookup = {}  # river id -> {seg_index: path}
    if segment_groups is not None:
        for g in segment_groups:
            seg_idx = g['segment_index']
            for river, path in zip(g['rivers'], g['paths']):
                rid = id(river)
                if rid not in matched_lookup:
                    matched_lookup[rid] = {}
                matched_lookup[rid][seg_idx] = path

    # Filter to valid rivers
    valid_rivers = [r for r in rivers if r._is_processed and r._processing_successful
                    and r.main_path is not None]
    if not valid_rivers:
        print("No valid processed rivers to plot.")
        return None, None

    nrows = int(np.ceil(len(valid_rivers) / ncols))
    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    cmap = plt.get_cmap(segment_cmap)

    # Build a stable color mapping: segment_index -> color
    if segment_groups is not None:
        all_seg_indices = [g['segment_index'] for g in segment_groups]
        seg_color_map = {si: cmap(i % cmap.N) for i, si in enumerate(all_seg_indices)}

    for idx, river in enumerate(valid_rivers):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        # Background
        if show_mndwi and river._mndwi is not None:
            ax.imshow(river._mndwi, extent=[river._left_utm_x, river._right_utm_x,
                                             river._lower_utm_y, river._upper_utm_y],
                      cmap=mndwi_cmap, alpha=mndwi_alpha, aspect='equal')

        rid = id(river)

        if segment_groups is not None and rid in matched_lookup:
            # Plot matched segments in color, everything else in grey
            # First draw the full path in grey as background
            for s, e, d in river.main_path:
                geom = river._D_primal[s][e][d]['geometry']
                xs, ys = geom.xy
                ax.plot(xs, ys, color=rejected_color, linewidth=linewidth)
            # Then overlay matched segments in their group color
            for seg_idx, path in matched_lookup[rid].items():
                color = seg_color_map[seg_idx]
                for s, e, d in path:
                    geom = river._D_primal[s][e][d]['geometry']
                    xs, ys = geom.xy
                    ax.plot(xs, ys, color=color, linewidth=linewidth + 0.5)
        else:
            # No segment_groups provided, or river not in any group:
            # fall back to coloring by split position
            try:
                segments, split_info = river.split_main_path_at_points(split_points)
            except Exception:
                segments = [river.main_path]

            if segment_groups is not None:
                # River didn't match — draw all grey
                for seg in segments:
                    for s, e, d in seg:
                        geom = river._D_primal[s][e][d]['geometry']
                        xs, ys = geom.xy
                        ax.plot(xs, ys, color=rejected_color, linewidth=linewidth)
            else:
                # No segment_groups at all — simple coloring by index
                for seg_idx, seg in enumerate(segments):
                    color = cmap(seg_idx % cmap.N)
                    for s, e, d in seg:
                        geom = river._D_primal[s][e][d]['geometry']
                        xs, ys = geom.xy
                        ax.plot(xs, ys, color=color, linewidth=linewidth)

        # Plot tributary branches for context
        if show_tributaries:
            for trib in river.tributary_confluences:
                if 'branch_utm_coords' in trib:
                    coords = trib['branch_utm_coords']
                    ax.plot(coords[:, 0], coords[:, 1], '-',
                            color=tributary_color, linewidth=tributary_linewidth)

        # Mark confluence points
        for c in common_confluences:
            cx, cy = c['utm_coords']
            ax.scatter(cx, cy, c=confluence_color, marker=confluence_marker,
                       s=confluence_size, zorder=5)

        # Label
        label = getattr(river, 'acquisition_date', None)
        if label is None:
            label = river.fname
        ax.set_title(str(label), fontsize=9)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    # Compute global extent across all panels and apply uniform limits
    all_xlims = []
    all_ylims = []
    for idx in range(len(valid_rivers)):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        all_xlims.append(ax.get_xlim())
        all_ylims.append(ax.get_ylim())
    if all_xlims:
        global_xlim = (min(xl[0] for xl in all_xlims), max(xl[1] for xl in all_xlims))
        global_ylim = (min(yl[0] for yl in all_ylims), max(yl[1] for yl in all_ylims))
        for idx in range(len(valid_rivers)):
            row, col = divmod(idx, ncols)
            axes[row][col].set_xlim(global_xlim)
            axes[row][col].set_ylim(global_ylim)

    # Hide unused panels
    for idx in range(len(valid_rivers), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    return fig, axes


def plot_centerline_comparison(river, delta_s=100, smoothing_factor=1e6,
                                path=None, pixel_size=None, ax=None,
                                figsize=(14, 10)):
    """
    Compare raw (D_primal) and smoothed centerlines against the banklines.

    Plots three layers on top of the water mask:
    - banklines (blue)
    - raw centerline from D_primal geometry (black)
    - smoothed/resampled centerline from resample_and_smooth (red)

    Parameters
    ----------
    river : River
        A processed River object.
    delta_s : float, optional
        Resampling distance for smooth centerline (default 100).
    smoothing_factor : float, optional
        Spline smoothing factor (default 1e6).
    path : list of tuples, optional
        Edge path to use. If None, uses river.main_path.
    pixel_size : float, optional
        Pixel size in metres.  Inferred from river._dataset if None.
    ax : matplotlib.Axes, optional
        Axes to plot on.  Creates a new figure if None.
    figsize : tuple, optional
        Figure size when creating a new figure (default (14, 10)).

    Returns
    -------
    fig, ax
    """
    from .utils import resample_and_smooth

    if path is None:
        path = river.main_path
    if pixel_size is None:
        if river._dataset is not None:
            pixel_size = river._dataset.transform[0]
        else:
            pixel_size = 30.0

    # --- Extract raw centerline from D_primal edges ---
    x_raw, y_raw = [], []
    for s, e, d in path:
        xc = list(river._D_primal[s][e][d]['geometry'].xy[0])
        yc = list(river._D_primal[s][e][d]['geometry'].xy[1])
        x_raw += xc
        y_raw += yc
    x_raw, y_raw = np.array(x_raw), np.array(y_raw)

    # --- Smoothed centerline (same logic as get_width_and_curvature) ---
    x_smooth, y_smooth = resample_and_smooth(
        x_raw, y_raw, delta_s, smoothing_factor)

    # --- Banklines ---
    banks = river.main_channel_banks
    bank1 = banks['left_bank'] if banks is not None else None
    bank2 = banks['right_bank'] if banks is not None else None

    # --- Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Water mask background
    if river._mndwi is not None:
        ax.imshow(river._mndwi,
                  extent=[river._left_utm_x, river._right_utm_x,
                          river._lower_utm_y, river._upper_utm_y],
                  cmap='Grays', alpha=0.5)

    # Banklines
    if bank1 is not None:
        ax.plot(*bank1.xy, color='tab:blue', linewidth=1.2, label='banklines')
    if bank2 is not None:
        ax.plot(*bank2.xy, color='tab:blue', linewidth=1.2)

    # Raw centerline
    ax.plot(x_raw, y_raw, 'k-', linewidth=1.0, alpha=0.7,
            label='raw centerline (D_primal)')

    # Smoothed centerline
    ax.plot(x_smooth, y_smooth, 'r-', linewidth=1.5,
            label=f'smoothed (s={smoothing_factor:.0e}, Δs={delta_s})')

    ax.set_aspect('equal')
    ax.legend(loc='best', fontsize=9)
    ax.set_title(f'{river.fname}  –  raw vs smoothed centerline')

    return fig, ax


def plot_pair(results, pair_idx, segment_group=None, km_interval=20,
              ax=None, figsize=(14, 10)):
    """
    Plot banklines and smoothed centerlines for a pair from segment analysis results.

    Parameters
    ----------
    results : dict
        Output of :func:`~rivabar.utils.analyze_segment_group`.
    pair_idx : int
        Index into ``results['pair_info']``.
    segment_group : dict, optional
        The segment group dict (from :func:`~rivabar.temporal_analysis.match_river_segments`).
        Used to look up river objects when ``results['rivers']`` is not
        available (older results).  If *None*, rivers are taken from
        ``results['rivers']``.
    km_interval : float, optional
        Spacing (in km) for distance markers along centerline 1 (default 20).
        Set to 0 or *None* to skip markers.
    ax : matplotlib.Axes, optional
        Axes to plot on.  Creates a new figure if *None*.
    figsize : tuple, optional
        Figure size when creating a new figure (default (14, 10)).

    Returns
    -------
    fig, ax
    """
    pair_info = results['pair_info'][pair_idx]
    cc = results['centerline_coords'][pair_idx]

    # Resolve river objects ------------------------------------------------
    river1_idx = pair_info['river1_index']
    river2_idx = pair_info['river2_index']

    if 'rivers' in results:
        sg_rivers = results['rivers']
    elif segment_group is not None:
        sg_rivers = segment_group['rivers']
    else:
        raise ValueError(
            "Cannot resolve river objects: results has no 'rivers' key "
            "and no segment_group was provided.  Re-run "
            "analyze_segment_group to get updated results, or pass the "
            "segment_group explicitly.")

    river1 = sg_rivers[river1_idx]
    river2 = sg_rivers[river2_idx]

    # Create axes ----------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Channel polygons (filled) ---------------------------------------------
    from .polygon_processing import create_channel_nw_polygon
    for river, color in [(river1, 'tab:blue'), (river2, 'tab:red')]:
        try:
            ch_poly = create_channel_nw_polygon(
                river._G_rook, buffer=10, dataset=river._dataset)
            if isinstance(ch_poly, Polygon):
                # Plot exterior
                ax.fill(*ch_poly.exterior.xy, color=color, alpha=0.15)
                ax.plot(*ch_poly.exterior.xy, color=color, alpha=0.4,
                        linewidth=0.8)
                # Plot island holes
                for interior in ch_poly.interiors:
                    ax.fill(*interior.xy, color='white', alpha=1.0)
                    ax.plot(*interior.xy, color=color, alpha=0.4,
                            linewidth=0.8)
        except Exception:
            # Fallback: plot bank polygon outlines
            for node in range(len(river._G_rook.nodes())):
                poly = river._G_rook.nodes()[node]['bank_polygon']
                if isinstance(poly, Polygon):
                    ax.plot(*poly.exterior.xy, color=color, alpha=0.4,
                            linewidth=0.8)
                else:
                    ax.plot(*poly.xy, color=color, alpha=0.4,
                            linewidth=0.8)

    # Smoothed centerlines from the pair results ---------------------------
    d1 = str(pair_info['date1'])[:10]
    d2 = str(pair_info['date2'])[:10]

    ax.plot(cc['x1'], cc['y1'], color='tab:blue', linewidth=1.5,
            label=f'{d1} (river 1)')
    ax.plot(cc['x2'], cc['y2'], color='tab:red', linewidth=1.5,
            label=f'{d2} (river 2)')

    # Distance markers along centerline 1 ---------------------------------
    if km_interval:
        s1 = pair_info['s1']
        interval_m = km_interval * 1000
        ref_dists = np.arange(0, s1[-1], interval_m)
        for rd in ref_dists:
            idx = np.argmin(np.abs(s1 - rd))
            ax.plot(cc['x1'][idx], cc['y1'][idx], 'ko', markersize=7,
                    markerfacecolor='yellow', markeredgecolor='black',
                    zorder=5)
            ax.text(cc['x1'][idx], cc['y1'][idx], f'{rd/1000:.0f} km',
                    fontsize=9, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.8),
                    zorder=6)

    ax.set_aspect('equal')
    ax.legend(loc='best', fontsize=9)
    gap_yr = pair_info['time_gap_years']
    ax.set_title(f'Pair {pair_idx}: {d1} vs {d2}  '
                 f'({gap_yr:.1f} yr, segment {results.get("segment_index", "")})')

    return fig, ax


def interactive_scatter(df, x, y, c=None, cmap=None, title=None,
                        hover_cols=None):
    """
    Create an interactive scatter plot for selecting data points from a DataFrame.

    Uses plotly for reliable interactive selection in Jupyter notebooks.
    Click on points to see their details in the hover tooltip. Use the
    **box select** or **lasso select** tools in the plotly toolbar to
    select multiple points.  The selected row indices are returned.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data (e.g. output of
        ``create_dataframe_from_results``).
    x : str
        Column name for the x-axis.
    y : str
        Column name for the y-axis.
    c : str or None, optional
        Column name used to colour the points.  Works with both numeric
        and categorical columns.
    cmap : dict or None, optional
        When *c* is categorical, a ``{category: color}`` mapping.
        Unmapped categories fall back to gray.
    title : str or None, optional
        Plot title.
    hover_cols : list of str or None, optional
        Extra column names to show in the hover tooltip.  If *None*,
        shows ``date1``, ``date2``, and the colour column (if any).

    Returns
    -------
    selector : object
        A selector object with a ``.selected`` attribute (list of int)
        that updates as you select points.  Access
        ``selector.selected`` after making your selection.

    Examples
    --------
    >>> sel = rb.interactive_scatter(
    ...     df, 'width_diff (m)', 'r_squared_curv_mr',
    ...     c='pair_class', title='Purus 2')
    >>> # Use box/lasso select in the plotly toolbar, then:
    >>> sel.selected  # [3, 17, 42]
    """
    import plotly.graph_objects as go
    import plotly.express as px

    # Build hover text
    if hover_cols is None:
        hover_cols = []
        for col_name in ['date1', 'date2']:
            if col_name in df.columns:
                hover_cols.append(col_name)
        if c is not None and c in df.columns and c not in hover_cols:
            hover_cols.append(c)

    hover_template = f"<b>{x}</b>: %{{x:.3f}}<br><b>{y}</b>: %{{y:.3f}}"
    custom_data_cols = []
    for i, col_name in enumerate(hover_cols):
        if col_name in df.columns:
            custom_data_cols.append(col_name)
            hover_template += f"<br><b>{col_name}</b>: %{{customdata[{i}]}}"
    hover_template += "<br><b>row</b>: %{text}<extra></extra>"

    is_categorical = (c is not None and c in df.columns
                      and (df[c].dtype == object or hasattr(df[c], 'cat')))

    if is_categorical:
        categories = sorted(df[c].unique(), key=str)
        if cmap is None:
            colors_px = px.colors.qualitative.Plotly
            cmap = {cat: colors_px[i % len(colors_px)]
                    for i, cat in enumerate(categories)}

        fig = go.Figure()
        for cat in categories:
            mask = df[c] == cat
            sub = df[mask]
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y],
                mode='markers',
                name=str(cat),
                marker=dict(size=8, color=cmap.get(cat, 'gray'),
                            line=dict(width=0.5, color='black')),
                text=sub.index.astype(str),
                customdata=sub[custom_data_cols].values if custom_data_cols else None,
                hovertemplate=hover_template,
                selectedpoints=[],
            ))
    else:
        marker_kw = dict(size=8, line=dict(width=0.5, color='black'))
        if c is not None and c in df.columns:
            marker_kw['color'] = df[c]
            marker_kw['colorscale'] = 'Viridis'
            marker_kw['colorbar'] = dict(title=c)

        fig = go.Figure(go.Scatter(
            x=df[x], y=df[y],
            mode='markers',
            marker=marker_kw,
            text=df.index.astype(str),
            customdata=df[custom_data_cols].values if custom_data_cols else None,
            hovertemplate=hover_template,
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        dragmode='lasso',
        width=900, height=600,
    )

    widget = go.FigureWidget(fig)

    # Store row indices on each trace for retrieval
    _trace_row_indices = []
    for trace in widget.data:
        _trace_row_indices.append([int(t) for t in trace.text])

    class Selector:
        """Reads selected points from the FigureWidget traces."""

        def __init__(self, widget, trace_row_indices):
            self._widget = widget
            self._trace_row_indices = trace_row_indices

        @property
        def selected(self):
            """Return DataFrame row indices of currently selected points."""
            result = []
            for trace, row_indices in zip(
                    self._widget.data, self._trace_row_indices):
                sel_pts = trace.selectedpoints
                if sel_pts:
                    for idx in sel_pts:
                        result.append(row_indices[idx])
            return result

    selector = Selector(widget, _trace_row_indices)

    from IPython.display import display
    display(widget)

    print("Use lasso/box select in the toolbar to select points. "
          "Then read selector.selected to get the row indices.")
    return selector


def plot_prediction_map(results, pair_idx, calibration, prediction=None,
                        delta_s=50, pixel_size=None, ax=None,
                        cmap='RdBu_r', vmax=None):
    """
    Map-view visualization of prediction residuals along the channel.

    Colors the channel centerline by the difference between predicted and
    observed migration rate (red = overprediction, blue = underprediction).
    Also shows the observed and predicted centerlines.

    Parameters
    ----------
    results : dict
        Results from ``analyze_river_pairs_filtered`` or
        ``analyze_segment_group``.
    pair_idx : int
        Index of the pair to visualize.
    calibration : dict
        Output of ``calibrate_segment``.
    prediction : dict, optional
        Output of ``predict_forward`` for this pair's river1. If *None*,
        it is computed internally.
    delta_s : float, optional
        Resampling interval (default 50).
    pixel_size : float, optional
        Pixel size for width conversion.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If *None*, a new figure is created.
    cmap : str, optional
        Colormap for residuals (default 'RdBu_r').
    vmax : float, optional
        Symmetric color scale limit. If *None*, uses the 95th percentile
        of absolute residuals.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    from .prediction import (nominal_migration_rate,
                             predicted_migration_rate)

    pair_info = results['pair_info'][pair_idx]
    curvature = results['curvatures'][pair_idx]
    distances = results['migration_distances'][pair_idx]
    s = results['along_channel_distances'][pair_idx]
    time_gap_years = pair_info['time_gap_years']
    W = np.mean(pair_info['width1'])
    observed_mr = distances / time_gap_years

    # centerline_coords x1/y1 are the same arrays as curvature/distances
    coords = results['centerline_coords'][pair_idx]
    x1, y1 = np.asarray(coords['x1']), np.asarray(coords['y1'])
    x2, y2 = np.asarray(coords['x2']), np.asarray(coords['y2'])

    # Compute predicted MR using calibrated parameters
    D = calibration['D']
    Cf = calibration['Cf_median']
    kl = calibration['kl_median']
    Omega = calibration.get('Omega', -1.0)
    Gamma = calibration.get('Gamma', 2.5)

    R0 = nominal_migration_rate(curvature, W, kl)
    R1 = predicted_migration_rate(R0, s, D, Cf, Omega, Gamma)

    # Residual = observed - predicted; positive means the model
    # underpredicted (actual migration was larger than predicted)
    residual = observed_mr - R1

    # Mask to stable segments only
    segment_results = pair_info.get('segment_results', [])
    if segment_results:
        mask = np.zeros(len(curvature), dtype=bool)
        for seg in segment_results:
            mask[seg['start_idx']:seg['end_idx'] + 1] = True
    else:
        mask = np.ones(len(curvature), dtype=bool)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))

    if vmax is None:
        vmax = np.percentile(np.abs(residual[mask]), 95)

    # Plot residual colors on the predicted centerline if available,
    # otherwise fall back to x1/y1
    # Color the river1 centerline by residual — x1/y1, residual, and
    # mask are all on the same grid, no interpolation needed
    points = np.column_stack([x1, y1]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_colors = (residual[:-1] + residual[1:]) / 2

    # Set unstable segments to gray
    seg_mask = mask[:-1] & mask[1:]
    seg_colors_masked = np.where(seg_mask, seg_colors, 0.0)

    lc = LineCollection(segments, cmap=cmap,
                        norm=Normalize(vmin=-vmax, vmax=vmax),
                        linewidths=3)
    lc.set_array(seg_colors_masked)
    colors = lc.to_rgba(seg_colors_masked)
    colors[~seg_mask] = [0.7, 0.7, 0.7, 1.0]
    lc.set_colors(colors)
    ax.add_collection(lc)

    # Plot observed centerlines
    ax.plot(x2, y2, 'k-', lw=1, alpha=0.5,
            label=f'Observed ({pair_info["date2"].year})')

    # Plot predicted centerline if available
    if prediction is not None:
        ax.plot(prediction['predicted_x'], prediction['predicted_y'],
                'r--', lw=1, label='Predicted position')

    ax.set_aspect('equal')
    ax.autoscale_view()

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label('Observed - Predicted MR (m/yr)\n'
                 'red = underpredicted, blue = overpredicted')

    ax.legend()
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')

    return ax