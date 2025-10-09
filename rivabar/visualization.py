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
                G_primal, D_primal, dataset=None, plot_main_banklines=True, plot_lines=True, plot_image=True,
                smoothing=False, start_x=None, start_y=None, end_x=None, end_y=None,
                cmap='Grays', alpha=0.5, ax=None, bankline_color='tab:blue', bankline_alpha=1.0):
    """
    Plots an image with overlaid lines representing bank polygons and edges from two graphs.

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
    plot_main_banklines : bool, optional
        If True, plot the main banklines (default is True).
    plot_lines : bool, optional
        If True, plot the centerlines (default is True).
    plot_image : bool, optional
        If True, plot the image (default is True).
    start_x : float, optional
        The x-coordinate of the starting point for smoothing (default is None).
    start_y : float, optional
        The y-coordinate of the starting point for smoothing (default is None).
    end_x : float, optional
        The x-coordinate of the ending point for smoothing (default is None).
    end_y : float, optional
        The y-coordinate of the ending point for smoothing (default is None).
    cmap : str, optional
        The colormap to use for the image (default is 'Blue_r').
    alpha : float, optional
        The transparency of the image (default is 0.5).
    bankline_color : str, optional
        The color of the main banklines (default is 'tab:blue').
    bankline_alpha : float, optional
        The transparency of the main banklines (default is 1.0).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if plot_image:
        plt.imshow(im, extent = [left_utm_x, right_utm_x, lower_utm_y, upper_utm_y], cmap=cmap, alpha=alpha)
    else:
        plt.axis('equal')
    x1, y1, x2, y2 = create_main_channel_banks(G_rook, G_primal, D_primal, dataset=dataset)
    plt.plot(x1, y1, color=bankline_color, alpha=bankline_alpha)
    plt.plot(x2, y2, color=bankline_color, alpha=bankline_alpha)
    # bank1_coords = D_primal.graph.get('main_channel_bank1_coords', None)
    # bank2_coords = D_primal.graph.get('main_channel_bank2_coords', None)
    # plt.plot(bank1_coords[:,0], bank1_coords[:,1], color=bankline_color, alpha=bankline_alpha)
    # plt.plot(bank2_coords[:,0], bank2_coords[:,1], color=bankline_color, alpha=bankline_alpha)
    for i in trange(2, len(G_rook.nodes)):
        x_interior = G_rook.nodes()[i]['bank_polygon'].exterior.xy[0]
        y_interior = G_rook.nodes()[i]['bank_polygon'].exterior.xy[1]
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