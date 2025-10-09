# Analysis Module - Complete implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import networkx as nx
from tqdm import tqdm, trange
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy import ndimage
from shapely.geometry import Point, LineString, MultiLineString, Polygon
from shapely.ops import split, nearest_points
from rasterio import features
from skimage.measure import find_contours

from .utils import resample_and_smooth, find_condition, compute_s_distance
from .geometry_utils import getExtrapolatedLine, find_matching_indices
from .graph_processing import find_start_node


def analyze_width_and_wavelength(D_primal, main_path, ax, delta_s=5, smoothing_factor=1e8, min_sinuosity=1.1, dx=30):
    """
    Analyze the width and wavelength of a river channel based on input data.

    Parameters
    ----------
    D_primal : dict
        Dictionary containing river channel data.
    main_path : list of tuples
        List of tuples representing the main path of the river channel.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object for plotting.
    delta_s : float, optional
        Resampling interval for smoothing (default is 5).
    smoothing_factor : float, optional
        Smoothing factor for the spline (default is 1e8).
    min_sinuosity : float, optional
        Minimum sinuosity to consider (default is 1.1).
    dx : float, optional
        Spatial resolution of the data (default is 30).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing wavelengths, sinuosities, mean widths, standard deviations of widths, and along-channel distances.
    curv : numpy.ndarray
        Array of curvature values.
    s : numpy.ndarray
        Array of along-channel distances.
    loc_zero_curv : numpy.ndarray
        Array of indices where curvature crosses zero.
    xsmooth : numpy.ndarray
        Smoothed x-coordinates of the river channel.
    ysmooth : numpy.ndarray
        Smoothed y-coordinates of the river channel.
    """
    xl = []
    yl = []
    w = []
    for s,e,d in main_path:
        key1 = list(D_primal[s][e][d]['half_widths'].keys())[0]
        key2 = list(D_primal[s][e][d]['half_widths'].keys())[1]
        w1 = D_primal[s][e][d]['half_widths'][key1]
        w2 = D_primal[s][e][d]['half_widths'][key2]
        x = D_primal[s][e][d]['geometry'].xy[0]
        y = D_primal[s][e][d]['geometry'].xy[1]
        xl += list(x)
        yl += list(y)
        w += list(np.array(w1)+np.array(w2))
    xsmooth, ysmooth = resample_and_smooth(xl[1:-1], yl[1:-1], delta_s, smoothing_factor)
    curv, s = compute_curvature(xsmooth, ysmooth)
    loc_zero_curv, loc_max_curv = find_zero_crossings(curv)
    spl = CubicSpline(np.arange(len(w[1:-1])), w[1:-1])
    xnew = np.linspace(0, len(w[1:-1]), num=len(curv))
    wnew = spl(xnew)
    # fig, ax = plt.subplots()
    # plt.imshow(mndwi, extent=[left_utm_x, right_utm_x, lower_utm_y, upper_utm_y], cmap='gray_r', alpha=0.3)
    ax.plot(xsmooth, ysmooth)
    ax.plot(xsmooth[loc_zero_curv], ysmooth[loc_zero_curv], 'ro', markersize=4)
    ax.axis('equal');
    sinuosities = []
    mean_widths = []
    half_wave_lengths = []
    std_widths = []
    along_ch_dist = []
    for i in range(len(loc_zero_curv)-1):
        arc_length = s[loc_zero_curv[i+1]]-s[loc_zero_curv[i]]
        x1 = xsmooth[loc_zero_curv[i]]
        y1 = ysmooth[loc_zero_curv[i]]
        x2 = xsmooth[loc_zero_curv[i+1]]
        y2 = ysmooth[loc_zero_curv[i+1]]
        half_wave_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        half_wave_lengths.append(half_wave_length)
        sinuosities.append(arc_length/half_wave_length)
        mean_widths.append(np.mean(wnew[loc_zero_curv[i]:loc_zero_curv[i+1]]))
        std_widths.append(np.std(wnew[loc_zero_curv[i]:loc_zero_curv[i+1]]))
        along_ch_dist.append(np.mean(s[loc_zero_curv[i]:loc_zero_curv[i+1]]))
    half_wave_lengths = np.array(half_wave_lengths)
    sinuosities = np.array(sinuosities)
    mean_widths = np.array(mean_widths)
    std_widths = np.array(std_widths)
    along_ch_dist = np.array(along_ch_dist)
    mean_widths = mean_widths[sinuosities > min_sinuosity]*dx
    std_widths = std_widths[sinuosities > min_sinuosity]*dx
    wave_lengths = 2*half_wave_lengths[sinuosities > min_sinuosity] # note that the half-wavelengths are converted to full wavelengths
    along_ch_dist = along_ch_dist[sinuosities > min_sinuosity]/1000
    sinuosities = sinuosities[sinuosities > min_sinuosity]
    df = pd.DataFrame(np.vstack((wave_lengths, sinuosities, mean_widths, std_widths, along_ch_dist)).T, columns=['wavelengths (m)', 'sinuosities', 'mean widths (m)', 'std. dev. of widths (m)', 'along-channel distance (km)'])
    plt.figure()
    plt.errorbar(mean_widths, wave_lengths, fmt='o', markersize=6, markeredgecolor='k', xerr=std_widths, capsize=5, ecolor='lightgray', elinewidth=1, zorder=1)
    scatter = plt.scatter(mean_widths, wave_lengths, c=along_ch_dist, cmap='viridis', edgecolor='black', zorder=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('channel width (m)')
    plt.ylabel('wavelength (m)');
    xticks = [100, 200, 300, 400, 500, 600, 800, 1000]
    plt.xticks(xticks, labels=np.array(xticks).astype('str'))
    yticks = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000]
    plt.yticks(yticks, labels=np.array(yticks).astype('str'))
    plt.colorbar(scatter, label='along-channel distance (km)');
    return df, curv, s, loc_zero_curv, xsmooth, ysmooth

def compute_curvature(x,y):
    """
    Compute the first derivatives and curvature of a curve (centerline).

    Parameters
    ----------
    x : array_like
        Cartesian x-coordinates of the curve.
    y : array_like
        Cartesian y-coordinates of the curve.

    Returns
    -------
    curvature : ndarray
        Curvature of the curve (in 1/units of x and y).
    s : ndarray
        Cumulative distance along the curve.

    Notes
    -----
    The function calculates the first and second derivatives of the input coordinates
    to determine the curvature and cumulative distance along the curve.
    """

    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)  
    ds = np.sqrt(dx**2+dy**2)
    s = np.cumsum(ds)
    ddx = np.gradient(dx) # second derivatives 
    ddy = np.gradient(dy) 
    curvature = (dx*ddy-dy*ddx)/((dx**2+dy**2)**1.5)
    return curvature, s

def find_zero_crossings(curve):
    """
    Find zero crossings of a curve.

    Parameters
    ----------
    curve : array_like
        A one-dimensional array that describes the curve.

    Returns
    -------
    loc_zero_curv : ndarray
        Indices of zero crossings.
    loc_max_curv : ndarray
        Indices of maximum values.

    Notes
    -----
    Zero crossings are points where the curve changes sign. The function also
    identifies the indices of the maximum values between zero crossings.
    """
    n_curv = abs(np.diff(np.sign(curve)))
    n_curv[find_condition(n_curv==2)] = 1
    loc_zero_curv = find_condition(n_curv)
    loc_zero_curv = loc_zero_curv +1
    loc_zero_curv = np.hstack((0,loc_zero_curv,len(curve)-1))
    n_infl = len(loc_zero_curv)
    max_curv = np.zeros(n_infl-1)
    loc_max_curv = np.zeros(n_infl-1, dtype=int)
    for i in range(1, n_infl):
        if np.mean(curve[loc_zero_curv[i-1]:loc_zero_curv[i]])>0:
            max_curv[i-1] = np.max(curve[loc_zero_curv[i-1]:loc_zero_curv[i]])
        if np.mean(curve[loc_zero_curv[i-1]:loc_zero_curv[i]])<0:
            max_curv[i-1] = np.min(curve[loc_zero_curv[i-1]:loc_zero_curv[i]])
        max_local_ind = find_condition(curve[loc_zero_curv[i-1]:loc_zero_curv[i]]==max_curv[i-1])
        if len(max_local_ind)>1:
            loc_max_curv[i-1] = loc_zero_curv[i-1] + max_local_ind[0]
        elif len(max_local_ind)==1:
            loc_max_curv[i-1] = loc_zero_curv[i-1] + max_local_ind
        else:
            loc_max_curv[i-1] = 0
    return loc_zero_curv, loc_max_curv

def filter_half_contours(contours, ch_map, threshold=0.25):
    """Simple one-liner to keep only 0.5-level contours."""
    return [contour for contour in contours 
            if np.mean(ch_map[np.clip(contour[::max(1, len(contour)//10), 0].astype(int), 0, ch_map.shape[0]-1),
                              np.clip(contour[::max(1, len(contour)//10), 1].astype(int), 0, ch_map.shape[1]-1)]) >= threshold]

def get_bank_coords(poly, mndwi, dataset, timer=False, filter_contours=False):
    """
    This function calculates the coordinates of river banks from a given polygon and MNDWI dataset.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        The polygon representing the area of interest.
    mndwi : numpy.ndarray
        The Modified Normalized Difference Water Index (MNDWI) array.
    dataset : rasterio.io.DatasetReader
        The raster dataset containing the spatial reference and transformation information.
    timer : bool, optional
        If True, uses a progress bar to show the progress of the loop (default is False).

    Returns
    -------
    x_utm : list
        List of x coordinates of the river bank in UTM.
    y_utm : list
        List of y coordinates of the river bank in UTM.
    ch_map : numpy.ndarray
        A binary map where the river channel is marked.
    """
    """this is the best solution so far for getting the bank coordinates"""
    tile_size = 500 # this should depend on the mean channel width (in pixels)
    row1, col1 = dataset.index(poly.bounds[0], poly.bounds[1])
    row2, col2 = dataset.index(poly.bounds[2], poly.bounds[3])
    row1 = max(row1+tile_size, 0)
    row1 = min(row1, mndwi.shape[0])
    row2 = max(row2-tile_size, 0)
    row2 = min(row2, mndwi.shape[0])
    col1 = max(col1-tile_size, 0)
    col1 = min(col1, mndwi.shape[1])
    col2 = max(col2+tile_size, 0)
    col2 = min(col2, mndwi.shape[1])
    rst_arr = np.zeros(np.shape(mndwi))
    shapes = ((geom, value) for geom, value in zip([poly], [1]))
    rasterized_poly = features.rasterize(shapes=shapes, fill=0, out=rst_arr, transform=dataset.transform)[row2:row1, col1:col2]
    mndwi_small = mndwi[row2:row1, col1:col2].copy()
    mndwi_small[rasterized_poly == 0] = 1
    mndwi_small_dist = ndimage.distance_transform_edt(mndwi_small)
    ch_map = np.zeros(np.shape(mndwi_small))
    # what follows here may seem unnecessary but you do need the 'for' loop if you want to deal with tributaries
    x = poly.exterior.xy[0]
    y = poly.exterior.xy[1]
    indices = np.array([dataset.index(x, y) for (x, y) in list(zip(x, y))])
    row = indices[:,0]
    col = indices[:,1]
    row = np.array(row)-row2
    row[row < 0] = 0
    col = np.array(col)-col1
    col[col < 0] = 0
    if timer:
        for i in trange(len(row)):
            if col[i]<mndwi_small.shape[1] and row[i]<mndwi_small.shape[0]:
                w = mndwi_small_dist[row[i], col[i]] # distance to closest channel bank at current location
                if w <= 500: # this probably shouldn't be hardcoded!
                    pad = int(w)+10
                    tile = np.ones((pad*2, pad*2))
                    tile[pad, pad] = 0
                    tile = ndimage.distance_transform_edt(tile)
                    tile[tile >= w] = 0 # needed to avoid issues with narrow channels
                    tile[tile > 0] = 1
                    r1 = max(0, row[i]-pad)
                    r2 = min(row[i]+pad, mndwi_small.shape[0])
                    c1 = max(0, col[i]-pad)
                    c2 = min(col[i]+pad, mndwi_small.shape[1])
                    tr1 = max(0, pad-row[i])
                    tr2 = min(2*pad, pad+mndwi_small.shape[0]-row[i])        
                    tc1 = max(0, pad-col[i])
                    tc2 = min(2*pad, pad+mndwi_small.shape[1]-col[i])
                    ch_map[r1:r2, c1:c2] = np.maximum(tile[tr1:tr2, tc1:tc2], ch_map[r1:r2, c1:c2])
    else:
        for i in range(len(row)):
            if col[i]<mndwi_small.shape[1] and row[i]<mndwi_small.shape[0]:
                w = mndwi_small_dist[row[i], col[i]] # distance to closest channel bank at current location
                if w <= 500:
                    pad = int(w)+10
                    tile = np.ones((pad*2, pad*2))
                    tile[pad, pad] = 0
                    tile = ndimage.distance_transform_edt(tile)
                    tile[tile >= w] = 0 # needed to avoid issues with narrow channels
                    tile[tile > 0] = 1
                    r1 = max(0, row[i]-pad)
                    r2 = min(row[i]+pad, mndwi_small.shape[0])
                    c1 = max(0, col[i]-pad)
                    c2 = min(col[i]+pad, mndwi_small.shape[1])
                    tr1 = max(0, pad-row[i])
                    tr2 = min(2*pad, pad+mndwi_small.shape[0]-row[i])
                    tc1 = max(0, pad-col[i])
                    tc2 = min(2*pad, pad+mndwi_small.shape[1]-col[i])
                    ch_map[r1:r2, c1:c2] = np.maximum(tile[tr1:tr2, tc1:tc2], ch_map[r1:r2, c1:c2])
    ch_map_cont = ch_map.copy()
    ch_map_cont[rasterized_poly == 0] = 1
    ch_map_cont = ~(ch_map_cont.astype('bool'))
    contours = find_contours(ch_map_cont, 0.5)
    if filter_contours:
        contours = filter_half_contours(contours, ch_map)
    # Calculate length and irregularity for each contour
    all_contour_data = []
    if len(contours) > 0:
        for i, contour_rc in enumerate(contours): # contour_rc is an (N, 2) array of (row, column)
            num_points = len(contour_rc)
            irregularity = 1.0  # Default for simple/problematic contours (less irregular)
            if num_points >= 3:
                try:
                    # ConvexHull needs at least 3 non-collinear points
                    hull = ConvexHull(contour_rc)
                    num_hull_vertices = len(hull.vertices)
                    if num_hull_vertices > 0:
                        irregularity = num_points / num_hull_vertices
                    # If num_hull_vertices is 0, it's problematic, stick to irregularity = 1.0
                except Exception: # Catches QhullError for collinear points, etc.
                    # For flat/collinear contours, irregularity is low (like a straight line)
                    irregularity = 1.0
            all_contour_data.append({
                'id': i,
                'length': num_points,
                'irregularity': irregularity,
            })
        if all_contour_data:
            lengths = np.array([data['length'] for data in all_contour_data])
            irregularities = np.array([data['irregularity'] for data in all_contour_data])

            # Normalize lengths (0 to 1)
            min_len, max_len = np.min(lengths), np.max(lengths)
            if max_len - min_len > 0:
                norm_lengths = (lengths - min_len) / (max_len - min_len)
            else:
                norm_lengths = np.zeros_like(lengths) if len(lengths) > 1 else np.array([0.5])

            # Normalize irregularities (0 to 1)
            min_irr, max_irr = np.min(irregularities), np.max(irregularities)
            if max_irr - min_irr > 0:
                norm_irregularities = (irregularities - min_irr) / (max_irr - min_irr)
            else:
                norm_irregularities = np.zeros_like(irregularities) if len(irregularities) > 1 else np.array([0.5])

            # Define weights for length and irregularity
            # Adjust these weights based on how much you value length vs. irregularity
            weight_length = 0.6  # 60% importance to length
            weight_irregularity = 0.4  # 40% importance to irregularity

            best_score = -1
            ind = -1
            for i in range(len(all_contour_data)):
                # Combined score
                score = (weight_length * norm_lengths[i] +
                        weight_irregularity * norm_irregularities[i])
                all_contour_data[i]['score'] = score
                if score > best_score:
                    best_score = score
                    ind = all_contour_data[i]['id']
            if ind != -1:
                # We found the best contour based on combined score
                x = contours[ind][:,1]
                y = contours[ind][:,0]
                # Apply UTM conversion as before
                x_utm = dataset.xy(row2 + np.array(y), col1 + np.array(x))[0]
                y_utm = dataset.xy(row2 + np.array(y), col1 + np.array(y))[1] # Corrected: use y for both dataset.xy args
            else:
                # Fallback if no suitable contour found (e.g., all contours were problematic)
                # This case should be rare if there are any contours.
                # Maybe handle as error?
                x_utm = []
                y_utm = []
                
        else: # No contours found initially
            x_utm = []
            y_utm = []
    else:
        x_utm = []
        y_utm = []
    return x_utm, y_utm, ch_map_cont

def compute_mndwi_small_dist(poly, dataset, mndwi, tile_size=500):
    """
    Computes the distance transform of a 'small' MNDWI tile, relative
    to a polygon.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        The polygon defining the area of interest.
    dataset : rasterio.io.DatasetReader
        The dataset containing the raster data.
    mndwi : numpy.ndarray
        The MNDWI (Modified Normalized Difference Water Index) array.
    tile_size : int, optional
        The size of the tile to be extracted (default is 500).

    Returns
    -------
    mndwi_small_dist : numpy.ndarray
        The distance transform of the small MNDWI tile.
    col1 : int
        The starting column index of the small tile in the original MNDWI array.
    col2 : int
        The ending column index of the small tile in the original MNDWI array.
    row1 : int
        The starting row index of the small tile in the original MNDWI array.
    row2 : int
        The ending row index of the small tile in the original MNDWI array.

    Notes
    -----
    The function extracts a smaller tile from the MNDWI array based on the 
    bounding box of the input polygon, rasterizes the polygon, and computes 
    the distance transform of the small tile. The units of the distance 
    transform are in pixels.
    """
    
    # tile_size = 500 # this should ideally depend on the mean channel width (in pixels), but not sure how that can be done
    row1, col1 = dataset.index(poly.bounds[0], poly.bounds[1])
    row2, col2 = dataset.index(poly.bounds[2], poly.bounds[3])
    row1, col1 = dataset.index(poly.bounds[0], poly.bounds[1])
    row2, col2 = dataset.index(poly.bounds[2], poly.bounds[3])
    row1 = max(row1+tile_size, 0)
    row2 = max(row2-tile_size, 0)
    col1 = max(col1-tile_size, 0)
    col2 = max(col2+tile_size, 0)
    rst_arr = np.zeros(np.shape(mndwi))
    shapes = ((geom, value) for geom, value in zip([poly], [1]))
    rasterized_poly = features.rasterize(shapes=shapes, fill=0, out=rst_arr, transform=dataset.transform)[row2:row1, col1:col2]
    mndwi_small = mndwi[row2:row1, col1:col2].copy()
    mndwi_small[rasterized_poly == 0] = 1
    mndwi_small_dist = ndimage.distance_transform_edt(mndwi_small)
    return mndwi_small_dist, col1, col2, row1, row2

def set_half_channel_widths(G_primal, G_rook, dataset, mndwi):
    """
    Set half channel widths for edges in the 'G_primal' graph.

    Parameters
    ----------
    G_primal : networkx.Graph
        The primal graph where edges represent the centerlines of channels.
    G_rook : networkx.Graph
        The rook graph where nodes represent polygons and edges represent adjacency between polygons.
    dataset : rasterio.DatasetReader
        The dataset containing the raster data.
    mndwi : numpy.ndarray
        The Modified Normalized Difference Water Index (MNDWI) array.

    Returns
    -------
    None
    """
    linestrings = []
    for s,e,d in G_primal.edges:
        linestrings.append(G_primal[s][e][d]['geometry'])
    for u, v in G_rook.edges(): # need this so that different empty lists are added to each edge
        G_rook[u][v]['G_primal_edges'] = []
    for s, e, d in G_primal.edges:
        G_primal[s][e][d]['half_widths'] = {}
    for node in tqdm(G_rook):
        poly = G_rook.nodes()[node]['cl_polygon']
        if type(poly) == Polygon:
            mndwi_small_dist, col1, col2, row1, row2 = compute_mndwi_small_dist(poly, dataset, mndwi)
            x1 = poly.exterior.xy[0]
            y1 = poly.exterior.xy[1]
            inds = []
            for i in range(len(linestrings)):
                x2 = linestrings[i].xy[0]
                y2 = linestrings[i].xy[1]
                indices = find_matching_indices(x1, y1, x2, y2)
                if len(indices) > 2:
                    inds.append(i)
            neighbors = list(nx.all_neighbors(G_rook, node))
            for neighbor in neighbors:
                for i in inds:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning) # ignore warnings about invalid intersections
                            common_cline = G_rook.nodes()[node]['cl_polygon'].intersection(G_rook.nodes()[neighbor]['cl_polygon'])
                        if common_cline.intersects(linestrings[i]):
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", RuntimeWarning) # ignore warnings about invalid intersections
                                intersection = G_rook.nodes()[node]['cl_polygon'].intersection(G_rook.nodes()[neighbor]['cl_polygon']).intersection(linestrings[i])
                        if type(intersection) == MultiLineString:
                            (s, e, d) = list(G_primal.edges)[i]
                            if (s, e, d) not in G_rook[node][neighbor]['G_primal_edges']:
                                G_rook[node][neighbor]['G_primal_edges'].append((s, e, d))
                            poly = G_rook.nodes()[node]['cl_polygon']
                            cl_x = G_primal[s][e][d]['geometry'].xy[0]
                            cl_y = G_primal[s][e][d]['geometry'].xy[1]
                            indices = np.array([dataset.index(x, y) for (x, y) in list(zip(cl_x, cl_y))])
                            cl_row = indices[:,0]
                            cl_col = indices[:,1]
                            cl_row = cl_row-row2
                            cl_col = cl_col-col1
                            r, c = np.shape(mndwi_small_dist)
                            cl_row = cl_row[(cl_col >= 0) & (cl_col < c)]
                            cl_col = cl_col[(cl_col >= 0) & (cl_col < c)]
                            cl_col = cl_col[(cl_row >= 0) & (cl_row < r)]
                            cl_row = cl_row[(cl_row >= 0) & (cl_row < r)]
                            w = mndwi_small_dist[cl_row, cl_col] # channel half width (number of pixels)
                            G_primal[s][e][d]['half_widths'][node] = w
                    except:
                        print('unable to set half width for edge', s, e, d)
                        pass

def get_bank_coords_for_main_channel(D_primal, mndwi, edge_path, dataset, cline_buffer=2000):
    """
    Extracts the coordinates of the banks for the main channel from the given dataset.

    Parameters
    ----------
    D_primal : networkx.DiGraph
        The directed graph representing the river network.
    mndwi : numpy.ndarray
        The Modified Normalized Difference Water Index (MNDWI) array.
    edge_path : list of tuples
        The list of edges representing the main channel path.
    dataset : rasterio.io.DatasetReader
        The dataset containing the spatial information.
    cline_buffer : int, optional
        The buffer distance around the centerline, by default 2000.

    Returns
    -------
    x : numpy.ndarray
        The x-coordinates of the main channel.
    y : numpy.ndarray
        The y-coordinates of the main channel.
    x_utm1 : numpy.ndarray
        The x-coordinates of the first bank.
    y_utm1 : numpy.ndarray
        The y-coordinates of the first bank.
    x_utm2 : numpy.ndarray
        The x-coordinates of the second bank.
    y_utm2 : numpy.ndarray
        The y-coordinates of the second bank.
    """

    if dataset is None:
        raise ValueError("Dataset is None - cannot extract bank coordinates")
    # Test dataset validity
    try:
        _ = dataset.transform
        _ = dataset.xy(0, 0)  # Test the xy method
    except Exception as e:
        raise ValueError(f"Dataset is invalid or closed: {str(e)}")
    
    x, y = [], []
    for s,e,d in edge_path:
        x_new = list(D_primal[s][e][d]['geometry'].xy[0][1:])
        x.extend(x_new)
        y_new = list(D_primal[s][e][d]['geometry'].xy[1][1:])
        y.extend(y_new)
    im_boundary = Polygon([dataset.xy(0,0), dataset.xy(0,mndwi.shape[1]), dataset.xy(mndwi.shape[0], mndwi.shape[1]), dataset.xy(mndwi.shape[0], 0)])
    if len(D_primal) > 2:
        point1 = Point(x[0], y[0])
        point2 = Point(x[-1], y[-1])
    else:
        s, inds = find_start_node(D_primal)
        e = list(nx.neighbors(D_primal, s))[0]
        x = D_primal[s][e][0]['geometry'].xy[0]
        y = D_primal[s][e][0]['geometry'].xy[1]
        point1 = Point(x[0], y[0])
        point2 = Point(x[-1], y[-1])
    nearest_point_on_polygon1 = nearest_points(im_boundary.exterior, point1)[0]
    nearest_point_on_polygon2 = nearest_points(im_boundary.exterior, point2)[0]
    x = np.array(x)
    xlong = np.hstack((nearest_point_on_polygon1.x, x, nearest_point_on_polygon2.x))
    y = np.array(y)
    ylong = np.hstack((nearest_point_on_polygon1.y, y, nearest_point_on_polygon2.y))
    cline = LineString(list(zip(xlong, ylong)))
    split_polygons = split(im_boundary, cline)
    x_utm1, y_utm1, ch_map = get_bank_coords(split_polygons.geoms[0], mndwi, dataset, timer=True)
    x_utm2, y_utm2, ch_map = get_bank_coords(split_polygons.geoms[1], mndwi, dataset, timer=True)
    bline1 = LineString(list(zip(x_utm1, y_utm1)))
    bline2 = LineString(list(zip(x_utm2, y_utm2)))
    non_overlap_1 = bline1.difference(bline2)
    non_overlap_2 = bline2.difference(bline1)
    if type(non_overlap_1) == MultiLineString:
        x_utm1, y_utm1 = [], []
        for geom in non_overlap_1.geoms:
            # only add linestrings that are not aligned with the x or y axis (average offset relative to the axes is larger than 0.1 m):
            # if (np.sum(np.abs(np.diff(geom.xy[0])))/len(geom.xy[0]) > 1) and (np.sum(np.abs(np.diff(geom.xy[1])))/len(geom.xy[1]) > 0.1):
            x_utm1.extend(geom.xy[0])
            y_utm1.extend(geom.xy[1])
    else:
        x_utm1, y_utm1 = non_overlap_1.xy[0], non_overlap_1.xy[1]
    if type(non_overlap_2) == MultiLineString:
        x_utm2, y_utm2 = [], []
        for geom in non_overlap_2.geoms:
            # only add linestrings that are not aligned with the x or y axis (average offset relative to the axes is larger than 0.1 m):
            # if (np.sum(np.abs(np.diff(geom.xy[0])))/len(geom.xy[0]) > 1) and (np.sum(np.abs(np.diff(geom.xy[1])))/len(geom.xy[1]) > 0.1):
            x_utm2.extend(geom.xy[0])
            y_utm2.extend(geom.xy[1])
    else:
        x_utm2, y_utm2 = non_overlap_2.xy[0], non_overlap_2.xy[1]
    buffered_centerline = LineString(np.vstack((x,y)).T).buffer(cline_buffer)
    bankline1 = buffered_centerline.intersection(LineString(np.vstack((x_utm1, y_utm1)).T))
    if type(bankline1) != LineString:
        lengths = []
        for line in bankline1.geoms:
            lengths.append(line.length)
        bankline1 = bankline1.geoms[np.argmax(lengths)]
    x_utm1 = bankline1.xy[0]; y_utm1 = bankline1.xy[1]
    bankline2 = buffered_centerline.intersection(LineString(np.vstack((x_utm2, y_utm2)).T))
    if type(bankline2) != LineString:
        lengths = []
        for line in bankline2.geoms:
            lengths.append(line.length)
        bankline2 = bankline2.geoms[np.argmax(lengths)]
    x_utm2 = bankline2.xy[0]; y_utm2 = bankline2.xy[1]
    return x, y, x_utm1, y_utm1, x_utm2, y_utm2

def get_channel_widths_along_path(D_primal, path):
    """
    Calculates the channel widths along a given path in the directed graph "D_primal".

    The path is a list of edges, where each edge is a tuple of two nodes and a key. The function retrieves the 
    'half_widths' attribute of each edge, which is a dictionary with two keys. The values corresponding to these 
    keys are lists of half-widths of the channel at various points along the edge. The function also calculates 
    the cumulative distance along the path.

    Parameters
    ----------
    D_primal : networkx.classes.digraph.DiGraph
        The directed graph.
    path : list
        The path, represented as a list of edges. Each edge is a tuple of two nodes and a key.

    Returns
    -------
    xl : list
        The x-coordinates of the points along the path.
    yl : list
        The y-coordinates of the points along the path.
    w1l : list
        The half-widths corresponding to the first key for each edge.
    w2l : list
        The half-widths corresponding to the second key for each edge.
    w : list
        The full widths of the channel at various points along the path.
    s : numpy.ndarray
        The cumulative distance along the path.
    """
    xl = []
    yl = []
    w1l = []
    w2l = []
    w = []
    for s,e,d in path:
        key1 = list(D_primal[s][e][d]['half_widths'].keys())[0]
        key2 = list(D_primal[s][e][d]['half_widths'].keys())[1]
        w1 = D_primal[s][e][d]['half_widths'][key1]
        w2 = D_primal[s][e][d]['half_widths'][key2]
        x = D_primal[s][e][d]['geometry'].xy[0]
        y = D_primal[s][e][d]['geometry'].xy[1]
        xl += list(x)
        yl += list(y)
        w += list(np.array(w1)+np.array(w2))
        w1l += list(w1)
        w2l += list(w2)
    dx = np.gradient(xl) # first derivatives
    dy = np.gradient(yl)  
    ds = np.sqrt(dx**2+dy**2)
    s = np.cumsum(ds)
    return xl, yl, w1l, w2l, w, s

def get_all_channel_widths(D_primal):
    """
    Extract all channel widths from a directed multigraph.
    
    This function iterates through all edges in the directed multigraph and
    extracts the channel widths by summing the half-widths from both sides
    of the channel.
    
    Parameters
    ----------
    D_primal : networkx.MultiDiGraph
        A directed multigraph where edges contain 'half_widths' attributes.
        Each 'half_widths' attribute is a dictionary with two keys, each
        corresponding to a list of half-width measurements.
        
    Returns
    -------
    widths : list
        A flattened list of all channel widths across all edges in the graph.
    """
    widths = []
    for s,e,d in D_primal.edges:
        key1 = list(D_primal[s][e][d]['half_widths'].keys())[0]
        key2 = list(D_primal[s][e][d]['half_widths'].keys())[1]
        w1 = D_primal[s][e][d]['half_widths'][key1]
        w2 = D_primal[s][e][d]['half_widths'][key2]
        widths += list(w1+w2)
    return widths

def get_channel_mouth_polygon(mndwi, dataset, points):
    """
    Create a polygon that defines the coastline when multiple channels reach the sea/lake (e.g., in a delta).
    It uses a line drawn roughly parallel to the coastline (defined by 'points') to create the polygon.

    Parameters
    ----------
    mndwi : numpy.ndarray
        A 2D array representing the Modified Normalized Difference Water Index (MNDWI).
    dataset : rasterio.io.DatasetReader
        A rasterio dataset object representing the image.
    points : list
        A list of points defining a line that runs roughly parallel to the coastline.

    Returns
    -------
    x_utm : list
        The x-coordinates of the vertices of the channel mouth polygon in UTM coordinates.
    y_utm : list
        The y-coordinates of the vertices of the channel mouth polygon in UTM coordinates.
    ch_map : numpy.ndarray
        A 2D array representing the 'channel' map - in this case, it is a map of the distance of the line from the coastline.

    Example
    -------
    points = plt.ginput(-1)  # create a line that runs roughly parallel to the coastline
    x_utm, y_utm = get_channel_mouth_polygon(mndwi, dataset, points)  # use this function to create the channel mouth polygon
    """
    a1, b1 = getExtrapolatedLine((points[1][0], points[1][1]), (points[0][0], points[0][1]), 2000)  # use the first two points
    a2, b2 = getExtrapolatedLine((points[-2][0], points[-2][1]), (points[-1][0], points[-1][1]), 2000)  # use the last two points
    line = LineString(np.vstack(((b1[0], b1[1]), points, (b2[0], b2[1]))))
    poly = line.buffer(1)
    tile_size = 5000  # this should depend on the mean channel width (in pixels)
    row1, col1 = dataset.index(poly.bounds[0], poly.bounds[1])
    row2, col2 = dataset.index(poly.bounds[2], poly.bounds[3])
    row1 = max(row1 + tile_size, 0)
    row1 = min(row1, mndwi.shape[0])
    row2 = max(row2 - tile_size, 0)
    row2 = min(row2, mndwi.shape[0])
    col1 = max(col1 - tile_size, 0)
    col1 = min(col1, mndwi.shape[1])
    col2 = max(col2 + tile_size, 0)
    col2 = min(col2, mndwi.shape[1])
    rst_arr = np.zeros(np.shape(mndwi))
    shapes = ((geom, value) for geom, value in zip([poly], [1]))
    rasterized_poly = features.rasterize(shapes=shapes, fill=0, out=rst_arr, transform=dataset.transform)[row2:row1, col1:col2]
    mndwi_small = mndwi[row2:row1, col1:col2].copy()

    im_boundary = Polygon([dataset.xy(0, 0), dataset.xy(0, mndwi.shape[1]), dataset.xy(mndwi.shape[0], mndwi.shape[1]), dataset.xy(mndwi.shape[0], 0)])
    geoms = split(im_boundary, line)
    areas = [geom.area for geom in geoms.geoms]
    corner_poly = geoms.geoms[np.argmin(areas)]
    rst_arr = np.zeros(np.shape(mndwi))
    shapes = ((geom, value) for geom, value in zip([corner_poly], [1]))
    rasterized_corner = features.rasterize(shapes=shapes, fill=0, out=rst_arr, transform=dataset.transform)[row2:row1, col1:col2]
    mndwi_small[rasterized_corner == 1] = 1

    mndwi_small_dist = ndimage.distance_transform_edt(mndwi_small)

    ch_map = np.zeros(np.shape(mndwi_small))
    ch_map[rasterized_corner == 1] = 1
    row = np.where(rasterized_poly)[0]
    col = np.where(rasterized_poly)[1]
    for i in trange(len(row)):
        if col[i] < mndwi_small.shape[1] and row[i] < mndwi_small.shape[0]:
            w = mndwi_small_dist[row[i], col[i]]  # distance to closest channel bank at current location
            if w <= 5000:
                pad = int(w) + 10
                tile = np.ones((pad * 2, pad * 2))
                tile[pad, pad] = 0
                tile = ndimage.distance_transform_edt(tile)
                tile[tile >= w] = 0  # needed to avoid issues with narrow channels
                tile[tile > 0] = 1
                r1 = max(0, row[i] - pad)
                r2 = min(row[i] + pad, mndwi_small.shape[0])
                c1 = max(0, col[i] - pad)
                c2 = min(col[i] + pad, mndwi_small.shape[1])
                tr1 = max(0, pad - row[i])
                tr2 = min(2 * pad, pad + mndwi_small.shape[0] - row[i])
                tc1 = max(0, pad - col[i])
                tc2 = min(2 * pad, pad + mndwi_small.shape[1] - col[i])
                ch_map[r1:r2, c1:c2] = np.maximum(tile[tr1:tr2, tc1:tc2], ch_map[r1:r2, c1:c2])
    contours = find_contours(ch_map, 0.5)
    contour_lengths = [len(contour) for contour in contours]
    if contour_lengths:
        ind = np.argmax(np.array(contour_lengths))
        x = contours[ind][:, 1]
        y = contours[ind][:, 0]
        x_utm = dataset.xy(row2 + np.array(y), col1 + np.array(x))[0]
        y_utm = dataset.xy(row2 + np.array(y), col1 + np.array(x))[1]
        # extend line so that it intersects the image boundary:
        a1, b1 = getExtrapolatedLine((x_utm[1], y_utm[1]), (x_utm[0], y_utm[0]), 20)  # use the first two points
        a2, b2 = getExtrapolatedLine((x_utm[-2], y_utm[-2]), (x_utm[-1], y_utm[-1]), 20)  # use the last two points
        xcoords = np.hstack((b1[0], x_utm, b2[0]))
        ycoords = np.hstack((b1[1], y_utm, b2[1]))
        line = LineString(np.vstack((xcoords, ycoords)).T)
        polys = split(im_boundary, line)
        areas = [geom.area for geom in polys.geoms]
        ch_mouth_poly = polys.geoms[np.argmin(areas)]
        x_utm = ch_mouth_poly.exterior.xy[0]
        y_utm = ch_mouth_poly.exterior.xy[1]
    else:
        x_utm = []
        y_utm = []
    return x_utm, y_utm, ch_map

def filter_outlier_paths(rivers, outlier_threshold=2.0, min_overlap_ratio=0.3, 
                       resample_points=100, plot_analysis=False):
    """
    Filter out main paths that are spatial outliers compared to the consensus.
    
    This method identifies centerlines that deviate significantly from the overall
    spatial trend while preserving shorter paths that follow the same general route.
    
    Parameters
    ----------
    rivers : list
        List of River instances with processed main paths
    outlier_threshold : float, optional
        Standard deviation threshold for outlier detection (default 2.0)
    min_overlap_ratio : float, optional
        Minimum overlap ratio with consensus path to be considered valid (default 0.3)
    resample_points : int, optional
        Number of points to resample each path to for comparison (default 100)
    plot_analysis : bool, optional
        Whether to plot the analysis results (default False)
        
    Returns
    -------
    filtered_rivers : list
        List of rivers with non-outlier paths
    outlier_rivers : list
        List of rivers identified as outliers
    analysis_results : dict
        Dictionary with analysis metrics
    """
    
    print(f"🔍 Analyzing {len(rivers)} river paths for spatial outliers...")
    
    # Extract main channel coordinates from all rivers
    valid_rivers = []
    all_paths = []
    
    for river in rivers:
        try:
            if hasattr(river, '_D_primal') and river._D_primal is not None:
                main_channel_coords = river._D_primal.graph.get('main_channel_cl_coords', None)
                if main_channel_coords is not None and len(main_channel_coords) > 10:
                    valid_rivers.append(river)
                    all_paths.append(main_channel_coords)
                else:
                    print(f"⚠️ Skipping river {getattr(river, 'scene_id', 'unknown')}: No valid main channel coords")
            else:
                print(f"⚠️ Skipping river {getattr(river, 'scene_id', 'unknown')}: No processed graph")
        except Exception as e:
            print(f"⚠️ Error processing river {getattr(river, 'scene_id', 'unknown')}: {e}")
    
    if len(valid_rivers) < 3:
        print("❌ Need at least 3 valid rivers for outlier detection")
        return rivers, [], {}
    
    print(f"✅ Found {len(valid_rivers)} rivers with valid main paths")
    
    # Find the overall bounding box for all paths
    all_x = np.concatenate([path[:, 0] for path in all_paths])
    all_y = np.concatenate([path[:, 1] for path in all_paths])
    
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    
    print(f"📏 Overall bounds: X=[{x_min:.0f}, {x_max:.0f}], Y=[{y_min:.0f}, {y_max:.0f}]")
    
    # Create a common reference line (approximate river direction)
    # Use the longest path as reference or create a synthetic one
    path_lengths = [len(path) for path in all_paths]
    longest_idx = np.argmax(path_lengths)
    reference_path = all_paths[longest_idx]
    
    print(f"📐 Using path from river {getattr(valid_rivers[longest_idx], 'scene_id', longest_idx)} as reference")
    
    # Resample all paths to common coordinate system
    resampled_paths = []
    path_metrics = []
    
    for i, (river, path) in enumerate(zip(valid_rivers, all_paths)):
        try:
            # Calculate cumulative distance along path
            distances = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
            distances = np.insert(distances, 0, 0)  # Add starting point
            
            # Create interpolation functions
            if len(distances) > 1 and distances[-1] > 0:
                fx = interp1d(distances, path[:, 0], kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
                fy = interp1d(distances, path[:, 1], kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
                
                # Resample to common number of points
                common_distances = np.linspace(0, distances[-1], resample_points)
                resampled_x = fx(common_distances)
                resampled_y = fy(common_distances)
                
                resampled_path = np.column_stack([resampled_x, resampled_y])
                resampled_paths.append(resampled_path)
                
                # Calculate metrics
                path_length = distances[-1]
                path_metrics.append({
                    'river_idx': i,
                    'river': river,
                    'length': path_length,
                    'n_points': len(path),
                    'resampled_path': resampled_path
                })
            else:
                print(f"⚠️ Skipping river {getattr(river, 'scene_id', i)}: Invalid path distances")
                
        except Exception as e:
            print(f"⚠️ Error resampling river {getattr(river, 'scene_id', i)}: {e}")
    
    if len(resampled_paths) < 3:
        print("❌ Not enough valid resampled paths for analysis")
        return rivers, [], {}
    
    print(f"✅ Successfully resampled {len(resampled_paths)} paths")
    
    # Calculate consensus path (median coordinates at each point)
    resampled_array = np.array(resampled_paths)  # Shape: (n_rivers, n_points, 2)
    consensus_path = np.median(resampled_array, axis=0)  # Shape: (n_points, 2)
    
    # Calculate deviations from consensus
    deviations = []
    for i, resampled_path in enumerate(resampled_paths):
        # Calculate point-wise distances to consensus
        point_distances = np.sqrt(np.sum((resampled_path - consensus_path)**2, axis=1))
        
        # Use different metrics for deviation
        mean_deviation = np.mean(point_distances)
        max_deviation = np.max(point_distances)
        median_deviation = np.median(point_distances)
        
        # Calculate overlap with consensus (how much of the path is close to consensus)
        close_points = np.sum(point_distances < np.std(point_distances)) / len(point_distances)
        
        deviations.append({
            'river_idx': path_metrics[i]['river_idx'],
            'river': path_metrics[i]['river'],
            'mean_deviation': mean_deviation,
            'max_deviation': max_deviation,
            'median_deviation': median_deviation,
            'overlap_ratio': close_points,
            'length': path_metrics[i]['length']
        })
    
    # Identify outliers using multiple criteria
    mean_devs = [d['mean_deviation'] for d in deviations]
    overlap_ratios = [d['overlap_ratio'] for d in deviations]
    
    # Calculate thresholds
    mean_dev_threshold = np.mean(mean_devs) + outlier_threshold * np.std(mean_devs)
    
    # Classify rivers
    filtered_rivers = []
    outlier_rivers = []
    
    for dev in deviations:
        is_outlier = (
            dev['mean_deviation'] > mean_dev_threshold or 
            dev['overlap_ratio'] < min_overlap_ratio
        )
        
        if is_outlier:
            outlier_rivers.append(dev['river'])
            print(f"🚫 Outlier: {getattr(dev['river'], 'scene_id', 'unknown')} "
                  f"(dev: {dev['mean_deviation']:.1f}m, overlap: {dev['overlap_ratio']:.2f})")
        else:
            filtered_rivers.append(dev['river'])
            print(f"✅ Keep: {getattr(dev['river'], 'scene_id', 'unknown')} "
                  f"(dev: {dev['mean_deviation']:.1f}m, overlap: {dev['overlap_ratio']:.2f})")
    
    # Analysis results
    analysis_results = {
        'n_input': len(rivers),
        'n_valid': len(valid_rivers),
        'n_filtered': len(filtered_rivers),
        'n_outliers': len(outlier_rivers),
        'mean_deviation_threshold': mean_dev_threshold,
        'min_overlap_threshold': min_overlap_ratio,
        'consensus_path': consensus_path,
        'deviations': deviations
    }
    
    print(f"\n📊 Filtering Results:")
    print(f"   Input rivers: {analysis_results['n_input']}")
    print(f"   Valid paths: {analysis_results['n_valid']}")
    print(f"   Kept rivers: {analysis_results['n_filtered']}")
    print(f"   Outlier rivers: {analysis_results['n_outliers']}")
    print(f"   Mean deviation threshold: {mean_dev_threshold:.1f}m")
    
    # Optional plotting
    if plot_analysis:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: All paths with consensus
        ax1 = axes[0, 0]
        for i, (river, path) in enumerate(zip(valid_rivers, all_paths)):
            color = 'red' if river in outlier_rivers else 'blue'
            alpha = 0.3 if river in outlier_rivers else 0.7
            ax1.plot(path[:, 0], path[:, 1], color=color, alpha=alpha, linewidth=1)
        
        ax1.plot(consensus_path[:, 0], consensus_path[:, 1], 'black', linewidth=3, label='Consensus')
        ax1.set_title('All Paths (Red=Outliers, Blue=Kept)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Deviation vs Overlap
        ax2 = axes[0, 1]
        colors = ['red' if d['river'] in outlier_rivers else 'blue' for d in deviations]
        ax2.scatter([d['mean_deviation'] for d in deviations], 
                   [d['overlap_ratio'] for d in deviations], c=colors, alpha=0.7)
        ax2.axvline(mean_dev_threshold, color='red', linestyle='--', label=f'Dev threshold: {mean_dev_threshold:.1f}m')
        ax2.axhline(min_overlap_ratio, color='red', linestyle='--', label=f'Overlap threshold: {min_overlap_ratio:.2f}')
        ax2.set_xlabel('Mean Deviation (m)')
        ax2.set_ylabel('Overlap Ratio')
        ax2.set_title('Outlier Detection Criteria')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Filtered paths only
        ax3 = axes[1, 0]
        for river in filtered_rivers:
            river_idx = next(i for i, r in enumerate(valid_rivers) if r == river)
            path = all_paths[river_idx]
            ax3.plot(path[:, 0], path[:, 1], 'blue', alpha=0.7, linewidth=1)
        ax3.plot(consensus_path[:, 0], consensus_path[:, 1], 'black', linewidth=3, label='Consensus')
        ax3.set_title('Filtered Paths (Outliers Removed)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Deviation histogram
        ax4 = axes[1, 1]
        ax4.hist([d['mean_deviation'] for d in deviations], bins=20, alpha=0.7, color='skyblue')
        ax4.axvline(mean_dev_threshold, color='red', linestyle='--', label=f'Threshold: {mean_dev_threshold:.1f}m')
        ax4.set_xlabel('Mean Deviation (m)')
        ax4.set_ylabel('Count')
        ax4.set_title('Distribution of Path Deviations')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return filtered_rivers, outlier_rivers, analysis_results

def classify_confluences_and_splits(D_primal):
    """
    Classifies nodes in a primal graph.
    If the graph is directed, classifies nodes as 'confluence', 'split', etc.
    If the graph is undirected, classifies nodes as 'junction', 'terminal', etc.

    Args:
        D_primal (nx.Graph): The centerline graph. Can be directed or undirected.

    Returns:
        dict: A dictionary mapping each node ID to its classification string.
    """
    classifications = {}
    is_directed = isinstance(D_primal, nx.DiGraph)

    for node in D_primal.nodes():
        if is_directed:
            in_degree = D_primal.in_degree(node)
            out_degree = D_primal.out_degree(node)

            if in_degree >= 2 and out_degree == 1:
                classifications[node] = 'confluence'
            elif in_degree == 1 and out_degree >= 2:
                classifications[node] = 'split'
            elif in_degree == 0 and out_degree > 0:
                classifications[node] = 'source'
            elif in_degree > 0 and out_degree == 0:
                classifications[node] = 'sink'
            elif in_degree == 1 and out_degree == 1:
                classifications[node] = 'channel'
            else:
                classifications[node] = 'other'
        else:  # Undirected graph
            degree = D_primal.degree(node)
            if degree >= 3:
                classifications[node] = 'junction'
            elif degree == 2:
                classifications[node] = 'channel'
            elif degree == 1:
                classifications[node] = 'terminal'
            else:
                classifications[node] = 'isolated'
            
    return classifications