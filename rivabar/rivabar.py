import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import Normalize
from scipy import ndimage
from scipy.signal import savgol_filter
import scipy.interpolate
from scipy.interpolate import CubicSpline
from tqdm import tqdm, trange
import pandas as pd
from skimage.morphology import skeletonize, remove_small_holes
from skimage.measure import label, regionprops_table, find_contours
import sknw # https://github.com/Image-Py/sknw
import networkx as nx
from sklearn.neighbors import KDTree
import rasterio
from rasterio.plot import adjust_band
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, MultiLineString, GeometryCollection
from shapely.geometry.polygon import InteriorRingSequence
from shapely.ops import polygonize_full, split, linemerge, nearest_points, unary_union
from libpysal import weights
import geopandas
import momepy
from itertools import combinations, permutations
import pickle
import warnings
from datetime import datetime
from scipy.spatial import ConvexHull

def convert_to_utm(x, y, left_utm_x, upper_utm_y, delta_x, delta_y):
    """
    Convert coordinates from pixel coordinates to UTM (Universal Transverse Mercator) coordinates.

    Args:
        x (array): The x-coordinates in the original projection system.
        y (array): The y-coordinates in the original projection system.
        left_utm_x (float): The x-coordinate of the leftmost point in the UTM coordinate system.
        upper_utm_y (float): The y-coordinate of the uppermost point in the UTM coordinate system.
        delta_x (float): The change in x-coordinate per unit distance in the UTM coordinate system.
        delta_y (float): The change in y-coordinate per unit distance in the UTM coordinate system.

    Returns:
        x_utm: the UTM x-coordinates corresponding to the input x-coordinates.
        y_utm: the UTM y-coordinates corresponding to the input y-coordinates.
    """
    x_utm = left_utm_x + 0.5*delta_x + x*delta_x 
    y_utm = upper_utm_y + 0.5*delta_y + y*delta_y 
    return x_utm, y_utm

def insert_node(graph, start_ind, end_ind, left_utm_x, upper_utm_y, delta_x, delta_y, start_x, start_y):
    """
    Inserts a new node into a graph, connecting two existing nodes with an edge.
    The new node will be located on an existing edge and will be as close as possible to (start_x, start_y).
    Updates the coordinates of the new node based on the given pixel coordinates and converts them to UTM coordinates.

    Parameters:
    - graph (networkx.Graph): The graph object to insert the new node into.
    - start_ind (int): The index of the starting node.
    - end_ind (int): The index of the ending node.
    - left_utm_x (float): The x-coordinate of the leftmost point in the UTM coordinate system.
    - upper_utm_y (float): The y-coordinate of the uppermost point in the UTM coordinate system.
    - delta_x (float): The change in x-coordinate per unit distance in the UTM coordinate system.
    - delta_y (float): The change in y-coordinate per unit distance in the UTM coordinate system.
    - start_x (float): The x-coordinate of the starting point in pixel coordinates.
    - start_y (float): The y-coordinate of the starting point in pixel coordinates.

    Returns:
    - graph (networkx.Graph): The updated graph with the new node and edges.
    - node (int): The index of the newly inserted node.
    """
    x = np.array(list(graph[start_ind][end_ind][0]['pts'][:, 1])) # x pixel coordinates of the edge
    y = np.array(list(graph[start_ind][end_ind][0]['pts'][:, 0])) # y pixel coordinates of the edge
    x, y = convert_to_utm(x, y, left_utm_x, upper_utm_y, delta_x, delta_y)
    tree = KDTree(np.vstack((x, y)).T)
    # index of point in edge geometry that is closest to (start_x, start_y):
    edge_ind = tree.query(np.reshape([start_x, start_y], (1, -1)))[1][0][0] 
    node = max(list(graph.nodes)) + 1 # index for new node
    # add the new node:
    graph.add_node(node, pts = graph[start_ind][end_ind][0]['pts'][edge_ind],
               o = graph[start_ind][end_ind][0]['pts'][edge_ind].astype('float'))
    # now we add the new edges:
    x = list(graph[start_ind][end_ind][0]['pts'][:, 1]) # x pixel coordinates of the old edge
    y = list(graph[start_ind][end_ind][0]['pts'][:, 0]) # y pixel coordinates of the old edge
    point1 = np.array([x[0], y[0]]) # coordinates of the first point on the old edge
    point2 = np.array([x[-1], y[-1]]) # coordinates of the last point on the old edge
    # coordinates of the start node of the old edge:
    point3 = np.array([graph.nodes()[start_ind]['o'][1], graph.nodes()[start_ind]['o'][0]])
    # if the start node is closer to the first point of the edge:
    if np.linalg.norm(point1 - point3) < np.linalg.norm(point2 - point3):
        graph.add_edge(start_ind, node, pts = graph[start_ind][end_ind][0]['pts'][:edge_ind])
        graph.add_edge(node, end_ind, pts = graph[start_ind][end_ind][0]['pts'][edge_ind:])
    else:
        graph.add_edge(start_ind, node, pts = graph[start_ind][end_ind][0]['pts'][edge_ind:])
        graph.add_edge(node, end_ind, pts = graph[start_ind][end_ind][0]['pts'][:edge_ind])
    graph.remove_edge(start_ind, end_ind) # remove the original edge
    return graph, node

def normalized_difference(b1, b2):
    """
    Calculate the normalized difference between two bands.

    This function computes the normalized difference index, a common
    calculation in remote sensing (e.g., NDVI, NDWI). It handles cases
    where both input bands are zero by setting the corresponding output
    pixels to NaN to avoid division by zero and undefined results.

    Parameters
    ----------
    b1 : numpy.ndarray
        The first input band (e.g., Near Infrared for NDVI).
        Should be a 2D NumPy array of numerical type.
    b2 : numpy.ndarray
        The second input band (e.g., Red for NDVI).
        Should be a 2D NumPy array of the same shape and numerical type as b1.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array containing the normalized difference values.
        The values typically range from -1 to 1. Pixels where both
        input bands were zero will be NaN.
    """
    band1 = np.where((b1==0) & (b2==0), np.nan, b1)
    band2 = np.where((b1==0) & (b2==0), np.nan, b2)
    return (band1 - band2) / (band1 + band2)

def find_graph_edges_close_to_start_and_end_points(graph, start_x, start_y, end_x, end_y, left_utm_x, upper_utm_y, delta_x, delta_y):
    """
    Find graph edges that are closest to specified start and end points.
    
    This function identifies the edges in a graph that are closest to given start and end points
    in UTM coordinates. It converts pixel coordinates to UTM, builds a KD-tree for efficient
    nearest neighbor search, and returns the node indices of the edges closest to the input points.
    
    Parameters
    ----------
    graph : networkx.Graph
        The graph containing edges with 'pts' attributes. Comes from skeletonization.
    start_x : float
        The x-coordinate (UTM) of the start point.
    start_y : float
        The y-coordinate (UTM) of the start point.
    end_x : float
        The x-coordinate (UTM) of the end point.
    end_y : float
        The y-coordinate (UTM) of the end point.
    left_utm_x : float
        The UTM x-coordinate of the left edge of the raster.
    upper_utm_y : float
        The UTM y-coordinate of the upper edge of the raster.
    delta_x : float
        The pixel width in UTM coordinates.
    delta_y : float
        The pixel height in UTM coordinates.
        
    Returns
    -------
    start_ind1 : int
        The source node index of the edge closest to the start point.
    end_ind1 : int
        The target node index of the edge closest to the start point.
    start_ind2 : int
        The source node index of the edge closest to the end point.
    end_ind2 : int
        The target node index of the edge closest to the end point.
    """
    # find reasonable start and end points on graph edges
    edge_utm_xs = []
    edge_utm_ys = []
    ss = []
    es = []    
    for s, e, d in graph.edges:
        x = np.array(list(graph[s][e][0]['pts'][:, 1]))
        y = np.array(list(graph[s][e][0]['pts'][:, 0]))
        x, y = convert_to_utm(x, y, left_utm_x, upper_utm_y, delta_x, delta_y)
        edge_utm_xs += list(x)
        edge_utm_ys += list(y)
        ss += [s] * len(x)
        es += [e] * len(x)
    tree = KDTree(np.vstack((edge_utm_xs, edge_utm_ys)).T)
    start_ind = tree.query(np.reshape([start_x, start_y], (1, -1)))[1][0][0]
    end_ind = tree.query(np.reshape([end_x, end_y], (1, -1)))[1][0][0]
    # start_dist = tree.query(np.reshape([start_x, start_y], (1, -1)))[0][0][0]
    # end_dist = tree.query(np.reshape([end_x, end_y], (1, -1)))[0][0][0]
    start_ind1 = ss[start_ind]
    end_ind1 = es[start_ind]
    start_ind2 = ss[end_ind]
    end_ind2 = es[end_ind]
    return start_ind1, end_ind1, start_ind2, end_ind2

def get_rid_of_extra_lines_at_beginning_and_end(G_primal, x1, y1, left_utm_x, upper_utm_y, delta_x, delta_y):
    """
    Remove extra lines at the beginning and end of a graph by finding the closest edge to a given point.
    
    Parameters
    ----------
    G_primal : networkx.Graph
        The graph containing edges with 'geometry' attributes.
    x1 : numpy.ndarray or float
        The x-coordinate(s) in pixel coordinates to be converted to UTM.
    y1 : numpy.ndarray or float
        The y-coordinate(s) in pixel coordinates to be converted to UTM.
    left_utm_x : float
        The UTM x-coordinate of the left edge of the raster.
    upper_utm_y : float
        The UTM y-coordinate of the upper edge of the raster.
    delta_x : float
        The pixel width in UTM coordinates.
    delta_y : float
        The pixel height in UTM coordinates.
        
    Returns
    -------
    G_primal : networkx.Graph
        The modified graph with extra lines removed.
    node : int
        The node index that is closest to the input point (x1, y1).
    """
    x1, y1 = convert_to_utm(x1, y1, left_utm_x, upper_utm_y, delta_x, delta_y)
    # collect UTM coordinates for edges in G_primal:
    edge_utm_xs = []
    edge_utm_ys = []
    ss = []
    es = []    
    for s, e, d in G_primal.edges:
        x = np.array(list(G_primal[s][e][0]['geometry'].xy[0]))
        y = np.array(list(G_primal[s][e][0]['geometry'].xy[1]))
        edge_utm_xs += list(x)
        edge_utm_ys += list(y)
        ss += [s] * len(x)
        es += [e] * len(x)
    # create KDTree for edge coordinates:
    tree = KDTree(np.vstack((edge_utm_xs, edge_utm_ys)).T)
    # find the edge that is closest to (x1, y1):
    si = tree.query(np.reshape([x1, y1], (1, -1)))[1][0][0]
    start_ind = ss[si]
    end_ind = es[si]
    # coordinates of the closest edge:
    x = np.array(list(G_primal[start_ind][end_ind][0]['geometry'].xy[0]))
    y = np.array(list(G_primal[start_ind][end_ind][0]['geometry'].xy[1]))
    # KDTree for the coordinates of the closest edge:
    tree = KDTree(np.vstack((x, y)).T)
    # find point that is closest to (x1, y1):
    ind = tree.query(np.reshape([x1, y1], (1, -1)))[1][0][0]
    start_ind_x = G_primal.nodes()[start_ind]['geometry'].xy[0][0]
    start_ind_y = G_primal.nodes()[start_ind]['geometry'].xy[1][0]
    end_ind_x = G_primal.nodes()[end_ind]['geometry'].xy[0][0]
    end_ind_y = G_primal.nodes()[end_ind]['geometry'].xy[1][0]
    start_dist = np.sqrt((start_ind_x - x1)**2 + (start_ind_y - y1)**2)
    end_dist = np.sqrt((end_ind_x - x1)**2 + (end_ind_y - y1)**2)
    if len(x) > 2:
        if ind == 1: # (x1,y1) is the second point on the G_primal edge of interest
            point = Point(x1, y1)
            line = LineString(np.vstack((x[ind:], y[ind:])).T)
            if len(list(nx.neighbors(G_primal, start_ind)))==1 or start_ind == 0:
                G_primal.nodes()[start_ind]['geometry'] = point
                node = start_ind
            else:
                G_primal.nodes()[end_ind]['geometry'] = point
                node = end_ind
            G_primal[start_ind][end_ind][0]['geometry'] = line
        elif ind == 0: # (x1,y1) is the first point on the G_primal edge of interest
            if start_dist < end_dist:
                G_primal.remove_node(end_ind)
                node = start_ind
            else:
                G_primal.remove_node(start_ind)
                node = end_ind
        else: # otherwise (x1, y1) is the end point
            point = Point(x1, y1)
            line = LineString(np.vstack((x[:ind+1], y[:ind+1])).T)
            G_primal.nodes()[end_ind]['geometry'] = point
            G_primal[start_ind][end_ind][0]['geometry'] = line
            node = end_ind
    if len(x) == 2:
        if ind == 0:
            G_primal.remove_node(end_ind)
            node = start_ind
        if ind == 1:
            G_primal.remove_node(start_ind)
            node = end_ind
    return G_primal, node

def convert_to_uint8(channel):
    """
    Convert a normalized float array to uint8 format.
    
    This function takes an array with values between 0 and 1 and
    converts it to uint8 format by scaling values to the range 0-255.
    
    Parameters
    ----------
    channel : numpy.ndarray
        Input array with normalized values (typically between 0 and 1).
        
    Returns
    -------
    numpy.ndarray
        Array converted to uint8 data type with values scaled to 0-255 range.
    """
    channel8 = channel * 255
    channel8 = channel8.astype('uint8')
    return channel8

def process_band(dirname, fname, band_numbers):
    """
    Helper function to open and read a specific band.

    Args:
        dirname: Path to folder that conatins the Landsat TIF files.
        fname: Name of Landsat tile or individual TIF file.
        band_numbers: Band numbers.

    Returns:
        bands: The band data.
        dataset: The rasterio dataset.
    """
    bands = {}
    if fname[-4:] == '.TIF' or fname[-4:] == '.tif': # single tif file
        with rasterio.open(os.path.join(dirname, fname), 'r') as dataset:
            for band_number in tqdm(band_numbers):
                bands[band_number] = dataset.read(band_number)
    else: # multiple TIF files
        for band_number in tqdm(band_numbers):
            with rasterio.open(os.path.join(dirname, fname, f'{fname}_B{band_number}.TIF')) as dataset:
                bands[band_number] = dataset.read(1)
    return bands, dataset

def read_landsat_data(dirname, fname, mndwi_threshold=0.01):
    """
    Read Landsat data from multiple TIF files and perform various operations.

    Args:
        dirname (str): Directory name where the TIF files are located.
        fname (str): File name prefix.
        mndwi_threshold (float): Threshold value for MNDWI.

    Returns:
        tuple: A tuple containing the processed image, MNDWI mask, and transformation parameters.
    """
    # this probably should be modified as not all recent landsat file names start with LC08
    if fname[:4] == 'LC08': # landsat 8
        band_numbers = [3, 6, 2, 4]
    else: # landsat 4 and 5
        band_numbers = [3, 5, 2, 1]
    bands, dataset = process_band(dirname, fname, band_numbers)

    if fname[:4] == 'LC08': # landsat 8
        rgb = np.stack([bands[4], bands[3], bands[2]], axis=-1)
        rgb_norm = adjust_band(rgb)
    else: # landsat 4 and 5
        rgb = np.stack([bands[3], bands[2], bands[1]], axis=-1)
        rgb_norm = adjust_band(rgb)

    left_utm_x = dataset.transform[2]
    upper_utm_y = dataset.transform[5]
    delta_x = dataset.transform[0]
    delta_y = dataset.transform[4]
    nxpix = rgb.shape[1]
    nypix = rgb.shape[0]
    right_utm_x = left_utm_x + delta_x*nxpix
    lower_utm_y = upper_utm_y + delta_y*nypix

    # compute true color image:
    R, G, B = cv2.split(rgb_norm)
    R8 = convert_to_uint8(R)
    G8 = convert_to_uint8(G)
    B8 = convert_to_uint8(B)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    equ = cv2.merge((clahe.apply(R8), clahe.apply(G8), clahe.apply(B8)))

    # compute mndwi:
    if fname[:4] == 'LC08': # landsat 8
        mndwi = normalized_difference(bands[3], bands[6])
    else: # landsat 4 and 5
        mndwi = normalized_difference(bands[2], bands[5])
    mndwi[mndwi > mndwi_threshold] = 1
    mndwi[mndwi != 1] = 0

    return equ, mndwi, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, delta_x, delta_y

def read_water_index(dirname, fname, mndwi_threshold):
    """
    Read a water index raster file and apply thresholding.
    
    This function opens a water index raster file, applies a threshold to create
    a binary water mask, and extracts the geospatial information.
    
    Parameters
    ----------
    dirname : str
        Directory path where the water index file is located.
    fname : str
        Filename of the water index raster.
    mndwi_threshold : float or str
        Threshold value for the water index. Values above this threshold
        will be set to 1 (water), others to 0 (non-water).
        
    Returns
    -------
    mndwi : numpy.ndarray
        Binary water mask where 1 represents water and 0 represents non-water.
    left_utm_x : float
        The UTM x-coordinate of the left edge of the raster.
    upper_utm_y : float
        The UTM y-coordinate of the upper edge of the raster.
    right_utm_x : float
        The UTM x-coordinate of the right edge of the raster.
    lower_utm_y : float
        The UTM y-coordinate of the lower edge of the raster.
    """
    with rasterio.open(dirname + fname) as dataset:
        mndwi = dataset.read(1)
        mndwi[mndwi > float(mndwi_threshold)] = 1
        mndwi[mndwi != 1] = 0
        left_utm_x = dataset.transform[2]
        upper_utm_y = dataset.transform[5]
        delta_x = dataset.transform[0]
        delta_y = dataset.transform[4]
    nxpix = mndwi.shape[1]
    nypix = mndwi.shape[0]
    right_utm_x = left_utm_x + delta_x*nxpix
    lower_utm_y = upper_utm_y + delta_y*nypix
    return mndwi, left_utm_x, upper_utm_y, right_utm_x, lower_utm_y

def create_mndwi(dirname, fname, file_type, mndwi_threshold=0.01, delete_pixels_polys=False, small_hole_threshold=64, remove_smaller_components=True, solidity_filter=False):
    """
    Create a Modified Normalized Difference Water Index (MNDWI) binary mask from input data.
    
    This function processes either a water index raster or Landsat data to create a binary
    water mask. It can filter out small holes, remove small components, and apply solidity
    filtering to improve the water mask quality.
    
    Parameters
    ----------
    dirname : str
        Directory path where the input file is located.
    fname : str
        Filename of the input raster.
    file_type : str
        Type of input file. Can be 'water_index' for a single water index raster
        or another value for Landsat TIF files.
    mndwi_threshold : float or str, optional
        Threshold value for the water index. Values above this threshold
        will be set to 1 (water), others to 0 (non-water). Default is 0.01.
    delete_pixels_polys : bool or list, optional
        List of polygons to mask out from the water index (e.g., bridges),
        or False to skip this step. Default is False.
    small_hole_threshold : int, optional
        Minimum size (in pixels) of holes to keep in the water mask.
        Smaller holes will be filled. Default is 64.
    remove_smaller_components : bool, optional
        Whether to remove small disconnected water bodies, keeping only
        the largest component. Default is True.
    solidity_filter : bool, optional
        Whether to filter objects based on solidity (area/convex hull area).
        Objects with solidity < 0.2 will be removed. Default is False.
        
    Returns
    -------
    mndwi : numpy.ndarray
        Binary water mask where 1 represents water and 0 represents non-water.
    left_utm_x : float
        The UTM x-coordinate of the left edge of the raster.
    upper_utm_y : float
        The UTM y-coordinate of the upper edge of the raster.
    right_utm_x : float
        The UTM x-coordinate of the right edge of the raster.
    lower_utm_y : float
        The UTM y-coordinate of the lower edge of the raster.
    delta_x : float
        The pixel width in UTM coordinates.
    delta_y : float
        The pixel height in UTM coordinates.
    dataset : rasterio.DatasetReader
        The opened raster dataset with metadata.
    """
    if file_type == 'water_index': # single water index raster
        with rasterio.open(dirname + fname) as dataset:
            mndwi = dataset.read(1)
            mndwi[mndwi > float(mndwi_threshold)] = 1
            mndwi[mndwi != 1] = 0
            left_utm_x = dataset.transform[2]
            upper_utm_y = dataset.transform[5]
            delta_x = dataset.transform[0]
            delta_y = dataset.transform[4]
        nxpix = mndwi.shape[1]
        nypix = mndwi.shape[0]
        right_utm_x = left_utm_x + delta_x*nxpix
        lower_utm_y = upper_utm_y + delta_y*nypix
    else: # single Landsat TIF file or multiple Landsat TIF files
        print('reading Landsat data')
        if type(mndwi_threshold) == str: # make sure that mndwi_threshold is a float
            mndwi_threshold = float(mndwi_threshold)
        equ, mndwi, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, delta_x, delta_y = read_landsat_data(dirname, fname, mndwi_threshold = mndwi_threshold)
    if delete_pixels_polys: # set pixels to zero in areas defined by polygons (e.g., bridges)
        rst_arr = mndwi.astype('uint32').copy()
        shapes = ((geom, value) for geom, value in zip(delete_pixels_polys, np.ones((len(delete_pixels_polys),))))
        mndwi = features.rasterize(shapes=shapes, fill=0, out=rst_arr, transform=dataset.transform)
    if max(np.shape(mndwi)) > 2**16//2:
        print('maximum dimension of input image needs to be smaller than 32768!')
        return None
    print('removing small holes')
    mndwi = remove_small_holes(mndwi.astype('bool'), small_hole_threshold) # remove small bars / islands
    print('removing small components') # remove small components (= lakes) from water index image
    if remove_smaller_components:
        mndwi_labels = label(mndwi)
        rp = regionprops_table(mndwi_labels, properties=['label', 'area', 'solidity'])
        df = pd.DataFrame(rp)
        mndwi = np.zeros(np.shape(mndwi))
        if solidity_filter: # remove objects with low solidity - these are most likely not part of the river
            df = df.sort_values('area', ascending=False)
            for ind in df.index[df['solidity'] < 0.2]:
                mndwi[mndwi_labels == ind+1] = 1
        else: # remove small objects
            df = df.sort_values('area', ascending=False, ignore_index=True)
            mndwi[mndwi_labels == df.loc[0, 'label']] = 1 # set the largest object to 1 (the rest is 0)
    return mndwi, left_utm_x, upper_utm_y, right_utm_x, lower_utm_y, delta_x, delta_y, dataset
    
def extract_centerline(fname, dirname, start_x, start_y, end_x, end_y, file_type, 
    ch_belt_smooth_factor=1e9, remove_smaller_components=False, delete_pixels_polys=False, 
    ch_belt_half_width=2000, mndwi_threshold=0.01, small_hole_threshold=64, plot_D_primal=False,
    min_g_primal_length=100000, solidity_filter=True, radius=50, min_main_path_length=2000, 
    flip_outlier_edges=False, check_edges=False):
    """
    Extract channel centrelines and banks from a georeferenced image.

    Parameters
    ----------
    fname : str
        Filename.
    dirname : str
        Name of directory.
    start_x : float
        Estimate of UTM x coordinate of start point.
    start_y : float
        Estimate of UTM y coordinate of start point.
    end_x : float
        Estimate of UTM x coordinate of end point.
    end_y : float
        Estimate of UTM y coordinate of end point.
    file_type : str
        Type of file used as input; it can be 'water_index' or something else if using Landsat bands.
    ch_belt_smooth_factor : float, optional
        Smoothing factor for getting a channel belt centerline (default is 1e9).
    remove_smaller_components : bool, optional
        Remove small components from water index image if true (default is False).
    delete_pixels_polys : bool or list, optional
        List of polygons or 'False'; set pixels to zero in areas defined by polygons (default is False).
    ch_belt_half_width : int, optional
        Half channel belt width in pixels, used to define area of interest along channel (default is 2000).
    mndwi_threshold : float, optional
        Value for thresholding water index image (default is 0.01).
    small_hole_threshold : int, optional
        Remove holes in water index image that are smaller than this value (in pixels); affects size of islands detected (default is 64).
    plot_D_primal : bool, optional
        Plot the directed multigraph (default is False).
    min_g_primal_length : int, optional
        Minimum length of centerline graph, in meters, measured over all edges; processing is stopped if graph is not long enough (default is 100000).
    solidity_filter : bool, optional
        If 'True', objects in the water index image that have a lower solidity than 0.2 will be removed; good for cleaning up complex water bodies but can be a problem when those bodies are connected to the river (e.g., parts of the Amazon) (default is True).
    radius : int, optional
        Defines the graph neighborhood of the main path in which nodes will be included. Default is 50; this might need to be increased when working with complex networks (default is 50).
    min_main_path_length : int, optional
        Minimum length of the main path, in meters (default is 2000).
    flip_outlier_edges : bool, optional
        Flip outlier edges (default is False). Should be set to 'True' for complex networks (e.g., Lena Delta, Brahmaputra).
    check_edges : bool, optional
        Check edges around each island for consistency in the direction of the flow (default is False). Should be set to 'True' for multithread rivers (e.g., Brahmaputra), but not for meandering rivers or for networks with unrealistic centerline orientations (e.g., Lena Delta).

    Returns
    -------
    D_primal : networkx.DiGraph
        Directed multigraph that describes the channel centerline network and tries to capture the flow directions.
    G_rook : networkx.Graph
        Rook neighborhood graph of the 'islands' that make up the channel belt; has only two elements (the two banks) if there are no islands / bars.
    G_primal : networkx.Graph
        Undirected multigraph that describes the channel centerline network; only returned for QC / testing purposes.
    mndwi : numpy.ndarray
        Water index image that was used in the processing.
    dataset : rasterio.DatasetReader
        Rasterio dataset.
    left_utm_x : float
        UTM x coordinate of the left edge of the image.
    right_utm_x : float
        UTM x coordinate of the right edge of the image.
    lower_utm_y : float
        UTM y coordinate of the lower edge of the image.
    upper_utm_y : float
        UTM y coordinate of the upper edge of the image.
    xs : numpy.ndarray
        x coordinates (UTM) of the smoothed channel belt centerline.
    ys : numpy.ndarray
        y coordinates (UTM) of the smoothed channel belt centerline.
    """
    mndwi, left_utm_x, upper_utm_y, right_utm_x, lower_utm_y, delta_x, delta_y, dataset = create_mndwi(dirname=dirname, fname=fname, file_type=file_type,\
                        mndwi_threshold=mndwi_threshold, delete_pixels_polys=delete_pixels_polys, small_hole_threshold=small_hole_threshold,\
                        remove_smaller_components=remove_smaller_components, solidity_filter=solidity_filter)
    if mndwi is None:
        return [], [], [], [], [], [], [], [], [], [], []

    print('running skeletonization')
    if np.nanmax(mndwi) != np.nanmin(mndwi):
        try:
            skeleton = skeletonize(mndwi) 
        except: # sometimes the default (Zhang) algorithm gives an error and the Lee method doesn't
            print('Skeletonization using the Zhang method failed; switching to the Lee method.')
            try:
                skeleton = skeletonize(mndwi, method='lee')
            except:
                print('skeletonization failed')
                return [], [], [], [], [], [], [], [], [], [], []
    else:
        print('skeletonization failed')
        return [], [], [], [], [], [], [], [], [], [], []

    print('building graph from skeleton')
    graph = sknw.build_sknw(skeleton, multi=True) # build multigraph from skeleton
    print('finding reasonable starting and ending points on graph edges')
    if type(start_x) == str:
        start_x = float(start_x)
    if type(start_y) == str:
        start_y = float(start_y)
    if type(end_x) == str:
        end_x = float(end_x)
    if type(end_y) == str:
        end_y = float(end_y)
    start_ind1, end_ind1, start_ind2, end_ind2 = \
        find_graph_edges_close_to_start_and_end_points(graph, start_x, start_y, end_x, end_y, \
                                    left_utm_x, upper_utm_y, delta_x, delta_y)
    try:
        graph, start_ind = insert_node(graph, start_ind1, end_ind1, left_utm_x, upper_utm_y, delta_x, delta_y, start_x, start_y)
        graph, end_ind = insert_node(graph, start_ind2, end_ind2, left_utm_x, upper_utm_y, delta_x, delta_y, end_x, end_y)
    except:
        print('could not find start and end points')
        return [], [], [], [], [], [], [], [], [], [], []
    
    # path between first and last nodes:
    try:
        path = nx.shortest_path(graph, source=start_ind, target=end_ind, weight='weight')
    except: # if there is no path between start_ind and end_ind
        print('no path between start_ind and end_ind')
        nodes1 = nx.shortest_path(graph, start_ind).keys()
        nodes2 = nx.shortest_path(graph, end_ind).keys()
        dists = []
        large_graph_inds = []
        if len(nodes1) <= len(nodes2):
            for node in nodes1:
                dist, closest_node = find_pixel_distance_between_nodes_and_other_node(graph, nodes2, node)
                large_graph_inds.append(closest_node)
                dists.append(dist)
            ind = np.where(dists == np.min(dists))[0][0]
            node1 = list(nodes1)[ind]
            node2 = large_graph_inds[ind]
        else:
            for node in nodes2:
                dist, closest_node = find_pixel_distance_between_nodes_and_other_node(graph, nodes1, node)
                large_graph_inds.append(closest_node)
                dists.append(dist)
            ind = np.where(dists == np.min(dists))[0][0]
            node1 = list(nodes2)[ind]
            node2 = large_graph_inds[ind]
        if dists[ind] <= 200: # distance less than 200 pixels
            img = np.array(mndwi * 255, dtype = np.uint8)
            cv2.line(img, (int(graph.nodes()[node1]['o'][1]), int(graph.nodes()[node1]['o'][0])), 
                (int(graph.nodes()[node2]['o'][1]), int(graph.nodes()[node2]['o'][0])), (255,255,255), 3)
            img = img//255
            mndwi = img.astype('bool')
            skeleton = skeletonize(mndwi) # skeletonization
            graph = sknw.build_sknw(skeleton, multi=True) # build multigraph from skeleton
            start_ind1, end_ind1, start_ind2, end_ind2 = \
                find_graph_edges_close_to_start_and_end_points(graph, start_x, start_y, end_x, end_y, \
                                                left_utm_x, upper_utm_y, delta_x, delta_y)
            graph, start_ind = insert_node(graph, start_ind1, end_ind1, left_utm_x, upper_utm_y, delta_x, delta_y, start_x, start_y)
            graph, end_ind = insert_node(graph, start_ind2, end_ind2, left_utm_x, upper_utm_y, delta_x, delta_y, end_x, end_y)
            try:
                path = nx.shortest_path(graph, source=start_ind, target=end_ind, weight='weight')
            except:
                print('still no path between start and end points')
                if len(nodes1) >= len(nodes2): # choose the longer path
                    path = nodes1
                    main_node = start_ind
                    other_node = end_ind
                else:
                    path = nodes2
                    main_node = end_ind
                    other_node = start_ind
                # need to recreate 'path' as a shortest path between two nodes on the main river channel:
                dist, other_node = find_distance_between_nodes_and_other_node(graph, path, other_node, 
                                                        left_utm_x, upper_utm_y, delta_x, delta_y)    
                start_ind_short_path = main_node
                end_ind_short_path = other_node
                try:
                    path = nx.shortest_path(graph, source=start_ind_short_path, target=end_ind_short_path, weight='weight')
                    if len(path) < 10:
                        print('path is too short!')
                        comps = sorted(nx.connected_components(graph), key=len, reverse=True) # list of connected components
                        dists = []
                        start_inds = []
                        end_inds = []
                        for i in range(5): # find the component that is closest to the start and end nodes
                            start_ind_dist, closest_node = find_distance_between_nodes_and_other_node(graph, comps[i], start_ind, 
                                                                                        left_utm_x, upper_utm_y, delta_x, delta_y)
                            start_inds.append(closest_node)
                            end_ind_dist, closest_node = find_distance_between_nodes_and_other_node(graph, comps[i], end_ind, 
                                                                                        left_utm_x, upper_utm_y, delta_x, delta_y)
                            end_inds.append(closest_node)
                            dists.append(0.5*start_ind_dist + 0.5*end_ind_dist)
                        comp_ind = np.where(dists == np.min(dists))[0][0]
                        start_ind = start_inds[comp_ind]
                        end_ind = end_inds[comp_ind]
                        path = nx.shortest_path(graph, source=start_ind, target=end_ind, weight='weight')
                    else:
                        start_ind = start_ind_short_path
                        end_ind = end_ind_short_path
                except:
                    print('could not find path')
                    return [], [], [], [], [], [], [], [], [], [], []
        else:
            print('distance too large between nodes1 and nodes2')
            if len(nodes1) >= len(nodes2): # choose the longer path
                path = nodes1
                main_node = start_ind
                other_node = end_ind
            else:
                path = nodes2
                main_node = end_ind
                other_node = start_ind
            # need to recreate 'path' as a shortest path between two nodes on the main river channel:
            dist, other_node = find_distance_between_nodes_and_other_node(graph, path, other_node, 
                                                    left_utm_x, upper_utm_y, delta_x, delta_y)    
            start_ind_short_path = main_node
            end_ind_short_path = other_node
            try:
                path = nx.shortest_path(graph, source=start_ind_short_path, target=end_ind_short_path, weight='weight')
                if len(path) < 10:
                    print('path is too short!')
                    comps = sorted(nx.connected_components(graph), key=len, reverse=True) # list of connected components
                    dists = []
                    start_inds = []
                    end_inds = []
                    for i in range(5): # find the component that is closest to the start and end nodes
                        start_ind_dist, closest_node = find_distance_between_nodes_and_other_node(graph, comps[i], start_ind, 
                                                                                    left_utm_x, upper_utm_y, delta_x, delta_y)
                        start_inds.append(closest_node)
                        end_ind_dist, closest_node = find_distance_between_nodes_and_other_node(graph, comps[i], end_ind, 
                                                                                    left_utm_x, upper_utm_y, delta_x, delta_y)
                        end_inds.append(closest_node)
                        dists.append(0.5*start_ind_dist + 0.5*end_ind_dist)
                    comp_ind = np.where(dists == np.min(dists))[0][0]
                    start_ind = start_inds[comp_ind]
                    end_ind = end_inds[comp_ind]
                    path = nx.shortest_path(graph, source=start_ind, target=end_ind, weight='weight')
                else:
                    start_ind = start_ind_short_path
                    end_ind = end_ind_short_path
            except:
                print('could not find path')
                return [], [], [], [], [], [], [], [], [], [], []
        
    print('finding nodes that are within a certain radius of the path')
    nodes = [] 
    for node in tqdm(path):
        test = nx.generators.ego_graph(graph, node, radius=int(radius))
        for n in test:
            if n not in nodes:
                nodes.append(n)
                
    # create smaller graph and remove dead ends from it:
    G = graph.subgraph(nodes).copy()
    G = remove_dead_ends(G, start_ind, end_ind)
    
    # remove edges that link a node to itself:
    edges_to_be_removed = []
    for node in G.nodes:
        for neighbor in list(nx.neighbors(G, node)):
            if neighbor == node:
                edges_to_be_removed.append((node, neighbor))
    G.remove_edges_from(edges_to_be_removed)
            
    # extend edges to the nodes:
    for s,e,d in G.edges: 
        for i in range(len(G[s][e])):
            x, y = extend_cline(G, s, e, i)
            G[s][e][i]['pts'] = np.vstack((y,x)).T

    # this fixes some problems that arise from superposition of edges from the skeletonization:
    for main_node in G.nodes:
        edge_sets = []
        edge_inds = []
        count = 0
        for node in list(nx.neighbors(G, main_node)):
            for i in range(len(G[main_node][node])):
                edge_inds.append((main_node, node, i, count))
                edge_set = set()
                for j in range(G[main_node][node][i]['pts'].shape[0]):
                    edge_set.add(tuple(G[main_node][node][i]['pts'][j,:]))
                edge_sets.append(edge_set)
                count += 1
        for combination in combinations(edge_inds, 2):
            common_points = edge_sets[combination[0][3]].intersection(edge_sets[combination[1][3]])
            if len(common_points) > 1:
                common_points.remove(tuple(G.nodes()[combination[0][0]]['o']))
                if not np.array_equal(G.nodes()[combination[0][1]]['o'], np.array(list(common_points)[0])):
                    p1 = G.nodes()[main_node]['o']
                    p2 = G.nodes()[combination[0][1]]['o']
                    p3 = np.array(list(common_points)[0])
                    l2 = np.sum((p1-p2)**2) # distance between p1 and p2
                    t = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / l2))
                    projection = p1 + t * (p2 - p1)
                    if np.linalg.norm(G[main_node][combination[0][1]][combination[0][2]]['pts'][0] - p1) \
                    < np.linalg.norm(G[main_node][combination[0][1]][combination[0][2]]['pts'][-1] - p1):
                        G[main_node][combination[0][1]][combination[0][2]]['pts'][1] = projection
                    else:
                        G[main_node][combination[0][1]][combination[0][2]]['pts'][-2] = projection

    # remove edges that link a node to itself:
    edges_to_be_removed = []
    for node in G.nodes:
        for neighbor in list(nx.neighbors(G, node)):
            if neighbor == node:
                edges_to_be_removed.append((node, neighbor))
    G.remove_edges_from(edges_to_be_removed)

    # find main path:
    path = nx.shortest_path(G, source = start_ind, target = end_ind)
    xcoords = []
    ycoords = []
    for i in range(len(path)-1):
        s = path[i]
        e = path[i+1]
        d = 0
        x = list(G[s][e][d]['pts'][:, 1])
        y = list(G[s][e][d]['pts'][:, 0])
        if len(x) > 0:
            # some of the path segments need to be flipped:
            point1 = np.array([x[0], y[0]])
            point2 = np.array([x[-1], y[-1]])
            point3 = np.array([G.nodes()[path[i]]['o'][1], G.nodes()[path[i]]['o'][0]])
            if np.linalg.norm(point1 - point3) < np.linalg.norm(point2 - point3):
                xcoords += x
                ycoords += y
            else:
                xcoords += x[::-1]
                ycoords += y[::-1] 
    xcoords = np.array(xcoords)
    ycoords = np.array(ycoords)
    xcoords_sm = savgol_filter(xcoords, 21, 3)
    ycoords_sm = savgol_filter(ycoords, 21, 3)
    xcoords_sm, ycoords_sm = resample_and_smooth(xcoords_sm, ycoords_sm, 25, 100000)
    main_path = LineString(np.vstack((xcoords, ycoords)).T)
    if main_path.length < min_main_path_length:  # this should be a parameter
        print('main path is too short')
        return [], [], [], [], [], [], [], [], [], [], []
    
    # creating list of linestrings for polygonization:
    clines = []
    for s,e,d in tqdm(G.edges): # extend edges to the nodes
        for i in range(len(G[s][e])):
            x = G[s][e][i]['pts'][:,1]
            y = G[s][e][i]['pts'][:,0]
            if len(x) > 1:
                line = LineString(np.vstack((x, y)).T)
                if not line.is_simple: # this is needed because a small number of edges are messy
                    x = G[s][e][i]['pts'][:,1][1:-1]
                    y = G[s][e][i]['pts'][:,0][1:-1]
                    line = LineString(np.vstack((x, y)).T)
                if line not in clines:
                    clines.append(line)            
    cline_polys = list(polygonize_full(clines)) # polygonize the centerline network
    
    if len(cline_polys[0].geoms) > 0 or len(cline_polys[3].geoms) > 0:
        # create polygons from polygonization result:
        polys = []
        for poly in cline_polys[0].geoms:
            if len(poly.interiors) > 0: # sometimes (rarely) there are holes and we want to get rid of them
                poly = Polygon(poly.exterior)
            polys.append(poly)
        for poly in cline_polys[3].geoms:
            poly = Polygon(poly)
            if poly.is_valid:
                polys.append(poly)
            else:
                poly = poly.buffer(0)
                polys.append(poly)
        polys_to_remove = []
        for i in range(len(polys)):
            for j in range(len(polys)):
                if polys[j].contains(polys[i]) and i != j:
                    polys_to_remove.append(i)
        if  len(polys_to_remove) > 0:
            new_polys = []
            for i in range(len(polys)):
                if i not in polys_to_remove:
                    new_polys.append(polys[i])
            polys = new_polys

        # get rid of polygons that are disconnected from the main fairway:
        gdf = geopandas.GeoDataFrame(polys, columns = ['geometry'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rook = weights.Rook.from_dataframe(gdf, use_index=False)
        main_inds = []
        for comp_label in range(np.max(rook.component_labels) + 1):
            inds = np.where(rook.component_labels == comp_label)[0]
            for ind in inds:
                if polys[ind].touches(main_path):
                    main_inds += list(inds)
                    break # if one polygon touches the main path, the whole component becomes part of the main fairway
        main_polys = [] # list for polygons that are part of the main fairway
        for ind in main_inds:
            main_polys.append(polys[ind])
        gdf = geopandas.GeoDataFrame(main_polys, columns = ['geometry'])
    else:
        gdf = geopandas.GeoDataFrame([], columns = ['geometry'])

    # channel belt centerline:
    xs, ys = resample_and_smooth(xcoords_sm, ycoords_sm, 25, float(ch_belt_smooth_factor))

    ch_belt_cl = LineString(np.vstack((xs, ys)).T) # smoothed channel belt centerline
    ch_belt_poly = ch_belt_cl.buffer(ch_belt_half_width) # channel belt polygon

    # these are needed so that later we can get rid of the extra line segments at the beginning and end of G_primal:
    xcoords1 = xcoords[0]
    xcoords2 = xcoords[-1]
    ycoords1 = ycoords[0]
    ycoords2 = ycoords[-1]

    # lengthen the channel centerline so that it intersects the channel belt polygon:
    ratio = 2*ch_belt_half_width/(np.sqrt((xs[1]-xs[0])**2 + (ys[1]-ys[0])**2))
    a1, b1 = getExtrapolatedLine((xs[1], ys[1]), (xs[0], ys[0]), ratio) # use the first two points
    ratio = 2*ch_belt_half_width/(np.sqrt((xs[-1]-xs[-2])**2 + (ys[-1]-ys[-2])**2))
    a2, b2 = getExtrapolatedLine((xs[-2], ys[-2]), (xs[-1], ys[-1]), ratio) # use the last two points
    xcoords = np.hstack((b1[0], xcoords, b2[0]))
    ycoords = np.hstack((b1[1], ycoords, b2[1]))
    main_path = LineString(np.vstack((xcoords, ycoords)).T) # extended main path (not smoothed)

    # create polygons for the outer boundaries of the channel belt:
    main_polys = split(ch_belt_poly, main_path) # split the channel belt polygon with the main path
    poly1 = main_polys.geoms[0]
    poly2 = main_polys.geoms[1]

    # trim down the boundary polygons so that they do not overlap with the channel belt polygons:
    if len(gdf) > 0:
        chb_poly = gdf.geometry.unary_union
        poly1_diff = poly1.difference(chb_poly)
        if type(poly1_diff) == MultiPolygon:
            poly1_diff = max(poly1_diff.geoms, key=lambda a: a.area)
        geoms_to_be_deleted = []
        if len(poly1_diff.interiors) > 0:
            for ind in range(len(poly1_diff.interiors)):
                count = 0
                for poly in gdf['geometry']:
                    if Polygon(poly1_diff.interiors[ind]).area == poly.area:
                        geoms_to_be_deleted.append(count)
                    count += 1
        poly1 = Polygon(poly1_diff.exterior)
        poly2_diff = poly2.difference(chb_poly)
        if type(poly2_diff) == MultiPolygon:
            poly2_diff = max(poly2_diff.geoms, key=lambda a: a.area)
        if len(poly2_diff.interiors) > 0:
            for ind in range(len(poly2_diff.interiors)):
                count = 0
                for poly in gdf['geometry']:
                    if Polygon(poly2_diff.interiors[ind]).area == poly.area:
                        geoms_to_be_deleted.append(count)
                    count += 1
        poly2 = Polygon(poly2_diff.exterior)
        gdf.drop(geoms_to_be_deleted, axis=0, inplace=True)

    # only keep the largest polygons from the resulting multipolygons:
    if type(poly1) != Polygon:
        poly1 = max(poly1.geoms, key=lambda a: a.area)
    if type(poly2) != Polygon:
        poly2 = max(poly2.geoms, key=lambda a: a.area)

    x, y = convert_to_utm(np.array(poly1.exterior.xy[0]), np.array(poly1.exterior.xy[1]),
                    left_utm_x, upper_utm_y, delta_x, delta_y)
    poly1_utm = Polygon(np.vstack((x, y)).T)
    im_boundary = Polygon([dataset.xy(0,0), dataset.xy(0,mndwi.shape[1]), dataset.xy(mndwi.shape[0], mndwi.shape[1]), dataset.xy(mndwi.shape[0], 0)])
    poly1_utm = im_boundary.intersection(poly1_utm) # trim polygon to image size
    if type(poly1_utm) == MultiPolygon:
        poly_areas = []
        for geom in poly1_utm.geoms:
            poly_areas.append(geom.area)
        poly1_utm = poly1_utm.geoms[np.argmax(poly_areas)]
    if type(poly1_utm) == GeometryCollection:
        for i in range(len(poly1_utm.geoms)):
            count = 0
            if type(poly1_utm.geoms[i]) == Polygon:
                main_poly_ind = i
                count += 1
        if count > 1:
            print('there are more than one polygons in poly1_utm!')
        poly1_utm = poly1_utm.geoms[main_poly_ind]
    x, y = convert_to_utm(np.array(poly2.exterior.xy[0]), np.array(poly2.exterior.xy[1]),
                    left_utm_x, upper_utm_y, delta_x, delta_y)
    poly2_utm = Polygon(np.vstack((x, y)).T)
    poly2_utm = im_boundary.intersection(poly2_utm)
    if type(poly2_utm) == MultiPolygon:
        poly_areas = []
        for geom in poly2_utm.geoms:
            poly_areas.append(geom.area)
        poly2_utm = poly2_utm.geoms[np.argmax(poly_areas)]
    if type(poly2_utm) == GeometryCollection:
        for i in range(len(poly2_utm.geoms)):
            count = 0
            if type(poly2_utm.geoms[i]) == Polygon:
                main_poly_ind = i
                count += 1
        if count > 1:
            print('there are more than one polygons in poly2_utm!')
        poly2_utm = poly2_utm.geoms[main_poly_ind]

    utm_polys = [poly1_utm, poly2_utm]
    # creating centerline polygons:
    for poly in tqdm(gdf['geometry']): # this takes a while
        if type(poly) == Polygon: # there are some cases when these are not polygons, but I think those cases can be ignored
            x, y = convert_to_utm(np.array(poly.exterior.xy[0]), np.array(poly.exterior.xy[1]),
                        left_utm_x, upper_utm_y, delta_x, delta_y)
            poly_utm = Polygon(np.vstack((x, y)).T)
            if not poly_utm.is_valid:
                poly_utm = poly_utm.buffer(0)
            utm_polys.append(poly_utm)

    # create geopandas dataframe from UTM polygons:
    gdf2 = geopandas.GeoDataFrame(utm_polys, columns = ['geometry'])
    gdf2.set_crs(dataset.crs)

    print('creating linestrings for primal graph')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rook = weights.Rook.from_dataframe(gdf2, use_index=False)
    linestrings = []
    for poly_ind, neighb_dict in tqdm(rook):
        for key in neighb_dict:
            if gdf2['geometry'][poly_ind].intersects(gdf2['geometry'][key]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning) # ignore warnings about invalid intersections
                    line = gdf2['geometry'][poly_ind].intersection(gdf2['geometry'][key])
                if type(line) == LineString:
                    if line not in linestrings:
                        linestrings.append(line)
                elif type(line) == MultiLineString:
                    line = linemerge(line)
                    if type(line) == LineString:
                        if line not in linestrings:
                            linestrings.append(line)
                    if type(line) == MultiLineString:
                        for geom in line.geoms:
                            if type(geom) == LineString:
                                if geom not in linestrings:
                                    linestrings.append(geom)
                else:
                    temp_lines = []
                    if type(line) != Polygon:
                        for geom in line.geoms:
                            if type(geom) == LineString:
                                temp_lines.append(geom)
                    line = linemerge(MultiLineString(temp_lines))
                    if type(line) == LineString:
                        if line not in linestrings:
                            linestrings.append(line)
                    if type(line) == MultiLineString:
                        for geom in line.geoms:
                            if type(geom) == LineString:
                                if geom not in linestrings:
                                    linestrings.append(geom)

    # create primal graph from linestrings:
    if len(linestrings) > 0:                        
        gdf_LS = geopandas.GeoDataFrame(linestrings, columns = ['geometry'])
        G_primal = momepy.gdf_to_nx(gdf_LS, approach="primal")
        degree = dict(nx.degree(G_primal))
        nx.set_node_attributes(G_primal, degree, 'degree')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            nodes, edges, sw = momepy.nx_to_gdf(G_primal, points=True, lines=True, spatial_weights=True)
    else:
        print('primal graph creation failed')
        return [], [], [], [], [], [], [], [], [], [], []

    # create a better, leaner primal graph (some edges are duplicated and they have to be removed)
    unique_node_pairs = edges[['node_start', 'node_end']][edges[['node_start', 'node_end']].duplicated() == False]
    inds_to_be_removed = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning) # ignore warnings about invalid intersections
        for i in range(len(unique_node_pairs)):
            inds = edges[['node_start', 'node_end']][edges[['node_start', 'node_end']] \
                                                    == unique_node_pairs.iloc[i]].dropna().index
            for combination in combinations(inds, 2):
                if edges['geometry'].loc[combination[0]].intersects(edges['geometry'].loc[combination[1]]):
                    intersection = edges['geometry'].loc[combination[0]].intersection(edges['geometry'].loc[combination[1]])
                    if type(intersection) == LineString:
                        inds_to_be_removed.append(combination[1])
                    elif len(intersection.geoms) > 2:
                        inds_to_be_removed.append(combination[1])
    edges2 = edges.drop(inds_to_be_removed)
    G_primal = nx.from_pandas_edgelist(edges2, source='node_start', target='node_end', edge_attr=['geometry', 'mm_len'], 
                        create_using=nx.MultiGraph, edge_key=None)
    for node in list(G_primal.nodes):
        G_primal.nodes()[node]['geometry'] = nodes['geometry'].iloc[node]

    G_length = 0
    for s,e,d in G_primal.edges:
        G_length += G_primal[s][e][d]['geometry'].length
    if G_length < min_g_primal_length:
        print('primal graph is too short')
        return [], [], [], [], [], [], [], [], [], [], []

    G_primal, primal_start_ind = get_rid_of_extra_lines_at_beginning_and_end(G_primal, xcoords1, ycoords1, left_utm_x, upper_utm_y, delta_x, delta_y)
    G_primal, primal_end_ind = get_rid_of_extra_lines_at_beginning_and_end(G_primal, xcoords2, ycoords2, left_utm_x, upper_utm_y, delta_x, delta_y)
    G_primal = remove_dead_ends(G_primal, primal_start_ind, primal_end_ind)
    print('start and end nodes in G_primal:')
    print(primal_start_ind, primal_end_ind)

    if len(G_primal) < 2:
        print('G_primal only has one node!')
        return [], [], [], [], [], [], [], [], [], [], []
    
    print('getting bank coordinates for the two main banks')
    utm_coords = []
    x1_poly, y1_poly, ch_map = get_bank_coords(poly1_utm, mndwi, dataset, timer=True)
    x2_poly, y2_poly, ch_map = get_bank_coords(poly2_utm, mndwi, dataset, timer=True)
    # need to lengthen the banklines so that they intersect the main centerline polygons:
    ratio = ch_belt_half_width*delta_x/(np.sqrt((x1_poly[1]-x1_poly[0])**2 + (y1_poly[1]-y1_poly[0])**2))
    a1, b1 = getExtrapolatedLine((x1_poly[1], y1_poly[1]), (x1_poly[0], y1_poly[0]), float(ratio))
    a2, b2 = getExtrapolatedLine((x1_poly[-2], y1_poly[-2]), (x1_poly[-1], y1_poly[-1]), float(ratio))
    x1_poly = np.hstack((b1[0], x1_poly, b2[0]))
    y1_poly = np.hstack((b1[1], y1_poly, b2[1]))
    a1, b1 = getExtrapolatedLine((x2_poly[1], y2_poly[1]), (x2_poly[0], y2_poly[0]), float(ratio))
    a2, b2 = getExtrapolatedLine((x2_poly[-2], y2_poly[-2]), (x2_poly[-1], y2_poly[-1]), float(ratio))
    x2_poly = np.hstack((b1[0], x2_poly, b2[0]))
    y2_poly = np.hstack((b1[1], y2_poly, b2[1]))

    poly1_split = split(poly1_utm, LineString(np.vstack((x1_poly, y1_poly)).T))
    areas = []
    for geom in poly1_split.geoms:
        areas.append(geom.area)
    x1_poly = poly1_split.geoms[np.argmax(areas)].exterior.xy[0]
    y1_poly = poly1_split.geoms[np.argmax(areas)].exterior.xy[1]

    poly2_split = split(poly2_utm, LineString(np.vstack((x2_poly, y2_poly)).T))
    areas = []
    for geom in poly2_split.geoms:
        areas.append(geom.area)
    x2_poly = poly2_split.geoms[np.argmax(areas)].exterior.xy[0]
    y2_poly = poly2_split.geoms[np.argmax(areas)].exterior.xy[1]

    utm_coords.append(np.vstack((x1_poly, y1_poly)).T)
    utm_coords.append(np.vstack((x2_poly, y2_poly)).T)

    print('getting bank coordinates for the rest of the islands')
    for i in trange(2, len(gdf2['geometry'])): # this takes a while
        if type(gdf2['geometry'].iloc[i]) == Polygon:
            x, y, ch_map = get_bank_coords(gdf2['geometry'].iloc[i], mndwi, dataset)
            if len(x) > 0:
                if x[-1] != x[0] or y[-1] != y[0]: # sometimes the contour needs to be closed
                    x = np.hstack((x, x[0]))
                    y = np.hstack((y, y[0]))
            utm_coords.append(np.vstack((x, y)).T)
        else:
            X = np.array([])
            Y = np.array([])
            for geom in gdf2['geometry'].iloc[i].geoms:
                x, y, ch_map = get_bank_coords(geom, mndwi, dataset)
                if len(x) > 0:
                    if x[-1] != x[0] or y[-1] != y[0]: # sometimes the contour needs to be closed
                        x = np.hstack((x, x[0]))
                        y = np.hstack((y, y[0]))
                X = np.hstack((X, x))
                Y = np.hstack((Y, y))
            utm_coords.append(np.vstack((X, Y)).T)

    # store results in a Rook graph:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rook = weights.Rook.from_dataframe(gdf2, use_index=False)
    centroids = np.column_stack((gdf2.centroid.x, gdf2.centroid.y))
    G_rook = rook.to_networkx()
    nx.set_node_attributes(G_rook, [], "bank_polygon")
    nx.set_node_attributes(G_rook, [], "centroid")
    nx.set_node_attributes(G_rook, [], "cl_polygon")
    for i in range(len(G_rook)):
        if len(utm_coords[i]) > 0:
            # if LineString(utm_coords[i]).is_closed:
            G_rook.nodes[i]["bank_polygon"] = Polygon(utm_coords[i])
            # else:
                # G_rook.nodes[i]["bank_polygon"] = LineString(utm_coords[i])
            G_rook.nodes[i]["centroid"] = centroids[i]
            G_rook.nodes[i]["cl_polygon"] = gdf2['geometry'].iloc[i]

    print('setting half channel widths')
    set_half_channel_widths(G_primal, G_rook, dataset, mndwi)

    # create a directed multigraph based on G_primal:
    xs, ys = convert_to_utm(np.array(xs), np.array(ys), left_utm_x, upper_utm_y, delta_x, delta_y)
    # sometimes xs and ys need to be flipped (I have no idea why):
    start_point_dist = np.sqrt((xs[0]-start_x)**2 + (ys[0]-start_y)**2)
    end_point_dist = np.sqrt((xs[0]-end_x)**2 + (ys[0]-end_y)**2)
    if end_point_dist < start_point_dist:
        xs = xs[::-1]; ys = ys[::-1]

    print('creating directed graph')
    D_primal, source_nodes, sink_nodes = create_directed_multigraph(G_primal, G_rook, xs, ys, primal_start_ind, primal_end_ind, 
                                                                    flip_outlier_edges=flip_outlier_edges, check_edges=check_edges)

    print('getting bank coordinates for main channel')
    start_node, inds = find_start_node(D_primal)
    if start_node is not None:
        edge_path = traverse_multigraph(D_primal, start_node)
        D_primal.graph['main_path'] = edge_path # store main path as a graph attribute
        x, y, x_utm1, y_utm1, x_utm2, y_utm2 = get_bank_coords_for_main_channel(D_primal, mndwi, edge_path, dataset)
        D_primal.graph['main_channel_cl_coords'] = np.vstack((x, y)).T
        D_primal.graph['main_channel_bank1_coords'] = np.vstack((x_utm1, y_utm1)).T
        D_primal.graph['main_channel_bank2_coords'] = np.vstack((x_utm2, y_utm2)).T

    if plot_D_primal:
        # fig, ax = plot_directed_graph(D_primal, mndwi, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, edge_path=edge_path, arrow_width=50)
        # fig, ax = plt.subplots()
        fig, ax = plot_im_and_lines(mndwi, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y,
            G_rook, G_primal, smoothing=False, start_x=start_x, start_y=start_y, end_x=end_x,
        end_y=end_y, plot_lines=False)
        plot_graph_w_colors(D_primal, ax)

    D_primal.name = fname # use filename as the graph name
    G_rook.name = fname # use filename as the graph name

    return D_primal, G_rook, G_primal, mndwi, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, xs, ys

def remove_dead_ends(graph, start_node, end_node):
    """
    Remove dead-end nodes from a graph.

    This function iteratively removes nodes from the graph that are not the 
    start or end node and have fewer than two neighbors, which are considered 
    dead ends.

    Parameters
    ----------
    graph : networkx.Graph
        The input graph from which dead-end nodes will be removed.
    start_node : node
        The starting node of the graph that should not be removed.
    end_node : node
        The ending node of the graph that should not be removed.

    Returns
    -------
    graph : networkx.Graph
        The graph with dead-end nodes removed.
    """
    while True:
        nodes_to_be_removed = []
        for node in graph:
            if node != start_node and node != end_node:
                if len(list(graph.neighbors(node))) < 2:
                    nodes_to_be_removed.append(node)
        if len(nodes_to_be_removed) == 0:
            break
        graph.remove_nodes_from(nodes_to_be_removed) # delete nodes
    return graph

def find_distance_between_nodes_and_other_node(graph, nodes, other_node, left_utm_x, upper_utm_y, delta_x, delta_y):
    """
    Finds the distance between a set of nodes and another node in a graph.

    Parameters
    ----------
    graph : networkx.Graph
        The graph containing the nodes.
    nodes : list
        List of node identifiers to compare against `other_node`.
    other_node : int
        The node identifier to find the distance to.
    left_utm_x : float
        The UTM x-coordinate of the left boundary.
    upper_utm_y : float
        The UTM y-coordinate of the upper boundary.
    delta_x : float
        The change in x-coordinate for UTM conversion.
    delta_y : float
        The change in y-coordinate for UTM conversion.

    Returns
    -------
    dist : float
        The distance between the closest node in `nodes` and `other_node`.
    closest_node : int
        The identifier of the closest node in `nodes` to `other_node`.
    """
    xs = [] # node coordinates
    ys = []
    for node in nodes:
        xs.append(graph.nodes()[node]['o'][1])
        ys.append(graph.nodes()[node]['o'][0])
    xs, ys = convert_to_utm(np.array(xs), np.array(ys), left_utm_x, upper_utm_y, delta_x, delta_y)
    tree = KDTree(np.vstack((xs, ys)).T)
    x = graph.nodes()[other_node]['o'][1]
    y = graph.nodes()[other_node]['o'][0]
    x, y = convert_to_utm(x, y, left_utm_x, upper_utm_y, delta_x, delta_y)
    dist = tree.query(np.reshape([x, y], (1, -1)))[0][0][0]
    closest_node_ind = tree.query(np.reshape([x, y], (1, -1)))[1][0][0]
    closest_node = list(nodes)[closest_node_ind]
    return dist, closest_node

def resample_and_smooth(x, y, delta_s, smoothing_factor):
    """
    Resample and smooth a given set of points.

    Parameters
    ----------
    x : array_like
        The x-coordinates of the points.
    y : array_like
        The y-coordinates of the points.
    delta_s : float
        The desired spacing between resampled points.
    smoothing_factor : float
        The smoothing factor used in the spline fitting.

    Returns
    -------
    xs : ndarray
        The x-coordinates of the resampled and smoothed points.
    ys : ndarray
        The y-coordinates of the resampled and smoothed points.

    Notes
    -----
    This function uses a parametric spline representation to smooth and resample the input points.
    If the spline fitting fails, the original points are returned.
    """
    x = np.array(x)
    y = np.array(y)
    okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
    dx = np.diff(x[okay]); dy = np.diff(y[okay])      
    ds = np.sqrt(dx**2+dy**2)
    try:
        tck, u = scipy.interpolate.splprep([x[okay], y[okay]], s=smoothing_factor) # parametric spline representation of curve
    except:
        return x, y
    unew = np.linspace(0,1,1+int(sum(ds)/delta_s)) # vector for resampling
    out = scipy.interpolate.splev(unew,tck) # resampling
    xs = out[0]
    ys = out[1]
    return xs, ys

def smooth_line(x, y, spline_ds = 25, spline_smoothing = 10000, savgol_window = 21, savgol_poly_order = 3):
    """
    Smooths the given line data using Savitzky-Golay filter and spline resampling.

    Parameters
    ----------
    x : array_like
        The x-coordinates of the data points.
    y : array_like
        The y-coordinates of the data points.
    spline_ds : int, optional
        The distance between points in the resampled spline, by default 25.
    spline_smoothing : int, optional
        The smoothing factor for the spline, by default 10000.
    savgol_window : int, optional
        The length of the filter window for the Savitzky-Golay filter, by default 21.
    savgol_poly_order : int, optional
        The order of the polynomial used in the Savitzky-Golay filter, by default 3.

    Returns
    -------
    xs : ndarray
        The smoothed x-coordinates.
    ys : ndarray
        The smoothed y-coordinates.
    """
    x = savgol_filter(x, savgol_window, savgol_poly_order)
    y = savgol_filter(y, savgol_window, savgol_poly_order)
    xs, ys = resample_and_smooth(x, y, spline_ds, spline_smoothing)
    return xs, ys

def compute_s_distance(x, y):
    """
    Compute the first derivatives of a curve (centerline) and the cumulative distance along the curve.

    Parameters
    ----------
    x : array_like
        Cartesian x-coordinates of the curve.
    y : array_like
        Cartesian y-coordinates of the curve.

    Returns
    -------
    s : ndarray
        Cumulative distance along the curve.

    Notes
    -----
    The function calculates the first derivatives of the x and y coordinates using numpy's gradient function.
    It then computes the distances between consecutive points along the curve and the cumulative distance.
    """
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)   
    ds = np.sqrt(dx**2+dy**2)
    s = np.hstack((0, np.cumsum(ds[1:])))
    return s

def find_pixel_distance_between_nodes_and_other_node(graph, nodes, other_node):
    """
    Find the pixel distance between a set of nodes and another node in a graph.

    Parameters
    ----------
    graph : networkx.Graph
        The graph containing the nodes.
    nodes : list
        A list of node identifiers for which the distance is to be calculated.
    other_node : int
        The node identifier of the other node.

    Returns
    -------
    dist : float
        The pixel distance between the closest node in `nodes` and `other_node`.
    closest_node : int
        The identifier of the closest node in `nodes` to `other_node`.

    """
    xs = [] # node coordinates
    ys = []
    for node in nodes:
        xs.append(graph.nodes()[node]['o'][1])
        ys.append(graph.nodes()[node]['o'][0])
    tree = KDTree(np.vstack((xs, ys)).T)
    x = graph.nodes()[other_node]['o'][1]
    y = graph.nodes()[other_node]['o'][0]
    dist = tree.query(np.reshape([x, y], (1, -1)))[0][0][0]
    closest_node_ind = tree.query(np.reshape([x, y], (1, -1)))[1][0][0]
    closest_node = list(nodes)[closest_node_ind]
    return dist, closest_node

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def extend_cline(graph, s, e, d):
    """
    Extends the edge (= centerline) in a graph by adding the coordinates of its start and end nodes.
    Needed because the edge outputs from 'sknw' stop before reaching the nodes.

    Parameters
    ----------
    graph : networkx.Graph
        The graph containing the nodes and edges.
    s : int
        The start node identifier.
    e : int
        The end node identifier.
    d : int
        The edge direction or identifier.

    Returns
    -------
    x : numpy.ndarray
        The extended x-coordinates of the edge.
    y : numpy.ndarray
        The extended y-coordinates of the edge.
    """
    x = graph[s][e][d]['pts'][:,1] # edge coordinates
    y = graph[s][e][d]['pts'][:,0]
    xs = graph.nodes()[s]['o'][1] # start node coordinates
    ys = graph.nodes()[s]['o'][0]
    xe = graph.nodes()[e]['o'][1] # end node coordinates
    ye = graph.nodes()[e]['o'][0]
    if len(x) > 0:
        dist1 = np.linalg.norm(np.array([x[0], y[0]]) - np.array([xs, ys]))
        dist2 = np.linalg.norm(np.array([x[-1], y[-1]]) - np.array([xs, ys]))
        if dist1 <= dist2:
            x = np.hstack((xs, x, xe))
            y = np.hstack((ys, y, ye))
        else:
            x = np.hstack((xs, x[::-1], xe))
            y = np.hstack((ys, y[::-1], ye))
    return x, y

def getExtrapolatedLine(p1, p2, ratio):
    """
    Creates a line extrapolated in the p1->p2 direction.

    Parameters
    ----------
    p1 : tuple of float
        The starting point of the line (x1, y1).
    p2 : tuple of float
        The ending point of the line (x2, y2).
    ratio : float
        The ratio by which to extrapolate the line, relative to the distance
        between P1 and p2

    Returns
    -------
    tuple of tuple of float
        A tuple containing the starting point `a` and the extrapolated point `b`.
    """
    a = p1
    b = (p1[0]+ratio*(p2[0]-p1[0]), p1[1]+ratio*(p2[1]-p1[1]))
    return a, b

def get_bank_coords(poly, mndwi, dataset, timer=False):
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
    ch_map[rasterized_poly == 0] = 1
    ch_map = ~(ch_map.astype('bool'))
    contours = find_contours(ch_map, 0.5)
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

    return x_utm, y_utm, ch_map

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

def find_matching_indices(x1, y1, x2, y2):
    """
    Find the indices of coordinates in the first set that match coordinates in the second set.

    Parameters
    ----------
    x1 : list or array-like
        The x-coordinates of the first set.
    y1 : list or array-like
        The y-coordinates of the first set.
    x2 : list or array-like
        The x-coordinates of the second set.
    y2 : list or array-like
        The y-coordinates of the second set.

    Returns
    -------
    matching_indices : list
        A list of indices from the first set where the coordinates match those in the second set.
    """
    # Convert the coordinates into NumPy arrays
    coords1 = np.array(list(zip(x1, y1)))
    coords2 = np.array(list(zip(x2, y2)))
    # Create an empty list to store matching indices
    matching_indices = []
    # Use a dictionary for faster lookup of coordinates in the second set
    coords2_set = {tuple(coord): i for i, coord in enumerate(coords2)}
    # Iterate over the first set of coordinates
    for i, coord in enumerate(coords1):
        if tuple(coord) in coords2_set:
            # If the coordinate is in the second set, add the index to the list
            matching_indices.append(i)
    return matching_indices

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

def save_shapefiles(dirname, fname, G_rook, dataset, fname_add_on=''):
    """
    Save shapefiles for bank polygons and centerline polygons from a graph structure.

    Parameters
    ----------
    dirname : str
        The directory name where the shapefiles will be saved.
    fname : str
        The base filename for the shapefiles.
    G_rook : networkx.Graph
        A graph structure containing nodes with 'bank_polygon' and 'cl_polygon' geometries.
    dataset : geopandas.GeoDataFrame
        A GeoDataFrame containing the coordinate reference system (CRS) information.
    fname_add_on : str, optional
        An additional string to append to the filenames (default is '').

    Returns
    -------
    None
    """
    gs = geopandas.GeoSeries(G_rook.nodes()[0]['bank_polygon'])
    gs.crs = 'epsg:'+str(dataset.crs.to_epsg())
    gs.to_file(dirname+fname[:-4]+fname_add_on+'_rb.shp')
    gs = geopandas.GeoSeries(G_rook.nodes()[1]['bank_polygon'])
    gs.crs = 'epsg:'+str(dataset.crs.to_epsg())    #dataset.crs.data['init']
    gs.to_file(dirname + fname[:-4]+ fname_add_on+'_lb.shp')
    if len(G_rook) > 2:
        bank_polys = []
        for i in range(2, len(G_rook)):
            bank_polys.append(G_rook.nodes()[i]['bank_polygon'])
        gdf = geopandas.GeoDataFrame(bank_polys, columns = ['geometry'])
        gdf['area'] = gdf.area
        gdf['length'] = gdf.length
        gdf.crs = 'epsg:'+str(dataset.crs.to_epsg())
        gdf.to_file(dirname + fname[:-4]+fname_add_on+'_bank_polygons.shp')
        cl_polys = []
        for i in range(2, len(G_rook)):
            cl_polys.append(G_rook.nodes()[i]['cl_polygon'])
        gdf = geopandas.GeoDataFrame(cl_polys, columns = ['geometry'])
        gdf['area'] = gdf.area
        gdf['length'] = gdf.length
        gdf.crs = 'epsg:'+str(dataset.crs.to_epsg())
        gdf.to_file(dirname + fname[:-4]+fname_add_on+'_cl_polygons.shp')

def plot_im_and_lines(im, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, G_rook, 
                G_primal, plot_main_banklines=True, plot_lines=True, plot_image=True,
                smoothing=False, start_x=None, start_y=None, end_x=None, end_y=None):
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

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure.
    """
    fig, ax = plt.subplots()
    if plot_image:
        plt.imshow(im, extent = [left_utm_x, right_utm_x, lower_utm_y, upper_utm_y], cmap='gray', alpha=1)
    for i in range(2):
        if type(G_rook.nodes()[i]['bank_polygon']) == Polygon:
            x = G_rook.nodes()[i]['bank_polygon'].exterior.xy[0]
            y = G_rook.nodes()[i]['bank_polygon'].exterior.xy[1]
            if smoothing:
                ind1 = find_closest_point(start_x, start_y, np.vstack((x, y)).T)
                ind2 = find_closest_point(end_x, end_y, np.vstack((x, y)).T)
                if ind1 < ind2:
                    x = x[ind1:ind2]
                    y = y[ind1:ind2]
                else:
                    x = x[ind2:ind1]
                    y = y[ind2:ind1]                
        else:
            x = G_rook.nodes()[i]['bank_polygon'].xy[0]
            y = G_rook.nodes()[i]['bank_polygon'].xy[1]
        if smoothing:
            x, y = smooth_line(x, y, spline_ds = 100, spline_smoothing = 10000, savgol_window = min(31, len(x)), 
                                     savgol_poly_order = 3)
        if i == 0 and plot_main_banklines:
            plt.plot(x, y, color='tab:blue')
        if i == 1 and plot_main_banklines:
            plt.plot(x, y, color='tab:blue')
    for i in trange(2, len(G_rook.nodes)):
        x = G_rook.nodes()[i]['bank_polygon'].exterior.xy[0]
        y = G_rook.nodes()[i]['bank_polygon'].exterior.xy[1]
        if smoothing:
            if len(x) > 3:
                x, y = smooth_line(x, y, spline_ds = 25, spline_smoothing = 5000, savgol_window = min(11, len(x)), 
                                        savgol_poly_order = 3)
        plt.plot(x, y, color='tab:blue')
    if plot_lines:
        for s,e,d in tqdm(G_primal.edges):
            x = G_primal[s][e][d]['geometry'].xy[0]
            y = G_primal[s][e][d]['geometry'].xy[1]
            if smoothing:
                if len(x) > 3:
                    x, y = smooth_line(x, y, spline_ds = 25, spline_smoothing = 5000, savgol_window = min(11, len(x)), 
                                            savgol_poly_order = 3)
            plt.plot(x, y, 'k')
    return fig, ax

def extend_line(x, y, ratio):
    """
    Extend a line by extrapolating its endpoints.

    Parameters
    ----------
    x : array-like
        The x-coordinates of the points defining the line.
    y : array-like
        The y-coordinates of the points defining the line.
    ratio : float
        The ratio by which to extend the line at both ends.

    Returns
    -------
    line : LineString
        A LineString object representing the extended line.

    Notes
    -----
    This function uses the first two points and the last two points of the 
    input coordinates to extrapolate the line at both ends.
    """
    a1, b1 = getExtrapolatedLine((x[1], y[1]), (x[0], y[0]), ratio) # use the first two points
    a2, b2 = getExtrapolatedLine((x[-2], y[-2]), (x[-1], y[-1]), ratio) # use the last two points
    x = np.hstack((b1[0], x, b2[0]))
    y = np.hstack((b1[1], y, b2[1]))
    line = LineString(np.vstack((x, y)).T)
    return line

def smooth_banklines(G_rook, dataset, mndwi, save_smooth_lines=False, spline_smoothing=1000):
    """
    Smooth banklines and return them as a list.

    Parameters
    ----------
    G_rook : networkx.Graph
        Graph containing nodes with 'bank_polygon' attributes representing the banklines.
    dataset : rasterio.DatasetReader
        Dataset containing the spatial reference and transformation information.
    mndwi : numpy.ndarray
        Array representing the Modified Normalized Difference Water Index (MNDWI) values.
    save_smooth_lines : bool, optional
        If True, the smoothed banklines will be saved back to the graph nodes (default is False).
    spline_smoothing : int, optional
        Smoothing factor for the spline interpolation (default is 1000).

    Returns
    -------
    list of shapely.geometry.Polygon
        List of smoothed bankline polygons.
    """
    polys = []
    im_boundary = Polygon([dataset.xy(0,0), dataset.xy(0,mndwi.shape[1]), dataset.xy(mndwi.shape[0], mndwi.shape[1]), dataset.xy(mndwi.shape[0], 0)])
    for i in range(2):
        # first need to isolate line that can be / should be smoothed:
        other_side = im_boundary.buffer(-20000).difference(G_rook.nodes()[i]['bank_polygon'])
        line = other_side.intersection(G_rook.nodes()[i]['bank_polygon'])
        new_line = []
        for geom in line.geoms:
            if type(geom) == LineString:
                new_line.append(geom)
        line = linemerge(new_line)
        if type(line) != LineString:
            lengths = []
            for geom in line.geoms:
                lengths.append(geom.length)
            line = line.geoms[np.argmax(lengths)]
        # print('smoothing main bankline')    
        x1, y1 = smooth_line(line.xy[0], line.xy[1], spline_ds = 100, spline_smoothing = spline_smoothing, savgol_window = min(11, len(line.xy[0])), 
                                savgol_poly_order = 3)
        line = extend_line(x1, y1, 2000)
        geoms = split(im_boundary, line)
        if geoms.geoms[0].intersection(G_rook.nodes[i]["bank_polygon"]).area > geoms.geoms[1].intersection(G_rook.nodes[i]["bank_polygon"]).area:
            if save_smooth_lines:
                G_rook.nodes()[i]['bank_polygon'] = geoms.geoms[0]
            polys.append(geoms.geoms[0])
        else:
            if save_smooth_lines:
                G_rook.nodes()[i]['bank_polygon'] = geoms.geoms[1]
            polys.append(geoms.geoms[1])
    for i in range(2, len(G_rook.nodes)):
        x = G_rook.nodes()[i]['bank_polygon'].exterior.xy[0]
        y = G_rook.nodes()[i]['bank_polygon'].exterior.xy[1]
        x, y = smooth_line(x, y, spline_ds = 50, spline_smoothing = spline_smoothing, savgol_window = min(11, len(x)), 
                                savgol_poly_order = 3)
        if save_smooth_lines:
            G_rook.nodes()[i]['bank_polygon'] = Polygon(np.vstack((x, y)).T)
        polys.append(Polygon(np.vstack((x, y)).T))
    return polys

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

def create_channel_nw_polygon(G_rook, buffer=10, ch_mouth_poly=None, dataset=None):
    """
    Creates a polygon representing the channel network.

    Parameters
    ----------
    G_rook : networkx.Graph
        A graph where nodes contain 'bank_polygon' attributes representing the bank polygons.
    buffer : int, optional
        The buffer distance to apply around the bank polygons (default is 10).
    ch_mouth_poly : shapely.geometry.Polygon, optional
        A polygon representing the channel mouth to be subtracted from the channel network polygon (default is None).
    dataset : rasterio.io.DatasetReader, optional
        A dataset object used to get the boundary coordinates (default is None).
    mndwi : numpy.ndarray, optional
        An array representing the Modified Normalized Difference Water Index (default is None).

    Returns
    -------
    shapely.geometry.Polygon
        A polygon representing the channel network.
    """
    both_banks = G_rook.nodes()[0]['bank_polygon'].buffer(buffer).union(G_rook.nodes()[1]['bank_polygon'].buffer(buffer))
    if type(both_banks) == Polygon and len(both_banks.interiors) > 0:
        ch_belt_pieces = both_banks.interiors
    else:
        im_boundary = Polygon([dataset.xy(0,0), dataset.xy(0, dataset.shape[1]), dataset.xy(dataset.shape[0], dataset.shape[1]), dataset.xy(dataset.shape[0], 0)])
        ch_belt_pieces= im_boundary.buffer(-10).difference(G_rook.nodes()[0]['bank_polygon'].buffer(10).union(G_rook.nodes()[1]['bank_polygon'].buffer(10)))
    # print(type(ch_belt_pieces))
    if type(ch_belt_pieces)==InteriorRingSequence:
        temp = []
        for i in range(len(ch_belt_pieces)):
            temp.append(Polygon(ch_belt_pieces[i]))
        ch_belt_pieces = temp
        if len(ch_belt_pieces) > 0:
            ch_nw_poly = ch_belt_pieces[0].buffer(buffer)
            if len(ch_belt_pieces) > 1:
                for ch_belt_piece in ch_belt_pieces[1:]:
                    ch_nw_poly = ch_nw_poly.union(ch_belt_piece.buffer(buffer))
    elif type(ch_belt_pieces) == Polygon: # if the channel belt is a single polygon
        ch_nw_poly = ch_belt_pieces.buffer(buffer)
    elif type(ch_belt_pieces) == MultiPolygon:
        for geom in ch_belt_pieces.geoms:
            for i in range(2,len(G_rook.nodes())):
                if geom.contains(G_rook.nodes()[i]['bank_polygon']):
                    ch_nw_poly = geom.buffer(buffer)
                    break
    if type(ch_nw_poly) == MultiPolygon:
        areas = []
        for geom in ch_nw_poly.geoms:
            areas.append(geom.area)
        ch_nw_poly = ch_nw_poly.geoms[np.argmax(areas)]
    if len(G_rook) > 2: # if there are islands, add the bank polygons as holes
        holes = [] # create list of holes
        for node in range(2, len(G_rook)):
            holes.append(G_rook.nodes()[node]['bank_polygon'].exterior)
        ch_nw_poly = Polygon(ch_nw_poly.exterior, holes)
    if ch_mouth_poly:
        outer_polygon = Polygon(ch_nw_poly.exterior).difference(ch_mouth_poly.buffer(10)).buffer(0)
        smaller_polygons = []
        for geom in ch_nw_poly.interiors:
            if outer_polygon.contains(geom):
                smaller_polygons.append(Polygon(geom).buffer(0))
            if outer_polygon.overlaps(Polygon(geom)):
                part_to_be_removed = Polygon(geom).buffer(0).difference(outer_polygon)
                smaller_polygons.append(Polygon(geom).buffer(0).difference(part_to_be_removed))
        if type(outer_polygon) == MultiPolygon:
            areas = []
            for geom in outer_polygon.geoms:
                areas.append(geom.area)
            outer_polygon = outer_polygon.geoms[np.argmax(areas)]
        ch_nw_poly = Polygon(outer_polygon.exterior, [p.exterior for p in smaller_polygons])
    return ch_nw_poly

def convert_geographic_proj_to_utm(dirname, fname, dstCrs):
    """
    Converts a geographic projection raster to UTM projection.

    Parameters
    ----------
    dirname : str
        Directory name where the source raster file is located.
    fname : str
        Filename of the source raster file.
    dstCrs : dict or str
        The destination coordinate reference system (CRS) for the UTM projection.

    Returns
    -------
    None
        The function saves the reprojected raster to a new file with '_UTM.tif' suffix in the same directory.
    """
    #open source raster
    srcRst = rasterio.open(dirname+fname)
    #calculate transform array and shape of reprojected raster
    transform, width, height = calculate_default_transform(
            srcRst.crs, dstCrs, srcRst.width, srcRst.height, *srcRst.bounds)
    #working of the meta for the destination raster
    kwargs = srcRst.meta.copy()
    kwargs.update({
            'crs': dstCrs,
            'transform': transform,
            'width': width,
            'height': height
        })
    #open destination raster
    dstRst = rasterio.open(dirname+fname[:-4]+'_UTM.tif', 'w', **kwargs)
    #reproject and save raster band data
    for i in range(1, srcRst.count + 1):
        reproject(
            source=rasterio.band(srcRst, i),
            destination=rasterio.band(dstRst, i),
            src_crs=srcRst.crs,
            dst_crs=dstCrs,
            resampling=Resampling.nearest)
    #close destination raster
    dstRst.close()

def closest_point_on_segment(p, a, b):
    """
    Calculate the closest point on a line segment to a given point.

    Parameters
    ----------
    p : numpy.ndarray
        The point from which the closest point on the segment is to be found.
    a : numpy.ndarray
        The starting point of the line segment.
    b : numpy.ndarray
        The ending point of the line segment.

    Returns
    -------
    numpy.ndarray
        The closest point on the line segment to the point `p`.

    Notes
    -----
    This function assumes that `p`, `a`, and `b` are numpy arrays of the same dimension.
    """
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    if t < 0.0:
        return a
    elif t > 1.0:
        return b
    return a + t * ab

def closest_segment(line, x, y):
    """
    Finds the segment from a list of points that is closest to the midpoint of a given line.

    Parameters
    ----------
    line : list of numpy.ndarray
        A list containing two numpy arrays representing the endpoints of the line.
    x : list of float
        A list of x-coordinates of the points.
    y : list of float
        A list of y-coordinates of the points.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two numpy arrays representing the endpoints of the closest segment.
    """
    mid_point = (line[0].flatten() + line[1].flatten()) / 2
    min_dist = float('inf')
    closest_seg = None
    for i in range(len(x) - 1):
        point_a = np.array([x[i], y[i]])
        point_b = np.array([x[i+1], y[i+1]])
        point_on_segment = closest_point_on_segment(mid_point, point_a, point_b)
        dist = np.linalg.norm(mid_point - point_on_segment)
        if dist < min_dist:
            min_dist = dist
            closest_seg = (point_a, point_b)
    return closest_seg

def angle_between(v1, v2):
    """
    Calculate the angle between two vectors.

    Parameters
    ----------
    v1 : array_like
        First input vector.
    v2 : array_like
        Second input vector.

    Returns
    -------
    float
        The angle between the two vectors in radians.

    Notes
    -----
    The angle is calculated using the dot product and the norms of the vectors.
    The result is clipped to the range [-1, 1] to avoid numerical issues with arccos.
    If either vector has zero length, the angle is undefined and np.nan is returned.

    Examples
    --------
    >>> import numpy as np
    >>> v1 = np.array([1, 0, 0])
    >>> v2 = np.array([0, 1, 0])
    >>> angle_between(v1, v2)
    1.5707963267948966  # π/2 radians or 90 degrees
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return np.nan  # or: raise ValueError("One or both vectors have zero length.")
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    return np.arccos(np.clip(cos_theta, -1, 1))  # In radians

def find_longer_segment_coords(polygon, i1, i2, xs, ys):
    """
    Find the longer segment between two points on a Shapely polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The polygon from which to find the segments.
    i1 : int
        The index of the first point on the polygon's exterior.
    i2 : int
        The index of the second point on the polygon's exterior.
    xs : int or array-like
        If an integer, it represents a specific point index. If array-like, it represents x-coordinates of points.
    ys : array-like
        The y-coordinates of points, only used if xs is array-like.

    Returns
    -------
    tuple of array-like
        The x and y coordinates of the longer segment. The coordinates are returned in the order that minimizes the distance to the provided points if xs is array-like.
    """
    if type(xs) == np.int_:
        xs0 = xs
    points = list(polygon.exterior.coords)
    # Ensure i1 < i2 for simplicity
    if i1 > i2:
        i1, i2 = i2, i1
    segment1_coords = points[i1:i2 + 1]
    segment2_coords = points[i2:] + points[:i1 + 1]
    segment1 = LineString(segment1_coords)
    segment2 = LineString(segment2_coords)
    # if len(segment1_coords) > len(segment2_coords):
    if len(segment1.simplify(100).xy[0]) > len(segment2.simplify(100).xy[0]):
        if type(xs) == np.int_:
            if segment1.xy[0][0] == points[xs0][0]:
                return segment1.xy[0], segment1.xy[1]
            else:
                return segment1.xy[0][::-1], segment1.xy[1][::-1]
        else:
            dist_to_start_point = np.linalg.norm(np.array([xs[0], ys[0]]) - np.array(segment1.xy)[:,0])
            dist_to_end_point = np.linalg.norm(np.array([xs[-1], ys[-1]]) - np.array(segment1.xy)[:,0])
            if dist_to_start_point < dist_to_end_point:
                return segment1.xy[0], segment1.xy[1]
            else:
                return segment1.xy[0][::-1], segment1.xy[1][::-1]
    else:
        if type(xs) == np.int_:
            if segment2.xy[0][0] == points[xs0][0]:
                return segment2.xy[0], segment2.xy[1]
            else:
                return segment2.xy[0][::-1], segment2.xy[1][::-1]
        else:
            dist_to_start_point = np.linalg.norm(np.array([xs[0], ys[0]]) - np.array(segment2.xy)[:,0])
            dist_to_end_point = np.linalg.norm(np.array([xs[-1], ys[-1]]) - np.array(segment2.xy)[:,0])
            if dist_to_start_point < dist_to_end_point:
                return segment2.xy[0], segment2.xy[1]
            else:
                return segment2.xy[0][::-1], segment2.xy[1][::-1]
            
def extract_coords(geometry):
    """
    Extract coordinates from a geometry object.
    
    This function extracts x and y coordinates from a geometry object,
    handling both LineString and MultiLineString geometries. For MultiLineString
    geometries, it adds NaN values between line segments to facilitate plotting.
    
    Parameters
    ----------
    geometry : shapely.geometry
        The geometry object from which to extract coordinates.
        Can be LineString, MultiLineString, or other geometry types.
        
    Returns
    -------
    x_all : list
        List of x-coordinates extracted from the geometry.
        For MultiLineString, includes NaN values between segments.
    y_all : list
        List of y-coordinates extracted from the geometry.
        For MultiLineString, includes NaN values between segments.
    """
    if geometry.is_empty:
        return [], []
    
    if geometry.geom_type == 'LineString':
        x, y = geometry.xy
        return list(x), list(y)
    elif geometry.geom_type == 'MultiLineString':
        x_all, y_all = [], []
        for line in geometry.geoms:
            x, y = line.xy
            x_all.extend(list(x))
            y_all.extend(list(y))
            # Add NaN to separate line segments when plotting
            x_all.append(float('nan'))
            y_all.append(float('nan'))
        return x_all, y_all
    else:
        return [], []

def create_directed_multigraph(G_primal, G_rook, xs, ys, primal_start_ind, primal_end_ind, flip_outlier_edges=False, check_edges=False, x_utm=None, y_utm=None):
    """
    Create a directed multigraph from the given primal and rook graphs.

    Parameters
    ----------
    G_primal : networkx.Graph
        The primal graph containing the original nodes and edges.
    G_rook : networkx.Graph
        The rook graph containing the centerline polygons.
    xs : list or numpy.ndarray
        The x-coordinates of the smoothed centerline.
    ys : list or numpy.ndarray
        The y-coordinates of the smoothed centerline.
    primal_start_ind : int
        The index of the starting node in the primal graph.
    primal_end_ind : int
        The index of the ending node in the primal graph.
    flip_outlier_edges : bool, optional
        Whether to flip the direction of outlier edges (default is False). Should be set to 'True' for complex networks (e.g., Lena Delta, Brahmaputra).
    check_edges : bool, optional
        Check edges around each island for consistency in the direction of the flow (default is False). Should be set to 'True' for multithread rivers (e.g., Brahmaputra), but not for meandering rivers or for networks with unrealistic centerline orientations (e.g., Lena Delta).

    Returns
    -------
    D_primal : networkx.MultiDiGraph
        The directed multigraph with edges added based on the banklines and centerline polygons.
    source_nodes : list
        The list of source nodes in the directed multigraph.
    sink_nodes : list
        The list of sink nodes in the directed multigraph.

    Notes
    -----
    This function constructs a directed multigraph by adding edges from the primal graph that overlap
    with the main banklines defined by the centerline polygons in the rook graph. It also ensures that
    the directions of the edges are consistent and flips outlier edges if specified.
    """
    D_primal = nx.MultiDiGraph()
    for node, data in G_primal.nodes(data=True):
        D_primal.add_node(node, **data)
    # x and y coordinates of first major centerline polygon:    
    x0 = G_rook.nodes()[0]['cl_polygon'].exterior.xy[0]
    y0 = G_rook.nodes()[0]['cl_polygon'].exterior.xy[1]
    # find closest points to first and last points on smoothed centerline (xs and ys):
    ind1 = find_closest_point(G_primal.nodes()[primal_start_ind]['geometry'].x, G_primal.nodes()[primal_start_ind]['geometry'].y, np.vstack((x0,y0)).T)
    ind2 = find_closest_point(G_primal.nodes()[primal_end_ind]['geometry'].x, G_primal.nodes()[primal_end_ind]['geometry'].y, np.vstack((x0,y0)).T)
    # trim the polygon to the segment of interest (= bankline):
    x0, y0 = find_longer_segment_coords(G_rook.nodes()[0]['cl_polygon'], ind1, ind2, xs, ys)
    # x and y coordinates of second major centerline polygon:
    x1 = G_rook.nodes()[1]['cl_polygon'].exterior.xy[0]
    y1 = G_rook.nodes()[1]['cl_polygon'].exterior.xy[1]
    # find closest points to first and last points on smoothed centerline (xs and ys):
    ind1 = find_closest_point(G_primal.nodes()[primal_start_ind]['geometry'].x, G_primal.nodes()[primal_start_ind]['geometry'].y, np.vstack((x1,y1)).T)
    ind2 = find_closest_point(G_primal.nodes()[primal_end_ind]['geometry'].x, G_primal.nodes()[primal_end_ind]['geometry'].y, np.vstack((x1,y1)).T)
    # trim the polygon to the segment of interest (= bankline):
    x1, y1 = find_longer_segment_coords(G_rook.nodes()[1]['cl_polygon'], ind1, ind2, xs, ys)

    if x_utm is not None:
        # Create LineString objects from your coordinates
        line1 = LineString(zip(x0, y0))
        line2 = LineString(zip(x1, y1))
        # Create Polygon object from your coordinates
        # Make sure the polygon is closed (first and last points are the same)
        if (x_utm[0] != x_utm[-1]) or (y_utm[0] != y_utm[-1]):
            # If not closed, append the first point to close it
            x_polygon = list(x_utm) + [x_utm[0]]
            y_polygon = list(y_utm) + [y_utm[0]]
        else:
            x_polygon = x_utm
            y_polygon = y_utm
        polygon = Polygon(zip(x_polygon, y_polygon))
        # Get the parts of the linestrings that are outside the polygon
        # Using the difference operation
        line1_outside = line1.difference(polygon)
        line2_outside = line2.difference(polygon)
        # Get coordinates for plotting
        x0, y0 = extract_coords(line1_outside)
        x1, y1 = extract_coords(line2_outside)

    # for the two main banklines (defined by (x0,y0) and (x1,y1)), find the G_primal edges that 
    # overlap with the banklines and add these edges to the directed graph D_primal:
    for s,e,d in tqdm(G_primal.edges):
        # x and y coordinates of the edge:
        xe = G_primal[s][e][d]['geometry'].xy[0]
        ye = G_primal[s][e][d]['geometry'].xy[1]
        indices = find_matching_indices(x0, y0, xe, ye)
        # it is tempting to use here the first and last elements of 'xe' and 'ye' but the 'geometry' coordinates in
        # 'G_primal' can be flipped relative to the order of the nodes; so instead you have to use the
        # node coordinates:
        start_xe = G_primal.nodes()[s]['geometry'].centroid.xy[0][0]
        start_ye = G_primal.nodes()[s]['geometry'].centroid.xy[1][0]
        end_xe = G_primal.nodes()[e]['geometry'].centroid.xy[0][0]
        end_ye = G_primal.nodes()[e]['geometry'].centroid.xy[1][0]
        point1 = np.array([start_xe, start_ye])
        point2 = np.array([end_xe, end_ye])
        if len(indices) > 2: # need at least 3 matching points for the overlap
            first_x0 = np.array(x0)[np.array(indices).astype('int')][0]
            first_y0 = np.array(y0)[np.array(indices).astype('int')][0]
            point0 = np.array([first_x0, first_y0])
            if np.linalg.norm(point0 - point1) < np.linalg.norm(point0 - point2):
                # no need to flip the direction of the edge
                if (s,e,d) not in D_primal.edges:
                    D_primal.add_edge(s,e,d)
                    for key in G_primal[s][e][d].keys(): # copy edge attributes
                        D_primal[s][e][d][key] = G_primal[s][e][d][key]
            else: # need to flip the direction of the edge
                if (e,s,d) not in D_primal.edges:
                    D_primal.add_edge(e,s,d)
                    for key in G_primal[s][e][d].keys(): # copy edge attributes
                        D_primal[e][s][d][key] = G_primal[s][e][d][key]
        indices = find_matching_indices(x1, y1, xe, ye)
        if len(indices) > 2: # need at least 3 matching points for the overlap
            first_x1 = np.array(x1)[np.array(indices).astype('int')][0]
            first_y1 = np.array(y1)[np.array(indices).astype('int')][0]
            point0 = np.array([first_x1, first_y1])
            if np.linalg.norm(point0 - point1) < np.linalg.norm(point0 - point2):
                # no need to flip the direction of the edge
                if (s,e,d) not in D_primal.edges:
                    D_primal.add_edge(s,e,d)
                    for key in G_primal[s][e][d].keys(): # copy edge attributes
                        D_primal[s][e][d][key] = G_primal[s][e][d][key]
            else: # need to flip the direction of the edge
                if (e,s,d) not in D_primal.edges:
                    D_primal.add_edge(e,s,d)
                    for key in G_primal[s][e][d].keys(): # copy edge attributes
                        D_primal[e][s][d][key] = G_primal[s][e][d][key]
    # a few edges still need to be added to D_primal because they don't overlap with the main banklines,
    # but have their end nodes on the main banklines:
    edges_to_add = []
    for s,e,d in D_primal.edges:
        if len(D_primal[s][e]) != len(G_primal[s][e]):
            edges_to_add.append((s,e)) 
    for s,e in edges_to_add:
        D_primal.add_edge(s,e,1)
        for key in G_primal[s][e][1].keys(): 
            D_primal[s][e][1][key] = G_primal[s][e][1][key]

    # Now we add the remaining edges to D_primal, going through each rook neighborhood node.
    # The directions of the new edges are determined by the sum of the existing directed edges 
    # that surround the bar (or rook node). The order of iterating through the edges makes sure
    # that the bar/island of interest already has some directed edges added.
    processed_nodes = [0, 1] # already processed the two main banklines
    nit = 0
    while len(processed_nodes) < len(G_rook):
        if nit > 1e8:
            print('something went wrong, breaking while loop!')
            break
        neighbors = []
        for node in processed_nodes: # collect neighbors of the already processed nodes
            for n in G_rook.neighbors(node):
                if n not in neighbors and n not in processed_nodes:
                    neighbors.append(n)
        for node in neighbors:
            if node != 0 and node != 1:
                vectors = []
                D_edges = []
                G_edges = []
                # find the edges that are already in D_primal:
                for s,e,d in D_primal.edges: 
                    if 'half_widths' in D_primal[s][e][d].keys():
                        keys = D_primal[s][e][d]['half_widths'].keys()
                        for neighbor in nx.neighbors(G_rook, node):
                            if node in keys and neighbor in keys:
                                x1 = D_primal.nodes()[s]['geometry'].xy[0][0]
                                y1 = D_primal.nodes()[s]['geometry'].xy[1][0]
                                x2 = D_primal.nodes()[e]['geometry'].xy[0][0]
                                y2 = D_primal.nodes()[e]['geometry'].xy[1][0]
                                vectors.append((np.array([x1, y1]), np.array([x2, y2])))
                                D_edges.append((s, e))
                # find the edges that are not in D_primal and need to be added:
                for s,e,d in G_primal.edges: 
                    if 'half_widths' in G_primal[s][e][d].keys():
                        keys = G_primal[s][e][d]['half_widths'].keys()
                        for neighbor in nx.neighbors(G_rook, node):
                            if (node in keys and neighbor in keys) and ((s, e) not in D_edges and (e, s) not in D_edges):
                                x1 = G_primal.nodes()[s]['geometry'].xy[0][0]
                                y1 = G_primal.nodes()[s]['geometry'].xy[1][0]
                                x2 = G_primal.nodes()[e]['geometry'].xy[0][0]
                                y2 = G_primal.nodes()[e]['geometry'].xy[1][0]
                                G_edges.append((s, e))
                # Initialize sums
                sum_x = 0
                sum_y = 0
                # Add up the components
                for vec in vectors:
                    sum_x += vec[1][0] - vec[0][0]
                    sum_y += vec[1][1] - vec[0][1]
                # The resultant vector of the existing directed edges:
                resultant_vector = (sum_x, sum_y)
                # compute the angle between the edges to be added and set their direction when adding them to D_primal:
                for s,e in G_edges:
                    x1 = G_primal.nodes()[s]['geometry'].xy[0][0]
                    y1 = G_primal.nodes()[s]['geometry'].xy[1][0]
                    x2 = G_primal.nodes()[e]['geometry'].xy[0][0]
                    y2 = G_primal.nodes()[e]['geometry'].xy[1][0]
                    seg_vec = (x2-x1, y2-y1)
                    ang = angle_between(resultant_vector, seg_vec)
                    if ang <= np.pi/2: # if angle is less than or equal to 90 degrees
                        for d in range(len(G_primal[s][e])):
                            D_primal.add_edge(s,e,d)
                            for key in G_primal[s][e][d].keys(): # copy edge attributes
                                D_primal[s][e][d][key] = G_primal[s][e][d][key]
                    else: # if angle is larger than 90 degrees, we need to flip the direction of the edge
                        for d in range(len(G_primal[s][e])):
                            D_primal.add_edge(e,s,d)
                            for key in G_primal[s][e][d].keys(): # copy edge attributes
                                D_primal[e][s][d][key] = G_primal[s][e][d][key]
                if node not in processed_nodes:
                    processed_nodes.append(node)
        nit += 1
    # finally, we go through all the 'G_primal' nodes and check if there are edges with 'weird' directions within a radius of 3
    # these outlier directions are then flipped (but only flipped once); this works well in complex networks (e.g., Lena Delta, Brahmaputra),
    # but not in mostly single-thread meandering rivers
    if flip_outlier_edges:
        edges_flipped = []
        for main_node in G_primal.nodes:
            nodes_within_radius = nx.single_source_shortest_path_length(G_primal, main_node, cutoff=3)
            nodes_list = list(nodes_within_radius.keys())
            edges_list = []
            vectors = []
            for node in nodes_list:
                for s,e in D_primal.edges(node):
                    x1 = D_primal.nodes()[s]['geometry'].xy[0][0]
                    y1 = D_primal.nodes()[s]['geometry'].xy[1][0]
                    x2 = D_primal.nodes()[e]['geometry'].xy[0][0]
                    y2 = D_primal.nodes()[e]['geometry'].xy[1][0]
                    vectors.append((np.array([x1, y1]), np.array([x2, y2])))
                    edges_list.append((s,e))              
            sum_x = 0
            sum_y = 0
            # Add up the components
            for vec in vectors:
                sum_x += vec[1][0] - vec[0][0]
                sum_y += vec[1][1] - vec[0][1]
            # The resultant vector of the existing directed edges:
            resultant_vector = (sum_x, sum_y)
            count = 0
            for vec in vectors:
                seg_vec = (vec[1][0]-vec[0][0], vec[1][1]-vec[0][1])
                ang = angle_between(resultant_vector, seg_vec)
                if ang > np.pi/1.8: # if angle is larger than ~100 degrees, need to flip the edge direction
                    s, e = edges_list[count]
                    if (s,e) not in edges_flipped:
                        D_primal.remove_edge(s, e) # remove edge
                        for d in range(len(G_primal[s][e])):
                            D_primal.add_edge(e,s,d)
                            for key in G_primal[s][e][d].keys(): # copy edge attributes
                                D_primal[e][s][d][key] = G_primal[s][e][d][key]
                        edges_flipped.append((s,e))
                count += 1
    # need to make sure that the linestring coordinates and the half widths are listed in sync with the edge directions:
    flip_coords_and_widths(D_primal)
    D_primal = set_width_weights(D_primal)
    if check_edges:
        D_primal = check_edges_around_islands(D_primal, G_rook)
    if x_utm is not None:
        D_primal = truncate_graph_by_polygon(D_primal, x_utm, y_utm)
    sources = [node for node in D_primal.nodes() if D_primal.in_degree(node) == 0]
    sinks = [node for node in D_primal.nodes() if D_primal.out_degree(node) == 0]
    return D_primal, sources, sinks

def truncate_graph_by_polygon(D_primal, x_utm, y_utm):
    """
    Remove nodes inside a polygon and truncate edges that cross the polygon boundary.
    
    Parameters
    ----------
    D_primal : networkx.MultiDiGraph
        The directed multigraph to be modified
    x_utm : array-like
        x-coordinates of polygon vertices
    y_utm : array-like
        y-coordinates of polygon vertices
    
    Returns
    -------
    D_primal_truncated : networkx.MultiDiGraph
        A new graph with nodes inside the polygon removed and edges truncated
    """
    import numpy as np
    import networkx as nx
    from shapely.geometry import Point, LineString, Polygon
    
    # Create a copy of the graph to modify
    D_primal_truncated = D_primal.copy()
    
    # Create Polygon object from coordinates
    if (x_utm[0] != x_utm[-1]) or (y_utm[0] != y_utm[-1]):
        # If not closed, append the first point to close it
        x_polygon = list(x_utm) + [x_utm[0]]
        y_polygon = list(y_utm) + [y_utm[0]]
    else:
        x_polygon = x_utm
        y_polygon = y_utm
    
    polygon = Polygon(zip(x_polygon, y_polygon))
    
    # 1. Find nodes inside the polygon
    nodes_to_remove = []
    for node in D_primal_truncated.nodes():
        x = D_primal_truncated.nodes[node]['geometry'].xy[0][0]
        y = D_primal_truncated.nodes[node]['geometry'].xy[1][0]
        point = Point(x, y)
        if polygon.contains(point):
            nodes_to_remove.append(node)
    
    # 2. Find edges that intersect the polygon but are not entirely inside
    edges_to_process = []
    for s, e, d in D_primal_truncated.edges(keys=True):
        # Skip edges where both endpoints are inside the polygon (they'll be removed later)
        if s in nodes_to_remove and e in nodes_to_remove:
            continue
        
        # Check if the edge intersects the polygon but is not entirely inside
        edge_geom = D_primal_truncated[s][e][d]['geometry']
        if edge_geom.intersects(polygon) and not polygon.contains(edge_geom):
            edges_to_process.append((s, e, d))
    
    # 3. Process each edge that crosses the polygon boundary
    next_node_id = max(D_primal_truncated.nodes()) + 1
    new_edges = []
    
    for s, e, d in edges_to_process:
        # Get the edge geometry
        edge_geom = D_primal_truncated[s][e][d]['geometry']
        
        # Get the part of the edge outside the polygon
        outside_part = edge_geom.difference(polygon)
        
        # Skip if the result is empty or a point
        if outside_part.is_empty or outside_part.geom_type == 'Point':
            continue
        
        # Handle both LineString and MultiLineString results
        if outside_part.geom_type == 'LineString':
            line_parts = [outside_part]
        elif outside_part.geom_type == 'MultiLineString':
            line_parts = list(outside_part.geoms)
        else:
            continue  # Skip other geometry types
        
        # Process each line part
        for line in line_parts:
            # Get the endpoints of the line
            start_point = Point(line.coords[0])
            end_point = Point(line.coords[-1])
            
            # Determine which nodes to keep
            start_node = None
            end_node = None
            
            # Check if the start point is at an existing node
            if Point(D_primal_truncated.nodes[s]['geometry'].xy[0][0], 
                    D_primal_truncated.nodes[s]['geometry'].xy[1][0]).distance(start_point) < 1e-6:
                start_node = s
            elif Point(D_primal_truncated.nodes[e]['geometry'].xy[0][0], 
                     D_primal_truncated.nodes[e]['geometry'].xy[1][0]).distance(start_point) < 1e-6:
                start_node = e
                
            # Check if the end point is at an existing node
            if Point(D_primal_truncated.nodes[s]['geometry'].xy[0][0], 
                    D_primal_truncated.nodes[s]['geometry'].xy[1][0]).distance(end_point) < 1e-6:
                end_node = s
            elif Point(D_primal_truncated.nodes[e]['geometry'].xy[0][0], 
                     D_primal_truncated.nodes[e]['geometry'].xy[1][0]).distance(end_point) < 1e-6:
                end_node = e
                
            # Create new nodes for the truncated ends
            if start_node is None:
                start_node = next_node_id
                next_node_id += 1
                D_primal_truncated.add_node(start_node, 
                                          geometry=Point(start_point.x, start_point.y))
            
            if end_node is None:
                end_node = next_node_id
                next_node_id += 1
                D_primal_truncated.add_node(end_node, 
                                          geometry=Point(end_point.x, end_point.y))
            
            # Only keep the edge if the original source node is retained
            if s not in nodes_to_remove:
                # If the original source is kept, preserve the edge direction
                if start_node == s:
                    new_edges.append((start_node, end_node, d, line, 
                                   D_primal_truncated[s][e][d].copy()))
            
            # If original target is kept and source was inside polygon
            elif e not in nodes_to_remove and s in nodes_to_remove:
                # New edge should point to the original target
                if end_node == e:
                    new_edges.append((start_node, end_node, d, line,
                                   D_primal_truncated[s][e][d].copy()))
    
    # 4. Remove nodes inside the polygon
    D_primal_truncated.remove_nodes_from(nodes_to_remove)
    
    # 5. Remove all edges connected to removed nodes
    edges_to_remove = []
    for s, e, d in D_primal_truncated.edges(keys=True):
        if s in nodes_to_remove or e in nodes_to_remove:
            edges_to_remove.append((s, e, d))
    
    D_primal_truncated.remove_edges_from(edges_to_remove)
    
    # 6. Add the new edges representing the truncated segments
    for s, e, d, geometry, attrs in new_edges:
        D_primal_truncated.add_edge(s, e, d, **attrs)
        D_primal_truncated[s][e][d]['geometry'] = geometry
    
    print(f"Removed {len(nodes_to_remove)} nodes inside the polygon")
    print(f"Added {next_node_id - max(D_primal.nodes()) - 1} new nodes at edge intersections")
    
    # 7. Find all sink nodes (nodes with out_degree=0) that were created as a result of truncation
    sinks = [node for node in D_primal_truncated.nodes() if D_primal_truncated.out_degree(node) == 0]
    print(f"Total sink nodes in truncated graph: {len(sinks)}")
    
    return D_primal_truncated

def set_width_weights(G_primal):
    """
    Set the width and weight attributes for the edges in a graph.

    This function iterates over the edges of the given graph `G_primal` and calculates the width
    for each edge based on the 'half_widths' attribute. If there are more than one 'half_widths'
    defined for an edge, it calculates the mean of the two half widths and sets it as the 'width'
    attribute of the edge. If no 'half_widths' are defined, it prints a message indicating so.

    Parameters
    ----------
    G_primal : networkx.Graph
        The input graph with edges that may have 'half_widths' attributes.

    Returns
    -------
    networkx.Graph
        The graph with updated 'width' attributes for the edges.

    Notes
    -----
    - The function assumes that the 'half_widths' attribute, if present, is a dictionary with at least
      two keys.
    - The 'weight' attribute calculation is commented out and can be customized as needed.
    """
    for s, e, d in G_primal.edges:
        if 'half_widths' in G_primal[s][e][d].keys():
            if len(list(G_primal[s][e][d]['half_widths'].keys())) > 1:
                key1 = list(G_primal[s][e][d]['half_widths'].keys())[0]
                key2 = list(G_primal[s][e][d]['half_widths'].keys())[1]
                w1 = G_primal[s][e][d]['half_widths'][key1]
                w2 = G_primal[s][e][d]['half_widths'][key2]
                G_primal[s][e][d]['width'] = np.mean(np.array(w1) + np.array(w2))
                # G_primal[s][e][d]['weight'] = 4*1/G_primal[s][e][d]['width']
                # G_primal[s][e][d]['weight'] = G_primal[s][e][d]['width']
            else:
                print('no half widths defined for edge ('+str(s)+','+str(e)+','+str(d)+')')
    return G_primal

def find_end_nodes(G_primal, xs, ys):
    """
    Find the end nodes in a graph that are closest to the given start and end points.

    Parameters
    ----------
    G_primal : networkx.Graph
        The input graph where nodes have 'geometry' attributes containing shapely geometries.
    xs : list of float
        List of x-coordinates of the points to find the closest end nodes to.
    ys : list of float
        List of y-coordinates of the points to find the closest end nodes to.

    Returns
    -------
    tuple
        A tuple containing two nodes:
        - node1: The end node closest to the first point (xs[0], ys[0]).
        - node2: The end node closest to the last point (xs[-1], ys[-1]).
    """
    end_nodes = []
    end_points = []
    for node in G_primal:
        if G_primal.degree[node] == 1:
            x1 = G_primal.nodes()[node]['geometry'].xy[0][0]
            y1 = G_primal.nodes()[node]['geometry'].xy[1][0]
            end_nodes.append(node)
            end_points.append((x1, y1))
    node1 = end_nodes[find_closest_point(xs[0], ys[0], end_points)]
    node2 = end_nodes[find_closest_point(xs[-1], ys[-1], end_points)]
    return node1, node2

def find_closest_point(x1, y1, other_points):
    """
    Find the index of the closest point to a given point from a list of other points.

    Parameters
    ----------
    x1 : float
        The x-coordinate of the given point.
    y1 : float
        The y-coordinate of the given point.
    other_points : list of tuples or list of lists
        A list of points where each point is represented as a tuple or list of two floats (x, y).

    Returns
    -------
    int
        The index of the closest point in the `other_points` list.
    """
    point = np.array([x1, y1])
    other_points = np.array(other_points)
    distances = np.linalg.norm(other_points - point, axis=1)
    min_index = np.argmin(distances)
    return min_index

def flip_coords_and_widths(D_primal):
    """
    Flip the coordinates and widths of edges in a primal graph if necessary.

    This function iterates over the edges of the given primal graph `D_primal`.
    For each edge, it checks if the distance from the start node to the first
    coordinate of the edge's geometry is greater than the distance from the end
    node to the first coordinate. If so, it flips the ordering of the coordinates
    in the edge's geometry and also flips the half-widths if they are defined.

    Parameters
    ----------
    D_primal : networkx.DiGraph
        A directed graph where each edge has a 'geometry' attribute containing
        a LineString and optionally a 'half_widths' attribute containing a dictionary
        of half-widths.

    Notes
    -----
    - The 'geometry' attribute of each edge is expected to be a shapely.geometry.LineString.
    - The 'half_widths' attribute, if present, is expected to be a dictionary with keys
      corresponding to different width types and values being lists of widths along the edge.
    - If an edge does not have 'half_widths' defined, a message is printed.
    """
    for s,e,d in D_primal.edges:
        x1 = D_primal[s][e][d]['geometry'].xy[0][0]
        y1 = D_primal[s][e][d]['geometry'].xy[1][0]
        xn1 = D_primal.nodes()[s]['geometry'].xy[0][0]
        yn1 = D_primal.nodes()[s]['geometry'].xy[1][0]
        xn2 = D_primal.nodes()[e]['geometry'].xy[0][0]
        yn2 = D_primal.nodes()[e]['geometry'].xy[1][0]
        if (x1-xn1)**2 + (y1-yn1)**2 > (x1-xn2)**2 + (y1-yn2)**2:
            # flip the ordering of coordinates in the linestrings:
            x = np.array(D_primal[s][e][d]['geometry'].xy[0])[::-1]
            y = np.array(D_primal[s][e][d]['geometry'].xy[1])[::-1]
            D_primal[s][e][d]['geometry'] = LineString(np.vstack((x, y)).T)
            if len(list(D_primal[s][e][d]['half_widths'].keys())) > 1:
                key1 = list(D_primal[s][e][d]['half_widths'].keys())[0]
                key2 = list(D_primal[s][e][d]['half_widths'].keys())[1]
                D_primal[s][e][d]['half_widths'][key1] = D_primal[s][e][d]['half_widths'][key1][::-1]
                D_primal[s][e][d]['half_widths'][key2] = D_primal[s][e][d]['half_widths'][key2][::-1]
            else:
                print('no half widths defined for edge ('+str(s)+','+str(e)+','+str(d)+')')

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
    wave_lengths = 2*half_wave_lengths[sinuosities > min_sinuosity]
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

def find_condition(condition):
    """
    Find the indices of elements that are non-zero in the flattened input array.

    Parameters
    ----------
    condition : array_like
        Input array. The function will find the non-zero elements in this array.

    Returns
    -------
    res : ndarray
        Indices of elements that are non-zero in the flattened input array.
    """
    res, = np.nonzero(np.ravel(condition))
    return res

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

def find_subpath(D_primal, root, depth_limit=10):
    """
    Finds the subpath in a directed graph from the root node up to a specified depth limit.

    Parameters
    ----------
    D_primal : networkx.DiGraph
        The directed graph in which to find the subpath.
    root : node
        The root node from which to start the search.
    depth_limit : int, optional
        The maximum depth to search from the root node (default is 10).

    Returns
    -------
    list or bool
        Returns the subpath with the maximum average width if any paths are found, otherwise returns False.

    Notes
    -----
    The function first finds all nodes within the depth limit from the root node. It then identifies the leaf nodes
    at the maximum depth. For each leaf node, it finds all simple edge paths from the root to the leaf. Among these
    paths, it calculates the average width of the edges and returns the path with the maximum average width.
    """
    nodes = nx.single_source_shortest_path_length(D_primal, root, cutoff=depth_limit)
    leaves = [node for node, depth in nodes.items() if depth == min(depth_limit, list(nodes.values())[-1])]
    D_primal_small = nx.subgraph(D_primal, nodes)
    all_paths = []
    for leaf in leaves:
        paths = nx.all_simple_edge_paths(D_primal_small, root, leaf)
        for path in paths:
            if len(path) > 0:
                all_paths.append(path)
    if len(all_paths) > 0:
        weights = []
        for path in all_paths:
            width = 0
            length = 0
            for s,e,d in path:
                width += D_primal[s][e][d]['width']
                length += D_primal[s][e][d]['mm_len']
            weights.append((width/len(path)))
        return all_paths[np.argmax(weights)]
    else:
        return False

def traverse_multigraph(G, start_node, subpath_depth=5):
    """
    Traverse a multigraph starting from a given node and return the path of edges.

    Parameters
    ----------
    G : networkx.MultiGraph
        The multigraph to traverse.
    start_node : node
        The starting node for the traversal.
    subpath_depth : int, optional
        The maximum depth of subpaths to consider during traversal (default is 5).

    Returns
    -------
    edge_path : list of tuples
        A list of edges representing the path traversed in the multigraph.

    Notes
    -----
    The function attempts to find subpaths of the given depth starting from the current node.
    If a cycle is detected (i.e., the nodes in the current path are the same as the nodes in the next path),
    the traversal is terminated to avoid infinite loops.
    """
    current_node = start_node
    edge_path = []
    subpath_depth = min(subpath_depth, len(G))
    current_path = []
    while True:
        next_path = find_subpath(G, current_node, subpath_depth)
        if next_path:
            # sometimes there is a cycle and we need to break it:
            # nodes_in_current_path = list(set(element for tuple_ in current_path for element in tuple_))
            nodes_in_current_path = set([element for tuple_ in [tuple_[:2] for tuple_ in current_path] for element in tuple_])
            # nodes_in_next_path = list(set(element for tuple_ in next_path for element in tuple_))
            nodes_in_next_path = set([element for tuple_ in [tuple_[:2] for tuple_ in next_path] for element in tuple_])
            if nodes_in_current_path == nodes_in_next_path:
                break
            edge_path.extend(next_path)
            current_node = edge_path[-1][1]
            current_path = next_path.copy()
        else:
            break
    return edge_path

def find_start_node(D_primal):
    """
    Find the start node in a directed graph.

    Parameters
    ----------
    D_primal : networkx.DiGraph
        A directed graph.

    Returns
    -------
    start_node : node or None
        The start node if found, otherwise None.
    inds : numpy.ndarray
        Indices of nodes with degree less than 3.

    Notes
    -----
    The function identifies nodes with degree less than 3 and nodes with no outgoing edges (sinks).
    It then checks for a path between pairs of these nodes and returns the first node of the pair
    if a path exists. If no such node is found, it prints a message and returns None.
    """
    inds = np.where(np.array(list(nx.degree(D_primal, D_primal.nodes)))[:,1] < 3)[0]
    nodes = np.array(list(nx.degree(D_primal, D_primal.nodes)))[:,0][inds]
    sinks = [node for node in D_primal.nodes() if D_primal.out_degree(node) == 0]
    nodes = np.unique(list(nodes) + sinks)
    pairs = list(permutations(nodes, 2))
    start_node = None
    for pair in pairs:
        if nx.has_path(D_primal, *pair):
            start_node = pair[0]
            break
    if start_node == None:
        print("could not find start node!")
        return start_node, inds
    return start_node, inds

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

def check_edges_around_islands(D_primal, G_rook):
    """
    Check for islands with multiple source or sink nodes and flip edges with outlier orientations.
    
    Parameters
    ----------
    D_primal : networkx.MultiDiGraph
        The directed multigraph representing the river network.
    G_rook : networkx.Graph
        The rook graph where each node represents an island.
        
    Returns
    -------
    D_primal : networkx.MultiDiGraph
        The modified directed multigraph with corrected edge orientations.
    """
    import numpy as np
    import networkx as nx
    from shapely.geometry import LineString
    
    # Find all source and sink nodes
    sources = [node for node in D_primal.nodes() if D_primal.in_degree(node) == 0]
    sinks = [node for node in D_primal.nodes() if D_primal.out_degree(node) == 0]
    
    print(f"Found {len(sources)} source nodes: {sources}")
    print(f"Found {len(sinks)} sink nodes: {sinks}")
    
    # Process each island (node in G_rook)
    for node in G_rook.nodes():
        # Step 1: Find all edges associated with this island
        island_edges = []
        for s, e, d in D_primal.edges(keys=True):
            if 'half_widths' in D_primal[s][e][d].keys():
                keys = D_primal[s][e][d]['half_widths'].keys()
                if node in keys:
                    x1 = D_primal.nodes()[s]['geometry'].xy[0][0]
                    y1 = D_primal.nodes()[s]['geometry'].xy[1][0]
                    x2 = D_primal.nodes()[e]['geometry'].xy[0][0]
                    y2 = D_primal.nodes()[e]['geometry'].xy[1][0]
                    vector = (x2-x1, y2-y1)
                    island_edges.append(((s, e, d), vector))
        if not island_edges:
            continue
        
        # Step 2: Calculate mean orientation of edges
        vectors = [edge[1] for edge in island_edges]
        sum_x = sum(v[0] for v in vectors)
        sum_y = sum(v[1] for v in vectors)
        mean_vector = (sum_x, sum_y)
                
        if np.linalg.norm(mean_vector) == 0:
            continue  # Skip islands with no clear mean direction
        
        # Step 3: Calculate angles between each edge and the mean orientation
        edge_angles = []
        for (s, e, d), vector in island_edges:
            ang = angle_between(mean_vector, vector)
            if np.isnan(ang):
                continue
            edge_angles.append(((s, e, d), ang))
        
        # Step 4: Sort edges by angle difference and identify outliers
        edge_angles.sort(key=lambda x: x[1], reverse=True)
        # Consider the top 25% of edges with the largest angle difference as outliers
        outlier_count = max(1, len(edge_angles) // 4)
        outliers = edge_angles[:outlier_count]
        
        # Step 5: Flip outlier edges
        for (s, e, d), angle in outliers:
            if angle > np.pi/2:  # Only flip if angle is greater than 90 degrees
                # print(f"Flipping edge ({s},{e},{d}) with angle {angle*180/np.pi:.1f} degrees")
                # Store edge attributes
                edge_attrs = dict(D_primal[s][e][d])
                # Remove the edge
                D_primal.remove_edge(s, e, d)
                # Add the edge in the opposite direction
                D_primal.add_edge(e, s, d, **edge_attrs)
                # Flip the geometry coordinates and half-widths
                if 'geometry' in D_primal[e][s][d]:
                    x = np.array(D_primal[e][s][d]['geometry'].xy[0])[::-1]
                    y = np.array(D_primal[e][s][d]['geometry'].xy[1])[::-1]
                    D_primal[e][s][d]['geometry'] = LineString(np.vstack((x, y)).T)
                if 'half_widths' in D_primal[e][s][d]:
                    for key in D_primal[e][s][d]['half_widths'].keys():
                        D_primal[e][s][d]['half_widths'][key] = D_primal[e][s][d]['half_widths'][key][::-1]
    
    # Final check for sources and sinks
    new_sources = [node for node in D_primal.nodes() if D_primal.in_degree(node) == 0]
    new_sinks = [node for node in D_primal.nodes() if D_primal.out_degree(node) == 0]
    
    print(f"After corrections, found {len(new_sources)} source nodes: {new_sources}")
    print(f"After corrections, found {len(new_sinks)} sink nodes: {new_sinks}")
    
    return D_primal

def calculate_iou(poly1, poly2):
    """
    Calculate the Intersection over Union (IoU) metric between two polygons.

    The IoU is defined as the area of the intersection of the two polygons 
    divided by the area of their union. If the area of the union is 0, the 
    function returns 0.

    Parameters
    ----------
    poly1 : shapely.geometry.Polygon
        The first polygon.
    poly2 : shapely.geometry.Polygon
        The second polygon.

    Returns
    -------
    float
        The IoU of the two polygons. If the area of the union is 0, it returns 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union != 0 else 0

def modified_iou(poly1, poly2):
    """
    Calculates a modified version of the Intersection over Union (IoU) metric between two polygons.

    Instead of dividing the area of the intersection by the area of the union of the two polygons, 
    it divides the area of the intersection by the area of the smaller polygon. This gives a measure 
    of the fraction of the smaller polygon that is inside the larger polygon.

    Parameters
    ----------
    poly1 : shapely.geometry.Polygon
        The first polygon.
    poly2 : shapely.geometry.Polygon
        The second polygon.

    Returns
    -------
    float
        The fraction of the smaller polygon that is inside the larger polygon. If the area of the 
        smaller polygon is 0, it returns 0.
    """
    # Calculate the intersection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        intersection = poly1.intersection(poly2).area
    # Determine the smaller polygon
    smaller_poly_area = min(poly1.area, poly2.area)
    # Calculate the fraction of the smaller polygon that is inside the larger polygon
    return intersection / smaller_poly_area if smaller_poly_area != 0 else 0

def cluster_polygons(gdf, iou_threshold, max_days = 2*365):
    """
    Cluster polygons based on Intersection over Union (IoU) and time difference.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing polygon geometries and associated attributes.
    iou_threshold : float
        The IoU threshold above which polygons are considered adjacent.
    max_days : int, optional
        The maximum number of days difference allowed between polygons to be considered for clustering (default is 2*365).

    Returns
    -------
    G : networkx.Graph
        A graph where nodes represent polygons and edges represent adjacency based on IoU and time difference.
    clusters : list of sets
        A list of sets, where each set contains the indices of polygons that form a cluster.
    """
    # Create a spatial index for the polygons
    sindex = gdf.sindex
    # Create a graph to represent adjacency (IoU above threshold)
    G = nx.Graph()
    for i, row1 in tqdm(gdf.iterrows()):
        if row1['type'] != 1 and row1['type'] != 0: # ignore the main banks
            poly1 = row1.geometry
            n_days1 = row1.n_days
            # Possible matches index
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                possible_matches_index = list(sindex.intersection(poly1.bounds))
                possible_matches = gdf.iloc[possible_matches_index]
                precise_matches = possible_matches[possible_matches.geometry.intersects(poly1)]        
            for j, row2 in precise_matches.iterrows():
                if i < j:  # Avoid duplicate pairs
                    poly2 = row2.geometry
                    n_days2 = row2.n_days
                    if np.abs(n_days2-n_days1) <= max_days: # time difference less than 'max_days'
                        iou = modified_iou(poly1, poly2)
                        if iou > iou_threshold:
                            G.add_edge(i, j)
    # Find clusters (connected components) in the graph
    clusters = list(nx.connected_components(G))
    return G, clusters

def find_numbers_between(start, end, decimal_places):
    """
    Returns a list of numbers between start and end, rounded to the specified decimal places.

    Parameters
    ----------
    start : float
        The starting number of the range.
    end : float
        The ending number of the range.
    decimal_places : int
        The number of decimal places to round each number to.

    Returns
    -------
    list of float
        A list of numbers between start and end, inclusive, rounded to the specified decimal places.
    """
    step = 10 ** -decimal_places  # Calculate the step size based on decimal places
    numbers = []
    current_number = round(start, decimal_places)  # Begin with start rounded to specified places
    while current_number <= end:
        numbers.append(current_number)
        current_number += step  # Increment by the step size
        current_number = round(current_number, decimal_places)
    return numbers

def group_edges_to_subpaths(edges):
    """
    Groups edges into subpaths in a directed graph.

    Parameters
    ----------
    edges : list of tuple
        A list of tuples where each tuple represents an edge in the format (start_node, end_node, data).

    Returns
    -------
    subpaths : list of list of tuple
        A list of subpaths, where each subpath is a list of edges. Each edge is represented as a tuple (start_node, end_node, data).
    """
    G = nx.DiGraph()
    for s, e, d in edges:
        G.add_edge(s, e, data=d)
    subpaths = []
    for node in G.nodes:
        # If the node has no incoming edges, it could be a start of a subpath
        if G.in_degree(node) == 0:
            for target in G.nodes:
                # If the target node has no outgoing edges, it could be an end of a subpath
                if G.out_degree(target) == 0 and nx.has_path(G, node, target):
                    path = nx.shortest_path(G, node, target)
                    # Convert node path to edge path with data
                    edge_path = [(path[i], path[i + 1], G.edges[path[i], path[i + 1]]['data']) for i in range(len(path) - 1)]
                    subpaths.append(edge_path)
    return subpaths

def find_matching_subpaths(subpaths1, subpaths2):
    """
    Find matching subpaths between two lists of subpaths.

    Parameters
    ----------
    subpaths1 : list of list of tuples
        The first list of subpaths, where each subpath is a list of tuples representing edges.
    subpaths2 : list of list of tuples
        The second list of subpaths, where each subpath is a list of tuples representing edges.

    Returns
    -------
    list of tuples
        A list of tuples, where each tuple contains a matching subpath from `subpaths1` and `subpaths2`.
        A subpath is considered matching if it has the same start and end nodes.
    """
    matching_subpaths = []
    for subpath1 in subpaths1:
        start1, end1 = subpath1[0][0], subpath1[-1][1]  # Start and end nodes of subpath1

        for subpath2 in subpaths2:
            start2, end2 = subpath2[0][0], subpath2[-1][1]  # Start and end nodes of subpath2

            if start1 == start2 and end1 == end2:
                # Found a match
                matching_subpaths.append((subpath1, subpath2))

    return matching_subpaths

def splice_paths(D_primal, path1, path2):
    """
    Splices two paths by replacing segments in the first path with improved segments from the second path based on edge widths.

    Parameters
    ----------
    D_primal : dict
        A dictionary representing the primal graph where keys are node pairs and values are dictionaries containing edge attributes.
    path1 : list of tuples
        The first path represented as a list of edges (tuples of nodes).
    path2 : list of tuples
        The second path represented as a list of edges (tuples of nodes).

    Returns
    -------
    list of tuples
        The spliced path with segments from the second path replacing segments in the first path where the edge widths are greater.

    Notes
    -----
    - The function assumes that the edges in the paths are tuples of the form (start_node, end_node, edge_data).
    - The edge_data dictionary must contain a 'width' key for comparing edge widths.
    - If a matching segment is not found in the original path for an improved subpath, a message is printed.
    """
    subpaths_in_path1 = group_edges_to_subpaths(set(path1) - set(path2))
    subpaths_in_path2 = group_edges_to_subpaths(set(path2) - set(path1))
    matched_subpaths = find_matching_subpaths(subpaths_in_path1, subpaths_in_path2)
    best_subpaths = []
    for match in matched_subpaths:
        width_1 = 0
        count = 0
        for s,e,d in match[0]:
            width_1 += D_primal[s][e][d]['width']
            count += 1
        width_1 = width_1/count
        width_2 = 0
        count = 0
        for s,e,d in match[1]:
            width_2 += D_primal[s][e][d]['width']
            count += 1
        width_2 = width_2/count
        if width_1 >= width_2:
            best_subpaths.append(match[0])
        else:
            best_subpaths.append(match[1])
    # Convert the original path to a list if it's not already
    spliced_path = list(path1)
    for improved_subpath in best_subpaths:
        # Get start and end nodes of the improved subpath
        start_node = improved_subpath[0][0]
        end_node = improved_subpath[-1][1]
        # Find the segment in the original path to replace
        start_index = end_index = None
        for i, edge in enumerate(spliced_path):
            if edge[0] == start_node:
                start_index = i
            if edge[1] == end_node:
                end_index = i
                break
        # Check if both indices were found
        if start_index is not None and end_index is not None:
            # Replace the segment in the original path with the improved subpath
            spliced_path[start_index:end_index + 1] = improved_subpath
        else:
            print("Matching segment not found for subpath starting at node", start_node)
    return spliced_path

def get_ch_and_bar_areas(gdf, xmin, xmax, ymin, ymax):
    """
    Calculate channel and bar areas within a specified area of interest (AOI) over time.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing geometries and attributes of river banks and bars.
    xmin : float
        Minimum x-coordinate of the AOI.
    xmax : float
        Maximum x-coordinate of the AOI.
    ymin : float
        Minimum y-coordinate of the AOI.
    ymax : float
        Maximum y-coordinate of the AOI.

    Returns
    -------
    dates : list
        List of unique dates corresponding to the n_days in the GeoDataFrame.
    all_bars : list
        List of geometries representing all bars within the AOI for each date.
    chs : list
        List of geometries representing channels within the AOI for each date.
    ch_belts : list
        List of geometries representing channel belts within the AOI for each date.
    bar_areas : list
        List of areas of bars within the AOI for each date.
    ch_areas : list
        List of areas of channels within the AOI for each date.
    """
    ch_areas = []
    bar_areas = []
    chs = []
    ch_belts = []
    all_bars = []
    dates = []
    aoi_poly = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    for n_days in tqdm(gdf.n_days.unique()):
        date = gdf[(gdf['n_days']== n_days) & (gdf['type']==0)].date.dt.year
        bank0 = gdf[(gdf['n_days']== n_days) & (gdf['type']==0)].geometry.values[0]
        bank0 = bank0.intersection(aoi_poly)
        # inds = []
        # if type(aoi_poly.difference(bank0)) == Polygon:
        #     geoms = [aoi_poly.difference(bank0)]
        # else:
        #     geoms = aoi_poly.difference(bank0).geoms
        # count = 0
        # for geom in geoms:
        #     if geom.touches(Point([xmin, ymin])) or geom.touches(Point([xmin, ymax])):
        #         inds.append(count)
        #     count += 1
        # for ind in inds:
        #     bank0 = bank0.union(geoms[ind])
        bank1 = gdf[(gdf['n_days']== n_days) & (gdf['type']==1)].geometry.values[0]
        bank1 = bank1.intersection(aoi_poly)
        # inds = []
        # if type(aoi_poly.difference(bank1)) == Polygon:
        #     geoms = [aoi_poly.difference(bank1)]
        # else:
        #     geoms = aoi_poly.difference(bank1).geoms
        # count = 0
        # for geom in geoms:
        #     if geom.touches(Point([xmax, ymin])) or geom.touches(Point([xmax, ymax])):
        #         inds.append(count)
        #     count += 1
        # for ind in inds:
        #     bank1 = bank1.union(geoms[ind])
        ch_belt = aoi_poly.difference(unary_union([bank0, bank1]))
        bars = []
        for i in gdf.index:
            if gdf.n_days[i] == n_days and gdf['type'][i] != 0 and gdf['type'][i] != 1:
                bars.append(gdf.geometry[i])
        bars = unary_union(bars)
        bars = bars.intersection(aoi_poly)
        bar_areas.append(bars.area)
        all_bars.append(bars)
        ch = ch_belt.difference(bars)
        ch_areas.append(ch.area)
        chs.append(ch)
        ch_belts.append(ch_belt)
        dates.append(date)
    return dates, all_bars, chs, ch_belts, bar_areas, ch_areas

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

def gdfs_from_D_primal(D_primal, dataset):
    """
    Create two GeoDataFrames (one for nodes and one for edges) from a primal graph.

    Parameters
    ----------
    D_primal : networkx.MultiDiGraph
        The primal graph containing nodes and edges with geometries and attributes.
    dataset : geopandas.GeoDataFrame
        The dataset containing the coordinate reference system (CRS) information.

    Returns
    -------
    node_df : geopandas.GeoDataFrame
        A GeoDataFrame containing the nodes with their geometries.
    edge_df : geopandas.GeoDataFrame
        A GeoDataFrame containing the edges with their geometries and attributes.

    Notes
    -----
    If you want to save the edge dataframe as a shapefile, you need to drop the 'widths' column.
    """
    # Extract nodes
    nodes = {node: data['geometry'] for node, data in D_primal.nodes(data=True)}
    node_df = geopandas.GeoDataFrame(nodes.items(), columns=['node_id', 'geometry'], 
                                     crs=dataset.crs.to_string())
    # Extract edges
    edges = []
    for s, e, d, data in D_primal.edges(data=True, keys=True):
        edge_attrs = {'source': s, 'target': e, 'key': d}
        for key, attr in data.items():
            if key == 'mm_len':
                edge_attrs['mm_len'] = attr
            if key == 'width':
                edge_attrs['width'] = attr
            if key == 'half_widths':
                key1 = list(attr.keys())[0]
                key2 = list(attr.keys())[1]
                edge_attrs['widths'] = attr[key1] + attr[key2]
        edge_geom = data['geometry']
        edges.append({'geometry': edge_geom, **edge_attrs})
    edge_df = geopandas.GeoDataFrame(edges, crs=dataset.crs.to_string())
    return node_df, edge_df

def crop_geotiff(input_file, output_file, row_off, col_off, max_rows=2**16//2, max_cols=2**16//2):
    """
    Crop a GeoTIFF file and save the cropped portion to a new file.
    Parameters
    ----------
    input_file : str
        Path to the input GeoTIFF file.
    output_file : str
        Path to the output cropped GeoTIFF file.
    row_off : int
        The row offset (starting row) for the cropping window.
    col_off : int
        The column offset (starting column) for the cropping window.
    max_rows : int, optional
        The maximum number of rows in the cropped image (default is 2**16 // 2).
    max_cols : int, optional
        The maximum number of columns in the cropped image (default is 2**16 // 2).
    Returns
    -------
    None
    """
    # Open the input GeoTIFF file
    with rasterio.open(input_file) as src:
        # Define the cropping window
        width = src.width
        height = src.height
        max_rows = min(height-row_off, max_rows)
        max_cols = min(width-col_off, max_cols)
        crop_window = rasterio.windows.Window(col_off, row_off, max_cols, max_rows)
        
        # Read the cropped portion of the image
        cropped_data = src.read(window=crop_window)
        
        # Update metadata for the cropped image
        crop_meta = src.meta.copy()
        crop_meta.update({
            'width': max_cols,
            'height': max_rows,
            'transform': src.window_transform(crop_window)
        })
        
        # Write the cropped image to the output GeoTIFF file
        with rasterio.open(output_file, 'w', **crop_meta) as dst:
            dst.write(cropped_data)

def write_shapefiles_and_graphs(G_rook, D_primal, dataset, dirname, rivername, ch_mouth_poly=None):
    """
    Writes shapefiles and graph data to specified directory.

    Parameters
    ----------
    G_rook : networkx.Graph
        The rook graph representing the river network.
    D_primal : networkx.Graph
        The primal graph representing the river network.
    dataset : geopandas.GeoDataFrame
        The dataset containing the river network data.
    dirname : str
        The directory name where the files will be saved.
    rivername : str
        The name of the river, used as a prefix for the filenames.
    ch_mouth_poly : shapely.geometry.Polygon, optional
        The polygon representing the channel mouth, by default None.

    Returns
    -------
    None
    """
    ch_nw_poly = create_channel_nw_polygon(G_rook, ch_mouth_poly=ch_mouth_poly, dataset=dataset)
    gs = geopandas.GeoSeries(ch_nw_poly)
    gs.crs = 'epsg:'+str(dataset.crs.to_epsg())
    gs.to_file(dirname + rivername + '_channels.shp')

    node_df, edge_df = gdfs_from_D_primal(D_primal, dataset)
    edge_df = edge_df.drop('widths', axis=1)
    node_df.to_file(dirname + rivername + "_node_df.shp")
    edge_df.to_file(dirname + rivername + "_edge_df.shp")

    with open(dirname + rivername + "_G_rook.pickle", "wb") as f:
        pickle.dump(G_rook, f)
    with open(dirname + rivername +"_D_primal.pickle", "wb") as f:
        pickle.dump(D_primal, f)
    with open(dirname + rivername +"_G_primal.pickle", "wb") as f:
        pickle.dump(D_primal, f)

def merge_and_plot_channel_polygons(fnames):
    """
    Merges multiple channel polygons from shapefiles and plots the resulting polygon.

    Parameters
    ----------
    fnames : list of str
        List of file paths to the shapefiles containing the channel polygons.

    Returns
    -------
    big_poly : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        The merged polygon resulting from the union of all input polygons.

    Notes
    -----
    - The function assumes that each shapefile contains at least one polygon.
    - If a shapefile contains multiple polygons, only the first one is considered.
    - The resulting merged polygon is plotted using matplotlib with the exterior in 'cornflowerblue' and interiors in white.
    """
    polys = []
    for fname in fnames:
        gdf = geopandas.read_file(fname)
        poly = gdf.iloc[0]['geometry'].buffer(0) # need to add case when there are multiple polygons
        polys.append(poly)
    big_poly = polys[0]
    for polygon in polys[1:]:
        big_poly = big_poly.union(polygon)
    plt.figure()
    for geom in big_poly.geoms:
        plt.fill(geom.exterior.xy[0], geom.exterior.xy[1], color='cornflowerblue')
        for interior in geom.interiors:
            plt.fill(interior.xy[0], interior.xy[1], 'w')
    plt.axis('equal')
    return big_poly

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

def create_and_plot_bars(G_rooks, ts, ax1=None, ax2=None, depo_cmap="Blues", erosion_cmap="Reds"):
    """
    Create preserved scroll bar polygons and erosion polygons from a list of rook neighborhood graphs 
    and plot them. It also handles the creation of the color map for the plot.

    Parameters
    ----------
    G_rooks : list
        A list of rook neighborhood graphs.
    ts : int
        The last time step to consider.
    ax1 : matplotlib.axes.Axes, optional
        The axes on which to plot the deposition polygons. Defaults to None.
    ax2 : matplotlib.axes.Axes, optional
        The axes on which to plot the erosion polygons. Defaults to None.
    depo_cmap : str, optional
        The name of the color map to use for the deposition polygons. Defaults to "Blues".
    erosion_cmap : str, optional
        The name of the color map to use for the erosion polygons. Defaults to "Reds".

    Returns
    -------
    chs : list
        A list of channel polygons.
    bars : list
        A list of scroll bar polygons.
    erosions_final : list
        A list of final erosion polygons.

    Example
    -------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,15), sharex=True, sharey=True)
    chs, bars, erosions = rb.create_and_plot_bars(G_rooks, len(G_rooks), ax1=ax1, ax2=ax2)
    """
    if ts < 2:
        print('ts must be larger than 1!')
        return None, None, None
    if ts > len(G_rooks):
        print('ts must be smaller than the length of G_rooks!')
        return None, None, None
    dates = []
    for G_rook in G_rooks[:ts]:
        dates.append(datetime.strptime(G_rook.name[12:20], "%Y%m%d"))
    df = pd.DataFrame(dates, columns=['date'])
    # Determine the range of dates
    min_date = df['date'].min()
    max_date = df['date'].max()
    # Generate a list of January 1st dates within the range
    jan_first_dates = pd.date_range(start=min_date, end=max_date, freq='AS')
    # Create a dataframe for the January 1st dates
    jan_first_df = pd.DataFrame(jan_first_dates, columns=['date'])
    # Concatenate the original dataframe with the January 1st dataframe
    combined_df = pd.concat([df, jan_first_df], ignore_index=True)
    combined_df['date_type'] = 0
    combined_df.iloc[len(df):, 1] = 1
    # Sort by datetime column
    df = combined_df.sort_values(by='date').reset_index(drop=True)
    df['timedelta'] = df['date'] - df['date'].min()
    df['n_days'] = df['timedelta'].dt.days
    n_days = df.n_days.values

    bars = [] # these are 'scroll' bars - shapely MultiPolygon objects that correspond to one time step
    chs = [] # list of channels - shapely Polygon objects
    all_chs = [] # list of merged channels (to be used for erosion)
    erosions = []
    erosion_all_chs = []
    cmap = plt.get_cmap(depo_cmap)
    # create list of channels:
    for i in trange(ts-1):
        ch1 = create_channel_nw_polygon(G_rooks[i])
        ch1 = ch1.buffer(0)
        if type(ch1) == MultiPolygon:
            areas = []
            for geom in ch1.geoms:
                areas.append(geom.area)
            ch1 = ch1.geoms[np.argmax(areas)]
        ch2 = create_channel_nw_polygon(G_rooks[i+1])
        ch2 = ch2.buffer(0)
        if type(ch2) == MultiPolygon:
            areas = []
            for geom in ch2.geoms:
                areas.append(geom.area)
            ch2 = ch2.geoms[np.argmax(areas)]
        chs.append(ch1)
    chs.append(ch2) # append last channel
    # create list of merged channels:
    all_ch = chs[ts-1]
    all_chs.append(all_ch)
    for i in trange(2, ts): 
        all_ch = all_ch.union(chs[ts-i])
        all_chs.append(all_ch)
    erosion_all_ch = chs[0]
    erosion_all_chs.append(erosion_all_ch)
    for i in trange(1, ts-1): 
        erosion_all_ch = erosion_all_ch.union(chs[i])
        erosion_all_chs.append(erosion_all_ch)
    # create scroll bars and plot them:
    for i in trange(ts-1): # create scroll bars
        bar = chs[i].difference(all_chs[ts-i-2]) # scroll bar defined by difference
        bars.append(bar)
        erosion = chs[i+1].difference(erosion_all_chs[i])
        erosions.append(erosion)
        if ax1: # plotting
            color = cmap(i/float(ts))
            if type(bar) == MultiPolygon or type(bar) == GeometryCollection:
                for b in bar.geoms:
                    if type(b) == Polygon:
                        if len(b.interiors) == 0:
                            ax1.fill(b.exterior.xy[0], b.exterior.xy[1], facecolor=color, edgecolor='k', linewidth=0.2)
                        else:
                            plot_polygon(ax1, b, facecolor=color, edgecolor='k', linewidth=0.2)
            if type(bar) == Polygon:
                if len(bar.interiors) == 0:
                    ax1.fill(bar.exterior.xy[0], bar.exterior.xy[1], facecolor=color, edgecolor='k', linewidth=0.2)
                else:
                    plot_polygon(ax1, bar, facecolor=color, edgecolor='k', linewidth=0.2)
    if ax1:
        plot_polygon(ax1, ch2, facecolor='lightblue', edgecolor='k')
        ax1.set_aspect('equal')
        date = datetime.strptime(G_rooks[ts-1].name[12:20], "%Y%m%d")
        ax1.set_title('Deposition, ' + date.strftime('%m')+'/'+date.strftime('%d')+'/'+date.strftime('%Y'))
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        dummy_data = np.vstack((n_days, n_days))
        dummy_im = ax1.imshow(dummy_data, cmap=cmap)
        dummy_im.remove() # remove the image
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        cbar = plt.colorbar(dummy_im, ax=ax1, shrink=0.5)
        cbar.set_label('age', fontsize=12)
        # Define the tick values and labels you want
        tick_values = df.n_days[df.date_type == 1].values
        tick_labels = df.date[df.date_type == 1].dt.year.values
        # Set the ticks and tick labels on the colorbar
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels(tick_labels)

    # compute final erosional polygons:
    erosions_final = []
    cmap = plt.get_cmap(erosion_cmap)
    for i in trange(ts-1):
        color = cmap(i/float(ts))
        if i==0:
            all_bars = chs[i]
        else:
            all_bars = all_bars.union(bars[i]) # all depositional area up until this point in time
        erosion = erosions[i].difference(all_bars) # remove net depositional areas from final erosion
        erosions_final.append(erosion)
        if ax2: # plotting
            if type(erosion) == MultiPolygon or type(erosion) == GeometryCollection:
                for b in erosion.geoms:
                    if type(b) == Polygon:
                        if len(b.interiors) == 0:
                            ax2.fill(b.exterior.xy[0], b.exterior.xy[1], facecolor=color, edgecolor='k', linewidth=0.2)
                        else:
                            plot_polygon(ax2, b, facecolor=color, edgecolor='k', linewidth=0.2)
            if type(erosion) == Polygon:
                if len(erosion.interiors) == 0:
                    ax2.fill(erosion.exterior.xy[0], erosion.exterior.xy[1], facecolor=color, edgecolor='k', linewidth=0.2)
                else:
                    plot_polygon(ax2, erosion, facecolor=color, edgecolor='k', linewidth=0.2)
    if ax2:
        plot_polygon(ax2, chs[0], facecolor='lightblue', edgecolor='k')
        ax2.set_aspect('equal')
        date = datetime.strptime(G_rooks[ts-1].name[12:20], "%Y%m%d")
        ax2.set_title('Erosion, ' + date.strftime('%m')+'/'+date.strftime('%d')+'/'+date.strftime('%Y'))
        xlim = ax2.get_xlim()
        ylim = ax2.get_ylim()
        dummy_data = np.vstack((n_days, n_days))
        dummy_im = ax2.imshow(dummy_data, cmap=cmap)
        dummy_im.remove() # remove the image
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        cbar = plt.colorbar(dummy_im, ax=ax2, shrink=0.5)
        cbar.set_label('age', fontsize=12)
        # Define the tick values and labels you want
        tick_values = df.n_days[df.date_type == 1].values
        tick_labels = df.date[df.date_type == 1].dt.year.values
        # Set the ticks and tick labels on the colorbar
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels(tick_labels)
        plt.tight_layout()
    return chs, bars, erosions_final

def create_geodataframe_from_bank_polygons(G_rooks, crs):
    """
    Creates a GeoDataFrame from bank polygons.

    Parameters
    ----------
    G_rooks : list
        A list of graph objects, each containing nodes with 'bank_polygon' attributes.
    crs : str
        Coordinate reference system in EPSG code format (e.g., '4326').

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the bank polygons with additional attributes:
        - 'geometry': The bank polygons.
        - 'date': The date extracted from the graph object names.
        - 'timedelta': The time difference from the earliest date.
        - 'n_days': The number of days since the earliest date.
        - 'length': The length of each polygon.
        - 'type': The type of bank (0, 1, or 2).
    """
    bank_polys = []
    bank_type = []
    dates = []
    for G_rook in G_rooks:
        dates.append(datetime.strptime(G_rook.name[12:20], "%Y%m%d"))
    date = []
    for j in range(len(G_rooks)):
        if len(G_rook) > 0:
            for i in range(0, len(G_rooks[j])):
                bank_polys.append(G_rooks[j].nodes()[i]['bank_polygon'])
                date.append(dates[j])           
                if i == 0:
                    bank_type.append(0)
                elif i == 1:
                    bank_type.append(1)
                else:
                    bank_type.append(2)
    gdf = geopandas.GeoDataFrame(bank_polys, columns = ['geometry'])
    gdf['date'] = date
    gdf['timedelta'] = gdf['date'] - gdf['date'].min()
    gdf['n_days'] = gdf['timedelta'].dt.days
    gdf['length'] = gdf.length
    gdf['type'] = bank_type
    gdf.crs = 'epsg:'+crs
    gdf = gdf.sort_values(by='date')
    gdf = gdf.reset_index(drop=True)
    return gdf

def create_geodataframe_from_deposition_erosion_polygons(G_rooks, polys, crs, min_area=10):
    """
    Create a GeoDataFrame from polygons of deposition or erosion that were generated with the 'create_and_plot_bars' function.

    Parameters
    ----------
    G_rooks : list
        A list of rook neighborhood graphs.
    polys : list
        A list of Polygon objects to be included in the GeoDataFrame.
    crs : str
        The coordinate reference system to be used for the GeoDataFrame.
    min_area : float, optional
        The minimum area for a polygon to be included in the GeoDataFrame. Defaults to 10.

    Returns
    -------
    gdf : GeoDataFrame
        A GeoDataFrame containing the polygons and their associated attributes.
    """
    dates = []
    for G_rook in G_rooks:
        dates.append(datetime.strptime(G_rook.name[12:20], "%Y%m%d"))
    final_polys = []
    start_dates = []
    end_dates = []
    count = 0
    for poly in tqdm(polys):
        if poly.area > min_area:
            if type(poly) == Polygon and poly.area > min_area:
                final_polys.append(poly)
                start_dates.append(dates[count])
                end_dates.append(dates[count + 1])
            else:
                for geom in poly.geoms:
                    if type(geom) == Polygon and geom.area > min_area:
                        final_polys.append(geom)
                        start_dates.append(dates[count])
                        end_dates.append(dates[count + 1])
        count += 1
    gdf = geopandas.GeoDataFrame(final_polys, columns=['geometry'])
    gdf['start_date'] = start_dates
    gdf['end_date'] = end_dates
    gdf['length'] = gdf.length
    gdf['area'] = gdf.area
    gdf.crs = 'epsg:' + crs
    gdf = gdf.sort_values(by='start_date')
    gdf = gdf.reset_index(drop=True)
    gdf['start_year'] = gdf['start_date'].dt.year
    gdf['start_month'] = gdf['start_date'].dt.month
    gdf['start_day'] = gdf['start_date'].dt.day
    gdf['end_year'] = gdf['end_date'].dt.year
    gdf['end_month'] = gdf['end_date'].dt.month
    gdf['end_day'] = gdf['end_date'].dt.day
    gdf['dt'] = gdf['start_date'] - gdf['start_date'].min()
    gdf['n_days'] = gdf['dt'].dt.days
    gdf['timedelta'] = gdf['end_date'] - gdf['start_date']
    gdf['duration'] = gdf['timedelta'].dt.days
    gdf.drop(['start_date', 'end_date', 'timedelta', 'dt'], axis=1, inplace=True)
    return gdf

def plot_polygon(ax, poly, **kwargs):
    """
    Plot a polygon on a given matplotlib axes.

    This function creates a compound path from the exterior coordinates and any interior coordinates 
    (for polygons with holes), creates a patch from this path, adds it to a collection, and then adds 
    this collection to the axes. It then autoscales the view of the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the polygon.
    poly : shapely.geometry.Polygon
        The polygon to plot.
    **kwargs : dict
        Arbitrary keyword arguments to be passed to the PathPatch and PatchCollection constructors.

    Returns
    -------
    collection : matplotlib.collections.PatchCollection
        The collection containing the patch created from the polygon.
    """
    path = Path.make_compound_path(
           Path(np.asarray(poly.exterior.coords)[:, :2]),
           *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])
 
    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
     
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

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
    plt.axis('equal')

def main(fname, dirname, start_x, start_y, end_x, end_y, file_type, **kwargs):
    """
    Main function to extract and process the centerline of a river.

    Parameters
    ----------
    fname : str
        The filename of the input data.
    dirname : str
        The directory name where the input data is located.
    start_x : float
        The starting x-coordinate.
    start_y : float
        The starting y-coordinate.
    end_x : float
        The ending x-coordinate.
    end_y : float
        The ending y-coordinate.
    file_type : str
        The type of the input file.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    None
    """
    for k, v in kwargs.items():
        print('keyword argument: {} = {}'.format(k, v))
    D_primal, G_rook, G_primal, mndwi, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, xs, ys = extract_centerline(fname, dirname, start_x, start_y, end_x, end_y, file_type, **kwargs)
    if len(D_primal) > 0:
        start_node, inds = find_start_node(D_primal)
        edge_path = traverse_multigraph(D_primal, start_node)
        x, y, x_utm1, y_utm1, x_utm2, y_utm2 = get_bank_coords_for_main_channel(D_primal, mndwi, edge_path, dataset)

if __name__=='__main__':
    main(sys.argv[1], # fname
         sys.argv[2], # dirname
         sys.argv[3], # start_x
         sys.argv[4], # start_y
         sys.argv[5], # end_x
         sys.argv[6], # end_y
         sys.argv[7], # file_type
         **dict(arg.split('=') for arg in sys.argv[8:])) # kwargs