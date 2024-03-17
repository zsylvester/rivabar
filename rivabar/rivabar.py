import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import savgol_filter
import scipy.interpolate
from scipy.interpolate import CubicSpline
from scipy.spatial import distance
from scipy.spatial import cKDTree
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
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, MultiLineString, GeometryCollection, MultiPoint
from shapely.ops import polygonize_full, split, linemerge, nearest_points, unary_union
from libpysal import weights
import geopandas
import momepy
from itertools import combinations, permutations
import pickle
import warnings

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
    band1 = np.where((b1==0) & (b2==0), np.nan, b1)
    band2 = np.where((b1==0) & (b2==0), np.nan, b2)
    return (band1 - band2) / (band1 + band2)

def find_graph_edges_close_to_start_and_end_points(graph, start_x, start_y, end_x, end_y, left_utm_x, upper_utm_y, delta_x, delta_y):
    # find reasonable starting and ending points on graph edges
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
    x1, y1 = convert_to_utm(x1, y1, left_utm_x, upper_utm_y, delta_x, delta_y)
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
    tree = KDTree(np.vstack((edge_utm_xs, edge_utm_ys)).T)
    si = tree.query(np.reshape([x1, y1], (1, -1)))[1][0][0]
    start_ind = ss[si]
    end_ind = es[si]
    x = np.array(list(G_primal[start_ind][end_ind][0]['geometry'].xy[0]))
    y = np.array(list(G_primal[start_ind][end_ind][0]['geometry'].xy[1]))
    tree = KDTree(np.vstack((x, y)).T)
    ind = tree.query(np.reshape([x1, y1], (1, -1)))[1][0][0]
    if ind == 1:
        point = Point(x1, y1)
        line = LineString(np.vstack((x[ind:], y[ind:])).T)
        G_primal.nodes()[start_ind]['geometry'] = point
        G_primal[start_ind][end_ind][0]['geometry'] = line
        node = start_ind
    else:
        point = Point(x1, y1)
        line = LineString(np.vstack((x[:ind+1], y[:ind+1])).T)
        G_primal.nodes()[end_ind]['geometry'] = point
        G_primal[start_ind][end_ind][0]['geometry'] = line
        node = end_ind
    return G_primal, node

def convert_to_uint8(channel):
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
    
def extract_centerline(fname, dirname, start_x, start_y, end_x, end_y, file_type, 
    ch_belt_smooth_factor = 1e9, remove_smaller_components = False, delete_pixels_polys = False, 
    ch_belt_half_width = 2000, mndwi_threshold = 0.01, small_hole_threshold = 64, 
    min_g_primal_length = 100000, solidity_filter=True, radius = 50):
    """
    Extract channel centrelines and banks from a georeferenced image.

    Args:

        fname: filename
        dirname: name of directory
        start_x: estimate of UTM x coordinate of start point
        start_y: estimate of UTM y coordinate of start point
        end_x: estimate of UTM x coordinate of end point
        end_y: estimate of UTM y coordinate of end point
        file_type: type of file used as input; it can be 'water_index' or something else if using Landsat bands
        ch_belt_smooth_factor: smoothing factor for getting a channel belt centerline
        remove_smaller_components: remove small components from water index image if true
        delete_pixels_polys: list of polygons or 'False'; set pixels to zero in areas defined by polygons
        ch_belt_half_width: half channel belt width in pixels, used to define area of interest along channel
        mndwi_threshold: value for thresholding water index image (default is 0.01)
        small_hole_threshold: remove holes in water index image that are smaller than this value (in pixels); affects size of islands detected
        min_g_primal_length: minimum length of centerline graph, in meters, measured over all edges; processing is stopped if graph is not long enough (default is 100 km)
        solidity_filter: if 'True', objects in the water index image that have a lower solidity than 0.2 will be removed; 
            good for cleaning up complex water bodies but can be a problem when those bodies are connected to the river (e.g., parts of the Amazon)
        radius: defines the graph neighborhood of the main path in which nodes will be included. Default is 50; this might need to be increased
            when working with complex networks

    Returns:

        D_primal: directed multigraph that describes the channel centerline network and tries to capture the flow directions
        G_rook: rook neighborhood graph of the 'islands' that make up the channel belt; has only two elements (the two banks) if there are no islands / bars
        G_primal: undirected multigraph that describes the channel centerline network; only returned for QC / testing purposes
        mndwi: water index image that was used in the processing
        dataset: rasterio dataset
        left_utm_x: UTM x coordinate of the left edge of the image
        right_utm_x: UTM x coordinate of the right edge of the image
        lower_utm_y: UTM y coordinate of the lower edge of the image
        upper_utm_y: UTM y coordinate of the upper edge of the image
        xs: x coordinates (UTM) of the smoothed channel belt centerline
        ys: y coordinates (UTM) of the smoothed channel belt centerline

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
        if type(mndwi_threshold) == str:
            mndwi_threshold = float(mndwi_threshold)
        equ, mndwi, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, delta_x, delta_y = read_landsat_data(dirname, fname, mndwi_threshold = mndwi_threshold)
    if delete_pixels_polys:
        rst_arr = mndwi.astype('uint32').copy()
        shapes = ((geom, value) for geom, value in zip(delete_pixels_polys, np.ones((len(delete_pixels_polys),))))
        mndwi = features.rasterize(shapes=shapes, fill=0, out=rst_arr, transform=dataset.transform)

    if max(np.shape(mndwi)) > 2**16//2:
        print('maximum dimension of input image needs to be smaller than 32768!')
        return [], [], [], [], [], [], [], [], [], [], []
    
    print('removing small holes')
    mndwi = remove_small_holes(mndwi.astype('bool'), small_hole_threshold) # remove small holes
    if remove_smaller_components:
        mndwi_labels = label(mndwi)
        rp = regionprops_table(mndwi_labels, properties=['label', 'area', 'solidity'])
        df = pd.DataFrame(rp)
        mndwi = np.zeros(np.shape(mndwi))
        if solidity_filter:
            df = df.sort_values('area', ascending=False)
            for ind in df.index[df['solidity'] < 0.2]:
                mndwi[mndwi_labels == ind+1] = 1
        else:
            df = df.sort_values('area', ascending=False, ignore_index=True)
            mndwi[mndwi_labels == df.loc[0, 'label']] = 1
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
        test = nx.generators.ego_graph(graph, node, radius=radius)
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
    if main_path.length < 2000:  # this should be a parameter
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
    cline_polys = list(polygonize_full(clines))
    
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

    ch_belt_cl = LineString(np.vstack((xs, ys)).T)
    ch_belt_poly = ch_belt_cl.buffer(ch_belt_half_width)

    # make the channel belt centerline longer at both ends:
    ratio = 2*ch_belt_half_width/(np.sqrt((xs[1]-xs[0])**2 + (ys[1]-ys[0])**2))
    a1, b1 = getExtrapolatedLine((xs[1], ys[1]), (xs[0], ys[0]), ratio) # use the first two points
    ratio = 2*ch_belt_half_width/(np.sqrt((xs[-1]-xs[-2])**2 + (ys[-1]-ys[-2])**2))
    a2, b2 = getExtrapolatedLine((xs[-2], ys[-2]), (xs[-1], ys[-1]), ratio) # use the last two points

    # these are needed so that later we can get rid of the extra line segments at the beginning and end of G_primal:
    xcoords1 = xcoords[0]
    xcoords2 = xcoords[-1]
    ycoords1 = ycoords[0]
    ycoords2 = ycoords[-1]

    # lengthen the channel centerline as well:
    xcoords = np.hstack((b1[0], xcoords, b2[0]))
    ycoords = np.hstack((b1[1], ycoords, b2[1]))
    main_path = LineString(np.vstack((xcoords, ycoords)).T)

    # create polygons for the outer boundaries of the channel belt:
    main_polys = split(ch_belt_poly, main_path)
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
    gdf2.crs = dataset.crs.data['init'] # set the CRS

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

    # delete the unnecessary bits from the main bankline polygons (so that they are proper banklines):
    # row1 = int(ycoords1)
    # col1 = int(xcoords1)
    # xcoord1, ycoord1 = convert_to_utm(col1, row1, left_utm_x, upper_utm_y, delta_x, delta_y)
    # row1 = int(ycoords2)
    # col1 = int(xcoords2)
    # xcoord2, ycoord2 = convert_to_utm(col1, row1, left_utm_x, upper_utm_y, delta_x, delta_y)
    # ind11 = find_closest_point(xcoord1, ycoord1, np.vstack((x1_poly, y1_poly)).T)
    # ind12 = find_closest_point(xcoord2, ycoord2, np.vstack((x1_poly, y1_poly)).T)
    # ind21 = find_closest_point(xcoord1, ycoord1, np.vstack((x2_poly, y2_poly)).T)
    # ind22 = find_closest_point(xcoord2, ycoord2, np.vstack((x2_poly, y2_poly)).T)
    # x1, y1 = find_longer_segment_coords(Polygon(np.vstack((x1_poly, y1_poly)).T), ind11, ind12, ind11, ind11)            
    # x2, y2 = find_longer_segment_coords(Polygon(np.vstack((x2_poly, y2_poly)).T), ind21, ind22, ind21, ind21)
    # utm_coords.append(np.vstack((x1, y1)).T)
    # utm_coords.append(np.vstack((x2, y2)).T)

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
    D_primal = create_directed_multigraph(G_primal, G_rook, xs, ys, primal_start_ind, primal_end_ind)

    return D_primal, G_rook, G_primal, mndwi, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, xs, ys

def remove_dead_ends(graph, start_node, end_node):
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
    x = np.array(x)
    y = np.array(y)
    okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
    dx = np.diff(x[okay]); dy = np.diff(y[okay])      
    ds = np.sqrt(dx**2+dy**2)
    tck, u = scipy.interpolate.splprep([x[okay], y[okay]], s=smoothing_factor) # parametric spline representation of curve
    unew = np.linspace(0,1,1+int(sum(ds)/delta_s)) # vector for resampling
    out = scipy.interpolate.splev(unew,tck) # resampling
    xs = out[0]
    ys = out[1]
    return xs, ys

def smooth_line(x, y, spline_ds = 25, spline_smoothing = 10000, savgol_window = 21, savgol_poly_order = 3):
    x = savgol_filter(x, savgol_window, savgol_poly_order)
    y = savgol_filter(y, savgol_window, savgol_poly_order)
    xs, ys = resample_and_smooth(x, y, spline_ds, spline_smoothing)
    return xs, ys

def compute_s_distance(x, y):
    """function for computing first derivatives of a curve (centerline)
    x,y are cartesian coodinates of the curve
    outputs:
    dx - first derivative of x coordinate
    dy - first derivative of y coordinate
    ds - distances between consecutive points along the curve
    s - cumulative distance along the curve"""
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)   
    ds = np.sqrt(dx**2+dy**2)
    s = np.hstack((0, np.cumsum(ds[1:])))
    return s

def find_pixel_distance_between_nodes_and_other_node(graph, nodes, other_node):
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
    'Creates a line extrapolated in p1->p2 direction'
    a = p1
    b = (p1[0]+ratio*(p2[0]-p1[0]), p1[1]+ratio*(p2[1]-p1[1]))
    return a, b

def get_bank_coords(poly, mndwi, dataset, timer=False):
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
    row, col = dataset.index(x, y)
    row = np.array(row)-row2
    row[row < 0] = 0
    col = np.array(col)-col1
    col[col < 0] = 0
    if timer:
        for i in trange(len(row)):
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
    contour_lengths = []
    for i in range(len(contours)):
        contour_lengths.append(len(contours[i]))
    if len(contour_lengths) > 0:
        ind = np.argmax(np.array(contour_lengths))
        x = contours[ind][:,1]
        y = contours[ind][:,0]
        x_utm = dataset.xy(row2+np.array(y), col1+np.array(x))[0]
        y_utm = dataset.xy(row2+np.array(y), col1+np.array(y))[1]
    else:
        x_utm = []
        y_utm = []
    return x_utm, y_utm, ch_map

def compute_mndwi_small_dist(poly, dataset, mndwi):
    """
    returns distance transform of 'small' mndwi tile; unist are pixels
    """
    tile_size = 500 # this should ideally depend on the mean channel width (in pixels), but not sure how that can be done
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

# warnings.filterwarnings('error')

def set_half_channel_widths(G_primal, G_rook, dataset, mndwi):
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
                            cl_row, cl_col = dataset.index(cl_x, cl_y)
                            cl_row = np.array(cl_row)-row2
                            cl_col = np.array(cl_col)-col1
                            r, c = np.shape(mndwi_small_dist)
                            cl_row = cl_row[(cl_col >= 0) & (cl_col < c)]
                            cl_col = cl_col[(cl_col >= 0) & (cl_col < c)]
                            cl_col = cl_col[(cl_row >= 0) & (cl_row < r)]
                            cl_row = cl_row[(cl_row >= 0) & (cl_row < r)]
                            w = mndwi_small_dist[cl_row, cl_col] # channel half width (number of pixels)
                            G_primal[s][e][d]['half_widths'][node] = w
                    except:
                        pass

def save_shapefiles(dirname, fname, G_rook, dataset, fname_add_on=''):
    # ch_nw_poly = create_channel_nw_polygon(G_rook)
    # gs = geopandas.GeoSeries(ch_nw_poly)
    # gs.crs = dataset.crs.data['init']
    # gs.to_file(dirname + fname[:-4] + '_channels.shp')
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

def plot_im_and_lines(im, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, G_rook, G_primal, smoothing=False):
    fig = plt.figure()
    plt.imshow(im, extent = [left_utm_x, right_utm_x, lower_utm_y, upper_utm_y], cmap='Blues', alpha=0.5)
    for i in range(2):
        if type(G_rook.nodes()[i]['bank_polygon']) == Polygon:
            x = G_rook.nodes()[i]['bank_polygon'].exterior.xy[0]
            y = G_rook.nodes()[i]['bank_polygon'].exterior.xy[1]
        else:
            x = G_rook.nodes()[i]['bank_polygon'].xy[0]
            y = G_rook.nodes()[i]['bank_polygon'].xy[1]
        if smoothing:
            x, y = smooth_line(x, y, spline_ds = 100, spline_smoothing = 10000, savgol_window = 31, 
                                     savgol_poly_order = 3)
        if i == 0:
            plt.plot(x, y, color='r', linewidth=3)
        if i == 1:
            plt.plot(x, y, color='r', linewidth=3)
    for i in trange(2, len(G_rook.nodes)):
        x = G_rook.nodes()[i]['bank_polygon'].exterior.xy[0]
        y = G_rook.nodes()[i]['bank_polygon'].exterior.xy[1]
        if smoothing:
            x, y = smooth_line(x, y, spline_ds = 25, spline_smoothing = 1000, savgol_window = min(11, len(x)), 
                                     savgol_poly_order = 3)
        plt.plot(x, y, color='tab:blue')
    for s,e,d in tqdm(G_primal.edges):
        x = G_primal[s][e][d]['geometry'].xy[0]
        y = G_primal[s][e][d]['geometry'].xy[1]
        if smoothing:
            x, y = smooth_line(x, y, spline_ds = 25, spline_smoothing = 1000, savgol_window = min(11, len(x)), 
                                     savgol_poly_order = 3)
        plt.plot(x, y, 'k')
    return fig

def read_and_plot_im(dirname, fname):
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

def create_channel_nw_polygon(G_rook):
    """
    create a channel network polygon - a single polygon with holes where there are islands / bars

    """
    two_main_banks = G_rook.nodes[0]['bank_polygon'].union(G_rook.nodes[1]['bank_polygon'])
    difference = two_main_banks.convex_hull.difference(two_main_banks)
    count = 0
    for geom in difference.geoms:
        if len(G_rook) > 2:
            for i in range(2, len(G_rook)):
                if geom.contains(G_rook.nodes[i]['bank_polygon']):
                    break
            if geom.contains(G_rook.nodes[i]['bank_polygon']):
                break
            count += 1
        else:
            areas = []
            for geom in difference.geoms:
                areas.append(geom.area)
            count = np.argmax(areas)
    ch_belt = difference.geoms[count]
    exterior = np.vstack((ch_belt.exterior.xy[0], ch_belt.exterior.xy[1])).T
    interior = []
    for i in range(2, len(G_rook)):
        x = G_rook.nodes()[i]['bank_polygon'].exterior.xy[0]
        y = G_rook.nodes()[i]['bank_polygon'].exterior.xy[1]
        interior.append(list(map(tuple, np.vstack((x,y)).T)))
    ch_nw_poly = Polygon(exterior, holes=interior)
    return ch_nw_poly

def convert_geographic_proj_to_utm(dirname, fname, dstCrs):
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
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    if t < 0.0:
        return a
    elif t > 1.0:
        return b
    return a + t * ab

def closest_segment(line, x, y):
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
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_theta, -1, 1))  # In radians

def find_longer_segment_coords(polygon, i1, i2, xs, ys):
    """Find the longer segment between two points on a Shapely polygon."""
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

def create_directed_multigraph(G_primal, G_rook, xs, ys, primal_start_ind, primal_end_ind, flip_outlier_edges=False):
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
    # that that bar/island already has some directed edges added for .
    processed_nodes = [0, 1]
    nit = 0
    while len(processed_nodes) < len(G_rook):
        if nit > 1e8:
            print('something went wrong, breaking while loop!')
            break
        neighbors = []
        for node in processed_nodes:
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
    # these outlier directions are then flipped (but only flipped once); this works well in complex networks (e.g., Lena Delta),
    # but not in meandering rivers
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
    # sometimes there are sink nodes in D_primal (in addition to the end point) that need to be eliminated:
    sinks = [node for node in D_primal.nodes() if D_primal.out_degree(node) == 0]
    if len(sinks) > 1:
        print('there is more than one sink in D_primal!')
        print('these sinks are: ' + str(sinks))
    cycles = list(nx.simple_cycles(D_primal))
    if len(cycles) > 0:
        print('there are ' + str(len(cycles)) + 'cycles in D_primal!')
        print('these cycles are: ' + str(cycles))
    # if len(sinks) > 0:
    #     for sink in sinks:
    #         if len([(u, v, k) for u, v, k in D_primal.edges(keys=True) if u == sink or v == sink]) > 1:
    #             edges = [(u, v, k) for u, v, k in D_primal.edges(keys=True) if u == sink or v == sink]
    #             D_temp = D_primal.copy()
    #             for edge in edges:
    #                 # Remove the existing edge
    #                 D_temp.remove_edge(edge[0], edge[1], edge[2])
    #                 # Add the edge in the opposite direction
    #                 D_temp.add_edge(edge[1], edge[0], edge[2])
    #                 if len(list(nx.simple_cycles(D_temp))) == 0:
    #                     print('flipping a flow direction in D_primal')
    #                     D_primal.add_edge(edge[1], edge[0], edge[2])
    #                     for key in D_primal[edge[0]][edge[1]][edge[2]].keys():
    #                         D_primal[edge[1]][edge[0]][edge[2]][key] = D_primal[edge[0]][edge[1]][edge[2]][key]
    #                     D_primal.remove_edge(edge[0], edge[1], edge[2])
    return D_primal

def set_width_weights(G_primal):
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
    point = np.array([x1, y1])
    other_points = np.array(other_points)
    distances = np.linalg.norm(other_points - point, axis=1)
    min_index = np.argmin(distances)
    return min_index

def flip_coords_and_widths(D_primal):
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
    return df

def compute_curvature(x,y):
    """function for computing first derivatives and curvature of a curve (centerline)
    x,y are cartesian coodinates of the curve
    outputs:
    curvature - curvature of the curve (in 1/units of x and y)
    s - cumulative distance along the curve"""
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)  
    ds = np.sqrt(dx**2+dy**2)
    s = np.cumsum(ds)
    ddx = np.gradient(dx) # second derivatives 
    ddy = np.gradient(dy) 
    curvature = (dx*ddy-dy*ddx)/((dx**2+dy**2)**1.5)
    return curvature, s

def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res

def find_zero_crossings(curve):
    """find zero crossings of a curve
    input: 
    a one-dimensional array that describes the curve
    outputs: 
    loc_zero_curv - indices of zero crossings
    loc_max_curv - indices of maximum values"""
    n_curv = abs(np.diff(np.sign(curve)))
    n_curv[find(n_curv==2)] = 1
    loc_zero_curv = find(n_curv)
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
        max_local_ind = find(curve[loc_zero_curv[i-1]:loc_zero_curv[i]]==max_curv[i-1])
        if len(max_local_ind)>1:
            loc_max_curv[i-1] = loc_zero_curv[i-1] + max_local_ind[0]
        elif len(max_local_ind)==1:
            loc_max_curv[i-1] = loc_zero_curv[i-1] + max_local_ind
        else:
            loc_max_curv[i-1] = 0
    return loc_zero_curv, loc_max_curv

def find_subpath(D_primal, root, depth_limit=10):
    nodes = nx.single_source_shortest_path_length(D_primal, root, cutoff=depth_limit)
    leaves = [node for node, depth in nodes.items() if depth == min(depth_limit, list(nodes.values())[-1])]
    D_primal_small = nx.subgraph(D_primal, nodes)
    all_paths = []
    for leaf in leaves:
        paths = nx.all_simple_edge_paths(D_primal_small, root, leaf)
        all_paths.extend(paths)
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
    current_node = start_node
    edge_path = []
    subpath_depth = min(subpath_depth, len(G))
    current_path = []
    while True:
        next_path = find_subpath(G, current_node, subpath_depth)
        if next_path:
            # sometimes there is a cycle and we need to break it:
            nodes_in_current_path = list(set(element for tuple_ in current_path for element in tuple_))
            nodes_in_next_path = list(set(element for tuple_ in next_path for element in tuple_))
            if nodes_in_current_path == nodes_in_next_path:
                break
            edge_path.extend(next_path)
            current_node = edge_path[-1][1]
            current_path = next_path.copy()
        else:
            break
    return edge_path

def find_start_node(D_primal):
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

def get_bank_coords_for_main_channel(D_primal, mndwi, edge_path, dataset):
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
            # only add linestrings that are not aligned with thex or y axis (average offset relative to the axes is larger than 0.1 m):
            if (np.sum(np.abs(np.diff(geom.xy[0])))/len(geom.xy[0]) > 1) and (np.sum(np.abs(np.diff(geom.xy[1])))/len(geom.xy[1]) > 0.1):
                x_utm1.extend(geom.xy[0])
                y_utm1.extend(geom.xy[1])
    else:
        x_utm1, y_utm1 = non_overlap_1.xy[0], non_overlap_1.xy[1]
    if type(non_overlap_2) == MultiLineString:
        x_utm2, y_utm2 = [], []
        for geom in non_overlap_2.geoms:
            # only add linestrings that are not aligned with thex or y axis (average offset relative to the axes is larger than 0.1 m):
            if (np.sum(np.abs(np.diff(geom.xy[0])))/len(geom.xy[0]) > 1) and (np.sum(np.abs(np.diff(geom.xy[1])))/len(geom.xy[1]) > 0.1):
                x_utm2.extend(geom.xy[0])
                y_utm2.extend(geom.xy[1])
    else:
        x_utm2, y_utm2 = non_overlap_2.xy[0], non_overlap_2.xy[1]
    return x, y, x_utm1, y_utm1, x_utm2, y_utm2

def get_channel_widths_along_path(D_primal, path):
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

def plot_directed_graph(D_primal, mndwi, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, edge_path=False):
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
            plt.plot(x, y, 'r', alpha=0.5)
    # plot main path:
    if edge_path:
        for s,e,d in edge_path:
            x = D_primal[s][e][d]['geometry'].xy[0]
            y = D_primal[s][e][d]['geometry'].xy[1]
            plt.plot(x, y, 'b', alpha=0.5)
    # plot arrows:
    for s,e,d in D_primal.edges:
        x1 = D_primal.nodes()[s]['geometry'].xy[0][0]
        y1 = D_primal.nodes()[s]['geometry'].xy[1][0]
        x2 = D_primal.nodes()[e]['geometry'].xy[0][0]
        y2 = D_primal.nodes()[e]['geometry'].xy[1][0]
        plt.arrow(x1, y1, x2-x1, y2-y1, width=100, length_includes_head=True, linewidth=None)

def calculate_iou(poly1, poly2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union != 0 else 0

def modified_iou(poly1, poly2):
    # Calculate the intersection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        intersection = poly1.intersection(poly2).area
    # Determine the smaller polygon
    smaller_poly_area = min(poly1.area, poly2.area)
    # Calculate the fraction of the smaller polygon that is inside the larger polygon
    return intersection / smaller_poly_area if smaller_poly_area != 0 else 0

def cluster_polygons(gdf, iou_threshold, max_days = 2*365):
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
    """Returns a list of numbers between start and end, rounded to the specified decimal places."""
    step = 10 ** -decimal_places  # Calculate the step size based on decimal places
    numbers = []
    current_number = round(start, decimal_places)  # Begin with start rounded to specified places
    while current_number <= end:
        numbers.append(current_number)
        current_number += step  # Increment by the step size
        current_number = round(current_number, decimal_places)
    return numbers

def group_edges_to_subpaths(edges):
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
    ch_areas = []
    bar_areas = []
    chs = []
    ch_belts = []
    all_bars = []
    dates = []
    aoi_poly = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    for n_days in tqdm(gdf.n_days.unique()):
        date = gdf[(gdf['n_days']== n_days) & (gdf['type']==0)].year
        bank0 = gdf[(gdf['n_days']== n_days) & (gdf['type']==0)].geometry.values[0]
        bank0 = bank0.intersection(aoi_poly)
        inds = []
        if type(aoi_poly.difference(bank0)) == Polygon:
            geoms = [aoi_poly.difference(bank0)]
        else:
            geoms = aoi_poly.difference(bank0).geoms
        count = 0
        for geom in geoms:
            if geom.touches(Point([xmin, ymin])) or geom.touches(Point([xmin, ymax])):
                inds.append(count)
            count += 1
        for ind in inds:
            bank0 = bank0.union(geoms[ind])
        bank1 = gdf[(gdf['n_days']== n_days) & (gdf['type']==1)].geometry.values[0]
        bank1 = bank1.intersection(aoi_poly)
        inds = []
        if type(aoi_poly.difference(bank1)) == Polygon:
            geoms = [aoi_poly.difference(bank1)]
        else:
            geoms = aoi_poly.difference(bank1).geoms
        count = 0
        for geom in geoms:
            if geom.touches(Point([xmax, ymin])) or geom.touches(Point([xmax, ymax])):
                inds.append(count)
            count += 1
        for ind in inds:
            bank1 = bank1.union(geoms[ind])
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
    Create two geodataframes (one for nodes and one for edges) from D_primal
    If you want to save the edge dataframe as a shapefile, you need to drop the 'widths' column
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
    # Open the input GeoTIFF file
    with rasterio.open(input_file) as src:
        # Define the cropping window
        width = src.width
        height = src.height
        max_rows = min(height-row_off, max_rows)
        max_cols = min(height-col_off, max_cols)
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

def write_shapefiles_and_graphs(G_rook, D_primal, dataset, dirname, rivername):
    ch_nw_poly = create_channel_nw_polygon(G_rook)
    gs = geopandas.GeoSeries(ch_nw_poly)
    gs.crs = dataset.crs.data['init']
    gs.to_file(dirname + rivername + '_channels.shp')

    node_df, edge_df = gdfs_from_D_primal(D_primal, dataset)
    edge_df = edge_df.drop('widths', axis=1)
    node_df.to_file(dirname + rivername + "_node_df.shp")
    edge_df.to_file(dirname + rivername + "_edge_df.shp")

    with open(dirname + rivername + "_G_rook.pickle", "wb") as f:
        pickle.dump(G_rook, f)
    with open(dirname + rivername +"_D_primal.pickle", "wb") as f:
        pickle.dump(D_primal, f)

def main(fname, dirname, start_x, start_y, end_x, end_y, file_type, **kwargs):
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