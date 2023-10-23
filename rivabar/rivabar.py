import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import savgol_filter
import scipy.interpolate
from scipy.spatial import distance
from scipy.spatial import cKDTree
from tqdm import tqdm, trange
import pandas as pd
from skimage.morphology import skeletonize, remove_small_holes
from skimage.measure import label, regionprops, regionprops_table, find_contours
import sknw # https://github.com/Image-Py/sknw
import networkx as nx
from sklearn.neighbors import KDTree
import glob
import rasterio
from rasterio.plot import adjust_band
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject, Resampling
from descartes import PolygonPatch
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, MultiLineString, GeometryCollection, MultiPoint
from shapely.ops import polygonize, polygonize_full, split, linemerge, nearest_points
from libpysal import weights
import geopandas
import momepy
import copy
from itertools import combinations

def convert_to_utm(x, y, left_utm_x, upper_utm_y, delta_x, delta_y):
    x_utm = left_utm_x + 0.5*delta_x + x*delta_x 
    y_utm = upper_utm_y + 0.5*delta_y + y*delta_y 
    return x_utm, y_utm

def insert_node(graph, start_ind, end_ind, left_utm_x, upper_utm_y, delta_x, delta_y, start_x, start_y):
    x = np.array(list(graph[start_ind][end_ind][0]['pts'][:, 1]))
    y = np.array(list(graph[start_ind][end_ind][0]['pts'][:, 0]))
    x, y = convert_to_utm(x, y, left_utm_x, upper_utm_y, delta_x, delta_y)
    tree = KDTree(np.vstack((x, y)).T)
    edge_ind = tree.query(np.reshape([start_x, start_y], (1, -1)))[1][0][0]
    node = max(list(graph.nodes)) + 1
    graph.add_node(node, pts = graph[start_ind][end_ind][0]['pts'][edge_ind],
               o = graph[start_ind][end_ind][0]['pts'][edge_ind].astype('float'))
    x = list(graph[start_ind][end_ind][0]['pts'][:, 1])
    y = list(graph[start_ind][end_ind][0]['pts'][:, 0])
    point1 = np.array([x[0], y[0]])
    point2 = np.array([x[-1], y[-1]])
    point3 = np.array([graph.nodes()[start_ind]['o'][1], graph.nodes()[start_ind]['o'][0]])
    if np.linalg.norm(point1 - point3) < np.linalg.norm(point2 - point3):
        graph.add_edge(start_ind, node, pts = graph[start_ind][end_ind][0]['pts'][:edge_ind])
        # ps = graph[start_ind][node][0]['pts']
        graph.add_edge(node, end_ind, pts = graph[start_ind][end_ind][0]['pts'][edge_ind:])
        # ps = graph[node][end_ind][0]['pts']
    else:
        graph.add_edge(start_ind, node, pts = graph[start_ind][end_ind][0]['pts'][edge_ind:])
        # ps = graph[start_ind][node][0]['pts']
        graph.add_edge(node, end_ind, pts = graph[start_ind][end_ind][0]['pts'][:edge_ind])
        # ps = graph[node][end_ind][0]['pts']
    graph.remove_edge(start_ind, end_ind)
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
    # ind = np.where((x==x1) & (y==y1))[0]
    print('ind: '+str(ind))
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

def read_landsat_data_multiple_tifs(dirname, fname, mndwi_threshold=0.01):
    if fname[:4] == 'LC08':
        dataset = rasterio.open(dirname+fname+'/'+fname+'_B3.TIF')
        band3 = dataset.read(1)
        dataset = rasterio.open(dirname+fname+'/'+fname+'_B6.TIF')
        band6 = dataset.read(1)
        dataset = rasterio.open(dirname+fname+'/'+fname+'_B2.TIF')
        band2 = dataset.read(1)
        dataset = rasterio.open(dirname+fname+'/'+fname+'_B4.TIF')
        band4 = dataset.read(1)
        rgb = np.stack([band4, band3, band2], axis=-1)
        rgb_norm = adjust_band(rgb)
        left_utm_x = dataset.transform[2]
        upper_utm_y = dataset.transform[5]
        delta_x = dataset.transform[0]
        delta_y = dataset.transform[4]
        nxpix = rgb.shape[1]
        nypix = rgb.shape[0]
        right_utm_x = left_utm_x + delta_x*nxpix
        lower_utm_y = upper_utm_y + delta_y*nypix
        R, G, B = cv2.split(rgb_norm)
        R8 = R*255
        R8 = R8.astype('uint8')
        G8 = G*255
        G8 = G8.astype('uint8')
        B8 = B*255
        B8 = B8.astype('uint8')
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        output1_R = clahe.apply(R8)
        output1_G = clahe.apply(G8)
        output1_B = clahe.apply(B8)
        equ = cv2.merge((output1_R, output1_G, output1_B))
        mndwi = normalized_difference(band3, band6)
        mndwi[mndwi > mndwi_threshold] = 1
        mndwi[mndwi != 1] = 0
    else: # landsat 4 and 5
        dataset = rasterio.open(dirname+fname+'/'+fname+'_B3.TIF')
        band3 = dataset.read(1)
        dataset = rasterio.open(dirname+fname+'/'+fname+'_B5.TIF')
        band5 = dataset.read(1)
        dataset = rasterio.open(dirname+fname+'/'+fname+'_B2.TIF')
        band2 = dataset.read(1)
        dataset = rasterio.open(dirname+fname+'/'+fname+'_B1.TIF')
        band1 = dataset.read(1)
        rgb = np.stack([band3, band2, band1], axis=-1)
        rgb_norm = adjust_band(rgb)
        left_utm_x = dataset.transform[2]
        upper_utm_y = dataset.transform[5]
        delta_x = dataset.transform[0]
        delta_y = dataset.transform[4]
        nxpix = rgb.shape[1]
        nypix = rgb.shape[0]
        right_utm_x = left_utm_x + delta_x*nxpix
        lower_utm_y = upper_utm_y + delta_y*nypix
        R, G, B = cv2.split(rgb_norm)
        R8 = R*255
        R8 = R8.astype('uint8')
        G8 = G*255
        G8 = G8.astype('uint8')
        B8 = B*255
        B8 = B8.astype('uint8')
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        output1_R = clahe.apply(R8)
        output1_G = clahe.apply(G8)
        output1_B = clahe.apply(B8)
        equ = cv2.merge((output1_R, output1_G, output1_B))
        mndwi = normalized_difference(band2, band5)
        mndwi[mndwi > mndwi_threshold] = 1
        mndwi[mndwi != 1] = 0
    return equ, mndwi, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, delta_x, delta_y

def read_landsat_data_single_tif(dirname, fname, mndwi_threshold=0.01):
    if fname[:4] == 'LC08':
        dataset = rasterio.open(dirname+fname)
        band2 = dataset.read(2)
        band3 = dataset.read(3)
        band4 = dataset.read(4)
        rgb = np.stack([band4, band3, band2], axis=-1)
        rgb_norm = adjust_band(rgb)
        left_utm_x = dataset.transform[2]
        upper_utm_y = dataset.transform[5]
        delta_x = dataset.transform[0]
        delta_y = dataset.transform[4]
        nxpix = rgb.shape[1]
        nypix = rgb.shape[0]
        right_utm_x = left_utm_x + delta_x*nxpix
        lower_utm_y = upper_utm_y + delta_y*nypix
        R, G, B = cv2.split(rgb_norm)
        R8 = R*255
        R8 = R8.astype('uint8')
        G8 = G*255
        G8 = G8.astype('uint8')
        B8 = B*255
        B8 = B8.astype('uint8')
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        output1_R = clahe.apply(R8)
        output1_G = clahe.apply(G8)
        output1_B = clahe.apply(B8)
        equ = cv2.merge((output1_R, output1_G, output1_B))
        B3 = dataset.read(3)
        B6 = dataset.read(6)
        mndwi = normalized_difference(B3, B6)
        mndwi[mndwi > mndwi_threshold] = 1
        mndwi[mndwi != 1] = 0
    else:
        dataset = rasterio.open(dirname+fname)
        band1 = dataset.read(1)
        band2 = dataset.read(2)
        band3 = dataset.read(3)
        rgb = np.stack([band3, band2, band1], axis=-1)
        rgb_norm = adjust_band(rgb)
        left_utm_x = dataset.transform[2]
        upper_utm_y = dataset.transform[5]
        delta_x = dataset.transform[0]
        delta_y = dataset.transform[4]
        nxpix = rgb.shape[1]
        nypix = rgb.shape[0]
        right_utm_x = left_utm_x + delta_x*nxpix
        lower_utm_y = upper_utm_y + delta_y*nypix
        R, G, B = cv2.split(rgb_norm)
        R8 = R*255
        R8 = R8.astype('uint8')
        G8 = G*255
        G8 = G8.astype('uint8')
        B8 = B*255
        B8 = B8.astype('uint8')
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        output1_R = clahe.apply(R8)
        output1_G = clahe.apply(G8)
        output1_B = clahe.apply(B8)
        equ = cv2.merge((output1_R, output1_G, output1_B))
        band5 = dataset.read(5)
        mndwi = normalized_difference(band2, band5)
        mndwi[mndwi > mndwi_threshold] = 1
        mndwi[mndwi != 1] = 0

    return equ, mndwi, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, delta_x, delta_y
    
def extract_centerline(fname, dirname, start_x, start_y, end_x, end_y, file_type, ratio = 10, 
    ch_belt_smooth_factor = 1e9, remove_smaller_components = False, delete_pixels_polys = False, 
    n_pixels_per_channel = 10, mndwi_threshold = 0.01, small_hole_threshold = 64):
    if file_type == 'multiple_tifs':
        equ, mndwi, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, delta_x, delta_y = read_landsat_data_multiple_tifs(dirname, fname, mndwi_threshold = mndwi_threshold)
    if file_type == 'single_tif':
        equ, mndwi, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, delta_x, delta_y = read_landsat_data_single_tif(dirname, fname, mndwi_threshold = mndwi_threshold)
    if file_type == 'water_index':
        dataset = rasterio.open(dirname + fname)
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

    if delete_pixels_polys:
        rst_arr = mndwi.astype('uint32').copy()
        shapes = ((geom, value) for geom, value in zip(delete_pixels_polys, np.ones((len(delete_pixels_polys),))))
        mndwi = features.rasterize(shapes=shapes, fill=0, out=rst_arr, transform=dataset.transform)

    print('removing small holes')
    mndwi = remove_small_holes(mndwi.astype('bool'), small_hole_threshold) # remove small holes
    if remove_smaller_components:
        mndwi_labels = label(mndwi)
        rp = regionprops_table(mndwi_labels, properties=['label', 'area'])
        df = pd.DataFrame(rp)
        ind = np.where(df['area'] == max(df['area']))[0][0]+1
        mndwi_labels[mndwi_labels != ind] = 0
        mndwi_labels[mndwi_labels > 0] = 1
        mndwi = mndwi_labels
    print('running skeletonization')
    try:
        skeleton = skeletonize(mndwi) 
    except: # sometimes the default (Zhang) algorithm gives an error and the Lee method doesn't
        print('Skeletonization using the Zhang method failed; switching to the Lee method.')
        skeleton = skeletonize(mndwi, method='lee') 

    print('building graph from skeleton')
    graph = sknw.build_sknw(skeleton, multi=True) # build multigraph from skeleton

    print('finding reasonable starting and ending points on graph edges')
    start_ind1, end_ind1, start_ind2, end_ind2 = \
        find_graph_edges_close_to_start_and_end_points(graph, start_x, start_y, end_x, end_y, \
                                       left_utm_x, upper_utm_y, delta_x, delta_y)
    graph, start_ind = insert_node(graph, start_ind1, end_ind1, left_utm_x, upper_utm_y, delta_x, delta_y, start_x, start_y)
    graph, end_ind = insert_node(graph, start_ind2, end_ind2, left_utm_x, upper_utm_y, delta_x, delta_y, end_x, end_y)
    
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

    print('start_ind: ' + str(start_ind))
    print('end_ind: ' + str(end_ind))
        
    print('finding nodes that are within a certain radius of the path')
    nodes = [] 
    for node in tqdm(path):
        test = nx.generators.ego_graph(graph, node, radius=200)
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
    
    print('creating list of linestrings for polygonization')
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
            polys.append(poly)
        for poly in cline_polys[3].geoms:
            poly = Polygon(poly)
            if poly.is_valid:
                polys.append(poly)
            else:
                poly = poly.buffer(0)
                polys.append(poly)

        # get rid of polygons that are disconnected from the main fairway:
        main_path = LineString(np.vstack((xcoords, ycoords)).T)
        gdf = geopandas.GeoDataFrame(polys, columns = ['geometry'])
        rook = weights.Rook.from_dataframe(gdf)
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

    # make the channel belt centerline longer at both ends:
    a2, b2 = getExtrapolatedLine((xs[-2], ys[-2]), (xs[-1], ys[-1]), float(ratio)) # use the last two points
    a1, b1 = getExtrapolatedLine((xs[1], ys[1]), (xs[0], ys[0]), float(ratio)) # use the first two points
    xs = np.hstack((b1[0], xs, b2[0]))
    ys = np.hstack((b1[1], ys, b2[1]))

    # create fairway boundariess:
    xbound1, ybound1, xbound2, ybound2 = get_fairway_bounds(xs, ys, 10*int(n_pixels_per_channel)*delta_x)

    # these are needed so that later we can get rid of the extra line segments at the beginning and end of G_primal:
    xcoords1 = xcoords[0]
    xcoords2 = xcoords[-1]
    ycoords1 = ycoords[0]
    ycoords2 = ycoords[-1]
    print('first point coords:')
    print(xcoords1, ycoords1)
    print('second point coords:')
    print(xcoords2, ycoords2)

    # lengthen the channel centerline as well:
    xcoords = np.hstack((b1[0], xcoords, b2[0]))
    ycoords = np.hstack((b1[1], ycoords, b2[1]))
    main_path = LineString(np.vstack((xcoords, ycoords)).T)

    # create polygons for the outer boundaries of the channel belt:
    poly1 = Polygon(linemerge(MultiLineString([main_path, LineString(np.vstack((xbound2, ybound2)).T), 
                    LineString(np.vstack((np.array([xcoords[0], xbound2[0]]), np.array([ycoords[0], ybound2[0]]))).T),
                    LineString(np.vstack((np.array([xcoords[-1], xbound2[-1]]), np.array([ycoords[-1], ybound2[-1]]))).T)])))
    poly2 = Polygon(linemerge(MultiLineString([main_path, LineString(np.vstack((xbound1, ybound1)).T), 
                    LineString(np.vstack((np.array([xcoords[0], xbound1[0]]), np.array([ycoords[0], ybound1[0]]))).T),
                    LineString(np.vstack((np.array([xcoords[-1], xbound1[-1]]), np.array([ycoords[-1], ybound1[-1]]))).T)])))
    poly1 = poly1.buffer(0)
    poly2 = poly2.buffer(0)

    # trim down the boundary polygons so that they do not overlap with the channel belt polygons:
    if len(gdf) > 0:
        for geom in gdf['geometry']:
            if not geom.is_valid:
                geom = geom.buffer(0)
            if poly1.intersection(geom).area > 0:
                poly1 = poly1.difference(geom)
            if poly2.intersection(geom).area > 0:
                poly2 = poly2.difference(geom)

        # only keep the largest polygons from the resulting multipolygons:
        if type(poly1) != Polygon:
            poly1 = max(poly1.geoms, key=lambda a: a.area)
        if type(poly2) != Polygon:
            poly2 = max(poly2.geoms, key=lambda a: a.area)

    x, y = convert_to_utm(np.array(poly1.exterior.xy[0]), np.array(poly1.exterior.xy[1]),
                      left_utm_x, upper_utm_y, delta_x, delta_y)
    poly1_utm = Polygon(np.vstack((x, y)).T)
    x, y = convert_to_utm(np.array(poly2.exterior.xy[0]), np.array(poly2.exterior.xy[1]),
                      left_utm_x, upper_utm_y, delta_x, delta_y)
    poly2_utm = Polygon(np.vstack((x, y)).T)

    utm_polys = [poly1_utm, poly2_utm]
    print('creating centerline polygons')
    for poly in tqdm(gdf['geometry']): # this takes a while
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
    rook = weights.Rook.from_dataframe(gdf2)
    linestrings = []
    for poly_ind, neighb_dict in tqdm(rook):
        for key in neighb_dict:
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
    gdf_LS = geopandas.GeoDataFrame(linestrings, columns = ['geometry'])
    G_primal = momepy.gdf_to_nx(gdf_LS, approach="primal")
    degree = dict(nx.degree(G_primal))
    nx.set_node_attributes(G_primal, degree, 'degree')
    nodes, edges, sw = momepy.nx_to_gdf(G_primal, points=True, lines=True, spatial_weights=True)

    # create a better, leaner primal graph (some edges are duplicated and they have to be removed)
    unique_node_pairs = edges[['node_start', 'node_end']][edges[['node_start', 'node_end']].duplicated() == False]
    inds_to_be_removed = []
    for i in range(len(unique_node_pairs)):
        inds = edges[['node_start', 'node_end']][edges[['node_start', 'node_end']] \
                                                 == unique_node_pairs.iloc[i]].dropna().index
        for combination in combinations(inds, 2):
            intersection = edges['geometry'].loc[combination[0]].intersection(edges['geometry'].loc[combination[1]])
            if type(intersection) == LineString:
                inds_to_be_removed.append(combination[1])
            elif len(intersection.geoms) > 2:
            # if len(edges['geometry'].loc[combination[0]].intersection(edges['geometry'].loc[combination[1]]).geoms) > 2:
                inds_to_be_removed.append(combination[1])
    edges2 = edges.drop(inds_to_be_removed)
    G_primal = nx.from_pandas_edgelist(edges2, source='node_start', target='node_end', edge_attr=['geometry', 'mm_len'], 
                        create_using=nx.MultiGraph, edge_key=None)
    for node in list(G_primal.nodes):
        G_primal.nodes()[node]['geometry'] = nodes['geometry'].iloc[node-1]

    # G_primal, primal_start_ind = get_rid_of_extra_lines_at_beginning_and_end(G_primal, xcoords1, ycoords1, left_utm_x, upper_utm_y, delta_x, delta_y)
    # G_primal, primal_end_ind = get_rid_of_extra_lines_at_beginning_and_end(G_primal, xcoords2, ycoords2, left_utm_x, upper_utm_y, delta_x, delta_y)
    # G_primal = remove_dead_ends(G_primal, primal_start_ind, primal_end_ind)
    # print('start and end nodes in G_primal:')
    # print(primal_start_ind, primal_end_ind)

    print('getting bank coordinates for the two main banks')
    utm_coords = []
    x1, y1, ch_map = get_bank_coords(poly1_utm, mndwi, dataset, gdf2, timer=True)
    x2, y2, ch_map = get_bank_coords(poly2_utm, mndwi, dataset, gdf2, timer=True)

    # delete the unnecessary bits from the bankline polygons (so that they are proper banklines):
    tree1 = cKDTree(np.vstack((x1, y1)).T)
    tree2 = cKDTree(np.vstack((x2, y2)).T)

    row1 = int(ycoords1)
    col1 = int(xcoords1)
    xcoord, ycoord = convert_to_utm(col1, row1, left_utm_x, upper_utm_y, delta_x, delta_y)
    pad = min(n_pixels_per_channel*10, row1, col1) # this probably shouldn't be hardcoded (number of pixels that should be larger than likely half channel width)
    mndwi_small = mndwi[row1-pad:row1+pad, col1-pad:col1+pad]
    mndwi_small_dist = ndimage.distance_transform_edt(mndwi_small)
    max_distance1 = np.ceil(mndwi_small_dist[pad, pad])*1.5
    points_within_distance1 = tree1.query_ball_point((xcoord, ycoord), max_distance1*delta_x)
    points_within_distance2 = tree2.query_ball_point((xcoord, ycoord), max_distance1*delta_x)

    row1 = int(ycoords2)
    col1 = int(xcoords2)
    xcoord, ycoord = convert_to_utm(col1, row1, left_utm_x, upper_utm_y, delta_x, delta_y)
    pad = min(n_pixels_per_channel*10, row1, col1)
    mndwi_small = mndwi[row1-pad:row1+pad, col1-pad:col1+pad]
    mndwi_small_dist = ndimage.distance_transform_edt(mndwi_small)
    max_distance1 = np.ceil(mndwi_small_dist[pad, pad])*1.5
    points_within_distance3 = tree1.query_ball_point((xcoord, ycoord), max_distance1*delta_x)
    points_within_distance4 = tree2.query_ball_point((xcoord, ycoord), max_distance1*delta_x)

    if len(points_within_distance1) > 0 and len(points_within_distance3) > 0:
        ind11 = min(max(points_within_distance1), max(points_within_distance3))
        ind12 = max(min(points_within_distance1), min(points_within_distance3))
        x1 = x1[ind11:ind12]
        y1 = y1[ind11:ind12]
    if len(points_within_distance2) > 0 and len(points_within_distance4) > 0:
        ind21 = min(max(points_within_distance2), max(points_within_distance4))
        ind22 = max(min(points_within_distance2), min(points_within_distance4))
        x2 = x2[ind21:ind22]
        y2 = y2[ind21:ind22]

    utm_coords.append(np.vstack((x1, y1)).T)
    utm_coords.append(np.vstack((x2, y2)).T)

    print('getting bank coordinates for the rest of the islands')
    for i in trange(2, len(gdf2['geometry'])): # this takes a while
        if type(gdf2['geometry'].iloc[i]) == Polygon:
            x, y, ch_map = get_bank_coords(gdf2['geometry'].iloc[i], mndwi, dataset, gdf2)
            if len(x) > 0:
                if x[-1] != x[0] or y[-1] != y[0]: # sometimes the contour needs to be closed
                    x = np.hstack((x, x[0]))
                    y = np.hstack((y, y[0]))
            utm_coords.append(np.vstack((x, y)).T)
        else:
            X = np.array([])
            Y = np.array([])
            for geom in gdf2['geometry'].iloc[i].geoms:
                x, y, ch_map = get_bank_coords(geom, mndwi, dataset, gdf2)
                if len(x) > 0:
                    if x[-1] != x[0] or y[-1] != y[0]: # sometimes the contour needs to be closed
                        x = np.hstack((x, x[0]))
                        y = np.hstack((y, y[0]))
                X = np.hstack((X, x))
                Y = np.hstack((Y, y))
            utm_coords.append(np.vstack((X, Y)).T)

    # store results in a Rook graph:
    rook = weights.Rook.from_dataframe(gdf2)
    centroids = np.column_stack((gdf2.centroid.x, gdf2.centroid.y))
    G_rook = rook.to_networkx()
    nx.set_node_attributes(G_rook, [], "bank_polygon")
    nx.set_node_attributes(G_rook, [], "centroid")
    nx.set_node_attributes(G_rook, [], "cl_polygon")
    for i in range(len(G_rook)):
        if len(utm_coords[i]) > 0:
            if LineString(utm_coords[i]).is_closed:
                G_rook.nodes[i]["bank_polygon"] = Polygon(utm_coords[i])
            else:
                G_rook.nodes[i]["bank_polygon"] = LineString(utm_coords[i])
            G_rook.nodes[i]["centroid"] = centroids[i]
            G_rook.nodes[i]["cl_polygon"] = gdf2['geometry'].iloc[i]

    print('setting half channel widths')
    for node in tqdm(G_rook.nodes):
        set_half_channel_widths(mndwi, gdf2, node, G_primal, G_rook, dataset)

    gdf2['area'] = gdf2.area
    gdf2['length'] = gdf2.length

    return mndwi, G, G_primal, G_rook, gdf2, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y

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
    dx = np.diff(x); dy = np.diff(y)      
    ds = np.sqrt(dx**2+dy**2)
    tck, u = scipy.interpolate.splprep([x,y],s=smoothing_factor) # parametric spline representation of curve
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

def get_fairway_bounds(x, y, W):
    """function for finding coordinates of channel banks, given a centerline and a channel width
    x,y - coordinates of centerline
    W - channel width
    outputs:
    xm, ym - coordinates of channel banks (both left and right banks)"""
    x1 = x.copy()
    y1 = y.copy()
    x2 = x.copy()
    y2 = y.copy()
    ns = len(x)
    dx = np.diff(x); dy = np.diff(y) 
    ds = np.sqrt(dx**2+dy**2)
    x1[:-1] = x[:-1] + 0.5*W*np.diff(y)/ds
    y1[:-1] = y[:-1] - 0.5*W*np.diff(x)/ds
    x2[:-1] = x[:-1] - 0.5*W*np.diff(y)/ds
    y2[:-1] = y[:-1] + 0.5*W*np.diff(x)/ds
    x1[ns-1] = x[ns-1] + 0.5*W*(y[ns-1]-y[ns-2])/ds[ns-2]
    y1[ns-1] = y[ns-1] - 0.5*W*(x[ns-1]-x[ns-2])/ds[ns-2]
    x2[ns-1] = x[ns-1] - 0.5*W*(y[ns-1]-y[ns-2])/ds[ns-2]
    y2[ns-1] = y[ns-1] + 0.5*W*(x[ns-1]-x[ns-2])/ds[ns-2]
    return x1, y1, x2, y2

def getExtrapolatedLine(p1, p2, ratio):
    'Creates a line extrapolated in p1->p2 direction'
    a = p1
    b = (p1[0]+ratio*(p2[0]-p1[0]), p1[1]+ratio*(p2[1]-p1[1]))
    return a, b

def get_bank_coords(poly, mndwi, dataset, gdf2, timer=False):
    """this is the best solution so far for getting the bank coordinates"""
    tile_size = 100 # this should depend on the mean channel width (in pixels)
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
    ch_map = np.zeros(np.shape(mndwi_small))
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
                pad = int(w)+10
                tile = np.ones((pad*2, pad*2))
                tile[pad, pad] = 0
                tile = ndimage.distance_transform_edt(tile)
                tile[tile >= w+0.5] = 0 # needed to avoid issues with narrow channels
                tile[tile > 0] = 1
                if col[i]>pad and col[i]<mndwi_small.shape[1]-pad and row[i]>pad and row[i]<mndwi_small.shape[0]-pad:
                    ch_map[row[i]-pad:row[i]+pad, col[i]-pad:col[i]+pad] = np.maximum(tile, ch_map[row[i]-pad:row[i]+pad, col[i]-pad:col[i]+pad])

    else:
        for i in range(len(row)):
            if col[i]<mndwi_small.shape[1] and row[i]<mndwi_small.shape[0]:
                w = mndwi_small_dist[row[i], col[i]] # distance to closest channel bank at current location
                pad = int(w)+10
                tile = np.ones((pad*2, pad*2))
                tile[pad, pad] = 0
                tile = ndimage.distance_transform_edt(tile)
                tile[tile >= w+0.5] = 0 # needed to avoid issues with narrow channels
                tile[tile > 0] = 1
                if col[i]>pad and col[i]<mndwi_small.shape[1]-pad and row[i]>pad and row[i]<mndwi_small.shape[0]-pad:
                    ch_map[row[i]-pad:row[i]+pad, col[i]-pad:col[i]+pad] = np.maximum(tile, ch_map[row[i]-pad:row[i]+pad, col[i]-pad:col[i]+pad])

    ch_map[rasterized_poly == 0] = 1
    ch_map = ~(ch_map.astype('bool'))
    ch_map = ndimage.binary_dilation(ch_map) # needed to counteract 'w+0.5' above
    contours = find_contours(ch_map, 0.5)
    # if timer:
    #     for i in range(len(contours)):
    #         x = contours[i][:,1]
    #         y = contours[i][:,0]
    #         x_utm = dataset.xy(row2+np.array(y), col1+np.array(x))[0]
    #         y_utm = dataset.xy(row2+np.array(y), col1+np.array(y))[1]
    #         if LineString(np.vstack((x_utm, y_utm)).T).buffer(200).intersects(GeometryCollection(list(gdf2['geometry'].iloc[2:]))):
    #             return x_utm, y_utm, ch_map
    #         elif poly == gdf2['geometry'].iloc[0]:
    #             if LineString(np.vstack((x_utm, y_utm)).T).buffer(200).intersects(gdf2['geometry'].iloc[1]):
    #                 return x_utm, y_utm, ch_map
    #         elif poly == gdf2['geometry'].iloc[1]:
    #             if LineString(np.vstack((x_utm, y_utm)).T).buffer(200).intersects(gdf2['geometry'].iloc[0]):
    #                 return x_utm, y_utm, ch_map
    # else:
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

def set_half_channel_widths(mndwi, gdf2, node, G_primal, G_rook, dataset):
    poly = gdf2['geometry'].iloc[node]
    tile_size = 500 # this should depend on the mean channel width (in pixels)
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
    for s, e, d in G_primal.edges:
        if type(G_rook.nodes()[node]['cl_polygon']) == Polygon:
            if G_primal[s][e][d]['geometry'].touches(G_rook.nodes()[node]['cl_polygon']):
                intersection = G_primal[s][e][d]['geometry'].intersection(G_rook.nodes()[node]['cl_polygon'])
                if type(intersection) != Point and type(intersection) != MultiPoint:
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
                    w = mndwi_small_dist[cl_row, cl_col] # channel half width
                    if 'w1' not in G_primal[s][e][d].keys():
                        G_primal[s][e][d]['w1'] = w
                    elif 'w2' not in G_primal[s][e][d].keys():
                        G_primal[s][e][d]['w2'] = w
        else: 
            print(node)

def save_shapefiles(dirname, fname, G_rook, gdf2, fname_add_on):
    # ch_nw_poly = create_channel_nw_polygon(G_rook)
    # gs = geopandas.GeoSeries(ch_nw_poly)
    # gs.crs = dataset.crs.data['init']
    # gs.to_file(dirname + fname[:-4] + '_channels.shp')
    gdf2.to_file(dirname + fname[:-4] + fname_add_on + '_cl_polygons.shp')
    gs = geopandas.GeoSeries(G_rook.nodes()[0]['bank_polygon'])
    gs.crs = 'epsg:'+str(gdf2.crs.to_epsg())
    gs.to_file(dirname+fname[:-4]+fname_add_on+'_rb.shp')
    gs = geopandas.GeoSeries(G_rook.nodes()[1]['bank_polygon'])
    gs.crs = 'epsg:'+str(gdf2.crs.to_epsg())    #dataset.crs.data['init']
    gs.to_file(dirname + fname[:-4]+ fname_add_on+'_lb.shp')
    if len(G_rook) > 2:
        bank_polys = []
        for i in range(2, len(G_rook)):
            bank_polys.append(G_rook.nodes()[i]['bank_polygon'])
            gdf = geopandas.GeoDataFrame(bank_polys, columns = ['geometry'])
            gdf['area'] = gdf.area
            gdf['length'] = gdf.length
            gdf.crs = 'epsg:'+str(gdf2.crs.to_epsg())
            gdf.to_file(dirname + fname[:-4]+fname_add_on+'_bank_polygons.shp')

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
    plt.imshow(im, extent = [left_utm_x, right_utm_x, lower_utm_y, upper_utm_y])
    return im, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y

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

def main(fname, dirname, start_x, start_y, end_x, end_y, file_type, **kwargs):
    for k, v in kwargs.items():
        print('keyword argument: {} = {}'.format(k, v))
    mndwi, G_primal, G_rook, gdf2, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y = extract_centerline(fname, dirname, start_x, start_y, end_x, end_y, file_type, **kwargs)

if __name__=='__main__':
    main(sys.argv[1], # fname
         sys.argv[2], # dirname
         sys.argv[3], # start_x
         sys.argv[4], # start_y
         sys.argv[5], # end_x
         sys.argv[6], # end_y
         sys.argv[7], # file_type
         **dict(arg.split('=') for arg in sys.argv[8:])) # kwargs