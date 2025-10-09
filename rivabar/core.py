import sys
import numpy as np
from matplotlib import pyplot as plt
import warnings
import cv2
from tqdm import tqdm, trange
from itertools import combinations
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
import sknw
import networkx as nx
import geopandas
from libpysal import weights
from shapely.geometry import (
    LineString, Polygon, MultiPolygon, Point, 
    GeometryCollection, MultiLineString
)
from shapely.ops import split, polygonize_full, linemerge
import momepy

from .data_io import create_mndwi, save_shapefiles, write_shapefiles_and_graphs
from .geometry_utils import (
    find_graph_edges_close_to_start_and_end_points, 
    insert_node, convert_to_utm, getExtrapolatedLine
)
from .graph_processing import (
    remove_dead_ends, create_directed_multigraph, 
    get_rid_of_extra_lines_at_beginning_and_end,
    find_distance_between_nodes_and_other_node,
    find_pixel_distance_between_nodes_and_other_node,
    extend_cline, find_start_node, traverse_multigraph
)
from .utils import resample_and_smooth, find_numbers_between
from .polygon_processing import smooth_banklines
from .analysis import (
    set_half_channel_widths, get_bank_coords, 
    get_bank_coords_for_main_channel,
    get_channel_widths_along_path,
    analyze_width_and_wavelength,
    filter_outlier_paths
)
from .visualization import plot_im_and_lines, plot_graph_w_colors, read_and_plot_im


def _initial_skeletonization_and_graph_setup(fname, dirname, start_x, start_y, end_x, end_y, file_type,
                                           water_index_type, mndwi_threshold, delete_pixels_polys, small_hole_threshold,
                                           remove_smaller_components, solidity_filter):
    """Phase 1: Create MNDWI, skeletonize, and set up initial graph with start/end nodes."""
    print('Phase 1: Initial setup and skeletonization')
    
    # Create MNDWI
    mndwi, left_utm_x, upper_utm_y, right_utm_x, lower_utm_y, delta_x, delta_y, dataset = create_mndwi(
        dirname=dirname, fname=fname, file_type=file_type,
        water_index_type=water_index_type,
        mndwi_threshold=mndwi_threshold, delete_pixels_polys=delete_pixels_polys, 
        small_hole_threshold=small_hole_threshold,
        remove_smaller_components=remove_smaller_components, solidity_filter=solidity_filter
    )
    
    if mndwi is None:
        print('could not create MNDWI!')
        return None, None, None, None, None, None, None, None, None, None
    
    # Skeletonization
    print('running skeletonization')
    if np.nanmax(mndwi) != np.nanmin(mndwi):
        try:
            skeleton = skeletonize(mndwi) 
        except:
            print('Skeletonization using the Zhang method failed; switching to the Lee method.')
            try:
                skeleton = skeletonize(mndwi, method='lee')
            except:
                print('skeletonization failed')
                return None, None, None, None, None, None, None, None, None, None
    else:
        print('skeletonization failed')
        return None, None, None, None, None, None, None, None, None, None

    # Build initial graph and insert start/end nodes
    # Building graph from skeleton
    # graph = sknw.build_sknw(skeleton, multi=True)
    try:
        graph = sknw.build_sknw(skeleton, multi=True)
    except Exception as e:
        print(f'Failed to build graph from skeleton: {e}')
        return None, None, None, mndwi, dataset, left_utm_x, upper_utm_y, right_utm_x, lower_utm_y, delta_x, delta_y
    
    # Finding reasonable starting and ending points on graph edges
    start_x, start_y, end_x, end_y = map(float, [start_x, start_y, end_x, end_y])
    
    start_ind1, end_ind1, start_ind2, end_ind2 = find_graph_edges_close_to_start_and_end_points(
        graph, start_x, start_y, end_x, end_y, left_utm_x, upper_utm_y, delta_x, delta_y
    )
    
    try:
        graph, start_ind = insert_node(graph, start_ind1, end_ind1, left_utm_x, upper_utm_y, delta_x, delta_y, start_x, start_y)
        graph, end_ind = insert_node(graph, start_ind2, end_ind2, left_utm_x, upper_utm_y, delta_x, delta_y, end_x, end_y)
    except:
        print('could not find start and end points')
        return None, None, None, mndwi, dataset, left_utm_x, upper_utm_y, right_utm_x, lower_utm_y, delta_x, delta_y
    
    return graph, start_ind, end_ind, mndwi, dataset, left_utm_x, upper_utm_y, right_utm_x, lower_utm_y, delta_x, delta_y


def _find_main_path_with_fallbacks(graph, start_x, start_y, end_x, end_y, start_ind, end_ind, mndwi, left_utm_x, upper_utm_y, delta_x, delta_y):
    """Phase 2: Find path between start/end points with complex fallback logic for disconnected components."""
    print('Phase 2: Finding main path with fallbacks')
    path = None
    try:
        path = nx.shortest_path(graph, source=start_ind, target=end_ind, weight='weight')
    except:
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
            
        if dists[ind] <= 200:  # Bridge gap between disconnected components if close enough
            img = np.array(mndwi * 255, dtype=np.uint8)
            cv2.line(img, (int(graph.nodes()[node1]['o'][1]), int(graph.nodes()[node1]['o'][0])), 
                    (int(graph.nodes()[node2]['o'][1]), int(graph.nodes()[node2]['o'][0])), (255,255,255), 3)
            img = img//255
            mndwi_new = img.astype('bool')
            skeleton = skeletonize(mndwi_new)
            graph = sknw.build_sknw(skeleton, multi=True)
            
            start_ind1, end_ind1, start_ind2, end_ind2 = find_graph_edges_close_to_start_and_end_points(graph, 
                start_x, start_y, end_x, end_y, left_utm_x, upper_utm_y, delta_x, delta_y)
            graph, start_ind_new = insert_node(graph, start_ind1, end_ind1, left_utm_x, upper_utm_y, delta_x, delta_y, start_ind, end_ind)
            graph, end_ind_new = insert_node(graph, start_ind2, end_ind2, left_utm_x, upper_utm_y, delta_x, delta_y, end_ind, start_ind)
            
            try:
                path = nx.shortest_path(graph, source=start_ind_new, target=end_ind_new, weight='weight')
                start_ind = start_ind_new
                end_ind = end_ind_new
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
    return graph, path, start_ind, end_ind

def _process_graph_and_extend_edges(graph, path, start_ind, end_ind, radius):
    """Phase 3: Find nodes within radius, remove dead ends, and extend edges."""
    print('Phase 3: Processing graph and extending edges')
    
    # Find nodes within radius of main path
    nodes = []
    for node in tqdm(path):
        test = nx.generators.ego_graph(graph, node, radius=int(radius))
        for n in test:
            if n not in nodes:
                nodes.append(n)
    
    # Create subgraph and clean it up
    G = graph.subgraph(nodes).copy()
    G = remove_dead_ends(G, start_ind, end_ind)
    
    # Remove edges that link a node to itself
    G = _remove_self_loops(G)
    
    # Extend edges to nodes
    for s, e, d in G.edges:
        for i in range(len(G[s][e])):
            x, y = extend_cline(G, s, e, i)
            G[s][e][i]['pts'] = np.vstack((y, x)).T
    
    # Fix edge superposition issues (complex geometric processing from original)
    _fix_edge_superposition(G)
    
    G = _remove_self_loops(G)
    
    return G


def _remove_self_loops(G):
    """Remove edges that link a node to itself."""
    edges_to_be_removed = []
    for node in G.nodes:
        for neighbor in list(nx.neighbors(G, node)):
            if neighbor == node:
                edges_to_be_removed.append((node, neighbor))
    G.remove_edges_from(edges_to_be_removed)
    return G


def _fix_edge_superposition(G):
    """Fix problems that arise from superposition of edges from skeletonization."""
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
                    l2 = np.sum((p1-p2)**2)
                    t = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / l2))
                    projection = p1 + t * (p2 - p1)
                    if np.linalg.norm(G[main_node][combination[0][1]][combination[0][2]]['pts'][0] - p1) \
                    < np.linalg.norm(G[main_node][combination[0][1]][combination[0][2]]['pts'][-1] - p1):
                        G[main_node][combination[0][1]][combination[0][2]]['pts'][1] = projection
                    else:
                        G[main_node][combination[0][1]][combination[0][2]]['pts'][-2] = projection


def _extract_and_smooth_main_path(G, start_ind, end_ind, min_main_path_length):
    """Phase 4: Extract main path coordinates and apply smoothing."""
    print('Phase 4: Extracting and smoothing main path')
    
    # Find main path
    path = nx.shortest_path(G, source=start_ind, target=end_ind)
    xcoords = []
    ycoords = []
    
    for i in range(len(path)-1):
        s = path[i]
        e = path[i+1]
        d = 0
        x = list(G[s][e][d]['pts'][:, 1])
        y = list(G[s][e][d]['pts'][:, 0])
        if len(x) > 0:
            # Check if segment needs to be flipped
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
    if main_path.length < min_main_path_length:
        print('main path is too short')
        return None, None, None, None, None
    
    return xcoords, ycoords, xcoords_sm, ycoords_sm, main_path


def _polygonize_and_process_centerlines(G, main_path):
    """Phase 5: Create polygons from centerline network and filter them."""
    print('Phase 5: Polygonizing centerline network')
    
    # Create linestrings for polygonization
    clines = []
    for s, e, d in tqdm(G.edges):
        for i in range(len(G[s][e])):
            x = G[s][e][i]['pts'][:,1]
            y = G[s][e][i]['pts'][:,0]
            if len(x) > 1:
                line = LineString(np.vstack((x, y)).T)
                if not line.is_simple:
                    x = G[s][e][i]['pts'][:,1][1:-1]
                    y = G[s][e][i]['pts'][:,0][1:-1]
                    line = LineString(np.vstack((x, y)).T)
                if line not in clines:
                    clines.append(line)
    
    # Polygonize
    cline_polys = list(polygonize_full(clines))
    
    if len(cline_polys[0].geoms) > 0 or len(cline_polys[3].geoms) > 0:
        # Process polygons
        polys = []
        for poly in cline_polys[0].geoms:
            if len(poly.interiors) > 0:
                poly = Polygon(poly.exterior)
            polys.append(poly)
        for poly in cline_polys[3].geoms:
            poly = Polygon(poly)
            if poly.is_valid:
                polys.append(poly)
            else:
                poly = poly.buffer(0)
                polys.append(poly)
        
        # Remove contained polygons
        polys_to_remove = []
        for i in range(len(polys)):
            for j in range(len(polys)):
                if polys[j].contains(polys[i]) and i != j:
                    polys_to_remove.append(i)
        if len(polys_to_remove) > 0:
            new_polys = []
            for i in range(len(polys)):
                if i not in polys_to_remove:
                    new_polys.append(polys[i])
            polys = new_polys
        
        # Filter polygons connected to main fairway
        gdf = geopandas.GeoDataFrame(polys, columns=['geometry'])
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
        gdf = geopandas.GeoDataFrame(main_polys, columns=['geometry'])
    else:
        gdf = geopandas.GeoDataFrame([], columns=['geometry'])
    
    return gdf

def _create_channel_belt_and_boundaries(xcoords, ycoords, xcoords_sm, ycoords_sm, ch_belt_smooth_factor, ch_belt_half_width, gdf):
    """Phase 6: Create channel belt centerline and boundary polygons."""
    print('Phase 6: Creating channel belt and boundaries')
    
    # Channel belt centerline
    xs, ys = resample_and_smooth(xcoords_sm, ycoords_sm, 25, float(ch_belt_smooth_factor))
    ch_belt_cl = LineString(np.vstack((xs, ys)).T)
    ch_belt_poly = ch_belt_cl.buffer(float(ch_belt_half_width))

    # These are needed so that later we can get rid of the extra line segments at the beginning and end of G_primal
    xcoords1 = xcoords[0]
    xcoords2 = xcoords[-1]
    ycoords1 = ycoords[0]
    ycoords2 = ycoords[-1]
    
    # Lengthen the channel centerline so that it intersects the channel belt polygon
    ratio = 2*float(ch_belt_half_width)/(np.sqrt((xs[1]-xs[0])**2 + (ys[1]-ys[0])**2))
    a1, b1 = getExtrapolatedLine((xs[1], ys[1]), (xs[0], ys[0]), ratio)
    ratio = 2*float(ch_belt_half_width)/(np.sqrt((xs[-1]-xs[-2])**2 + (ys[-1]-ys[-2])**2))
    a2, b2 = getExtrapolatedLine((xs[-2], ys[-2]), (xs[-1], ys[-1]), ratio)
    xcoords_ext = np.hstack((b1[0], xcoords, b2[0])) if len(xcoords) > 0 else np.array([b1[0], b2[0]])
    ycoords_ext = np.hstack((b1[1], ycoords, b2[1])) if len(ycoords) > 0 else np.array([b1[1], b2[1]])
    main_path_ext = LineString(np.vstack((xcoords_ext, ycoords_ext)).T)
    
    # Create polygons for the outer boundaries of the channel belt
    main_polys = split(ch_belt_poly, main_path_ext)
    poly1 = main_polys.geoms[0]
    poly2 = main_polys.geoms[1]
    
    # Trim the boundary polygons so that they do not overlap with the channel belt polygons
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
    
    return xs, ys, poly1, poly2, xcoords1, xcoords2, ycoords1, ycoords2

def _create_UTM_geodataframe(gdf, poly1, poly2, left_utm_x, upper_utm_y, delta_x, delta_y, dataset, mndwi):
    """Phase 7: Create UTM geodataframe"""
    print('Phase 7: Create UTM geodataframe')

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

    return gdf2, poly1_utm, poly2_utm

def _create_primal_graph(gdf2, xcoords1, ycoords1, xcoords2, ycoords2, left_utm_x, 
                         upper_utm_y, delta_x, delta_y, min_g_primal_length):
    """Phase 8: Create primal graph"""
    print('Phase 8: Create primal graph')
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
        return None

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
        return None

    G_primal, primal_start_ind = get_rid_of_extra_lines_at_beginning_and_end(G_primal, xcoords1, ycoords1, left_utm_x, upper_utm_y, delta_x, delta_y)
    G_primal, primal_end_ind = get_rid_of_extra_lines_at_beginning_and_end(G_primal, xcoords2, ycoords2, left_utm_x, upper_utm_y, delta_x, delta_y)
    G_primal = remove_dead_ends(G_primal, primal_start_ind, primal_end_ind)
    print('start and end nodes in G_primal:')
    print(primal_start_ind, primal_end_ind)

    if len(G_primal) < 2:
        print('G_primal only has one node!')
        return None

    return G_primal, primal_start_ind, primal_end_ind

def _create_rook_graph(gdf2, poly1_utm, poly2_utm, mndwi, dataset, ch_belt_half_width, delta_x, filter_contours=False):
    """Phase 9: Create rook graph"""
    print('Phase 9: Create rook graph')
    utm_coords = []
    x1_poly, y1_poly, ch_map = get_bank_coords(poly1_utm, mndwi, dataset, timer=True, filter_contours=filter_contours)
    x2_poly, y2_poly, ch_map = get_bank_coords(poly2_utm, mndwi, dataset, timer=True, filter_contours=filter_contours)
    # need to lengthen the banklines so that they intersect the main centerline polygons:
    ratio = float(ch_belt_half_width)*delta_x/(np.sqrt((x1_poly[1]-x1_poly[0])**2 + (y1_poly[1]-y1_poly[0])**2))
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

    # Getting bank coordinates for the rest of the islands
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

    # Store results in a Rook graph
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
            G_rook.nodes[i]["bank_polygon"] = Polygon(utm_coords[i])
            G_rook.nodes[i]["centroid"] = centroids[i]
            G_rook.nodes[i]["cl_polygon"] = gdf2['geometry'].iloc[i]

    return G_rook


def map_river_banks(fname, dirname, start_x, start_y, end_x, end_y, file_type, 
    ch_belt_smooth_factor=1e9, remove_smaller_components=True, delete_pixels_polys=False, 
    ch_belt_half_width=2000, water_index_type='mndwi', mndwi_threshold=0.01, small_hole_threshold=64, plot_D_primal=False,
    min_g_primal_length=100000, solidity_filter=True, radius=50, min_main_path_length=2000, 
    flip_outlier_edges=False, check_edges=False, filter_contours=False):
    """
    Map channel centerlines and banks from a georeferenced image.
    
    This function has been refactored into multiple phases for better maintainability.
    """
    
    # Phase 1: Initial setup and skeletonization
    result = _initial_skeletonization_and_graph_setup(
        fname, dirname, start_x, start_y, end_x, end_y, file_type,
        water_index_type, mndwi_threshold, delete_pixels_polys, small_hole_threshold,
        remove_smaller_components, solidity_filter
    )
    
    if result[0] is None:
        return None, None, None, None, None, None, None, None, None, None, None
    
    graph, start_ind, end_ind, mndwi, dataset, left_utm_x, upper_utm_y, right_utm_x, lower_utm_y, delta_x, delta_y = result
    if dataset is None:
        print('dataset is not defined!')
    
    # Phase 2: Find main path with fallback logic
    graph, path, start_ind, end_ind = _find_main_path_with_fallbacks(
        graph, start_x, start_y, end_x, end_y, 
        start_ind, end_ind, mndwi, left_utm_x, upper_utm_y, delta_x, delta_y
    )
    if path is None:
        return None, None, None, None, None, None, None, None, None, None, None
    
    # Phase 3: Process graph and extend edges
    G = _process_graph_and_extend_edges(graph, path, start_ind, end_ind, radius)
    
    # Phase 4: Extract and smooth main path
    path_result = _extract_and_smooth_main_path(G, start_ind, end_ind, min_main_path_length)
    if path_result[0] is None:
        return None, None, None, None, None, None, None, None, None, None, None
    
    xcoords, ycoords, xcoords_sm, ycoords_sm, main_path = path_result
    
    # Phase 5: Polygonize centerline network
    gdf = _polygonize_and_process_centerlines(G, main_path)
    
    # Phase 6: Create channel belt and boundaries
    xs, ys, poly1, poly2, xcoords1, xcoords2, ycoords1, ycoords2 = _create_channel_belt_and_boundaries(
        xcoords, ycoords, xcoords_sm, ycoords_sm, ch_belt_smooth_factor, ch_belt_half_width, gdf
    )
    
    # Phase 7: Create UTM geodataframe
    gdf2, poly1_utm, poly2_utm = _create_UTM_geodataframe(gdf, poly1, poly2, left_utm_x, upper_utm_y, delta_x, delta_y, dataset, mndwi)
    
    # Phase 8: Create primal graph
    G_primal, primal_start_ind, primal_end_ind = _create_primal_graph(gdf2, xcoords1, ycoords1, xcoords2, ycoords2, left_utm_x, 
                         upper_utm_y, delta_x, delta_y, min_g_primal_length)
    if G_primal is None:
        return None, None, None, None, None, None, None, None, None, None, None
    
    # Phase 9: Create rook graph
    G_rook = _create_rook_graph(gdf2, poly1_utm, poly2_utm, mndwi, dataset, ch_belt_half_width, delta_x, filter_contours=filter_contours)

    if G_rook is None:
        return None, None, None, None, None, None, None, None, None, None, None
    
    polys = smooth_banklines(G_rook, dataset, mndwi, save_smooth_lines=True)
    
    # Phase 10: Set half channel widths
    print('Phase 10: Set half channel widths')
    set_half_channel_widths(G_primal, G_rook, dataset, mndwi)

    # Phase 11: Create directed graph
    print('Phase 11: Create directed graph')
    xs, ys = convert_to_utm(np.array(xs), np.array(ys), left_utm_x, upper_utm_y, delta_x, delta_y)
    # sometimes xs and ys need to be flipped (I have no idea why):
    start_point_dist = np.sqrt((xs[0]-start_x)**2 + (ys[0]-start_y)**2)
    end_point_dist = np.sqrt((xs[0]-end_x)**2 + (ys[0]-end_y)**2)
    if end_point_dist < start_point_dist:
        xs = xs[::-1]; ys = ys[::-1]
    D_primal, source_nodes, sink_nodes = create_directed_multigraph(G_primal, G_rook, xs, ys, primal_start_ind, primal_end_ind, 
                                                                    flip_outlier_edges=flip_outlier_edges, check_edges=check_edges)

    # Phase 12: Get bank coordinates for main channel
    print('Phase 12: Get bank coordinates for main channel')
    start_node, inds = find_start_node(D_primal)
    if start_node is not None:
        edge_path = traverse_multigraph(D_primal, start_node)
        D_primal.graph['main_path'] = edge_path # store main path as a graph attribute
        x, y, x_utm1, y_utm1, x_utm2, y_utm2 = get_bank_coords_for_main_channel(D_primal, mndwi, edge_path, dataset)
        D_primal.graph['main_channel_cl_coords'] = np.vstack((x, y)).T
        D_primal.graph['main_channel_bank1_coords'] = np.vstack((x_utm1, y_utm1)).T
        D_primal.graph['main_channel_bank2_coords'] = np.vstack((x_utm2, y_utm2)).T
    
    # Set graph names
    D_primal.name = fname
    G_rook.name = fname
    G_primal.name = fname

    if plot_D_primal:
        fig, ax = plot_im_and_lines(mndwi, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y,
            G_rook, G_primal, D_primal, dataset=dataset, smoothing=False, start_x=start_x, start_y=start_y, end_x=end_x,
            end_y=end_y, plot_lines=False)
        plot_graph_w_colors(D_primal, ax)

    return D_primal, G_rook, G_primal, mndwi, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, xs, ys


def main(fname, dirname, start_x, start_y, end_x, end_y, file_type, **kwargs):
    """Main function to extract centerlines from command line arguments."""
    return map_river_banks(
        fname=fname,
        dirname=dirname, 
        start_x=float(start_x),
        start_y=float(start_y),
        end_x=float(end_x), 
        end_y=float(end_y),
        file_type=file_type,
        **kwargs
    )


if __name__=='__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], 
         sys.argv[5], sys.argv[6], sys.argv[7],
         **dict(arg.split('=') for arg in sys.argv[8:])) 