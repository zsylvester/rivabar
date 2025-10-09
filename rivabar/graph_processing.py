"""Graph processing and manipulation functions for river centerline extraction."""

import numpy as np
import networkx as nx
import warnings
from shapely.geometry import LineString, Point, Polygon
from scipy.spatial import KDTree
from tqdm import tqdm
import pandas as pd
from itertools import combinations, permutations

from .geometry_utils import (convert_to_utm, find_matching_indices, find_closest_point, 
                           find_longer_segment_coords, extract_coords, angle_between)
from .utils import find_condition


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
    dist = tree.query(np.reshape([x, y], (1, -1)))[0][0]
    closest_node_ind = tree.query(np.reshape([x, y], (1, -1)))[1][0]
    closest_node = list(nodes)[closest_node_ind]
    return dist, closest_node

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
    dist = tree.query(np.reshape([x, y], (1, -1)))[0][0]
    closest_node_ind = tree.query(np.reshape([x, y], (1, -1)))[1][0]
    closest_node = list(nodes)[closest_node_ind]
    return dist, closest_node

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
    if start_node is None:
        print("could not find start node!")
        return start_node, inds
    return start_node, inds

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
        Whether to flip the direction of outlier edges (default is False). 
        Should be set to 'True' for complex networks (e.g., Lena Delta, Brahmaputra).
    check_edges : bool, optional
        Check edges around each island for consistency in the direction of the flow 
        (default is False). Should be set to 'True' for multithread rivers (e.g., Brahmaputra), 
        but not for meandering rivers or for networks with unrealistic centerline 
        orientations (e.g., Lena Delta).

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
        A new graph with nodes inside the polygon removed and edges truncated.
    """
    
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
    new_boundary_nodes = []  # Track new nodes created at polygon boundary
    
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
                # Track new boundary nodes for reporting
                if polygon.boundary.distance(start_point) < 1e-6:
                    new_boundary_nodes.append(start_node)
            
            if end_node is None:
                end_node = next_node_id
                next_node_id += 1
                D_primal_truncated.add_node(end_node, 
                                          geometry=Point(end_point.x, end_point.y))
                # Track new boundary nodes for reporting
                if polygon.boundary.distance(end_point) < 1e-6:
                    new_boundary_nodes.append(end_node)
            
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
    print(f"Created {len(new_boundary_nodes)} new boundary nodes")
    
    # 7. Find all sink nodes (nodes with out_degree=0) for reporting
    sinks = [node for node in D_primal_truncated.nodes() if D_primal_truncated.out_degree(node) == 0]
    sources = [node for node in D_primal_truncated.nodes() if D_primal_truncated.in_degree(node) == 0]
    print(f"Total sink nodes in truncated graph: {len(sinks)}")
    print(f"Total source nodes in truncated graph: {len(sources)}")
    
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
    si = tree.query(np.reshape([x1, y1], (1, -1)))[1][0]
    start_ind = ss[si]
    end_ind = es[si]
    # coordinates of the closest edge:
    x = np.array(list(G_primal[start_ind][end_ind][0]['geometry'].xy[0]))
    y = np.array(list(G_primal[start_ind][end_ind][0]['geometry'].xy[1]))
    # KDTree for the coordinates of the closest edge:
    tree = KDTree(np.vstack((x, y)).T)
    # find point that is closest to (x1, y1):
    ind = tree.query(np.reshape([x1, y1], (1, -1)))[1][0]
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

def find_subpath(D_primal, root, depth_limit=10):
    """
    Finds the subpath with the largest average width in a directed graph from the root node up to a specified depth limit.

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
    """
    current_node = start_node
    edge_path = []
    subpath_depth = min(subpath_depth, len(G))
    current_path = []
    while True:
        next_path = find_subpath(G, current_node, subpath_depth)
        if next_path:
            # nodes_in_current_path = list(set(element for tuple_ in current_path for element in tuple_))
            nodes_in_current_path = set([element for tuple_ in [tuple_[:2] for tuple_ in current_path] for element in tuple_])
            # nodes_in_next_path = list(set(element for tuple_ in next_path for element in tuple_))
            nodes_in_next_path = set([element for tuple_ in [tuple_[:2] for tuple_ in next_path] for element in tuple_])
            # sometimes there is a cycle and we need to break it:
            if nodes_in_current_path == nodes_in_next_path:
                break
            edge_path.extend(next_path)
            current_node = edge_path[-1][1]
            current_path = next_path.copy()
        else:
            break
    return edge_path

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

def find_subpath(D_primal, root, depth_limit=10):
    """
    Finds the subpath with the largest average width in a directed graph from the root node up to a specified depth limit.

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