# Temporal Analysis Module - Complete implementation

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
import networkx as nx
from tqdm import tqdm, trange
from datetime import datetime
from shapely.geometry import Polygon, MultiPolygon, Point, GeometryCollection
from shapely.ops import unary_union
from pyproj import Transformer
import ee
import ipywidgets as widgets
from IPython.display import display
import geopandas
from scipy.spatial import cKDTree

# Internal imports
from .polygon_processing import create_channel_nw_polygon, plot_polygon

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

def cluster_polygons(gdf, iou_threshold, max_days=2*365):
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
        The spliced path where segments from path1 have been replaced with corresponding segments from path2 if they have better (higher) widths.
    """
    # Find the start and end nodes that are common between the two paths
    start_node = None
    end_node = None
    
    for edge1 in path1:
        for edge2 in path2:
            if edge1[0] == edge2[0]:  # Same start node
                start_node = edge1[0]
                break
        if start_node:
            break
    
    for edge1 in reversed(path1):
        for edge2 in reversed(path2):
            if edge1[1] == edge2[1]:  # Same end node
                end_node = edge1[1]
                break
        if end_node:
            break
    
    if not start_node or not end_node:
        # If no common start or end nodes, return the original path1
        return path1
    
    # Extract the subpaths from start_node to end_node for both paths
    subpath1 = []
    subpath2 = []
    
    recording1 = False
    for edge in path1:
        if edge[0] == start_node:
            recording1 = True
        if recording1:
            subpath1.append(edge)
        if edge[1] == end_node:
            break
    
    recording2 = False
    for edge in path2:
        if edge[0] == start_node:
            recording2 = True
        if recording2:
            subpath2.append(edge)
        if edge[1] == end_node:
            break
    
    # Calculate the average width for each subpath
    avg_width1 = np.mean([D_primal[edge[0]][edge[1]][edge[2]]['width'] for edge in subpath1 if 'width' in D_primal[edge[0]][edge[1]][edge[2]]])
    avg_width2 = np.mean([D_primal[edge[0]][edge[1]][edge[2]]['width'] for edge in subpath2 if 'width' in D_primal[edge[0]][edge[1]][edge[2]]])
    
    # If subpath2 has a higher average width, replace subpath1 with subpath2
    if avg_width2 > avg_width1:
        # Find the indices of start_node and end_node in path1
        start_index = next(i for i, edge in enumerate(path1) if edge[0] == start_node)
        end_index = next(i for i, edge in enumerate(path1) if edge[1] == end_node)
        
        # Replace the subpath in path1 with subpath2
        spliced_path = path1[:start_index] + subpath2 + path1[end_index+1:]
        return spliced_path
    else:
        return path1

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

def create_and_plot_bars(rivers, ts1, ts2, ax1=None, ax2=None, depo_cmap="Blues", erosion_cmap="Reds", alpha=0.5, aoi=None, colorbar=True, color_scale_timestep=None):
    """
    Create preserved scroll bar polygons and erosion polygons from a list of rook neighborhood graphs 
    and plot them. It also handles the creation of the color map for the plot.

    Parameters
    ----------
    rivers : list
        A list of river objects with rook neighborhood graphs.
    ts1 : int
        The first time step to consider (inclusive).
    ts2 : int
        The last time step to consider (inclusive).
    ax1 : matplotlib.axes.Axes, optional
        The axes on which to plot the deposition polygons. Defaults to None.
    ax2 : matplotlib.axes.Axes, optional
        The axes on which to plot the erosion polygons. Defaults to None.
    depo_cmap : str, optional
        The name of the color map to use for the deposition polygons. Defaults to "Blues".
    erosion_cmap : str, optional
        The name of the color map to use for the erosion polygons. Defaults to "Reds".
    aoi : list or tuple, optional
        Area of interest defined as [xmin, xmax, ymin, ymax]. If provided, all channel polygons
        will be cropped to this area before processing. Defaults to None.
    colorbar : bool, optional
        Whether to show colorbar. Defaults to True.
    color_scale_timestep : int, optional
        Alternative timestep to use for color scaling instead of ts2. If provided,
        uses ts2 = min(ts1 + color_scale_timestep, len(rivers)-1) for color scaling only.
        The actual analysis still uses the original ts1 and ts2. Defaults to None.

    Returns
    -------
    chs : list
        A list of channel polygons (cropped to AOI if provided).
    bars : list
        A list of scroll bar polygons.
    erosions_final : list
        A list of final erosion polygons.
    aoi_dates : list
        A list of datetime objects corresponding to the channels that remain after AOI cropping.
        If no AOI is provided, this equals the original dates list.
    aoi_centerlines : list
        A list of main channel centerline segments (LineStrings) corresponding to the AOI.
        If no AOI is provided, returns all available centerlines.

    Example
    -------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,15), sharex=True, sharey=True)
    chs, bars, erosions, dates, centerlines = rb.create_and_plot_bars(rivers, 0, len(rivers)-1, ax1=ax1, ax2=ax2)
    
    # With area of interest:
    aoi = [xmin, xmax, ymin, ymax]
    chs, bars, erosions, aoi_dates, aoi_centerlines = rb.create_and_plot_bars(rivers, 5, 15, ax1=ax1, ax2=ax2, aoi=aoi)
    """
    # Validate input parameters
    if ts1 < 0:
        print('ts1 must be >= 0!')
        return None, None, None, None, None
    if ts2 <= ts1:
        print('ts2 must be greater than ts1!')
        return None, None, None, None, None
    if ts2 > len(rivers):
        print('ts2 must be <= length of "rivers"!')
        return None, None, None, None, None
    if ts2 - ts1 < 2:
        print('Need at least 2 time steps (ts2 - ts1 >= 2)!')
        return None, None, None, None, None
    
    # Calculate the number of time steps to process
    ts = ts2 - ts1 + 1
    
    # Calculate alternative timestep for color scaling if provided
    if color_scale_timestep is not None:
        color_ts = color_scale_timestep - ts1 + 1
    else:
        color_ts = ts
    
    # Create AOI polygon if provided
    aoi_polygon = None
    if aoi is not None:
        from shapely.geometry import box
        xmin, xmax, ymin, ymax = aoi
        aoi_polygon = box(xmin, ymin, xmax, ymax)
        print(f"Using AOI: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
    
    dates = []
    for river in rivers[ts1:ts2+1]:
        dates.append(datetime.strptime(river.acquisition_date, "%Y-%m-%d"))
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
        ch1 = create_channel_nw_polygon(rivers[ts1+i]._G_rook, buffer=10, dataset=rivers[ts1+i]._dataset)
        ch1 = ch1.buffer(0)
        if type(ch1) == MultiPolygon:
            areas = []
            for geom in ch1.geoms:
                areas.append(geom.area)
            ch1 = ch1.geoms[np.argmax(areas)]
        # print(ts1+i+1)
        ch2 = create_channel_nw_polygon(rivers[ts1+i+1]._G_rook, buffer=10, dataset=rivers[ts1+i+1]._dataset)
        ch2 = ch2.buffer(0)
        if type(ch2) == MultiPolygon:
            areas = []
            for geom in ch2.geoms:
                areas.append(geom.area)
            ch2 = ch2.geoms[np.argmax(areas)]
        chs.append(ch1)
    chs.append(ch2) # append last channel
    
    # Extract centerlines from rivers
    centerlines = []
    for i, river in enumerate(rivers[ts1:ts2+1]):
        try:
            centerline = river.main_channel_centerline
            if centerline is not None:
                centerlines.append(centerline)
            else:
                print(f"Warning: No centerline available for river {ts1+i}")
                centerlines.append(None)
        except AttributeError:
            print(f"Warning: River {ts1+i} does not have main_channel_centerline attribute")
            centerlines.append(None)
    
    # Crop channels and centerlines to AOI if provided
    if aoi_polygon is not None:
        print(f"Cropping {len(chs)} channel polygons and {len(centerlines)} centerlines to AOI...")
        cropped_chs = []
        cropped_centerlines = []
        
        for i, (ch, centerline) in enumerate(zip(chs, centerlines)):
            try:
                # Crop channel polygon
                cropped_ch = ch.intersection(aoi_polygon)
                
                # Handle different geometry types returned by intersection
                if cropped_ch.is_empty:
                    print(f"Warning: Channel {ts1+i} has no intersection with AOI")
                    cropped_chs.append(None)
                elif type(cropped_ch) == MultiPolygon:
                    # Keep the largest polygon if multiple are returned
                    areas = [geom.area for geom in cropped_ch.geoms if type(geom) == Polygon]
                    if areas:
                        largest_idx = np.argmax(areas)
                        largest_poly = [geom for geom in cropped_ch.geoms if type(geom) == Polygon][largest_idx]
                        cropped_chs.append(largest_poly)
                    else:
                        print(f"Warning: Channel {ts1+i} intersection contains no valid polygons")
                        cropped_chs.append(None)
                elif type(cropped_ch) == Polygon:
                    cropped_chs.append(cropped_ch)
                else:
                    print(f"Warning: Channel {ts1+i} intersection returned unexpected geometry type: {type(cropped_ch)}")
                    cropped_chs.append(None)
                
                # Crop centerline
                if centerline is not None:
                    try:
                        cropped_centerline = centerline.intersection(aoi_polygon)
                        if cropped_centerline.is_empty:
                            cropped_centerlines.append(None)
                        elif hasattr(cropped_centerline, 'geoms'):
                            # MultiLineString - take the longest segment
                            segments = [geom for geom in cropped_centerline.geoms if hasattr(geom, 'length')]
                            if segments:
                                longest_segment = max(segments, key=lambda x: x.length)
                                cropped_centerlines.append(longest_segment)
                            else:
                                cropped_centerlines.append(None)
                        else:
                            # Single LineString
                            cropped_centerlines.append(cropped_centerline)
                    except Exception as e:
                        print(f"Warning: Error cropping centerline {ts1+i}: {e}")
                        cropped_centerlines.append(None)
                else:
                    cropped_centerlines.append(None)
                    
            except Exception as e:
                print(f"Error cropping channel {ts1+i}: {e}")
                cropped_chs.append(None)
                cropped_centerlines.append(None)
        
        # Filter out None values and update lists
        valid_indices = [i for i, ch in enumerate(cropped_chs) if ch is not None]
        chs = [cropped_chs[i] for i in valid_indices]
        aoi_centerlines = [cropped_centerlines[i] for i in valid_indices]
        
        # Update dates to match the valid channels
        aoi_dates = [dates[i] for i in valid_indices]
        
        # Update ts to reflect the number of valid cropped channels
        original_ts = ts
        ts = len(chs)
        
        if ts < 2:
            print(f"Error: Only {ts} valid channels after cropping to AOI. Need at least 2 channels for analysis.")
            return None, None, None, None, None
            
        if ts != original_ts:
            print(f"Note: Using {ts} channels after AOI cropping (originally {original_ts})")
        
        # Print centerline summary
        valid_centerlines = [cl for cl in aoi_centerlines if cl is not None]
        print(f"Note: {len(valid_centerlines)} valid centerlines in AOI (out of {len(aoi_centerlines)})")
    else:
        # If no AOI provided, use original dates and centerlines
        aoi_dates = dates
        aoi_centerlines = centerlines
    
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
            color = cmap(i/float(color_ts))
            if type(bar) == MultiPolygon or type(bar) == GeometryCollection:
                for b in bar.geoms:
                    if type(b) == Polygon:
                        if len(b.interiors) == 0:
                            ax1.fill(b.exterior.xy[0], b.exterior.xy[1], facecolor=color, edgecolor='k', linewidth=0.2, alpha=alpha)
                        else:
                            plot_polygon(ax1, b, facecolor=color, edgecolor='k', linewidth=0.2, alpha=alpha)
            if type(bar) == Polygon:
                if len(bar.interiors) == 0:
                    ax1.fill(bar.exterior.xy[0], bar.exterior.xy[1], facecolor=color, edgecolor='k', linewidth=0.2, alpha=alpha)
                else:
                    plot_polygon(ax1, bar, facecolor=color, edgecolor='k', linewidth=0.2, alpha=alpha)
    if ax1:
        # Use the last channel (accounting for potential AOI cropping)
        final_channel = chs[-1] if chs else ch2
        plot_polygon(ax1, final_channel, facecolor='lightblue', edgecolor='k', alpha=alpha)
        ax1.set_aspect('equal')
        # date = datetime.strptime(rivers[min(ts2, len(rivers)-1)].acquisition_date, "%Y-%m-%d")
        # ax1.set_title('Deposition, ' + date.strftime('%m')+'/'+date.strftime('%d')+'/'+date.strftime('%Y'))
        if colorbar:
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
        color = cmap(i/float(color_ts))
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
                            ax2.fill(b.exterior.xy[0], b.exterior.xy[1], facecolor=color, edgecolor='k', linewidth=0.2, alpha=alpha)
                            # ax2.fill(b.exterior.xy[0], b.exterior.xy[1], facecolor='xkcd:faded red', edgecolor='k', linewidth=0.2, alpha=alpha)
                        else:
                            plot_polygon(ax2, b, facecolor=color, edgecolor='k', linewidth=0.2, alpha=alpha)
                            # plot_polygon(ax2, b, facecolor='xkcd:faded red', edgecolor='k', linewidth=0.2, alpha=alpha)
            if type(erosion) == Polygon:
                if len(erosion.interiors) == 0:
                    ax2.fill(erosion.exterior.xy[0], erosion.exterior.xy[1], facecolor=color, edgecolor='k', linewidth=0.2, alpha=alpha)
                    # ax2.fill(erosion.exterior.xy[0], erosion.exterior.xy[1], facecolor='xkcd:faded red', edgecolor='k', linewidth=0.2, alpha=alpha)
                else:
                    plot_polygon(ax2, erosion, facecolor=color, edgecolor='k', linewidth=0.2, alpha=alpha)
                    # plot_polygon(ax2, erosion, facecolor='xkcd:faded red', edgecolor='k', linewidth=0.2, alpha=alpha)
    if ax2:
        plot_polygon(ax2, chs[0], facecolor='lightblue', edgecolor='k', alpha=alpha)
        ax2.set_aspect('equal')
        # date = datetime.strptime(rivers[min(ts2, len(rivers)-1)].acquisition_date, "%Y-%m-%d")
        # ax2.set_title('Erosion, ' + date.strftime('%m')+'/'+date.strftime('%d')+'/'+date.strftime('%Y'))
        if colorbar:
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
        # plt.tight_layout()
    return chs, bars, erosions_final, aoi_dates, aoi_centerlines

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
    gdf = gpd.GeoDataFrame(bank_polys, columns = ['geometry'])
    gdf['date'] = date
    gdf['timedelta'] = gdf['date'] - gdf['date'].min()
    gdf['n_days'] = gdf['timedelta'].dt.days
    gdf['length'] = gdf.length
    gdf['type'] = bank_type
    gdf.crs = 'epsg:'+crs
    gdf = gdf.sort_values(by='date')
    gdf = gdf.reset_index(drop=True)
    return gdf

def create_dataframe_from_bank_polygons(rivers):
    """
    Creates a GeoDataFrame from bank polygons extracted from river objects.

    Parameters
    ----------
    rivers : list
        A list of river objects, each containing a graph (_G_rook) with nodes 
        that have 'bank_polygon' attributes and an acquisition_date property.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the bank polygons with additional attributes:
        - 'geometry': The bank polygons.
        - 'year', 'month', 'day': Individual date components.
        - 'date': The full date as datetime object.
        - 'timedelta': The time difference from the earliest date.
        - 'n_days': The number of days since the earliest date.
        - 'length': The length of each polygon.
        - 'type': The type of bank (0, 1, or 2).

    Notes
    -----
    Invalid geometries are fixed using the buffer(0) technique.
    """
    bank_polys = []
    year = []
    month = []
    day = []
    bank_type = []
    for j in range(len(rivers)):
        for i in range(0, len(rivers[j]._G_rook)):
            bank_polys.append(rivers[j]._G_rook.nodes()[i]['bank_polygon'])
            year.append(int(rivers[j].acquisition_date[:4]))
            month.append(int(rivers[j].acquisition_date[5:7]))
            day.append(int(int(rivers[j].acquisition_date[8:])))             
            if i == 0:
                bank_type.append(0)
            elif i == 1:
                bank_type.append(1)
            else:
                bank_type.append(2)
    gdf = geopandas.GeoDataFrame(bank_polys, columns = ['geometry'])
    gdf['year'] = year
    gdf['month'] = month
    gdf['day'] = day
    gdf['date'] = pd.to_datetime(gdf[['year', 'month', 'day']])
    gdf['timedelta'] = gdf['date'] - gdf['date'].min()
    gdf['n_days'] = gdf['timedelta'].dt.days
    gdf['length'] = gdf.length
    gdf['type'] = bank_type
    gdf.crs = 'epsg:'+str(rivers[0]._dataset.crs.to_epsg())
    gdf = gdf.sort_values(by='date')
    gdf = gdf.reset_index(drop=True)
    for i in range(0, len(gdf.geometry)):
        if not gdf.geometry[i].is_valid:
            gdf.geometry[i] = gdf.geometry[i].buffer(0)
    return gdf

def get_landsat_scene_crs(path_number, row_number, year=2020):
    """
    Get the CRS used by Landsat scenes for a specific path/row.
    
    Parameters
    ----------
    path_number, row_number : int
        WRS path and row
    year : int, optional
        Year to sample (default 2020)
        
    Returns
    -------
    crs_string : str
        EPSG code or CRS string used by Landsat scenes
    """
    
    # Get a sample scene to extract CRS
    if year < 2013:
        collection_id = "LANDSAT/LT05/C02/T1_L2"
    elif year < 2021:
        collection_id = "LANDSAT/LC08/C02/T1_L2"
    else:
        collection_id = "LANDSAT/LC09/C02/T1_L2"
    
    collection = (ee.ImageCollection(collection_id)
                  .filterDate(f'{year}-01-01', f'{year}-12-31')
                  .filter(ee.Filter.eq('WRS_PATH', path_number))
                  .filter(ee.Filter.eq('WRS_ROW', row_number))
                  .first())
    
    # Get the CRS
    crs = collection.projection().crs().getInfo()
    return crs

def convert_to_landsat_crs(point0_lonlat, point1_lonlat, path_number, row_number):
    """
    Convert coordinates to match the Landsat scene's CRS.
    """
    
    # Get the Landsat scene CRS
    landsat_crs = get_landsat_scene_crs(path_number, row_number)
    print(f"Landsat scene CRS: {landsat_crs}")
    
    # Create transformer
    transformer = Transformer.from_crs("EPSG:4326", landsat_crs, always_xy=True)
    
    # Convert coordinates
    start_x, start_y = transformer.transform(point0_lonlat[0], point0_lonlat[1])
    end_x, end_y = transformer.transform(point1_lonlat[0], point1_lonlat[1])
    
    print(f"Converted coordinates:")
    print(f"  Start: ({start_x:.1f}, {start_y:.1f})")
    print(f"  End: ({end_x:.1f}, {end_y:.1f})")
    
    return start_x, start_y, end_x, end_y, landsat_crs

def collect_river_endpoints(m):
    """Interactive map tool to collect start and end points for river analysis."""
    
    # Data storage
    endpoints = {'start': None, 'end': None}
    
    # Status displays
    status = widgets.HTML(value="<b>Click to set river START point</b>")
    start_point = widgets.Text(description='Start point:')
    end_point = widgets.Text(description='End point:')
    
    # Create button to confirm points
    confirm_button = widgets.Button(
        description='Confirm Points',
        disabled=True,
        button_style='success'
    )
    
    # Create button to reset points
    reset_button = widgets.Button(
        description='Reset',
        button_style='warning'
    )
    
    def handle_click(**kwargs):
        if kwargs.get('type') == 'click':
            coords = kwargs.get('coordinates')
            lon, lat = round(coords[0], 6), round(coords[1], 6)
            
            # Add marker based on which point we're capturing
            if endpoints['start'] is None:
                # Mark start point
                endpoints['start'] = [lon, lat]
                m.add_marker(location=[lat, lon]) #, popup=widgets.HTML("START"))
                start_point.value = f"[{lon}, {lat}]"
                status.value = "<b>Click to set river END point</b>"
                
            elif endpoints['end'] is None:
                # Mark end point
                endpoints['end'] = [lon, lat]
                m.add_marker(location=[lat, lon]) #, popup=widgets.HTML("END"))
                end_point.value = f"[{lon}, {lat}]"
                status.value = "<b>Points captured! Click 'Confirm' to use these points.</b>"
                confirm_button.disabled = False
    
    def handle_reset(b):
        """Reset the map and points."""
        endpoints['start'] = None
        endpoints['end'] = None
        start_point.value = ""
        end_point.value = ""
        status.value = "<b>Click to set river START point</b>"
        confirm_button.disabled = True
        # Reset map
        # m.clear_markers()
    
    def handle_confirm(b):
        """Return the confirmed endpoints."""
        status.value = "<b>✅ Points confirmed! Ready to use in your analysis.</b>"
        
        # Format points as code snippet for easy copy-paste
        point_code = f"""
        # River endpoints (longitude, latitude)
        point0 = {endpoints['start']}  # Start point
        point1 = {endpoints['end']}  # End point
        """
        
        # Display copy-pasteable code
        display(widgets.HTML(f"<pre>{point_code}</pre>"))
    
    # Attach event handlers
    m.on_interaction(handle_click)
    reset_button.on_click(handle_reset)
    confirm_button.on_click(handle_confirm)
    
    # Display map and controls
    display(m)
    display(widgets.VBox([
        status,
        widgets.HBox([start_point, end_point]),
        widgets.HBox([confirm_button, reset_button])
    ]))
    
    return endpoints

def plot_deposition_erosion_with_dates(bars, erosions, dates, centerlines, figsize=(15, 8)):
    """
    Plot deposition and erosion with bar widths proportional to time intervals.
    Rates are normalized by centerline length.
    
    Parameters
    ----------
    bars : list
        List of deposition polygons for each time step (length n-1)
    erosions : list  
        List of erosion polygons for each time step (length n-1)
    dates : list
        List of acquisition dates (length n)
    centerlines : list
        List of centerline geometries (LineString or MultiLineString) for each time step (length n)
    figsize : tuple
        Figure size (width, height)
    """
    import pandas as pd
    import matplotlib.dates as mdates
    from shapely.geometry import LineString, MultiLineString
    
    # Convert dates to pandas datetime if they aren't already
    dates_pd = pd.to_datetime(dates)
    
    # Calculate time intervals between consecutive dates
    time_intervals = []
    interval_centers = []
    
    for i in range(len(dates_pd) - 1):
        # Time interval between consecutive acquisitions
        interval_days = (dates_pd[i+1] - dates_pd[i]).days
        time_intervals.append(interval_days)
        
        # Center point of the interval (where we'll place the bar)
        center_date = dates_pd[i] + (dates_pd[i+1] - dates_pd[i]) / 2
        interval_centers.append(center_date)
    
    # Calculate centerline lengths for normalization
    centerline_lengths = []
    for centerline in centerlines:
        if centerline is not None:
            if isinstance(centerline, MultiLineString):
                # Sum lengths of all line segments
                total_length = sum(line.length for line in centerline.geoms)
            elif isinstance(centerline, LineString):
                total_length = centerline.length
            else:
                total_length = 0
            centerline_lengths.append(total_length)
        else:
            centerline_lengths.append(0)
    
    # For each time interval, use the average centerline length
    interval_centerline_lengths = []
    for i in range(len(centerline_lengths) - 1):
        # Average length between consecutive time steps
        avg_length = (centerline_lengths[i] + centerline_lengths[i+1]) / 2
        interval_centerline_lengths.append(avg_length)
    
    # Calculate areas for each time step
    deposition_areas = []
    erosion_areas = []
    
    for i, (bar_polys, erosion_polys) in enumerate(zip(bars, erosions)):
        # Calculate total deposition area
        depo_area = 0
        if bar_polys is not None:
            if hasattr(bar_polys, 'geoms'):  # MultiPolygon
                for poly in bar_polys.geoms:
                    if hasattr(poly, 'area'):
                        depo_area += poly.area
            elif hasattr(bar_polys, 'area'):  # Single Polygon
                depo_area = bar_polys.area
        
        # Calculate total erosion area  
        ero_area = 0
        if erosion_polys is not None:
            if hasattr(erosion_polys, 'geoms'):  # MultiPolygon
                for poly in erosion_polys.geoms:
                    if hasattr(poly, 'area'):
                        ero_area += poly.area
            elif hasattr(erosion_polys, 'area'):  # Single Polygon
                ero_area = erosion_polys.area
        
        deposition_areas.append(depo_area)
        erosion_areas.append(ero_area)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars with widths proportional to time intervals
    for i, (center_date, interval_days, depo_area, ero_area, centerline_length) in enumerate(
        zip(interval_centers, time_intervals, deposition_areas, erosion_areas, interval_centerline_lengths)):
        
        # Bar width proportional to time interval
        width = interval_days * 1.0
        
        # Normalize rates by centerline length (avoid division by zero)
        if centerline_length > 0:
            depo_rate_normalized = (depo_area/interval_days) / centerline_length
            ero_rate_normalized = (ero_area/interval_days) / centerline_length
        else:
            depo_rate_normalized = 0
            ero_rate_normalized = 0
        
        # Plot deposition (positive)
        ax.bar(center_date, depo_rate_normalized, width=width, 
               color='xkcd:blue', edgecolor='k', alpha=0.8, label='Deposition' if i == 0 else "")
        
        # Plot erosion (negative)
        ax.bar(center_date, -ero_rate_normalized, width=width, 
               color='xkcd:faded red', edgecolor='k', alpha=0.8, label='Erosion' if i == 0 else "")
    
    # Customize the plot
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rate of Change (m/day)', fontsize=12)
    ax.set_title('Deposition and Erosion Over Time', fontsize=14)
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # # Add vertical lines at acquisition times
    # for date in dates_pd:
    #     ax.axvline(x=date, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    # Add minor ticks for better resolution
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # Rotate labels for better readability
    plt.xticks(rotation=45)
    
    # Add legend
    ax.legend()
    
    # Set axis limits with some padding
    ax.set_xlim(dates_pd[0] - pd.Timedelta(days=30), 
                dates_pd[-1] + pd.Timedelta(days=30))
    
    # Adjust layout
    plt.tight_layout()
    
    # Print summary statistics
    print(f"Number of time intervals: {len(bars)}")
    print(f"Date range: {dates_pd[0].strftime('%Y-%m-%d')} to {dates_pd[-1].strftime('%Y-%m-%d')}")
    print(f"Average time interval: {np.mean(time_intervals):.1f} days")
    print(f"Average centerline length: {np.mean(interval_centerline_lengths):.2f} m")
    print(f"Total deposition: {np.sum(deposition_areas):.2f} m²")
    print(f"Total erosion: {np.sum(erosion_areas):.2f} m²")
    print(f"Net change: {np.sum(deposition_areas) - np.sum(erosion_areas):.2f} m²")
    
    # Create summary DataFrame with normalized rates
    normalized_depo_rates = []
    normalized_ero_rates = []
    
    for i, (depo_area, ero_area, interval_days, centerline_length) in enumerate(
        zip(deposition_areas, erosion_areas, time_intervals, interval_centerline_lengths)):
        
        if centerline_length > 0:
            normalized_depo_rates.append((depo_area/interval_days) / centerline_length)
            normalized_ero_rates.append((ero_area/interval_days) / centerline_length)
        else:
            normalized_depo_rates.append(0)
            normalized_ero_rates.append(0)
    
    summary_df = pd.DataFrame({
        'start_date': dates_pd[:-1],
        'end_date': dates_pd[1:],
        'center_date': interval_centers,
        'interval_days': time_intervals,
        'centerline_length_m': interval_centerline_lengths,
        'deposition_m2': deposition_areas,
        'erosion_m2': erosion_areas,
        'deposition_rate_m_per_day': normalized_depo_rates,
        'erosion_rate_m_per_day': normalized_ero_rates,
        'net_change_m2': np.array(deposition_areas) - np.array(erosion_areas),
        'net_rate_m_per_day': np.array(normalized_depo_rates) - np.array(normalized_ero_rates)
    })
    
    return fig, ax, summary_df

def _map_rook_nodes(G_rook_t1, G_rook_t2, iou_threshold=0.5):
    """Maps nodes in G_rook between two timesteps based on polygon IoU."""
    node_mapping = {}
    
    nodes2_data = list(G_rook_t2.nodes(data=True))
    if not nodes2_data:
        return {}

    nodes2, data2 = zip(*nodes2_data)
    polys2 = [d['cl_polygon'] for d in data2]

    for n1, data1 in tqdm(G_rook_t1.nodes(data=True), desc="Mapping G_rook nodes"):
        poly1 = data1.get('cl_polygon')
        if not poly1:
            continue

        best_match = None
        max_iou = 0.0

        for i, poly2 in enumerate(polys2):
            if poly1.intersects(poly2):
                intersection_area = poly1.intersection(poly2).area
                union_area = poly1.union(poly2).area
                if union_area > 0:
                    iou = intersection_area / union_area
                    if iou > max_iou:
                        max_iou = iou
                        best_match = nodes2[i]

        if max_iou > iou_threshold:
            node_mapping[n1] = best_match
            
    return node_mapping

def _map_primal_nodes(D_primal_t1, D_primal_t2, distance_threshold=100.0):
    """Maps nodes in D_primal between two timesteps based on proximity."""
    node_mapping = {}
    
    nodes1_data = list(D_primal_t1.nodes(data=True))
    nodes2_data = list(D_primal_t2.nodes(data=True))

    if not nodes1_data or not nodes2_data:
        return {}
        
    nodes2, data2 = zip(*nodes2_data)
    coords2 = [np.array(d['geometry'].coords)[0] for d in data2]
    tree2 = cKDTree(coords2)

    for n1, data1 in tqdm(nodes1_data, desc="Mapping D_primal nodes"):
        point1_geom = data1.get('geometry')
        if not point1_geom:
            continue
            
        point1 = np.array(point1_geom.coords)[0]
        dist, idx = tree2.query(point1)
        
        if dist < distance_threshold:
            best_match = nodes2[idx]
            node_mapping[n1] = best_match
            
    return node_mapping

def _map_primal_edges(D_primal_t1, D_primal_t2, similarity_threshold=0.5, buffer_dist=50.0):
    """Maps edges in D_primal between two timesteps based on line similarity."""
    edge_mapping = {}

    edges2_data = list(D_primal_t2.edges(keys=True, data=True))
    if not edges2_data:
        return {}

    edges2, data2 = [], []
    for u, v, k, d in edges2_data:
        edges2.append((u,v,k))
        data2.append(d)

    lines2 = [d.get('geometry') for d in data2]

    for e1_u, e1_v, e1_k, data1 in tqdm(D_primal_t1.edges(keys=True, data=True), desc="Mapping D_primal edges"):
        line1 = data1.get('geometry')
        if not line1 or line1.is_empty:
            continue

        best_match = None
        max_similarity = 0.0
        
        buffered_line1 = line1.buffer(buffer_dist)

        for i, line2 in enumerate(lines2):
            if not line2 or line2.is_empty:
                continue

            if buffered_line1.intersects(line2):
                # A simple similarity metric based on intersection length
                intersection_len = buffered_line1.intersection(line2).length
                similarity = intersection_len / line2.length
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = edges2[i]
        
        if max_similarity > similarity_threshold:
            edge_mapping[(e1_u, e1_v, e1_k)] = best_match

    return edge_mapping


def map_graphs_over_time(D_primal_t1, G_rook_t1, D_primal_t2, G_rook_t2,
                         rook_iou_threshold=0.5, primal_dist_threshold=100.0,
                         primal_edge_sim_threshold=0.5, primal_edge_buffer=50.0):
    """
    Maps graph components (nodes and edges) between two time steps (t1 and t2).
    Mapping is based on the similarity of the locations and shapes of centerlines, 
    centerline nodes, and banklines/islands.
    Args:
        D_primal_t1 (nx.MultiDiGraph): Directed centerline graph at time t1.
        G_rook_t1 (nx.Graph): Bankline rook graph at time t1.
        D_primal_t2 (nx.MultiDiGraph): Directed centerline graph at time t2.
        G_rook_t2 (nx.Graph): Bankline rook graph at time t2.
        rook_iou_threshold (float): IoU threshold for mapping G_rook nodes.
        primal_dist_threshold (float): Distance threshold for mapping D_primal nodes.
        primal_edge_sim_threshold (float): Similarity threshold for mapping D_primal edges.
        primal_edge_buffer (float): Buffer distance for D_primal edge similarity calculation.
    Returns:
        dict: A dictionary containing mappings for G_rook nodes, D_primal nodes, and D_primal edges.
    """
    print("Mapping G_rook nodes...")
    rook_node_mapping = _map_rook_nodes(G_rook_t1, G_rook_t2, iou_threshold=rook_iou_threshold)

    print("Mapping D_primal nodes...")
    primal_node_mapping = _map_primal_nodes(D_primal_t1, D_primal_t2, distance_threshold=primal_dist_threshold)

    print("Mapping D_primal edges...")
    primal_edge_mapping = _map_primal_edges(D_primal_t1, D_primal_t2, 
                                            similarity_threshold=primal_edge_sim_threshold, 
                                            buffer_dist=primal_edge_buffer)
    
    mappings = {
        'rook_nodes': rook_node_mapping,
        'primal_nodes': primal_node_mapping,
        'primal_edges': primal_edge_mapping,
    }

    print("Graph mapping complete.")
    return mappings

def calculate_node_displacement_deviation(D_primal_t1, D_primal_t2, primal_node_mapping):
    """
    Calculates how much the node displacement vector deviates from the average 
    orientation of the connected centerline edges, and the displacement distance.

    For each mapped node, this function computes two vectors:
    1. The displacement vector, from the node's position at t1 to its new position at t2.
    2. The average orientation vector of all connected centerline edges at t1.

    It then calculates the angle between these two vectors and the length of the displacement.

    Args:
        D_primal_t1 (nx.MultiGraph): Centerline graph at time t1.
        D_primal_t2 (nx.MultiGraph): Centerline graph at time t2.
        primal_node_mapping (dict): A dictionary mapping node IDs from t1 to t2.

    Returns:
        tuple: A tuple containing two dictionaries:
               - deviations (dict): Node IDs from t1 to deviation angles in degrees.
               - distances (dict): Node IDs from t1 to displacement distances.
    """
    deviations = {}
    distances = {}

    for n1, n2 in tqdm(primal_node_mapping.items(), desc="Calculating node deviations"):
        if not D_primal_t1.has_node(n1) or not D_primal_t2.has_node(n2):
            continue

        p1 = np.array(D_primal_t1.nodes[n1]['geometry'].coords[0])
        p2 = np.array(D_primal_t2.nodes[n2]['geometry'].coords[0])
        displacement_vector = p2 - p1
        displacement_norm = np.linalg.norm(displacement_vector)
        distances[n1] = displacement_norm

        if displacement_norm == 0:
            deviations[n1] = 0.0
            continue
        
        normalized_displacement = displacement_vector / displacement_norm

        orientation_vectors = []
        # The graph may be a MultiGraph, which is undirected. Use .edges()
        for u, v, _, _ in D_primal_t1.edges(n1, data=True, keys=True):
            other_node = v if u == n1 else u
            
            p_other = np.array(D_primal_t1.nodes[other_node]['geometry'].coords[0])
            vector = p_other - p1
            
            vector_norm = np.linalg.norm(vector)
            if vector_norm > 0:
                orientation_vectors.append(vector / vector_norm)
        
        if not orientation_vectors:
            deviations[n1] = None
            continue

        avg_orientation_vector = np.mean(orientation_vectors, axis=0)
        avg_orientation_norm = np.linalg.norm(avg_orientation_vector)

        if avg_orientation_norm == 0:
            deviations[n1] = None
            continue
            
        normalized_avg_orientation = avg_orientation_vector / avg_orientation_norm

        dot_product = np.clip(np.dot(normalized_displacement, normalized_avg_orientation), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        
        deviations[n1] = np.degrees(angle_rad)
        
    return deviations, distances