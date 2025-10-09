import numpy as np
from scipy.signal import savgol_filter
from shapely.geometry import Polygon, LineString, MultiPolygon, LinearRing, MultiLineString
from shapely.geometry.polygon import InteriorRingSequence
from shapely.ops import unary_union, split, linemerge, orient
import matplotlib.pyplot as plt
from .geometry_utils import find_closest_point, getExtrapolatedLine
from shapely.geometry import Polygon, LineString, LinearRing
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

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
    poly = orient(poly, sign=1.0) # polygon needs to be oriented correctly (exterior is counterclockwise, holes are clockwise)
    path = Path.make_compound_path(
           Path(np.asarray(poly.exterior.coords)[:, :2]),
           *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])
 
    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
     
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def one_time_step(chs, ind, plotting=False):
    """
    Calculate erosion, deposition, and unchanged channel areas between two channel polygons.
    
    Parameters
    ----------
    chs : list
        List of channel polygons (Shapely Polygon or MultiPolygon objects) for different time steps.
    ind : int
        Index of the first time step to compare. The function compares chs[ind] with chs[ind+1].
    plotting : bool, optional
        If True, creates a map of the erosion, deposition, and unchanged channel areas.
        Default is False.
        
    Returns
    -------
    erosion : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Areas that were not water in time step ind but became water in time step ind+1.
    deposition : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Areas that were water in time step ind but became land in time step ind+1.
    channel : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Areas that remained as water in both time steps.
    """
    erosion = chs[ind+1].difference(chs[ind])
    deposition = chs[ind].difference(chs[ind+1])
    channel = chs[ind].intersection(chs[ind+1])

    if plotting:
        _, ax = plt.subplots()
        if type(channel) == MultiPolygon:
            for poly in channel.geoms:
                plt.fill(poly.exterior.xy[0], poly.exterior.xy[1], 'xkcd:light blue', alpha=1, edgecolor='k', linewidth=0.5)
        if type(channel) == Polygon:
            plot_polygon(ax, channel, color='xkcd:light blue', edgecolor='k', linewidth=0.5)
        if type(erosion) == MultiPolygon:
            for poly in erosion.geoms:
                plt.fill(poly.exterior.xy[0], poly.exterior.xy[1], 'xkcd:faded red', alpha=1, edgecolor='k', linewidth=0.5)
        if type(erosion) == Polygon:
            plot_polygon(ax, erosion, color='xkcd:faded red', edgecolor='k', linewidth=0.5)
        if type(deposition) == MultiPolygon:
            for poly in deposition.geoms:
                plt.fill(poly.exterior.xy[0], poly.exterior.xy[1], 'xkcd:sand', alpha=1, edgecolor='k', linewidth=0.5)
        if type(deposition) == Polygon:
            plot_polygon(ax, deposition, color='xkcd:sand', edgecolor='k', linewidth=0.5)
        plt.axis('equal')

    return erosion, deposition, channel


def create_main_channel_banks(G_rook, G_primal, D_primal, dataset):
    """
    Create main channel banks from graph structures.
    
    Parameters
    ----------
    G_rook : networkx.Graph
        Graph containing bank polygons.
    G_primal : networkx.Graph
        Graph containing node geometries.
    D_primal : networkx.DiGraph
        Directed graph containing the main path information.
        
    Returns
    -------
    tuple
        Four arrays representing x and y coordinates of the two main banks.
    """
    
    ch_nw_poly = create_channel_nw_polygon(G_rook, dataset=dataset)
    start_ind = D_primal.graph['main_path'][0][0]
    end_ind = D_primal.graph['main_path'][-1][1]
    x_start = G_primal.nodes()[start_ind]['geometry'].xy[0][0]
    y_start = G_primal.nodes()[start_ind]['geometry'].xy[1][0]
    x_end = G_primal.nodes()[end_ind]['geometry'].xy[0][0]
    y_end = G_primal.nodes()[end_ind]['geometry'].xy[1][0]

    ind1 = find_closest_point(x_start, y_start, np.vstack((ch_nw_poly.exterior.xy[0], ch_nw_poly.exterior.xy[1])).T)
    ind2 = find_closest_point(x_end, y_end, np.vstack((ch_nw_poly.exterior.xy[0], ch_nw_poly.exterior.xy[1])).T)

    if ind1 <= ind2:
        x1 = ch_nw_poly.exterior.xy[0][ind1:ind2]
        y1 = ch_nw_poly.exterior.xy[1][ind1:ind2]
        x2 = ch_nw_poly.exterior.xy[0][ind2:]
        y2 = ch_nw_poly.exterior.xy[1][ind2:]
    else:
        x1 = ch_nw_poly.exterior.xy[0][ind2:ind1]
        y1 = ch_nw_poly.exterior.xy[1][ind2:ind1]
        x2 = ch_nw_poly.exterior.xy[0][ind1:]
        y2 = ch_nw_poly.exterior.xy[1][ind1:]
    return x1, y1, x2, y2


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
    
    a1, b1 = getExtrapolatedLine((x[10], y[10]), (x[0], y[0]), ratio) # use the first two points
    a2, b2 = getExtrapolatedLine((x[-10], y[-10]), (x[-1], y[-1]), ratio) # use the last two points
    x = np.hstack((b1[0], x, b2[0]))
    y = np.hstack((b1[1], y, b2[1]))
    line = LineString(np.vstack((x, y)).T)
    return line


def smooth_polygon(poly, savgol_window=21, remove_count=1):
    """
    Smooth a polygon boundary using Savitzky-Golay filter and remove edge artifacts.
    
    This function applies Savitzky-Golay smoothing to the x and y coordinates of a 
    polygon's exterior boundary, removes potentially problematic edge points, and 
    returns a simplified smooth polygon. The smoothing helps eliminate noise while 
    preserving the overall shape characteristics.
    
    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Input polygon whose exterior boundary will be smoothed. The polygon should
        have a valid exterior ring with sufficient points for smoothing.
    savgol_window:
        Window length used in the Savitzky-Golay filter. Default is 21.
    remove_count : int, optional
        Number of points to remove from both the beginning and end of the smoothed
        coordinate arrays to eliminate edge artifacts from the filtering process.
        Default is 1. Must be less than half the number of boundary points.
    
    Returns
    -------
    shapely.geometry.Polygon
        A new smoothed polygon with simplified geometry. The polygon is automatically
        closed (first point equals last point) and simplified with a tolerance of 3
        coordinate units to reduce unnecessary vertices while preserving shape.
    
    Notes
    -----
    - Uses Savitzky-Golay filter with default window length 21 and polynomial order 3
    - Automatically closes the polygon by appending the first coordinate to the end
    - Applies simplification with tolerance=3 to reduce vertex count
    - Edge points are removed to avoid artifacts from the filtering boundary effects
    - The function assumes the input polygon has at least 25+ points for effective smoothing
    
    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> import numpy as np
    >>> 
    >>> # Create a noisy polygon
    >>> theta = np.linspace(0, 2*np.pi, 100)
    >>> x = np.cos(theta) + 0.1 * np.random.random(100)
    >>> y = np.sin(theta) + 0.1 * np.random.random(100)
    >>> noisy_poly = Polygon(zip(x, y))
    >>> 
    >>> # Smooth the polygon
    >>> smooth_poly = smooth_polygon(noisy_poly, remove_count=1)
    >>> 
    >>> # Compare areas
    >>> print(f"Original area: {noisy_poly.area:.3f}")
    >>> print(f"Smoothed area: {smooth_poly.area:.3f}")
    """
    x = poly.exterior.xy[0]
    y = poly.exterior.xy[1]
    # Use Savitzky-Golay filter on x and y
    x_sm = savgol_filter(x, savgol_window, 3)
    y_sm = savgol_filter(y, savgol_window, 3)
    # Remove first and last few points
    x_trimmed = x_sm[remove_count:-remove_count]
    y_trimmed = y_sm[remove_count:-remove_count]
    # Close the polygon
    x_sm = np.append(x_trimmed, x_trimmed[0])
    y_sm = np.append(y_trimmed, y_trimmed[0])
    # Create the smooth polygon and simplify it
    smooth_poly = Polygon(zip(x_sm, y_sm))
    tolerance = vertex_density_tolerance(smooth_poly, multiplier=0.5)
    smooth_poly = smooth_poly.simplify(tolerance)
    return smooth_poly


def smooth_line(x, y, savgol_window=21, multiplier=1.5): #, spline_ds = 25, spline_smoothing = 10000, savgol_window = 21, savgol_poly_order = 3):
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
    # Use Savitzky-Golay filter on x and y
    x_sm = savgol_filter(x, savgol_window, 3)
    y_sm = savgol_filter(y, savgol_window, 3)
    # Create the smooth polygon and simplify it
    smooth_line = LineString(zip(x_sm, y_sm))
    tolerance = vertex_density_tolerance(smooth_line, multiplier=multiplier)
    smooth_line = smooth_line.simplify(tolerance)
    return smooth_line.xy[0], smooth_line.xy[1]


def vertex_density_tolerance(geometry, multiplier=1.5):
    """
    Tolerance based on average edge length - adapts to geometry resolution.
    
    Parameters:
    -----------
    geometry : shapely.geometry.Polygon, LineString, or LinearRing
        Input geometry to analyze
    multiplier : float
        Multiplier for average edge length (default 1.5)
        
    Returns:
    --------
    float
        Tolerance value based on average edge length
    """

    
    # Get coordinates based on geometry type
    if isinstance(geometry, Polygon):
        coords = list(geometry.exterior.coords)
    elif isinstance(geometry, (LineString, LinearRing)):
        coords = list(geometry.coords)
    else:
        # Handle other geometry types or raise error
        try:
            coords = list(geometry.coords)  # Try direct access
        except AttributeError:
            raise TypeError(f"Unsupported geometry type: {type(geometry)}")
    
    # Calculate edge lengths
    edge_lengths = []
    for i in range(len(coords) - 1):
        dx = coords[i+1][0] - coords[i][0]
        dy = coords[i+1][1] - coords[i][1]
        edge_lengths.append(np.sqrt(dx**2 + dy**2))
    
    if not edge_lengths:  # Handle case with insufficient points
        return 0.0
        
    return np.mean(edge_lengths) * multiplier


def smooth_banklines(G_rook, dataset, mndwi, save_smooth_lines=False):
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
    im_boundary = Polygon([dataset.xy(0,0), dataset.xy(0, mndwi.shape[1]), dataset.xy(mndwi.shape[0], mndwi.shape[1]), dataset.xy(mndwi.shape[0], 0)])
    for i in range(2): # deal with the main banklines first - they are more complicated
        # first need to isolate line that can be / should be smoothed:
        other_side = im_boundary.buffer(-100).difference(G_rook.nodes()[i]['bank_polygon'])
        line = other_side.intersection(G_rook.nodes()[i]['bank_polygon'])
        new_line = []
        for geom in line.geoms:
            if type(geom) == LineString:
                new_line.append(geom)
        line = linemerge(new_line)
        if type(line) != LineString:
            lengths = []
            for geom in line.geoms:
                # lengths.append(geom.length)
                lengths.append(len(geom.coords))
            line = line.geoms[np.argmax(lengths)]
        # line = extend_line(line.xy[0], line.xy[1], 10000)
        # x1, y1 = smooth_line(line.xy[0], line.xy[1], multiplier=0.1)
        # line = LineString(np.vstack((x1, y1)).T)
        x1, y1 = smooth_line(line.xy[0], line.xy[1], multiplier=0.1) # smooth line first
        line = extend_line(x1, y1, 10000) # then do the extension needed to cut the image boundary rectangle
        geoms = split(im_boundary, line)
        if len(geoms.geoms) > 1:
            if geoms.geoms[0].intersection(G_rook.nodes[i]["bank_polygon"]).area > geoms.geoms[1].intersection(G_rook.nodes[i]["bank_polygon"]).area:
                if save_smooth_lines:
                    G_rook.nodes()[i]['bank_polygon'] = geoms.geoms[0]
                polys.append(geoms.geoms[0])
            else:
                if save_smooth_lines:
                    G_rook.nodes()[i]['bank_polygon'] = geoms.geoms[1]
                polys.append(geoms.geoms[1])
        else:
            print("Warning: only one polygon found after splitting when trying to smooth banklines")
    for i in range(2, len(G_rook.nodes)):
        smooth_bank_poly = smooth_polygon(G_rook.nodes()[i]['bank_polygon'])
        if save_smooth_lines:
            G_rook.nodes()[i]['bank_polygon'] = smooth_bank_poly
        polys.append(smooth_bank_poly)
    return polys


def simplify_if_needed(geom, simplify_tolerance):
    """Apply simplification to reduce vertices."""
    if hasattr(geom, 'exterior') and len(geom.exterior.coords) > 100:
        return geom.simplify(simplify_tolerance, preserve_topology=True)
    return geom


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
    simplify_tolerance = buffer * 0.1  # 10% of buffer distance
    both_banks = simplify_if_needed(G_rook.nodes()[0]['bank_polygon'].buffer(buffer).union(G_rook.nodes()[1]['bank_polygon'].buffer(buffer)), simplify_tolerance)
    if type(both_banks) == Polygon and len(both_banks.interiors) > 0:
        ch_belt_pieces = both_banks.interiors
    else:
        # Create image boundary polygon using transform instead of xy method
        transform = dataset.transform
        # Convert pixel coordinates to world coordinates using the transform
        ul_x, ul_y = transform * (0, 0)  # Upper left
        ur_x, ur_y = transform * (dataset.shape[1], 0)  # Upper right
        lr_x, lr_y = transform * (dataset.shape[1], dataset.shape[0])  # Lower right
        ll_x, ll_y = transform * (0, dataset.shape[0])  # Lower left
        im_boundary = Polygon([(ul_x, ul_y), (ur_x, ur_y), (lr_x, lr_y), (ll_x, ll_y)])
        # Create buffers for bank polygons once to avoid redundant calculations
        bank0_buffer = simplify_if_needed(G_rook.nodes()[0]['bank_polygon'].buffer(buffer), simplify_tolerance)
        bank1_buffer = simplify_if_needed(G_rook.nodes()[1]['bank_polygon'].buffer(buffer), simplify_tolerance)
        # Combine bank buffers and subtract from image boundary buffer
        ch_belt_pieces = im_boundary.buffer(-10).difference(bank0_buffer.union(bank1_buffer))
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
    ch_nw_poly = ch_nw_poly.simplify(0.1, preserve_topology=True)
    return ch_nw_poly

def create_channel_nw_polygon_old(G_rook, buffer=10, ch_mouth_poly=None, dataset=None):
    """
    Creates a polygon representing the channel network.

    Parameters
    ----------
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



def straighten_channel(xl, yl, xls, yls):
    """
    Straighten a channel while preserving local shapes.
    
    Parameters
    ----------
    xl : array_like
        The x-coordinates of the original sinuous channel.
    yl : array_like
        The y-coordinates of the original sinuous channel.
    xls : array_like
        The x-coordinates of the smoothed centerline.
    yls : array_like
        The y-coordinates of the smoothed centerline.
        
    Returns
    -------
    xl_straight : ndarray
        The x-coordinates of the straightened channel.
    yl_straight : ndarray
        The y-coordinates of the straightened channel.
    """
    xl = np.array(xl)
    yl = np.array(yl)
    xls = np.array(xls)
    yls = np.array(yls)
    
    # Calculate the distances along the smoothed centerline
    dxs = np.diff(xls)
    dys = np.diff(yls)
    ds = np.sqrt(dxs**2 + dys**2)
    s = np.zeros(len(xls))
    s[1:] = np.cumsum(ds)
    
    # Create a straight reference line with the same total length
    xref = np.zeros_like(xls)
    yref = s
    
    # For each point in the original centerline, find the closest point on the smoothed centerline
    xl_straight = np.zeros_like(xl)
    yl_straight = np.zeros_like(yl)
    
    for i in range(len(xl)):
        # Find the closest point on the smoothed centerline
        dist_to_smooth = np.sqrt((xl[i] - xls)**2 + (yl[i] - yls)**2)
        closest_idx = np.argmin(dist_to_smooth)
        
        # If closest point is at the start or end, just use that point
        if closest_idx == 0:
            tangent_angle = np.arctan2(dys[0], dxs[0])
            normal_angle = tangent_angle + np.pi/2
            
            # Vector from smoothed centerline to original point
            dx = xl[i] - xls[0]
            dy = yl[i] - yls[0]
            
            # Project onto normal direction to get signed distance
            normal_dist = dx * np.cos(normal_angle) + dy * np.sin(normal_angle)
            
            # Map to the straight reference line
            xl_straight[i] = normal_dist
            yl_straight[i] = 0
            
        elif closest_idx == len(xls) - 1:
            tangent_angle = np.arctan2(dys[-1], dxs[-1])
            normal_angle = tangent_angle + np.pi/2
            
            # Vector from smoothed centerline to original point
            dx = xl[i] - xls[-1]
            dy = yl[i] - yls[-1]
            
            # Project onto normal direction to get signed distance
            normal_dist = dx * np.cos(normal_angle) + dy * np.sin(normal_angle)
            
            # Map to the straight reference line
            xl_straight[i] = normal_dist
            yl_straight[i] = s[-1]
            
        else:
                # For interior points, interpolate between the two nearest segments
                tangent_angle = np.arctan2(dys[closest_idx], dxs[closest_idx])
                normal_angle = tangent_angle + np.pi/2
                
                # Vector from smoothed centerline to original point
                dx = xl[i] - xls[closest_idx]
                dy = yl[i] - yls[closest_idx]
                
                # Project onto normal direction to get signed distance
                normal_dist = dx * np.cos(normal_angle) + dy * np.sin(normal_angle)
                
                # Map to the straight reference line
                xl_straight[i] = normal_dist
                yl_straight[i] = s[closest_idx]

    return xl_straight, yl_straight


def straighten_polygon(polygon, xls, yls):
    """
    Straighten a polygon along a centerline.
    
    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The polygon to straighten.
    xls : array_like
        The x-coordinates of the smoothed centerline.
    yls : array_like
        The y-coordinates of the smoothed centerline.
        
    Returns
    -------
    straight_polygon : shapely.geometry.Polygon
        The straightened polygon.
    """
    # Extract polygon exterior coordinates
    xl_exterior = np.array(polygon.exterior.xy[0])
    yl_exterior = np.array(polygon.exterior.xy[1])
    
    # Straighten the exterior coordinates
    xl_straight, yl_straight = straighten_channel(xl_exterior, yl_exterior, xls, yls)
    
    # Create a new exterior ring
    exterior_straight = LinearRing(np.column_stack([xl_straight, yl_straight]))
    
    # Straighten each interior ring (hole) if any
    interior_rings_straight = []
    for interior in polygon.interiors:
        xl_interior = np.array(interior.xy[0])
        yl_interior = np.array(interior.xy[1])
        
        # Straighten the interior coordinates
        xl_interior_straight, yl_interior_straight = straighten_channel(xl_interior, yl_interior, xls, yls)
        
        # Create a new interior ring
        interior_straight = LinearRing(np.column_stack([xl_interior_straight, yl_interior_straight]))
        interior_rings_straight.append(interior_straight)
    
    # Create the straightened polygon
    straight_polygon = Polygon(exterior_straight, interior_rings_straight)
    
    return straight_polygon


def count_vertices(geometry):
    """Count total number of vertices in a geometry"""
    from shapely.geometry import Polygon, MultiPolygon
    
    if isinstance(geometry, Polygon):
        count = len(geometry.exterior.coords)
        for interior in geometry.interiors:
            count += len(interior.coords)
        return count
    elif isinstance(geometry, MultiPolygon):
        return sum(count_vertices(poly) for poly in geometry.geoms)
    else:
        return 0


def remove_endpoints(poly_sm, remove_count=3):
    """
    remove problematic start/end points
    """
    x, y = poly_sm.xy
    x, y = np.array(x), np.array(y)
    
    # Remove first and last few points
    x_trimmed = x[remove_count:-remove_count]
    y_trimmed = y[remove_count:-remove_count]
    
    # Close the polygon
    x_result = np.append(x_trimmed, x_trimmed[0])
    y_result = np.append(y_trimmed, y_trimmed[0])
    
    return x_result, y_result


def polygon_to_svg(geometry, filename, width=4000, height=800, 
                   fill_color="white", stroke_color="black", stroke_width=0.25,
                   interior_fill="black", interior_stroke="black", 
                   smooth=True, savgol_window=7, 
                   scalebar=True, scalebar_length_km=100, scalebar_position="bottom-left",
                   scalebar_color="black", scalebar_width=3, scalebar_text_size=32,
                   slice_polygon=False, num_slices=4):
    """
    Save a Shapely Polygon or MultiPolygon to SVG file with improved smoothing.
    
    Parameters
    ----------
    geometry : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        The geometry to save
    filename : str
        Output SVG filename
    width, height : int
        Dimensions of the SVG canvas
    fill_color : str or list
        Fill color(s) for polygons. If a list, colors are assigned to each polygon in a MultiPolygon
    stroke_color : str or list
        Stroke color(s) for polygons
    stroke_width : int
        Stroke width for all paths
    interior_fill : str
        Fill color for interior holes
    interior_stroke : str
        Stroke color for interior holes
    smooth : bool
        Whether to apply smoothing
    ds : float
        Distance between points after resampling (for smoothing)
    smoothing : float
        Smoothing factor for splines (higher values = smoother curves)
    scalebar : bool
        Whether to include a scalebar
    scalebar_length_km : float
        Length of scalebar in kilometers
    scalebar_position : str
        Position of scalebar: "bottom-left", "bottom-right", "top-left", "top-right"
    scalebar_color : str
        Color of the scalebar
    scalebar_width : int
        Width of the scalebar line
    scalebar_text_size : int
        Size of the scalebar text
    slice_polygon : bool
        Whether to slice the polygon into multiple pieces
    num_slices : int
        Number of slices to create (4-10 recommended)
    """
    import numpy as np
    from shapely.geometry import Polygon, MultiPolygon, LineString
    from shapely.ops import split
    
    # Make a copy to avoid modifying the original
    import copy
    working_geom = copy.deepcopy(geometry)
    
    # Get original vertex count
    orig_vertices = count_vertices(working_geom)
    
    # Apply smoothing if requested
    if smooth:
        x = working_geom.exterior.xy[0]
        y = working_geom.exterior.xy[1]
        exterior_smooth = LineString(zip(x, y))

        # Smooth interiors
        interiors_smooth = []
        for interior in working_geom.interiors:
            if Polygon(interior).area>0:
                x_interior, y_interior = interior.xy
                if len(x_interior) >= savgol_window + 3:
                    x_interior = savgol_filter(x_interior, savgol_window, 3)
                    y_interior = savgol_filter(y_interior, savgol_window, 3)
                interior_lstr = LineString(zip(x_interior, y_interior))
                interior_lstr = interior_lstr.simplify(3)
                x_smooth, y_smooth = remove_endpoints(interior_lstr, remove_count=1)
                interior_lstr = LineString(zip(x_smooth, y_smooth))
                if interior_lstr.is_valid:
                    interiors_smooth.append(interior_lstr)
        working_geom = Polygon(exterior_smooth, interiors_smooth)

    # Convert to list of polygons for consistent handling
    if isinstance(working_geom, Polygon):
        polygons = [working_geom]
    elif isinstance(working_geom, MultiPolygon):
        polygons = list(working_geom.geoms)
    else:
        raise ValueError("Input geometry must be Polygon or MultiPolygon")
    
    # Slice polygons if requested
    if slice_polygon and num_slices > 1:
        sliced_polygons = []
        for polygon in polygons:
            # Get polygon bounds for slicing (note: we use original coordinates here, not swapped)
            minx, miny, maxx, maxy = polygon.bounds
            slice_height = (maxy - miny) / num_slices
            
            current_parts = [polygon]
            
            # Create vertical slicing lines
            for i in range(1, num_slices):
                slice_y = miny + i * slice_height
                # Create a vertical line that extends beyond the polygon bounds
                slice_line = LineString([(minx - 1000, slice_y), (maxx + 1000, slice_y)])
                
                # Split all current parts with this line
                new_parts = []
                for part in current_parts:
                    try:
                        split_result = split(part, slice_line)
                        if hasattr(split_result, 'geoms'):
                            new_parts.extend(list(split_result.geoms))
                        else:
                            new_parts.append(part)  # If split failed, keep original
                    except:
                        new_parts.append(part)  # If split failed, keep original
                current_parts = new_parts
            
            # Filter out non-polygon results and add to sliced_polygons
            for part in current_parts:
                if isinstance(part, Polygon) and not part.is_empty:
                    sliced_polygons.append(part)
        
        polygons = sliced_polygons
        print(f"Sliced into {len(polygons)} pieces")
    
    # Print info about smoothing results
    new_vertices = count_vertices(MultiPolygon(polygons) if len(polygons) > 1 else polygons[0])
    print(f"Vertices reduced from {orig_vertices:,} to {new_vertices:,} " + 
          f"({(1 - new_vertices/orig_vertices)*100:.1f}% reduction)")
    
    # Ensure fill_color and stroke_color are lists of the right length
    if isinstance(fill_color, str):
        fill_color = [fill_color] * len(polygons)
    if isinstance(stroke_color, str):
        stroke_color = [stroke_color] * len(polygons)
    
    # Calculate bounds from all polygons for scaling - SWAP X AND Y
    y_min = min(polygon.bounds[0] for polygon in polygons)  # was x_min
    y_max = max(polygon.bounds[2] for polygon in polygons)  # was x_max
    x_min = min(polygon.bounds[1] for polygon in polygons)  # was y_min
    x_max = max(polygon.bounds[3] for polygon in polygons)  # was y_max
    
    # Calculate scaling to fit the canvas with margins
    margin = 50
    x_scale = (width - 2*margin) / (x_max - x_min) if x_max > x_min else 1
    y_scale = (height - 2*margin) / (y_max - y_min) if y_max > y_min else 1
    
    # Use the smaller scale to maintain aspect ratio
    scale = min(x_scale, y_scale)
    
    # Center the plot
    x_center = width / 2
    y_center = height / 2
    
    with open(filename, 'w') as f:
        # SVG header
        f.write(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n')
        
        # Process each polygon
        count = 0
        for poly_idx, polygon in enumerate(polygons):
            # Write exterior polygon
            f.write(f'  <g id="polygon_{poly_idx}_exterior" fill="{fill_color[poly_idx]}" ' + 
                    f'stroke="{stroke_color[poly_idx]}" stroke-width="{stroke_width}">\n')
            
            # Extract and transform exterior points - SWAP X AND Y
            exterior_x, exterior_y = polygon.exterior.xy
            # Swap coordinates: original x becomes SVG y, original y becomes SVG x
            # Flip y-axis to fix upside-down issue (SVG y increases downward)
            x_svg = x_center + (np.array(exterior_y) - (x_min + x_max)/2) * scale
            y_svg = y_center - (np.array(exterior_x) - (y_min + y_max)/2) * scale
            
            # Create path data for exterior
            path_data = f'M {x_svg[0]:.3f},{y_svg[0]:.3f}'
            for i in range(1, len(x_svg)):
                path_data += f' L {x_svg[i]:.3f},{y_svg[i]:.3f}'
            path_data += ' Z'  # Close the path
            
            f.write(f'    <path d="{path_data}" />\n')
            f.write('  </g>\n')
            
            # Write interior holes
            for i, interior in enumerate(polygon.interiors):
                int_x, int_y = interior.xy
                
                f.write(f'  <g id="polygon_{poly_idx}_interior_{i}" fill="{interior_fill}" ' + 
                        f'stroke="{interior_stroke}" stroke-width="{stroke_width}">\n')
                
                # Swap coordinates: original x becomes SVG y, original y becomes SVG x
                # Flip y-axis to fix upside-down issue
                x_svg = x_center + (np.array(int_y) - (x_min + x_max)/2) * scale
                y_svg = y_center - (np.array(int_x) - (y_min + y_max)/2) * scale
                
                # Create path data for interior
                path_data = f'M {x_svg[0]:.3f},{y_svg[0]:.3f}'
                for j in range(1, len(x_svg)):
                    path_data += f' L {x_svg[j]:.3f},{y_svg[j]:.3f}'
                path_data += ' Z'  # Close the path
                
                f.write(f'    <path d="{path_data}" />\n')
                f.write('  </g>\n')
        
        # Add scalebar if requested
        if scalebar:
            # Calculate scalebar length in SVG pixels
            scalebar_length_m = scalebar_length_km * 1000  # Convert km to meters
            scalebar_length_svg = scalebar_length_m * scale
            
            # Determine scalebar position
            scalebar_margin = 30
            if scalebar_position == "bottom-left":
                scalebar_x = scalebar_margin
                scalebar_y = height - scalebar_margin - 20
            elif scalebar_position == "bottom-right":
                scalebar_x = width - scalebar_margin - scalebar_length_svg
                scalebar_y = height - scalebar_margin - 20
            elif scalebar_position == "top-left":
                scalebar_x = scalebar_margin
                scalebar_y = scalebar_margin + 20
            elif scalebar_position == "top-right":
                scalebar_x = width - scalebar_margin - scalebar_length_svg
                scalebar_y = scalebar_margin + 20
            else:
                # Default to bottom-left
                scalebar_x = scalebar_margin
                scalebar_y = height - scalebar_margin - 20
            
            # Add scalebar group
            f.write(f'  <g id="scalebar">\n')
            
            # Scalebar line
            f.write(f'    <line x1="{scalebar_x:.1f}" y1="{scalebar_y:.1f}" ' + 
                    f'x2="{scalebar_x + scalebar_length_svg:.1f}" y2="{scalebar_y:.1f}" ' + 
                    f'stroke="{scalebar_color}" stroke-width="{scalebar_width}" />\n')
            
            # Scalebar end markers (short vertical lines)
            marker_height = 5
            f.write(f'    <line x1="{scalebar_x:.1f}" y1="{scalebar_y - marker_height:.1f}" ' + 
                    f'x2="{scalebar_x:.1f}" y2="{scalebar_y + marker_height:.1f}" ' + 
                    f'stroke="{scalebar_color}" stroke-width="{scalebar_width}" />\n')
            f.write(f'    <line x1="{scalebar_x + scalebar_length_svg:.1f}" y1="{scalebar_y - marker_height:.1f}" ' + 
                    f'x2="{scalebar_x + scalebar_length_svg:.1f}" y2="{scalebar_y + marker_height:.1f}" ' + 
                    f'stroke="{scalebar_color}" stroke-width="{scalebar_width}" />\n')
            
            # Scalebar text
            text_x = scalebar_x + scalebar_length_svg / 2
            text_y = scalebar_y - 10
            f.write(f'    <text x="{text_x:.1f}" y="{text_y:.1f}" ' + 
                    f'text-anchor="middle" font-family="Arial, sans-serif" ' + 
                    f'font-size="{scalebar_text_size}" fill="{scalebar_color}">' + 
                    f'{scalebar_length_km:.0f} km</text>\n')
            
            f.write('  </g>\n')
            
        # SVG footer
        f.write('</svg>\n')
    
    print(f"Created SVG file: {filename}")
    if scalebar:
        print(f"Added scalebar: {scalebar_length_km} km at {scalebar_position}") 