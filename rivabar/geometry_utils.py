import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from sklearn.neighbors import KDTree


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

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([1, 2, 3])
    >>> extended_line = extend_line(x, y, 0.5)
    """
    p1 = (x[0], y[0])
    p2 = (x[1], y[1])
    a, b = getExtrapolatedLine(p1, p2, -ratio)
    p1 = (x[-2], y[-2])
    p2 = (x[-1], y[-1])
    c, d = getExtrapolatedLine(p1, p2, 1 + ratio)
    x_new = [b[0]] + list(x) + [d[0]]
    y_new = [b[1]] + list(y) + [d[1]]
    line = LineString(list(zip(x_new, y_new)))
    return line

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
    
    print(len(segment1_coords), len(segment2_coords))
    
    # Handle cases where segments have only 1 point
    if len(segment1_coords) == 1 and len(segment2_coords) == 1:
        # Both segments have only 1 point - return the first one
        coord = segment1_coords[0]
        return [coord[0]], [coord[1]]
    
    elif len(segment1_coords) == 1:
        # segment1 has only 1 point, so segment2 is longer
        if len(segment2_coords) >= 2:
            segment2 = LineString(segment2_coords)
            return _get_oriented_coords(segment2, xs, ys, xs0 if type(xs) == np.int_ else None, points)
        else:
            # This shouldn't happen, but handle gracefully
            coord = segment2_coords[0]
            return [coord[0]], [coord[1]]
    
    elif len(segment2_coords) == 1:
        # segment2 has only 1 point, so segment1 is longer
        if len(segment1_coords) >= 2:
            segment1 = LineString(segment1_coords)
            return _get_oriented_coords(segment1, xs, ys, xs0 if type(xs) == np.int_ else None, points)
        else:
            # This shouldn't happen, but handle gracefully
            coord = segment1_coords[0]
            return [coord[0]], [coord[1]]
    
    else:
        # Both segments have 2+ points - use original logic
        segment1 = LineString(segment1_coords)
        segment2 = LineString(segment2_coords)
        
        # Compare simplified lengths to determine longer segment
        if len(segment1.simplify(100).xy[0]) > len(segment2.simplify(100).xy[0]):
            return _get_oriented_coords(segment1, xs, ys, xs0 if type(xs) == np.int_ else None, points)
        else:
            return _get_oriented_coords(segment2, xs, ys, xs0 if type(xs) == np.int_ else None, points)

def _get_oriented_coords(segment, xs, ys, xs0, points):
    """
    Helper function to get coordinates in the correct orientation.
    
    Parameters
    ----------
    segment : shapely.geometry.LineString
        The segment to get coordinates from
    xs : int or array-like
        x-coordinates or point index
    ys : array-like
        y-coordinates (used if xs is array-like)
    xs0 : int or None
        Point index if xs was originally an integer
    points : list
        List of all polygon points
        
    Returns
    -------
    tuple
        (x_coords, y_coords) in the correct orientation
    """
    if xs0 is not None:  # xs was originally an integer
        if segment.xy[0][0] == points[xs0][0]:
            return segment.xy[0], segment.xy[1]
        else:
            return segment.xy[0][::-1], segment.xy[1][::-1]
    else:  # xs is array-like
        dist_to_start_point = np.linalg.norm(np.array([xs[0], ys[0]]) - np.array(segment.xy)[:,0])
        dist_to_end_point = np.linalg.norm(np.array([xs[-1], ys[-1]]) - np.array(segment.xy)[:,0])
        if dist_to_start_point < dist_to_end_point:
            return segment.xy[0], segment.xy[1]
        else:
            return segment.xy[0][::-1], segment.xy[1][::-1]

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