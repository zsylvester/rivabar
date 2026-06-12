import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import adjust_band
from rasterio import features
from tqdm import tqdm
from skimage.morphology import remove_small_holes
from skimage.measure import label, regionprops_table
import geopandas
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from rasterio.enums import Resampling

from .utils import convert_to_uint8, normalized_difference
from .polygon_processing import create_channel_nw_polygon


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
    if fname[:4] == 'LC08' or fname[:4] == 'LC09': # landsat 8 and 9
        band_numbers = [3, 6, 2, 4]
    else: # landsat 4 and 5
        band_numbers = [3, 5, 2, 1]
    bands, dataset = process_band(dirname, fname, band_numbers)

    if fname[:4] == 'LC08' or fname[:4] == 'LC09': # landsat 8 and 9
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
    if fname[:4] == 'LC08' or fname[:4] == 'LC09': # landsat 8 and 9
        mndwi = normalized_difference(bands[3], bands[6])
    else: # landsat 4 and 5
        mndwi = normalized_difference(bands[2], bands[5])
    mndwi[mndwi > mndwi_threshold] = 1
    mndwi[mndwi != 1] = 0

    return equ, mndwi, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, delta_x, delta_y


def read_water_index(dirname, fname, mndwi_threshold, type='mndwi'):
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
        if type == 'mndwi':
            mndwi = dataset.read(1)
            mndwi[mndwi > float(mndwi_threshold)] = 1
            mndwi[mndwi != 1] = 0
        elif type == 'ddwi':
            mndwi = dataset.read(1) # no need for thresholding with DDWI image
        left_utm_x = dataset.transform[2]
        upper_utm_y = dataset.transform[5]
        delta_x = dataset.transform[0]
        delta_y = dataset.transform[4]
    nxpix = mndwi.shape[1]
    nypix = mndwi.shape[0]
    right_utm_x = left_utm_x + delta_x*nxpix
    lower_utm_y = upper_utm_y + delta_y*nypix
    return mndwi, left_utm_x, upper_utm_y, right_utm_x, lower_utm_y


def create_mndwi(dirname, fname, file_type, water_index_type='mndwi', mndwi_threshold=0.01, delete_pixels_polys=False, small_hole_threshold=64, remove_smaller_components=True, solidity_filter=False, max_water_fraction=0.15):
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
    max_water_fraction : float or None, optional
        Maximum fraction of valid pixels that can be classified as water.
        If exceeded, the scene is assumed to be cloud-contaminated (clouds
        produce positive MNDWI values that get misclassified as water) and
        the function returns None. Set to None to disable. Default is 0.15
        (normal river scenes typically have 1-3% water).

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
        try:
            file_path = os.path.join(dirname, fname)
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                return None, None, None, None, None, None, None, None
            
            with rasterio.open(file_path) as src:
                mndwi = src.read(1)
                # Extract transform information while the dataset is open
                left_utm_x = src.transform[2]
                upper_utm_y = src.transform[5]
                delta_x = src.transform[0]
                delta_y = src.transform[4]
            
            # Re-open the dataset to keep it available for later use
            dataset = rasterio.open(file_path)
            
        except Exception as e:
            print(f"Error opening file {dirname + fname}: {str(e)}")
            return None, None, None, None, None, None, None, None
        
        if water_index_type == 'mndwi':
            mndwi[mndwi > float(mndwi_threshold)] = 1
            mndwi[mndwi != 1] = 0
        elif water_index_type == 'ddwi': # no need for thresholding with DDWI image
            pass

        nxpix = mndwi.shape[1]
        nypix = mndwi.shape[0]
        right_utm_x = left_utm_x + delta_x*nxpix
        lower_utm_y = upper_utm_y + delta_y*nypix
    else: # single Landsat TIF file or multiple Landsat TIF files
        if type(mndwi_threshold) == str: # make sure that mndwi_threshold is a float
            mndwi_threshold = float(mndwi_threshold)
        equ, mndwi, dataset, left_utm_x, right_utm_x, lower_utm_y, upper_utm_y, delta_x, delta_y = read_landsat_data(dirname, fname, mndwi_threshold = mndwi_threshold)

    if delete_pixels_polys: # set pixels to zero in areas defined by polygons (e.g., bridges)
        rst_arr = mndwi.astype('uint32').copy()
        shapes = ((geom, value) for geom, value in zip(delete_pixels_polys, np.ones((len(delete_pixels_polys),))))
        mndwi = features.rasterize(shapes=shapes, fill=0, out=rst_arr, transform=dataset.transform)
    # if max(np.shape(mndwi)) > 2**16//2:
    #     print('maximum dimension of input image needs to be smaller than 32768!')
    #     return None
    # removing small holes
    mndwi = remove_small_holes(mndwi.astype('bool'), float(small_hole_threshold)) # remove small bars / islands

    # Check for cloud contamination: clouds produce positive MNDWI values
    # that get misclassified as water, resulting in abnormally high water fraction.
    # This check runs after hole removal, which fills gaps between cloud patches
    # and gives a more accurate picture of what will be skeletonized.
    if max_water_fraction is not None:
        water_fraction = mndwi.sum() / mndwi.size
        if water_fraction > max_water_fraction:
            print(f'Scene appears cloud-contaminated: {water_fraction:.1%} of pixels classified as water (threshold: {max_water_fraction:.0%})')
            return None, None, None, None, None, None, None, None

    if remove_smaller_components:
        # remove small components (= lakes) from water index image
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


def downsample_raster(input_path, output_path, scale_factor=0.5):
    """
    Downsample a raster by a given scale factor.
    
    Args:
        input_path (str): Path to input raster
        output_path (str): Path to output raster  
        scale_factor (float): Scale factor (0.5 = half size)
    """
    with rasterio.open(input_path) as dataset:
        # Calculate new dimensions
        new_height = int(dataset.height * scale_factor)
        new_width = int(dataset.width * scale_factor)
        
        # Read and resample the data
        data = dataset.read(
            out_shape=(dataset.count, new_height, new_width),
            resampling=Resampling.bilinear
        )
        
        # Update the transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / new_width),
            (dataset.height / new_height)
        )
        
        # Update metadata
        meta = dataset.meta.copy()
        meta.update({
            'height': new_height,
            'width': new_width,
            'transform': transform
        })
        
        # Write the downsampled raster
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data)


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

def save_planetscope_river_result(river, source_files, save_dir='planetscope_results', 
                                scene_id=None, cloud_cover=None, **extra_metadata):
    """
    Convenience function to save PlanetScope river results with automatic metadata extraction.
    
    This function extracts metadata from PlanetScope filenames and saves the river object
    with all relevant information in the same format as Landsat batch processing.
    
    Parameters
    ----------
    river : River
        The river object to save
    source_files : list
        List of source PlanetScope files used to create the mosaic
    save_dir : str, optional
        Directory to save results (default: 'planetscope_results')
    scene_id : str, optional
        Custom scene identifier. If None, will be auto-generated
    cloud_cover : float, optional
        Cloud cover percentage
    **extra_metadata : dict
        Additional metadata to store
        
    Returns
    -------
    str
        Path to saved pickle file
        
    Examples
    --------
    >>> # From your notebook:
    >>> fnames = glob.glob('/path/to/20250612*AnalyticMS_SR_8b_clip.tif')
    >>> result_file = rb.save_planetscope_river_result(
    ...     river, 
    ...     source_files=fnames,
    ...     scene_id='rio_beni_20250612',
    ...     cloud_cover=10.5,
    ...     location='Rio Beni',
    ...     processing_notes='DDWI threshold auto-detected'
    ... )
    """
    import os
    from datetime import datetime
    
    # Extract metadata from source files
    metadata = river.extract_planetscope_metadata_from_files(source_files)
    
    # Generate scene_id if not provided
    if scene_id is None:
        if metadata['unique_dates']:
            primary_date = metadata['primary_acquisition_date'].replace('-', '')
            scene_id = f"planetscope_{primary_date}_{len(source_files)}tiles"
        else:
            scene_id = f"planetscope_mosaic_{datetime.now().strftime('%Y%m%d')}"
    
    # Prepare processing metadata
    processing_metadata = {
        'n_source_tiles': len(source_files),
        'source_filenames': [os.path.basename(f) for f in source_files],
        'metadata_extracted': metadata,
        **extra_metadata
    }
    
    # Save using the river method (MNDWI will be automatically excluded)
    result_file = river.save_planetscope_result(
        save_dir=save_dir,
        scene_id=scene_id,
        acquisition_date=metadata['primary_acquisition_date'],
        cloud_cover=cloud_cover,
        source_files=source_files,
        processing_metadata=processing_metadata
    )
    
    return result_file 

class MinimalDataset:
    """
    A minimal dataset placeholder that stores essential raster information.
    
    This class provides the same interface as rasterio datasets for the basic
    attributes needed by river objects (crs, transform, shape).
    """
    def __init__(self, crs, transform, shape):
        """
        Initialize a minimal dataset with basic raster information.
        
        Parameters
        ----------
        crs : str or rasterio.crs.CRS
            Coordinate reference system
        transform : rasterio.transform.Affine
            Affine transformation matrix
        shape : tuple
            Shape of the raster (height, width)
        """
        from rasterio.crs import CRS
        
        if isinstance(crs, str):
            self.crs = CRS.from_string(crs)
        else:
            self.crs = crs
        self.transform = transform
        self.shape = shape

    def xy(self, row, col, offset='center'):
        """Return the (x, y) coordinates of a pixel, like rasterio datasets."""
        from rasterio.transform import xy as transform_xy
        return transform_xy(self.transform, row, col, offset=offset)

    def __repr__(self):
        return f"MinimalDataset(crs={self.crs}, transform={self.transform}, shape={self.shape})"


def create_water_mask_from_mapping(G_rook, dataset, output_path=None, 
                                   include_islands=True, buffer_distance=0,
                                   fill_value=1, nodata_value=0):
    """
    Create a water mask raster from rivabar mapping results.
    
    This function creates a water mask by:
    1. Creating a bounding polygon from the image extent
    2. Subtracting all land polygons (main banks and islands) from it
    3. The remaining area is water
    
    Parameters
    ----------
    G_rook : networkx.Graph
        Graph from rivabar mapping containing 'bank_polygon' attributes on nodes.
        Nodes 0 and 1 are the main channel banks (land), nodes 2+ are islands/bars (land).
    dataset : rasterio.DatasetReader or MinimalDataset
        Dataset with georeferencing information (crs, transform, shape/bounds).
    output_path : str, optional
        Path to save the water mask as a GeoTIFF. If None, only returns the array.
    include_islands : bool, optional
        If True, subtract island/bar areas from water (they become land). Default is True.
    buffer_distance : float, optional
        Buffer distance to apply to land polygons (positive = expand land, negative = shrink).
        Default is 0.
    fill_value : int, optional
        Value for water pixels (default is 1).
    nodata_value : int, optional
        Value for land/nodata pixels (default is 0).
    
    Returns
    -------
    water_mask : numpy.ndarray
        2D binary array where fill_value indicates water and nodata_value indicates land.
    
    Examples
    --------
    >>> # After running rivabar mapping
    >>> D_primal, G_rook, G_primal, mndwi, dataset, ... = rb.extract_centerline(...)
    >>> 
    >>> # Create water mask
    >>> water_mask = rb.create_water_mask_from_mapping(G_rook, dataset)
    >>> 
    >>> # Save to file
    >>> water_mask = rb.create_water_mask_from_mapping(G_rook, dataset, 
    ...                                                output_path='water_mask.tif')
    """
    from rasterio.features import rasterize
    from shapely.geometry import box

    # Get raster dimensions and transform from dataset
    if hasattr(dataset, 'shape'):
        if isinstance(dataset.shape, tuple) and len(dataset.shape) >= 2:
            height, width = dataset.shape[-2], dataset.shape[-1]
        else:
            height, width = dataset.shape, dataset.shape
    elif hasattr(dataset, 'height') and hasattr(dataset, 'width'):
        height, width = dataset.height, dataset.width
    else:
        raise ValueError("Cannot determine raster dimensions from dataset")
    
    transform = dataset.transform
    crs = dataset.crs
    
    # Calculate bounds from transform
    left = transform.c
    top = transform.f
    right = left + width * transform.a
    bottom = top + height * transform.e  # Note: transform.e is typically negative
    
    # Ensure proper min/max
    min_x, max_x = min(left, right), max(left, right)
    min_y, max_y = min(top, bottom), max(top, bottom)
    
    # Create bounding box polygon for the entire image extent
    bbox = box(min_x, min_y, max_x, max_y)
    
    # Collect ALL land polygons (main banks AND islands)
    land_polygons = []
    
    for node_id in G_rook.nodes():
        node_data = G_rook.nodes()[node_id]
        if 'bank_polygon' not in node_data:
            continue
            
        polygon = node_data['bank_polygon']
        
        if polygon is None:
            continue
        
        # Skip islands if not including them
        if node_id >= 2 and not include_islands:
            continue
            
        # Ensure it's a valid Polygon
        if isinstance(polygon, Polygon):
            if polygon.is_valid:
                land_polygons.append(polygon)
            else:
                # Try to fix invalid polygon
                fixed = polygon.buffer(0)
                if fixed.is_valid and not fixed.is_empty:
                    land_polygons.append(fixed)
        elif isinstance(polygon, MultiPolygon):
            for geom in polygon.geoms:
                if isinstance(geom, Polygon):
                    if geom.is_valid:
                        land_polygons.append(geom)
                    else:
                        fixed = geom.buffer(0)
                        if fixed.is_valid and not fixed.is_empty:
                            land_polygons.append(fixed)
        elif hasattr(polygon, 'geoms'):
            # Handle other geometry collections
            for geom in polygon.geoms:
                if isinstance(geom, Polygon) and geom.is_valid:
                    land_polygons.append(geom)
    
    if not land_polygons:
        print("Warning: No valid land polygons found in G_rook")
        # Return all water if no land polygons found
        return np.full((height, width), fill_value, dtype=np.uint8)
    
    # Create the water area by subtracting land from bounding box
    try:
        # Union all land polygons
        all_land = unary_union(land_polygons)
        
        # Apply buffer if specified
        if buffer_distance != 0:
            all_land = all_land.buffer(buffer_distance)
        
        # Water area = bounding box - land areas
        water_area = bbox.difference(all_land)
            
    except Exception as e:
        print(f"Warning: Error processing polygons: {e}")
        print("Attempting to fix geometries...")
        try:
            # Try to fix geometries
            fixed_polygons = [p.buffer(0) for p in land_polygons if p.buffer(0).is_valid]
            all_land = unary_union(fixed_polygons)
            water_area = bbox.difference(all_land)
        except Exception as e2:
            print(f"Error: Could not create water mask: {e2}")
            return np.full((height, width), nodata_value, dtype=np.uint8)
    
    # Rasterize the water area
    if water_area.is_empty:
        print("Warning: Water area is empty after polygon operations")
        water_mask = np.full((height, width), nodata_value, dtype=np.uint8)
    else:
        # Convert to list of geometries for rasterize
        if isinstance(water_area, Polygon):
            geometries = [(water_area, fill_value)]
        elif isinstance(water_area, MultiPolygon):
            geometries = [(geom, fill_value) for geom in water_area.geoms if geom.is_valid]
        elif hasattr(water_area, 'geoms'):
            geometries = [(geom, fill_value) for geom in water_area.geoms 
                         if isinstance(geom, Polygon) and geom.is_valid]
        else:
            print(f"Warning: Unexpected geometry type: {type(water_area)}")
            geometries = [(water_area, fill_value)]
        
        water_mask = rasterize(
            geometries,
            out_shape=(height, width),
            transform=transform,
            fill=nodata_value,
            dtype=np.uint8
        )
    
    # Save to file if output path is provided
    if output_path is not None:
        import rasterio
        from rasterio.crs import CRS
        
        # Ensure CRS is proper type
        if isinstance(crs, str):
            crs = CRS.from_string(crs)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=water_mask.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata_value
        ) as dst:
            dst.write(water_mask, 1)
        print(f"Water mask saved to: {output_path}")
    
    return water_mask


def create_water_mask_from_river(river, output_path=None, **kwargs):
    """
    Create a water mask from a River object.
    
    This is a convenience wrapper around create_water_mask_from_mapping
    that extracts the necessary data from a River object.
    
    Parameters
    ----------
    river : River
        A processed River object with G_rook and dataset attributes.
    output_path : str, optional
        Path to save the water mask as a GeoTIFF.
    **kwargs : dict
        Additional keyword arguments passed to create_water_mask_from_mapping.
    
    Returns
    -------
    water_mask : numpy.ndarray
        2D binary array where 1 indicates water and 0 indicates land.
    
    Examples
    --------
    >>> river = rb.River(fname=..., dirname=..., ...)
    >>> river.map_river_banks(...)
    >>> water_mask = rb.create_water_mask_from_river(river, output_path='mask.tif')
    """
    if not hasattr(river, '_G_rook') or river._G_rook is None:
        raise ValueError("River object has not been processed. Call map_river_banks() first.")
    
    if not hasattr(river, '_dataset') or river._dataset is None:
        raise ValueError("River object does not have dataset information.")
    
    return create_water_mask_from_mapping(
        river._G_rook, 
        river._dataset, 
        output_path=output_path,
        **kwargs
    ) 

def normalize_image(image, saturation_factor=0.5, percentile_stretch=True):
    """
    Normalize an image for display, optionally with reduced saturation.

    Parameters
    ----------
    image : numpy.ndarray
        Image array (height, width, bands).
    saturation_factor : float, optional
        Factor to reduce saturation (0.0 = grayscale, 1.0 = full saturation).
        Default 0.5.
    percentile_stretch : bool, optional
        Whether to apply a 2-98 percentile stretch per band for better
        contrast (default True).

    Returns
    -------
    numpy.ndarray
        Normalized image with values in [0, 1], ready for display.
    """
    image = image.astype(np.float32)
    image = np.nan_to_num(image, nan=0, posinf=0, neginf=0)

    for i in range(image.shape[2]):
        band = image[:, :, i]
        if percentile_stretch:
            valid = band[band > 0]
            if valid.size == 0:
                image[:, :, i] = 0.0
                continue
            p2, p98 = np.percentile(valid, [2, 98])
            if p98 > p2:
                image[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        else:
            min_val, max_val = band.min(), band.max()
            if max_val > min_val:
                image[:, :, i] = (band - min_val) / (max_val - min_val)

    # Reduce saturation by blending with grayscale
    if saturation_factor < 1.0:
        gray = np.mean(image, axis=2, keepdims=True)
        gray = np.repeat(gray, image.shape[2], axis=2)
        image = saturation_factor * image + (1 - saturation_factor) * gray

    return np.clip(image, 0, 1)


def crop_image_to_aoi(image_path, aoi_bounds, n_bands=3):
    """
    Read an image cropped to an area of interest.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to a georeferenced image file.
    aoi_bounds : list or tuple
        (left, right, bottom, top) in the image's coordinate system.
    n_bands : int, optional
        Number of bands to keep (default 3, for RGB display).

    Returns
    -------
    numpy.ndarray
        Cropped image array with shape (height, width, bands).
    """
    from rasterio.windows import from_bounds
    with rasterio.open(image_path) as src:
        left, right, bottom, top = aoi_bounds
        window = from_bounds(left, bottom, right, top, src.transform)
        cropped = src.read(window=window)
        if cropped.shape[0] > n_bands:
            cropped = cropped[:n_bands]
        return np.transpose(cropped, (1, 2, 0))


def prepare_image_stack(image_files, aoi_bounds, saturation_factor=0.5,
                        percentile_stretch=True):
    """
    Crop and normalize a list of images to a common area of interest.

    Parameters
    ----------
    image_files : list of str or pathlib.Path
        Image files (e.g., from :func:`match_rivers_to_images`).
    aoi_bounds : list or tuple
        (left, right, bottom, top) in the images' coordinate system.
    saturation_factor : float, optional
        Saturation reduction factor passed to :func:`normalize_image`
        (default 0.5).
    percentile_stretch : bool, optional
        Whether to percentile-stretch each band (default True).

    Returns
    -------
    list of numpy.ndarray or None
        One normalized (height, width, bands) array per input file, in the
        same order. Entries are None for images that failed to load, so the
        list stays aligned with the input (and with any matched river list).
    """
    image_stack = []
    for image_path in tqdm(image_files):
        try:
            cropped = crop_image_to_aoi(image_path, aoi_bounds)
            image_stack.append(normalize_image(cropped, saturation_factor,
                                               percentile_stretch))
        except Exception as e:
            print(f"Failed to crop {image_path}: {e}")
            image_stack.append(None)
    return image_stack
