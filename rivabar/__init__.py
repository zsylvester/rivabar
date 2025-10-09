"""
rivabar - A Python package to automatically extract channel centerlines and banklines from water index images of rivers
"""

# Rivabar - River Analysis and Centerline Extraction Package

# Import main function
from .core import map_river_banks, main
from .rivabar_legacy import extract_centerline

# Import River class for object-oriented API
from .river import River

# Import utility functions
from .utils import (
    convert_to_uint8, normalized_difference, get_cmap, find_condition,
    count_vertices, remove_endpoints, find_numbers_between,
    resample_and_smooth, compute_s_distance, find_closest_negative_minimum,
    visualize_curvature_circles, correlate_curves, visualize_dtw_correlations,
    resample_channel_width_to_centerline, get_width_and_curvature,
    compute_migration_distances, analyze_river_pairs_filtered
)

# Import geometry utilities
from .geometry_utils import (
    convert_to_utm, convert_geographic_proj_to_utm, closest_point_on_segment,
    closest_segment, angle_between, extract_coords, find_matching_indices,
    getExtrapolatedLine, extend_line, find_closest_point, find_longer_segment_coords,
    find_graph_edges_close_to_start_and_end_points, insert_node
)

# Import data I/O functions
from .data_io import (
    process_band, read_landsat_data, read_water_index, create_mndwi,
    save_shapefiles, crop_geotiff, read_and_plot_im,
    downsample_raster, save_planetscope_river_result, MinimalDataset
)

# Import polygon processing functions
from .polygon_processing import (
    smooth_polygon, smooth_line, vertex_density_tolerance, smooth_banklines,
    simplify_if_needed, create_channel_nw_polygon, straighten_channel,
    straighten_polygon, polygon_to_svg, plot_polygon, one_time_step, create_main_channel_banks
)

# Import graph processing functions
from .graph_processing import (
    remove_dead_ends, create_directed_multigraph, truncate_graph_by_polygon,
    set_width_weights, find_end_nodes, flip_coords_and_widths,
    find_distance_between_nodes_and_other_node, find_pixel_distance_between_nodes_and_other_node,
    extend_cline, traverse_multigraph, find_start_node, check_edges_around_islands,
    group_edges_to_subpaths, find_matching_subpaths, splice_paths, find_subpath
)

# Import analysis functions
from .analysis import (
    analyze_width_and_wavelength, compute_curvature, find_zero_crossings,
    get_bank_coords, compute_mndwi_small_dist, set_half_channel_widths,
    get_bank_coords_for_main_channel, get_channel_widths_along_path,
    get_all_channel_widths, get_channel_mouth_polygon, filter_outlier_paths,
    classify_confluences_and_splits
)
from .temporal_analysis import map_graphs_over_time, calculate_node_displacement_deviation

# Import temporal analysis functions
from .temporal_analysis import (
    calculate_iou, modified_iou, cluster_polygons, get_ch_and_bar_areas,
    create_and_plot_bars, create_geodataframe_from_bank_polygons,
    create_dataframe_from_bank_polygons, convert_to_landsat_crs,
    collect_river_endpoints, plot_deposition_erosion_with_dates
)

# Import visualization functions
from .visualization import (
    plot_im_and_lines, plot_graph_w_colors,
    plot_graph_mappings, plot_deviation_histogram
)

# Import additional I/O functions
from .data_io import (
    gdfs_from_D_primal, write_shapefiles_and_graphs, merge_and_plot_channel_polygons
)

__version__ = "1.0.0"
__author__ = "Zoltan Sylvester"
__email__ = "zoltan.sylvester@beg.utexas.edu"

__all__ = [
    # Core functions
    'map_river_banks', 'main', 'extract_centerline',
    
    # River class (object-oriented API)
    'River',
    
    # Utility functions
    'convert_to_uint8', 'normalized_difference', 'get_cmap', 'find_condition',
    'count_vertices', 'remove_endpoints', 'find_numbers_between',
    'resample_and_smooth', 'compute_s_distance', 'find_closest_negative_minimum',
    'visualize_curvature_circles', 'correlate_curves', 'visualize_dtw_correlations',
    'resample_channel_width_to_centerline', 'get_width_and_curvature',
    'compute_migration_distances', 'analyze_river_pairs_filtered',
    
    # Geometry utilities
    'convert_to_utm', 'convert_geographic_proj_to_utm', 'closest_point_on_segment',
    'closest_segment', 'angle_between', 'extract_coords', 'find_matching_indices',
    'getExtrapolatedLine', 'extend_line', 'find_closest_point', 'find_longer_segment_coords',
    'find_graph_edges_close_to_start_and_end_points', 'insert_node',
    
    # Data I/O functions
    'process_band', 'read_landsat_data', 'read_water_index', 'create_mndwi',
    'save_shapefiles', 'crop_geotiff', 'read_and_plot_im', 'gdfs_from_D_primal',
    'write_shapefiles_and_graphs', 'merge_and_plot_channel_polygons', 
    'downsample_raster', 'save_planetscope_river_result', 'MinimalDataset',
    
    # Polygon processing
    'smooth_polygon', 'smooth_line', 'vertex_density_tolerance', 'smooth_banklines',
    'simplify_if_needed', 'create_channel_nw_polygon', 'straighten_channel',
    'straighten_polygon', 'polygon_to_svg', 'one_time_step', 'create_main_channel_banks',
    
    # Graph processing
    'remove_dead_ends', 'create_directed_multigraph', 'truncate_graph_by_polygon',
    'set_width_weights', 'find_end_nodes', 'flip_coords_and_widths',
    'find_distance_between_nodes_and_other_node', 'find_pixel_distance_between_nodes_and_other_node',
    'extend_cline', 'traverse_multigraph', 'find_start_node', 'check_edges_around_islands',
    'group_edges_to_subpaths', 'find_matching_subpaths', 'splice_paths', 'find_subpath',
    
    # Analysis functions
    'analyze_width_and_wavelength', 'compute_curvature', 'find_zero_crossings',
    'get_bank_coords', 'compute_mndwi_small_dist', 'set_half_channel_widths',
    'get_bank_coords_for_main_channel', 'get_channel_widths_along_path',
    'get_all_channel_widths', 'get_channel_mouth_polygon', 'filter_outlier_paths',
    'classify_confluences_and_splits',
    
    # Temporal analysis
    'calculate_iou', 'modified_iou', 'cluster_polygons', 'get_ch_and_bar_areas',
    'create_and_plot_bars', 'create_geodataframe_from_bank_polygons',
    'create_dataframe_from_bank_polygons', 'convert_to_landsat_crs',
    'collect_river_endpoints', 'plot_deposition_erosion_with_dates',
    
    # Visualization
    'plot_im_and_lines', 'plot_graph_w_colors',
    'plot_graph_mappings', 'plot_deviation_histogram'
] 