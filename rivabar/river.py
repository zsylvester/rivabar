import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import warnings
import os
import datetime

from .core import map_river_banks
from .analysis import get_channel_widths_along_path, analyze_width_and_wavelength
from .visualization import plot_im_and_lines, plot_graph_w_colors
from .data_io import create_mndwi


class River:
    """
    A River class that provides a clean, object-oriented interface to rivabar functionality.
    
    This class wraps the existing functional API without breaking any current functionality.
    All existing functions remain available and unchanged.
    """
    
    def __init__(self, fname=None, dirname=None, start_x=None, start_y=None, 
                 end_x=None, end_y=None, file_type=None, **kwargs):
        """
        Initialize a River object.
        
        Parameters
        ----------
        fname : str, optional
            Image filename
        dirname : str, optional  
            Directory containing the image
        start_x, start_y : float, optional
            Starting point coordinates (UTM)
        end_x, end_y : float, optional
            Ending point coordinates (UTM)
        file_type : str, optional
            Type of input file ('single_tif', 'multiple_tifs', etc.)
        **kwargs : dict
            Additional parameters for map_river_banks function
        """
        
        # Store initialization parameters
        self.fname = fname
        self.dirname = dirname
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.file_type = file_type
        self.kwargs = kwargs
        
        # Graph storage - will be populated after processing
        self._D_primal = None
        self._G_rook = None  
        self._G_primal = None
        self._mndwi = None
        self._dataset = None
        
        # Coordinate bounds
        self._left_utm_x = None
        self._right_utm_x = None
        self._lower_utm_y = None
        self._upper_utm_y = None
        self._xs = None
        self._ys = None
        
        # Processing flags
        self._is_processed = False
        self._processing_successful = False
        
        # Add new attributes
        self._mndwi_created = False
        self._delta_x = None
        self._delta_y = None
    
    def map_river_banks(self, **override_kwargs):
        """
        Extract river centerlines and banklines using the existing map_river_banks function.
        
        Parameters
        ----------
        **override_kwargs : dict
            Parameters to override the initialization kwargs
            
        Returns
        -------
        bool
            True if processing was successful, False otherwise
        """
        
        if not self._validate_inputs():
            raise ValueError("Missing required parameters. Need fname, dirname, start/end coordinates, and file_type.")
        
        # Merge initialization kwargs with any overrides
        processing_kwargs = {**self.kwargs, **override_kwargs}
        
        print(f"Processing river: {self.fname}")
        print(f"Start point: ({self.start_x}, {self.start_y})")
        print(f"End point: ({self.end_x}, {self.end_y})")
        
        try:
            # Call the existing map_river_banks function
            result = map_river_banks(
                fname=self.fname,
                dirname=self.dirname,
                start_x=self.start_x,
                start_y=self.start_y, 
                end_x=self.end_x,
                end_y=self.end_y,
                file_type=self.file_type,
                **processing_kwargs
            )
            
            # Unpack results
            (self._D_primal, self._G_rook, self._G_primal, self._mndwi, self._dataset,
             self._left_utm_x, self._right_utm_x, self._lower_utm_y, self._upper_utm_y, 
             self._xs, self._ys) = result
            
            # Check if processing was successful
            if self._D_primal is not None:
                self._processing_successful = True
                self._is_processed = True
                print(f"✓ Successfully processed {self.fname}")
                print(f"  - Directed graph: {len(self._D_primal.nodes)} nodes, {len(self._D_primal.edges)} edges")
                print(f"  - Rook graph: {len(self._G_rook.nodes)} polygons")
                print(f"  - Primal graph: {len(self._G_primal.nodes)} nodes")
                return True
            else:
                self._processing_successful = False
                self._is_processed = True
                print(f"✗ Failed to process {self.fname}")
                return False
                
        except Exception as e:
            import traceback
            import sys
            
            print(f"✗ Error processing {self.fname}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Processing parameters:")
            print(f"  - File type: {self.file_type}")
            print(f"  - Start point: ({self.start_x}, {self.start_y})")
            print(f"  - End point: ({self.end_x}, {self.end_y})")
            print(f"  - Directory: {self.dirname}")
            print("Full traceback:")
            traceback.print_exc()
            
            self._processing_successful = False
            self._is_processed = True
            return False
    
    def _validate_inputs(self):
        """Validate that all required inputs are provided."""
        required = [self.fname, self.dirname, self.file_type]
        return all(param is not None for param in required)
    
    def _check_processed(self):
        """Check if the river has been successfully processed."""
        if not self._is_processed:
            raise RuntimeError("River has not been processed yet. Call map_river_banks() first.")
        if not self._processing_successful:
            raise RuntimeError("River processing failed. Cannot access results.")
    
    # Properties to access the graphs (read-only)
    @property
    def directed_graph(self):
        """Get the directed centerline graph (D_primal)."""
        self._check_processed()
        return self._D_primal
    
    @property
    def centerline_graph(self):
        """Get the primal centerline graph (G_primal).""" 
        self._check_processed()
        return self._G_primal
    
    @property
    def bankline_graph(self):
        """Get the bankline (rook) graph (G_rook)."""
        self._check_processed()
        return self._G_rook
    
    @property
    def mndwi(self):
        """Get the MNDWI water mask."""
        self._check_processed()
        return self._mndwi
    
    @property
    def dataset(self):
        """Get the raster dataset."""
        self._check_processed()
        return self._dataset
    
    @property
    def bounds(self):
        """Get the UTM coordinate bounds as a dict."""
        self._check_processed()
        return {
            'left': self._left_utm_x,
            'right': self._right_utm_x, 
            'lower': self._lower_utm_y,
            'upper': self._upper_utm_y
        }
    
    @property
    def centerline_coords(self):
        """Get the main centerline coordinates."""
        self._check_processed()
        return np.column_stack([self._xs, self._ys])
    
    @property
    def main_path(self):
        """Get the main path as a list of edge tuples."""
        self._check_processed()
        if 'main_path' in self._D_primal.graph:  # Check for dictionary key
            return self._D_primal.graph['main_path']
        return None
    
    @property
    def main_channel_centerline(self):
        """Get main channel centerline coordinates as a LineString."""
        self._check_processed()
        coords = self._D_primal.graph.get('main_channel_cl_coords', None)
        if coords is not None:
            return LineString(coords)
        return None
    
    @property
    def main_channel_banks(self):
        """Get main channel bank coordinates as LineStrings."""
        self._check_processed()
        bank1_coords = self._D_primal.graph.get('main_channel_bank1_coords', None)
        bank2_coords = self._D_primal.graph.get('main_channel_bank2_coords', None)
        
        if bank1_coords is not None and bank2_coords is not None:
            return {
                'left_bank': LineString(bank1_coords),
                'right_bank': LineString(bank2_coords)
            }
        return None
    
    # Analysis methods
    def get_channel_widths(self, path=None):
        """
        Get channel widths along the main path.
        
        Parameters
        ----------
        path : list, optional
            Custom path as list of edge tuples. If None, uses main path.
            
        Returns
        -------
        np.ndarray
            Channel widths along the path
        """
        self._check_processed()
        
        if path is None:
            path = self.main_path
        if path is None:
            raise ValueError("No main path available and no custom path provided.")
            
        xl, yl, w1l, w2l, w, s = get_channel_widths_along_path(self._D_primal, path)

        return s, np.array(w)*self._dataset.transform[0] # convert to meters

    
    def analyze_wavelength_and_width(self, path=None, ax=None, **kwargs):
        """
        Analyze channel wavelength and width statistics.
        
        Parameters
        ----------
        path : list, optional
            Custom path as list of edge tuples. If None, uses main path.
        ax : matplotlib.Axes, optional
            Axes to plot on. If None, creates new figure.
            
        Returns
        -------
        dict
            Dictionary with analysis results
        """
        self._check_processed()
        
        if path is None:
            path = self.main_path
        if path is None:
            raise ValueError("No main path available and no custom path provided.")
            
        return analyze_width_and_wavelength(self._D_primal, path, ax, **kwargs)
    
    # Visualization methods
    def plot_overview(self, figsize=(12, 8), **kwargs):
        """
        Plot an overview of the river with centerlines and banks.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height)
        **kwargs : dict
            Additional arguments passed to plot_im_and_lines
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects
        """
        self._check_processed()
        
        fig, ax = plot_im_and_lines(
            self._mndwi, 
            self._left_utm_x, self._right_utm_x, 
            self._lower_utm_y, self._upper_utm_y,
            self._G_rook, self._G_primal, self._D_primal,
            start_x=self.start_x, start_y=self.start_y,
            end_x=self.end_x, end_y=self.end_y,
            **kwargs
        )
        
        ax.set_title(f'River: {self.fname}')
        fig.set_size_inches(figsize)
        
        return fig, ax
    
    def plot_directed_graph(self, ax=None, **kwargs):
        """
        Plot the directed centerline graph with colors.
        
        Parameters
        ----------
        ax : matplotlib.Axes, optional
            Axes to plot on. If None, uses current axes.
        **kwargs : dict
            Additional arguments passed to plot_graph_w_colors
            
        Returns
        -------
        ax
            Matplotlib axes object
        """
        self._check_processed()
        
        if ax is None:
            ax = plt.gca()
            
        return plot_graph_w_colors(self._D_primal, ax, **kwargs)
    
    def create_mndwi(self, mndwi_threshold=0.01, delete_pixels_polys=False, 
                     small_hole_threshold=64, remove_smaller_components=True, 
                     solidity_filter=False, **kwargs):
        """
        Create MNDWI water mask from the river's input data.
        
        This allows you to create and visualize the water mask before running
        the full centerline extraction process.
        
        Parameters
        ----------
        mndwi_threshold : float, optional
            Threshold for water detection. Default 0.01.
        delete_pixels_polys : bool or list, optional
            Polygons to mask out (e.g., bridges). Default False.
        small_hole_threshold : int, optional
            Minimum hole size to keep. Default 64.
        remove_smaller_components : bool, optional
            Remove small water bodies. Default True.
        solidity_filter : bool, optional
            Filter by solidity. Default False.
        **kwargs : dict
            Additional parameters for create_mndwi
            
        Returns
        -------
        numpy.ndarray
            Binary water mask
        """
        
        if not self._validate_inputs():
            raise ValueError("Missing required parameters. Need fname, dirname, and file_type.")
        
        print(f"Creating MNDWI for: {self.fname}")
        
        # Call the create_mndwi function
        result = create_mndwi(
            dirname=self.dirname,
            fname=self.fname,
            file_type=self.file_type,
            mndwi_threshold=mndwi_threshold,
            delete_pixels_polys=delete_pixels_polys,
            small_hole_threshold=small_hole_threshold,
            remove_smaller_components=remove_smaller_components,
            solidity_filter=solidity_filter,
            **kwargs
        )
        
        # Unpack and store results
        (self._mndwi, self._left_utm_x, self._upper_utm_y, self._right_utm_x, 
         self._lower_utm_y, self._delta_x, self._delta_y, self._dataset) = result
        
        # Update processing flags
        self._mndwi_created = True
        
        print(f"✓ MNDWI created: {self._mndwi.shape}")
        print(f"  - Water pixels: {np.sum(self._mndwi):,}")
        print(f"  - Total pixels: {self._mndwi.size:,}")
        print(f"  - Water percentage: {100*np.sum(self._mndwi)/self._mndwi.size:.1f}%")
        
        return self._mndwi
    
    def plot_mndwi(self, figsize=(12, 8), add_start_end_points=True, **kwargs):
        """
        Plot the MNDWI water mask.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default (12, 8).
        add_start_end_points : bool, optional
            Whether to plot start/end points. Default True.
        **kwargs : dict
            Additional arguments for plt.imshow
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axes
        """
        
        if self._mndwi is None:
            raise RuntimeError("MNDWI not created yet. Call create_mndwi() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot MNDWI
        im = ax.imshow(
            self._mndwi, 
            extent=[self._left_utm_x, self._right_utm_x, 
                   self._lower_utm_y, self._upper_utm_y],
            cmap='gray_r',
            **kwargs
        )
        
        # Add start/end points if provided
        if add_start_end_points and all([self.start_x, self.start_y, self.end_x, self.end_y]):
            ax.plot(self.start_x, self.start_y, 'go', markersize=8, label='Start')
            ax.plot(self.end_x, self.end_y, 'ro', markersize=8, label='End')
            ax.legend()
        
        ax.set_xlabel('UTM X (m)')
        ax.set_ylabel('UTM Y (m)')
        ax.set_title(f'MNDWI Water Mask: {self.fname}')
        
        return fig, ax
    
    def get_start_end_points_interactive(self):
        """
        Interactively select start and end points on the MNDWI plot.
        
        Returns
        -------
        tuple
            (start_x, start_y, end_x, end_y)
        """
        
        if self._mndwi is None:
            raise RuntimeError("MNDWI not created yet. Call create_mndwi() first.")
        
        fig, ax = self.plot_mndwi(add_start_end_points=False)
        
        print("Click two points: first for START, second for END")
        points = plt.ginput(n=2, timeout=-1)  # Set timeout to -1 to prevent timing out
        
        if len(points) == 2:
            self.start_x, self.start_y = points[0]
            self.end_x, self.end_y = points[1]
            
            # Add points to plot
            ax.plot(self.start_x, self.start_y, 'go', markersize=8, label='Start')
            ax.plot(self.end_x, self.end_y, 'ro', markersize=8, label='End')
            ax.legend()
            plt.draw()
            
            print(f"Start point: ({self.start_x:.1f}, {self.start_y:.1f})")
            print(f"End point: ({self.end_x:.1f}, {self.end_y:.1f})")
            
            return self.start_x, self.start_y, self.end_x, self.end_y
        else:
            print("Need exactly 2 points!")
            return None
    
    # Utility methods
    def summary(self):
        """Print a summary of the river analysis."""
        print(f"\n=== River Summary: {self.fname} ===")
        print(f"Processed: {self._is_processed}")
        print(f"Successful: {self._processing_successful}")
        
        if self._processing_successful:
            print(f"Start point: ({self.start_x:.1f}, {self.start_y:.1f})")
            print(f"End point: ({self.end_x:.1f}, {self.end_y:.1f})")
            print(f"Directed graph: {len(self._D_primal.nodes)} nodes, {len(self._D_primal.edges)} edges")
            print(f"Rook graph: {len(self._G_rook.nodes)} polygons") 
            print(f"Primal graph: {len(self._G_primal.nodes)} nodes")
            print(f"MNDWI shape: {self._mndwi.shape}")
            
            # Main path info
            if self.main_path:
                print(f"Main path: {len(self.main_path)} edges")
                s, widths = self.get_channel_widths()
                print(f"Channel width: {np.mean(widths):.1f} ± {np.std(widths):.1f} m")
                print(f"Channel length: {s[-1]:.1f} m")
            
        print("=" * (20 + len(self.fname)))

    def clear_raster_data(self):
        """
        Clear heavy raster data to save memory while keeping graph results.
        
        This removes the MNDWI array but preserves all graph data,
        coordinates, and analysis results.
        """
        # Clear heavy raster data
        self._mndwi = None
        
        # Keep everything else: graphs, coordinates, bounds, etc.
        print(f"Cleared raster data for {self.fname}")
    
    def to_geopandas(self):
        """
        Export river data to GeoDataFrames.
        
        Returns
        -------
        dict
            Dictionary containing GeoDataFrames for different components
        """
        self._check_processed()
        
        result = {}
        
        # Main centerline
        if self.main_channel_centerline:
            result['centerline'] = gpd.GeoDataFrame(
                [{'geometry': self.main_channel_centerline, 'name': self.fname}],
                crs=self._dataset.crs
            )
        
        # Banks
        banks = self.main_channel_banks
        if banks:
            bank_data = [
                {'geometry': banks['left_bank'], 'side': 'left', 'name': self.fname},
                {'geometry': banks['right_bank'], 'side': 'right', 'name': self.fname}
            ]
            result['banks'] = gpd.GeoDataFrame(bank_data, crs=self._dataset.crs)
        
        # Channel polygons (from rook graph)
        polygons = []
        for node_id, data in self._G_rook.nodes(data=True):
            if 'bank_polygon' in data and data['bank_polygon'] is not None:
                polygons.append({
                    'geometry': data['bank_polygon'],
                    'node_id': node_id,
                    'name': self.fname
                })
        
        if polygons:
            result['polygons'] = gpd.GeoDataFrame(polygons, crs=self._dataset.crs)
        
        return result
    
    # Add these methods to the River class

    def save_results(self, filepath):
        """Save river analysis results to file."""
        import pickle
        
        # Create a results dictionary without heavy raster data
        results = {
            'fname': self.fname,
            'dirname': self.dirname,
            'start_x': self.start_x, 'start_y': self.start_y,
            'end_x': self.end_x, 'end_y': self.end_y,
            'file_type': self.file_type,
            'kwargs': self.kwargs,
            
            # Graph data (these are the important results)
            'D_primal': self._D_primal,
            'G_rook': self._G_rook,
            'G_primal': self._G_primal,
            
            # Coordinate data
            'left_utm_x': self._left_utm_x,
            'right_utm_x': self._right_utm_x,
            'lower_utm_y': self._lower_utm_y,
            'upper_utm_y': self._upper_utm_y,
            'xs': self._xs,
            'ys': self._ys,
            
            # Processing flags
            'processing_successful': self._processing_successful,
            'is_processed': self._is_processed
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Saved results to {filepath}")

    @classmethod
    def load_results(cls, filepath):
        """Load river results from file and create a River instance."""
        import pickle
        
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        # Create river instance
        river = cls(
            fname=results['fname'],
            dirname=results['dirname'],
            start_x=results['start_x'],
            start_y=results['start_y'],
            end_x=results['end_x'],
            end_y=results['end_y'],
            file_type=results['file_type'],
            **results['kwargs']
        )
        
        # Restore processed data
        river._D_primal = results['D_primal']
        river._G_rook = results['G_rook']
        river._G_primal = results['G_primal']
        river._left_utm_x = results['left_utm_x']
        river._right_utm_x = results['right_utm_x']
        river._lower_utm_y = results['lower_utm_y']
        river._upper_utm_y = results['upper_utm_y']
        river._xs = results['xs']
        river._ys = results['ys']
        river._processing_successful = results['processing_successful']
        river._is_processed = results['is_processed']
        
        # MNDWI and dataset are not loaded (saving memory)
        river._mndwi = None
        river._dataset = None
        
        print(f"Loaded results from {filepath}")
        return river
    
    def __repr__(self):
        """String representation of the River object."""
        status = "✓ Processed" if self._processing_successful else ("✗ Failed" if self._is_processed else "○ Not processed")
        return f"River('{self.fname}', {status})"
    
    # Add property to check if MNDWI exists
    @property
    def has_mndwi(self):
        """Check if MNDWI has been created."""
        return self._mndwi is not None

    def get_memory_usage(self):
        """Get approximate memory usage of stored data."""
        import sys
        
        memory_mb = 0
        if self._mndwi is not None:
            memory_mb += self._mndwi.nbytes / 1024**2
        if self._dataset is not None:
            # Rough estimate for dataset
            memory_mb += 50  # Approximate
        
        return f"{memory_mb:.1f} MB"
    
    def get_memory_breakdown(self):
        """Get detailed memory usage breakdown."""
        breakdown = {}
        total_mb = 0
        
        if self._mndwi is not None:
            mndwi_mb = self._mndwi.nbytes / 1024**2
            breakdown['MNDWI'] = f"{mndwi_mb:.1f} MB"
            total_mb += mndwi_mb
        else:
            breakdown['MNDWI'] = "Not loaded"
        
        if self._dataset is not None:
            dataset_mb = 50  # Rough estimate
            breakdown['Dataset'] = f"{dataset_mb:.1f} MB"
            total_mb += dataset_mb
        else:
            breakdown['Dataset'] = "Not loaded"
        
        # Graph data is typically small
        if self._D_primal is not None:
            breakdown['Graphs'] = "~1-5 MB"
            total_mb += 3  # Rough estimate
        else:
            breakdown['Graphs'] = "Not processed"
        
        breakdown['Total'] = f"{total_mb:.1f} MB"
        return breakdown

    @classmethod
    def batch_process_landsat_scenes(cls, path_number, row_number, start_x, start_y, end_x, end_y,
                                   years=None, max_cloud_cover=10, n_scenes_per_year=3, 
                                   clear_rasters=True, save_individual=True, save_dir='river_results',
                                   skip_scenes=None, download_mndwi=False, mndwi_dir='mndwi_rasters',
                                   download_false_color=False, false_color_dir='false_color_images',
                                   **processing_kwargs):
        """
        Batch process multiple Landsat scenes from Google Earth Engine.
        
        Parameters
        ----------
        path_number : int
            Landsat WRS path number
        row_number : int
            Landsat WRS row number  
        start_x, start_y : float
            Starting point coordinates (UTM)
        end_x, end_y : float
            Ending point coordinates (UTM)
        years : list or range, optional
            Years to process. If None, uses 1984-2023
        max_cloud_cover : float, optional
            Maximum cloud cover percentage (default 10)
        n_scenes_per_year : int, optional
            Maximum scenes per year (default 3)
        clear_rasters : bool, optional
            Whether to clear raster data after processing (default True)
        save_individual : bool, optional
            Whether to save each river individually to prevent data loss (default True)
        save_dir : str, optional
            Directory to save individual river files (default 'river_results')
        skip_scenes : list, optional
            List of scene IDs to skip (useful for avoiding problematic scenes that crash the kernel)
        download_mndwi : bool, optional
            Whether to download MNDWI rasters to local folder (default False)
        mndwi_dir : str, optional
            Directory to save MNDWI rasters (default 'mndwi_rasters')
        download_false_color : bool, optional
            Whether to download false color images (SWIR2-NIR-Green) to local folder (default False)
        false_color_dir : str, optional
            Directory to save false color images (default 'false_color_images')
        **processing_kwargs : dict
            Additional arguments for map_river_banks
            
        Returns
        -------
        successful_rivers : list
            List of successfully processed River instances (or file paths if save_individual=True)
        failed_scenes : list
            List of scene IDs that failed processing
        """
        import ee
        import tempfile
        import os
        import gc
        import geemap
        import pickle
        from datetime import datetime
        
        if years is None:
            years = range(1984, 2024)
        
        if skip_scenes is None:
            skip_scenes = []
        
        # Create save directory if saving individually
        if save_individual:
            os.makedirs(save_dir, exist_ok=True)
            print(f"💾 Individual rivers will be saved to: {save_dir}")
        
        # Create MNDWI directory if downloading rasters
        if download_mndwi:
            os.makedirs(mndwi_dir, exist_ok=True)
            print(f"🗺️  MNDWI rasters will be saved to: {mndwi_dir}")
        
        # Create false color directory if downloading images
        if download_false_color:
            os.makedirs(false_color_dir, exist_ok=True)
            print(f"🌈 False color images will be saved to: {false_color_dir}")
        
        if skip_scenes:
            print(f"⏭️  Will skip {len(skip_scenes)} problematic scenes: {skip_scenes}")
        
        successful_rivers = []
        failed_scenes = []
        skipped_scenes = []
        scene_count = 0
        
        print(f"🛰️  Batch processing Landsat scenes for Path {path_number}, Row {row_number}")
        print(f"Years: {min(years)}-{max(years)}, Max cloud cover: {max_cloud_cover}%")
        print(f"Processing parameters: {processing_kwargs}")
        
        # Process each year
        for year in years:
            start_date = f'{year}-01-01'
            end_date = f'{year}-12-31'
            
            print(f"\n--- Processing year {year} ---")
            
            # Determine which Landsat mission to use
            if year < 2013:
                # Landsat 5
                collection_id = "LANDSAT/LT05/C02/T1_L2"
                green_band = 'SR_B2'
                nir_band = 'SR_B4'
                swir_band = 'SR_B5'
                swir2_band = 'SR_B7'
            elif year < 2021:
                # Landsat 8
                collection_id = "LANDSAT/LC08/C02/T1_L2" 
                green_band = 'SR_B3'
                nir_band = 'SR_B5'
                swir_band = 'SR_B6'
                swir2_band = 'SR_B7'
            else:
                # Landsat 9
                collection_id = "LANDSAT/LC09/C02/T1_L2"
                green_band = 'SR_B3'
                nir_band = 'SR_B5'
                swir_band = 'SR_B6'
                swir2_band = 'SR_B7'
            
            # Get image collection
            try:
                collection = (ee.ImageCollection(collection_id)
                            .filterDate(start_date, end_date)
                            .filter(ee.Filter.eq('WRS_PATH', path_number))
                            .filter(ee.Filter.eq('WRS_ROW', row_number))
                            .filter(ee.Filter.lt("CLOUD_COVER", max_cloud_cover))
                            .sort('CLOUD_COVER', True)
                            .limit(n_scenes_per_year))
                
                imgs = collection.toList(collection.size())
                n_images = collection.size().getInfo()
                
                print(f"Found {n_images} scenes for {year}")
                
                # Process each image
                for i in range(n_images):
                    scene_count += 1
                    
                    try:
                        im = ee.Image(imgs.get(i))
                        scene_id = im.getString('system:index').getInfo()
                        acquisition_date = im.getString('DATE_ACQUIRED').getInfo()
                        cloud_cover = im.getNumber('CLOUD_COVER').getInfo()
                        
                        print(f"  Processing scene {scene_count}: {scene_id}")
                        print(f"    Date: {acquisition_date}, Cloud cover: {cloud_cover:.1f}%")
                        
                        # Check if this scene should be skipped
                        if scene_id in skip_scenes:
                            print(f"    ⏭️  Skipping {scene_id} (in skip list)")
                            skipped_scenes.append(scene_id)
                            continue
                        
                        # Check if this scene was already processed (if saving individually)
                        if save_individual:
                            river_file = os.path.join(save_dir, f"river_{scene_id}.pkl")
                            if os.path.exists(river_file):
                                print(f"    ⏭️  Skipping {scene_id} (already processed)")
                                successful_rivers.append(river_file)
                                continue
                        
                        # Calculate MNDWI
                        mndwi = im.normalizedDifference([green_band, swir_band])
                        
                        # Download to temporary file
                        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                        
                        try:
                            # Download the MNDWI image
                            print(f"    Downloading MNDWI...")
                            geemap.download_ee_image(mndwi, tmp_path)
                            
                            # Save MNDWI raster to permanent location if requested
                            if download_mndwi:
                                mndwi_filename = f"mndwi_{scene_id}.tif"
                                mndwi_path = os.path.join(mndwi_dir, mndwi_filename)
                                
                                # Check if MNDWI file already exists
                                if not os.path.exists(mndwi_path):
                                    import shutil
                                    shutil.copy2(tmp_path, mndwi_path)
                                    print(f"    🗺️  Saved MNDWI: {mndwi_path}")
                                else:
                                    print(f"    🗺️  MNDWI already exists: {mndwi_path}")
                            
                            # Create and save false color image if requested
                            if download_false_color:
                                false_color_filename = f"false_color_{scene_id}.tif"
                                false_color_path = os.path.join(false_color_dir, false_color_filename)
                                
                                # Check if false color image already exists
                                if not os.path.exists(false_color_path):
                                    print(f"    Creating false color image...")
                                    
                                    # Create false color composite: SWIR2, NIR, Green
                                    false_color = im.select([swir2_band, nir_band, green_band])
                                    
                                    # Create temporary file for false color image
                                    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_fc_file:
                                        tmp_fc_path = tmp_fc_file.name
                                    
                                    try:
                                        print(f"    Downloading false color image...")
                                        geemap.download_ee_image(false_color, tmp_fc_path)
                                        
                                        import shutil
                                        shutil.copy2(tmp_fc_path, false_color_path)
                                        print(f"    🌈 Saved false color: {false_color_path}")
                                        
                                        # Clean up temporary false color file
                                        os.unlink(tmp_fc_path)
                                        
                                    except Exception as e:
                                        print(f"    ❌ Failed to download false color image: {e}")
                                        if os.path.exists(tmp_fc_path):
                                            os.unlink(tmp_fc_path)
                                else:
                                    print(f"    🌈 False color already exists: {false_color_path}")
                            
                            # Create River instance and process
                            print(f"    Processing with rivabar...")
                            
                            # Filter out batch-specific parameters that shouldn't go to map_river_banks
                            filtered_kwargs = {k: v for k, v in processing_kwargs.items() 
                                             if k not in ['skip_scenes']}
                            
                            river = cls(
                                fname=os.path.basename(tmp_path),
                                dirname=os.path.dirname(tmp_path) + '/',
                                start_x=start_x,
                                start_y=start_y,
                                end_x=end_x,
                                end_y=end_y,
                                file_type='water_index',
                                **filtered_kwargs
                            )
                            
                            # Add metadata
                            river.scene_id = scene_id
                            river.acquisition_date = acquisition_date
                            river.cloud_cover = cloud_cover
                            river.landsat_mission = collection_id.split('/')[1]
                            river.year = year
                            
                            # Process the river
                            memory_before = river.get_memory_usage()
                            success = river.map_river_banks()
                            
                            if success:
                                print(f"    ✅ Success! Memory: {memory_before} → {river.get_memory_usage()}")
                                
                                # Clear raster data to save memory
                                if clear_rasters:
                                    river.clear_raster_data()
                                    print(f"    🧹 Cleared rasters, memory: {river.get_memory_usage()}")
                                
                                # Save individual river if requested
                                if save_individual:
                                    river_file = os.path.join(save_dir, f"river_{scene_id}.pkl")
                                    
                                    # Create a safe copy of the river for pickling
                                    # Store the dataset temporarily and remove it
                                    temp_dataset = river._dataset
                                    river._dataset = None
                                    
                                    # Create a results dictionary for saving
                                    river_data = {
                                        'river': river,
                                        'scene_id': scene_id,
                                        'acquisition_date': acquisition_date,
                                        'cloud_cover': cloud_cover,
                                        'landsat_mission': collection_id.split('/')[1],
                                        'year': year,
                                        'processing_timestamp': datetime.now(),
                                        'path_number': path_number,
                                        'row_number': row_number,
                                        # Store dataset info separately
                                        'dataset_crs': str(temp_dataset.crs) if temp_dataset else None,
                                        'dataset_transform': temp_dataset.transform if temp_dataset else None,
                                        'dataset_shape': temp_dataset.shape if temp_dataset else None
                                    }
                                    
                                    try:
                                        with open(river_file, 'wb') as f:
                                            pickle.dump(river_data, f)
                                        print(f"    💾 Saved: {river_file}")
                                        successful_rivers.append(river_file)  # Store file path
                                    except Exception as pickle_error:
                                        print(f"    ⚠️ Error saving {river_file}: {pickle_error}")
                                        # If saving fails, still add to successful_rivers list
                                        successful_rivers.append(river)
                                    finally:
                                        # Restore the dataset reference
                                        river._dataset = temp_dataset
                                else:
                                    successful_rivers.append(river)  # Store river object
                                
                                print(f"    📊 Collected river (total: {len(successful_rivers)})")
                            else:
                                print(f"    ❌ Processing failed")
                                failed_scenes.append(scene_id)
                                
                        finally:
                            # Clean up temporary file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                        
                    except Exception as e:
                        print(f"    ❌ Error processing scene: {str(e)}")
                        failed_scenes.append(f"{year}_{i}")
                    
                    # Force garbage collection
                    gc.collect()
                    
            except Exception as e:
                print(f"❌ Error accessing collection for {year}: {str(e)}")
                continue
        
        print(f"\n🎉 Batch processing complete!")
        print(f"✅ Successful: {len(successful_rivers)} rivers")
        print(f"❌ Failed: {len(failed_scenes)} scenes")
        print(f"⏭️  Skipped: {len(skipped_scenes)} scenes")
        
        if failed_scenes:
            print(f"Failed scenes: {failed_scenes}")
        
        if skipped_scenes:
            print(f"Skipped scenes: {skipped_scenes}")
        
        if save_individual:
            print(f"💾 Individual river files saved in: {save_dir}")
            print(f"📁 To load all rivers: rivers = River.load_batch_results('{save_dir}')")
        
        if download_mndwi:
            print(f"🗺️  MNDWI rasters saved in: {mndwi_dir}")
        
        return successful_rivers, failed_scenes

    @classmethod
    def load_batch_results(cls, save_dir, min_file_size_kb=0):
        """
        Load all saved rivers from a batch processing directory.
        
        Parameters
        ----------
        save_dir : str
            Directory containing saved river files
        min_file_size_kb : float, optional
            Minimum file size in kilobytes to load (default 0, loads all files)
            
        Returns
        -------
        list
            List of River instances loaded from files
        """
        import pickle
        import os
        import glob
        
        river_files = glob.glob(os.path.join(save_dir, "river_*.pkl"))
        rivers = []
        skipped_files = 0
        
        print(f"📂 Found {len(river_files)} river files in {save_dir}")
        if min_file_size_kb > 0:
            print(f"🔍 Only loading files larger than {min_file_size_kb} KB")
        
        for river_file in sorted(river_files):
            # Check file size if minimum size is specified
            if min_file_size_kb > 0:
                file_size_kb = os.path.getsize(river_file) / 1024  # Convert bytes to KB
                if file_size_kb < min_file_size_kb:
                    skipped_files += 1
                    continue
            
            try:
                with open(river_file, 'rb') as f:
                    river_data = pickle.load(f)
                
                # Extract river object and restore metadata
                if isinstance(river_data, dict) and 'river' in river_data:
                    river = river_data['river']
                    # Restore metadata attributes
                    river.scene_id = river_data.get('scene_id', 'unknown')
                    river.acquisition_date = river_data.get('acquisition_date', 'unknown')
                    river.cloud_cover = river_data.get('cloud_cover', 0)
                    river.landsat_mission = river_data.get('landsat_mission', 'unknown')
                    river.year = river_data.get('year', 0)
                    
                    # Create a minimal dataset-like object for CRS info if available
                    if river_data.get('dataset_crs') and river._dataset is None:
                        # Create a simple object to hold CRS info for to_geopandas()
                        class MinimalDataset:
                            def __init__(self, crs_str, transform=None, shape=None):
                                from rasterio.crs import CRS
                                self.crs = CRS.from_string(crs_str) if crs_str else None
                                self.transform = transform
                                self.shape = shape
                        
                        river._dataset = MinimalDataset(
                            river_data.get('dataset_crs'),
                            river_data.get('dataset_transform'),
                            river_data.get('dataset_shape')
                        )
                else:
                    # Backward compatibility - if it's just a river object
                    river = river_data
                
                rivers.append(river)
                
            except Exception as e:
                print(f"❌ Error loading {river_file}: {e}")
        
        print(f"✅ Successfully loaded {len(rivers)} rivers")
        if min_file_size_kb > 0:
            print(f"⏭️ Skipped {skipped_files} files smaller than {min_file_size_kb} KB")
        return rivers

    @classmethod  
    def batch_process_from_file_list(cls, dirname, fnames, start_x, start_y, end_x, end_y,
                                   clear_rasters=True, **processing_kwargs):
        """
        Batch process rivers from a list of existing files.
        
        Parameters
        ----------
        dirname : str
            Directory containing the files
        fnames : list
            List of filenames to process
        start_x, start_y : float
            Starting point coordinates (UTM)
        end_x, end_y : float
            Ending point coordinates (UTM)
        clear_rasters : bool, optional
            Whether to clear raster data after processing (default True)
        **processing_kwargs : dict
            Additional arguments for map_river_banks
            
        Returns
        -------
        successful_rivers : list
            List of successfully processed River instances
        failed_files : list
            List of filenames that failed processing
        """
        import gc
        
        successful_rivers = []
        failed_files = []
        
        print(f"🗂️  Batch processing {len(fnames)} files from {dirname}")
        
        for i, fname in enumerate(fnames):
            print(f"\n--- Processing {i+1}/{len(fnames)}: {fname} ---")
            
            try:
                # Create River instance
                river = cls(
                    fname=fname,
                    dirname=dirname,
                    start_x=start_x,
                    start_y=start_y,
                    end_x=end_x,
                    end_y=end_y,
                    file_type='water_index',
                    **processing_kwargs
                )
                
                # Process the river
                memory_before = river.get_memory_usage()
                success = river.map_river_banks()
                
                if success:
                    print(f"✅ Success! Memory: {memory_before} → {river.get_memory_usage()}")
                    
                    # Clear raster data to save memory
                    if clear_rasters:
                        river.clear_raster_data()
                        print(f"🧹 Cleared rasters, memory: {river.get_memory_usage()}")
                    
                    successful_rivers.append(river)
                    print(f"📊 Collected river (total: {len(successful_rivers)})")
                else:
                    print(f"❌ Processing failed")
                    failed_files.append(fname)
                    
            except Exception as e:
                print(f"❌ Error processing {fname}: {str(e)}")
                failed_files.append(fname)
            
            # Force garbage collection
            gc.collect()
        
        print(f"\n🎉 Batch processing complete!")
        print(f"✅ Successful: {len(successful_rivers)} rivers")
        print(f"❌ Failed: {len(failed_files)} files")
        
        return successful_rivers, failed_files 
    
    def save_planetscope_result(self, save_dir='planetscope_results', scene_id=None, 
                              acquisition_date=None, cloud_cover=None, 
                              source_files=None, processing_metadata=None):
        """
        Save river mapping result for PlanetScope data as a pickle file.
        
        Parameters
        ----------
        save_dir : str, optional
            Directory to save the result file (default: 'planetscope_results')
        scene_id : str, optional
            Scene identifier. If None, will be extracted from filename or generated
        acquisition_date : str, optional
            Acquisition date in 'YYYY-MM-DD' format. If None, will be extracted from filename
        cloud_cover : float, optional
            Cloud cover percentage
        source_files : list, optional
            List of source files used to create the mosaic
        processing_metadata : dict, optional
            Additional processing metadata
            
        Returns
        -------
        str
            Path to the saved pickle file
            
        Examples
        --------
        >>> river.save_planetscope_result(
        ...     scene_id='rio_beni_20250612',
        ...     acquisition_date='2025-06-12',
        ...     cloud_cover=5.2,
        ...     source_files=['20250612_151136_17_24f4_3B_AnalyticMS_SR_8b_clip.tif']
        ... )
        """
        import os
        import pickle
        from datetime import datetime
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract metadata from filename if not provided
        if scene_id is None:
            if hasattr(self, 'fname') and self.fname:
                # Extract date from filename like 'ddwi_mosaic_small.tif' or PlanetScope pattern
                fname_base = os.path.splitext(self.fname)[0]
                if 'mosaic' in fname_base.lower():
                    scene_id = f"planetscope_mosaic_{datetime.now().strftime('%Y%m%d')}"
                else:
                    scene_id = fname_base
            else:
                scene_id = f"planetscope_river_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if acquisition_date is None and source_files:
            # Try to extract date from first source file
            for fname in source_files:
                if fname.startswith('20'):  # PlanetScope format: 20250612_...
                    try:
                        date_str = fname[:8]  # YYYYMMDD
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        acquisition_date = f"{year:04d}-{month:02d}-{day:02d}"
                        break
                    except (ValueError, IndexError):
                        continue
        
        if acquisition_date is None:
            acquisition_date = datetime.now().strftime('%Y-%m-%d')
        
        # Create filename for saving
        safe_scene_id = scene_id.replace('/', '_').replace('\\', '_')
        river_file = os.path.join(save_dir, f"river_{safe_scene_id}.pkl")
        
        # Store dataset and MNDWI temporarily and remove them for pickling to save space
        temp_dataset = self._dataset
        temp_mndwi = self._mndwi
        self._dataset = None
        self._mndwi = None
        
        # Store original file size info for comparison
        original_memory = self.get_memory_usage()
        
        try:
            # Create results dictionary for saving
            river_data = {
                'river': self,
                'scene_id': scene_id,
                'acquisition_date': acquisition_date,
                'cloud_cover': cloud_cover or 0.0,
                'satellite_mission': 'PlanetScope',
                'processing_timestamp': datetime.now(),
                'source_files': source_files or [],
                'processing_metadata': processing_metadata or {},
                # Store dataset info separately
                'dataset_crs': str(temp_dataset.crs) if temp_dataset else None,
                'dataset_transform': temp_dataset.transform if temp_dataset else None,
                'dataset_shape': temp_dataset.shape if temp_dataset else None,
                # MNDWI info without storing the actual array
                'mndwi_shape': temp_mndwi.shape if temp_mndwi is not None else None,
                'mndwi_dtype': str(temp_mndwi.dtype) if temp_mndwi is not None else None,
                'water_pixel_count': int(np.sum(temp_mndwi)) if temp_mndwi is not None else None,
                # PlanetScope specific metadata
                'data_type': 'planetscope',
                'mosaic_info': {
                    'n_tiles': len(source_files) if source_files else 1,
                    'water_index_type': getattr(self, 'water_index_type', 'ddwi'),
                    'file_type': getattr(self, 'file_type', 'water_index')
                }
            }
            
            # Add river metadata attributes for consistency with Landsat processing
            self.scene_id = scene_id
            self.acquisition_date = acquisition_date
            self.cloud_cover = cloud_cover or 0.0
            self.satellite_mission = 'PlanetScope'
            
            # Save the pickle file
            with open(river_file, 'wb') as f:
                pickle.dump(river_data, f)
            
            # Get file size for reporting
            import os
            file_size_mb = os.path.getsize(river_file) / (1024**2)
            
            print(f"💾 Saved PlanetScope river result: {river_file}")
            print(f"   Scene ID: {scene_id}")
            print(f"   Acquisition date: {acquisition_date}")
            if source_files:
                print(f"   Source tiles: {len(source_files)}")
            print(f"   File size: {file_size_mb:.1f} MB (MNDWI excluded)")
            if temp_mndwi is not None:
                mndwi_size_mb = temp_mndwi.nbytes / (1024**2)
                print(f"   MNDWI would add: {mndwi_size_mb:.1f} MB")
            
            return river_file
            
        except Exception as e:
            print(f"⚠️ Error saving PlanetScope result: {e}")
            raise
        finally:
            # Restore the dataset and MNDWI references
            self._dataset = temp_dataset
            self._mndwi = temp_mndwi

    @staticmethod
    def extract_planetscope_metadata_from_files(file_list):
        """
        Extract metadata from PlanetScope file names.
        
        Parameters
        ----------
        file_list : list
            List of PlanetScope file paths
            
        Returns
        -------
        dict
            Dictionary containing extracted metadata
        """
        metadata = {
            'acquisition_dates': [],
            'item_ids': [],
            'unique_dates': set()
        }
        
        for filepath in file_list:
            filename = os.path.basename(filepath)
            
            # PlanetScope pattern: YYYYMMDD_HHMMSS_XX_XXXX_3B_AnalyticMS_SR_8b_clip.tif
            if filename.startswith('20') and len(filename) > 8:
                try:
                    # Extract date
                    date_str = filename[:8]  # YYYYMMDD
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    acquisition_date = f"{year:04d}-{month:02d}-{day:02d}"
                    
                    # Extract time if available
                    if len(filename) > 15 and filename[8] == '_':
                        time_str = filename[9:15]  # HHMMSS
                        if time_str.isdigit() and len(time_str) == 6:
                            hour = int(time_str[:2])
                            minute = int(time_str[2:4])
                            second = int(time_str[4:6])
                            full_datetime = f"{acquisition_date}T{hour:02d}:{minute:02d}:{second:02d}"
                        else:
                            full_datetime = acquisition_date
                    else:
                        full_datetime = acquisition_date
                    
                    metadata['acquisition_dates'].append(full_datetime)
                    metadata['unique_dates'].add(acquisition_date)
                    
                    # Extract item ID (everything before the first underscore after date_time)
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        item_id = '_'.join(parts[:4])  # YYYYMMDD_HHMMSS_XX_XXXX
                        metadata['item_ids'].append(item_id)
                    
                except (ValueError, IndexError):
                    continue
        
        # Set primary acquisition date (most common or first)
        if metadata['unique_dates']:
            metadata['primary_acquisition_date'] = sorted(metadata['unique_dates'])[0]
        else:
            metadata['primary_acquisition_date'] = datetime.now().strftime('%Y-%m-%d')
        
        return metadata 