import geojson
import numpy as np
from sentinelhub import BBox, CRS
from shapely.geometry import shape, box
from sentinelhub import BBox
from shapely.ops import transform
from pyproj import Transformer

def bbox_intersects_geometry(bbox: BBox, geometry_dict: dict=None, geometry_3857=None) -> bool:
    """
    Check if a bbox intersects with a geometry, handling CRS transformations.
    
    Parameters:
    -----------
    bbox : BBox
        BBox object in Web Mercator (EPSG:3857)
    geometry_dict : dict
        Geometry dictionary from GeoJSON (assumed to be in EPSG:4326)
    geometry_3857 : shapely.geometry, optional
        Pre-transformed geometry in EPSG:3857 for performance
    
    Returns:
    --------
    bool
        True if bbox intersects geometry, False otherwise
    """
    # Create bbox geometry in Web Mercator
    bbox_geom = box(*bbox)
    
    if geometry_3857 is None:
        # Create geometry from dict (assumed to be in EPSG:4326)
        geom_4326 = shape(geometry_dict)
        
        # Transform geometry from EPSG:4326 to EPSG:3857 to match bbox CRS
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        geometry_3857 = transform(transformer.transform, geom_4326)
    
    return bbox_geom.intersects(geometry_3857)

def transform_geometry_to_3857(geometry_dict: dict):
    """
    Transform a geometry from EPSG:4326 to EPSG:3857.
    Use this to pre-transform geometry for better performance.
    """
    geom_4326 = shape(geometry_dict)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return transform(transformer.transform, geom_4326)

def retrieve_geometry(geojson_path) -> dict:
    with open(geojson_path) as f:
        geo_file = geojson.load(f)
        
    return geo_file['features'][0]['geometry']

def get_bbox(i: int, j: int , tiles: np.ndarray) -> BBox:
    
    tile_coords = np.array([[tiles[i, j], tiles[i, j+1]], 
                       [tiles[i+1, j], tiles[i+1, j+1]]])
    
    flat_coords = tile_coords.reshape(-1, 2)
    xs = flat_coords[:, 0]
    ys = flat_coords[:, 1]
    
    return BBox(bbox=[xs.min(), ys.min(), xs.max(), ys.max()], crs=CRS.POP_WEB)
    
def get_pixels(bbox: BBox, resolution: int = 20) -> tuple[int, int]:

    width_m = bbox.max_x - bbox.min_x
    height_m = bbox.max_y - bbox.min_y

    width_px = int(width_m / resolution)
    height_px = int(height_m / resolution)
    
    return width_px, height_px