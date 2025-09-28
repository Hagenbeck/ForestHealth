import geojson
import geopandas as gpd
import numpy as np
from affine import Affine
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from sentinelhub import CRS, BBox
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from data_models import CRSType


def bbox_intersects_geometry(
    bbox: BBox, geometry_dict: dict = None, geometry_3857: BaseGeometry = None
) -> bool:
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
    bbox_geom = box(*bbox)

    if geometry_3857 is None:
        geom_4326 = shape(geometry_dict)

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        geometry_3857 = transform(transformer.transform, geom_4326)

    return bbox_geom.intersects(geometry_3857)


def transform_geometry_to_3857(geometry_dict: dict) -> BaseGeometry:
    """
    Transform a geometry from EPSG:4326 to EPSG:3857.
    Use this to pre-transform geometry for better performance.
    """
    geom_4326 = shape(geometry_dict)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return transform(transformer.transform, geom_4326)


def retrieve_geometry(geojson_path: str) -> dict:
    """
    Retrieve the geometry from a geojson file

    Args:
        geojson_path (str): Path to the .geojson

    Returns:
        dict: geometry of geojson
    """
    with open(geojson_path) as f:
        geo_file = geojson.load(f)

    return geo_file["features"][0]["geometry"]


def get_bbox(y: int, x: int, tiles: np.ndarray) -> BBox:
    """
    Calculates the bounding box for a tile of a split request

    Args:
        y (int): y index of tiles
        x (int): x index of tiles
        tiles (np.ndarray): array with coords of boundaries

    Returns:
        BBox: _description_
    """
    tile_coords = np.array(
        [[tiles[y, x], tiles[y, x + 1]], [tiles[y + 1, x], tiles[y + 1, x + 1]]]
    )

    flat_coords = tile_coords.reshape(-1, 2)
    xs = flat_coords[:, 0]
    ys = flat_coords[:, 1]

    return BBox(bbox=[xs.min(), ys.min(), xs.max(), ys.max()], crs=CRS.POP_WEB)


def get_pixels(bbox: BBox, resolution: int = 20) -> tuple[int, int]:
    """
    Calculate the width and height of a bbox in pixels

    Args:
        bbox (BBox): bounding Box
        resolution (int, optional): resolution in meters. Defaults to 20.

    Returns:
        tuple[int, int]: width, height of bbox in pixels
    """

    width_m = bbox.max_x - bbox.min_x
    height_m = bbox.max_y - bbox.min_y

    width_px = int(width_m / resolution)
    height_px = int(height_m / resolution)

    return width_px, height_px


def load_mask_from_geojson(
    path: str, crs_of_file: CRSType = "EPSG:3857"
) -> tuple[np.array, Affine]:
    """
    This function generates a mask based on the polygon labels of a geojson

    Args:
        path (str): Path to geojson
        crs_of_file (CRSType, optional): Coordinate Reference System of the geojson. Defaults to "EPSG:3857".

    Returns:
        np.array: mask of the polygon labels
        Affine: transform that defines the bounds of the mask
    """
    gdf = gpd.read_file(path)

    if gdf.crs is None:
        gdf = gdf.set_crs(crs_of_file)

    if gdf.crs.to_string() != "EPSG:3857":
        gdf = gdf.to_crs("EPSG:3857")

    minx, miny, maxx, maxy = gdf.total_bounds

    pixel_size = 20

    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)

    transform = from_origin(minx, maxy, pixel_size, pixel_size)

    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],  # 1 = Wald
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    return mask, transform
