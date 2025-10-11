import math

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

from core.paths import get_data_path
from data_sourcing.data_models import CRSType


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


def set_geopandas_crs_to_epsg_3857(
    gdf: gpd.GeoDataFrame, crs_of_file: CRSType = "EPSG:3857"
) -> gpd.GeoDataFrame:
    """
    This method transforms a GeoDataFrame safely to a CRS referenced Dataframe

    Args:
        gdf (gpd.GeoDataFrame): the GeoDataFrame to be transformed
        crs_of_file (_type_, optional): the original crs of the GeoDataFrame. Defaults to "EPSG:3857".

    Returns:
        gpd.GeoDataFrame: transformed GeoDataFrame
    """
    if gdf.crs is None:
        gdf = gdf.set_crs(crs_of_file)

    if gdf.crs.to_string() != "EPSG:3857":
        gdf = gdf.to_crs("EPSG:3857")

    return gdf


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

    gdf_3857 = set_geopandas_crs_to_epsg_3857(gdf, crs_of_file)

    minx, miny, maxx, maxy = gdf_3857.total_bounds

    pixel_size = 20

    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)

    transform = from_origin(minx, maxy, pixel_size, pixel_size)

    mask = rasterize(
        [(geom, 1) for geom in gdf_3857.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    return mask, transform


def get_slicing_for_subarray(
    path_to_labels: str = "forest_labels.geojson",
    path_to_aoi: str = "blackForestPoly.geojson",
    pixel_size: int = 20,
    crs_of_labels: CRSType = "EPSG:3857",
) -> tuple[int, int, int, int]:
    """
    Returns the slicing that represents the pixels of the labels area in the complete AOI

    Args:
        path_to_labels (str, optional): Path of the labels file. Defaults to "forest_labels.geojson".
        path_to_aoi (str, optional): Path of the file for the AOI. Defaults to "blackForestPoly.geojson".
        pixel_size (int, optional): Pixel size of data. Defaults to 20.
        crs_of_labels (_type_, optional): Coordinate Reference System of labels. Defaults to "EPSG:3857".

    Returns:
        tuple[int, int, int, int]: min_col, max_col, min_row, max_row of slicing
    """
    data_path_to_aoi = get_data_path(path_to_aoi)
    geometry = retrieve_geometry(data_path_to_aoi)
    geometry_3857 = transform_geometry_to_3857(geometry)

    data_path_to_labels = get_data_path(path_to_labels)
    gdf = gpd.read_file(data_path_to_labels)

    gdf_3857 = set_geopandas_crs_to_epsg_3857(gdf, crs_of_labels)

    minx_aoi, miny_aoi, maxx_aoi, maxy_aoi = geometry_3857.bounds
    minx_labels, miny_labels, maxx_labels, maxy_labels = gdf_3857.total_bounds

    pixel_size = 20

    width = int((maxx_labels - minx_labels) / pixel_size)
    height = int((maxy_labels - miny_labels) / pixel_size)

    min_col = math.ceil((minx_labels - minx_aoi) / 20)
    max_col = math.ceil((minx_labels - minx_aoi) / 20 + width)
    min_row = math.ceil((maxy_aoi - maxy_labels) / 20)
    max_row = math.ceil((maxy_aoi - maxy_labels) / 20 + height)

    return min_col, max_col, min_row, max_row
