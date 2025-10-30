import numpy as np
import rasterio
from sentinelhub import BBox

from core.paths import get_data_path


def load_raster_layer(raster_file: str) -> rasterio.io.DatasetReader:
    """Load a raster file in tiff format with rasterio

    Args:
        raster_file (str): file name of tiff File

    Returns:
        rasterio.io.DatasetReader: rasterio DatasetReader Object of the specified tiff
    """
    path_to_raster_file = get_data_path(raster_file)

    dataset = rasterio.open(path_to_raster_file, mode="r")

    return dataset


def get_subsection_of_raster_layer(
    aoi_bbox: BBox, aoi_crs: str, raster_layer: rasterio.io.DatasetReader
) -> np.ndarray:
    """Get the subsection of a raster_layer for a specified AOI

    Args:
        aoi_bbox (BBox): Bounding Box of AOI
        aoi_crs (str): CRS of aoi_bbox (e.g., "EPSG:3857")
        raster_layer (rasterio.io.DatasetReader): raster layer

    Returns:
        np.ndarray: extracted subsection fitting to bbox of AOI
    """
    from rasterio.warp import transform_bounds

    raster_crs = raster_layer.crs.to_string()
    if aoi_crs != raster_crs:
        transformed_bounds = transform_bounds(
            aoi_crs,
            raster_crs,
            aoi_bbox.min_x,
            aoi_bbox.min_y,
            aoi_bbox.max_x,
            aoi_bbox.max_y,
        )
    else:
        transformed_bounds = (
            aoi_bbox.min_x,
            aoi_bbox.min_y,
            aoi_bbox.max_x,
            aoi_bbox.max_y,
        )

    row_upper, col_left = raster_layer.index(
        transformed_bounds[0], transformed_bounds[3]
    )
    row_lower, col_right = raster_layer.index(
        transformed_bounds[2], transformed_bounds[1]
    )

    row_min, row_max = min(row_upper, row_lower), max(row_upper, row_lower)
    col_min, col_max = min(col_left, col_right), max(col_left, col_right)

    data = raster_layer.read(1)
    return data[row_min:row_max, col_min:col_max]
