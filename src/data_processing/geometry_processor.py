import math

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_bounds,
)
from rasterio.windows import from_bounds
from sentinelhub import BBox

import config as cf
from core.paths import get_data_path
from data_sourcing.data_models import CRSType
from data_sourcing.geometry_toolkit import GeometryToolkit


class GeometryProcessor:
    aoi_geometry: any
    aoi_crs: CRSType
    aoi_bbox: BBox
    aoi_worldcover: np.ndarray
    monthly_observations: np.ndarray
    worldcover: rasterio.io.DatasetReader
    resolution: int

    def __init__(self, data_file: str = cf.OBSERVATION_SAVE_FILE):
        self.monthly_observation = np.load(get_data_path(data_file))
        self.aoi_geometry = GeometryToolkit.retrieve_geometry(
            get_data_path(cf.GEOMETRY_FILE)
        )
        self.aoi_crs = cf.GEOMETRY_FILE_CRS
        self.aoi_bbox = GeometryProcessor.extract_bbox_from_geometry(
            geometry=self.aoi_geometry,
            geometry_crs=self.aoi_crs,
            bbox_crs="EPSG:3857",
        )
        self.worldcover = GeometryProcessor.load_raster_layer(cf.WORLDCOVER_FILE)
        self.resolution = cf.RESOLUTION

    @staticmethod
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

    @staticmethod
    def extract_bbox_from_geometry(
        geometry: dict, geometry_crs: CRSType, bbox_crs: CRSType
    ):
        """_summary_

        Args:
            geometry (dict): _description_
            geometry_crs (CRSType): _description_
            bbox_crs (CRSType): _description_
        """

        coords = geometry["coordinates"][0][0]

        minx = min(x for x, y in coords)
        miny = min(y for x, y in coords)
        maxx = max(x for x, y in coords)
        maxy = max(y for x, y in coords)

        if geometry_crs != bbox_crs:
            try:
                minx, miny, maxx, maxy = transform_bounds(
                    geometry_crs, bbox_crs, minx, miny, maxx, maxy, densify_pts=21
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to transform bounds from {geometry_crs} to {bbox_crs}: {exc}"
                )

        return BBox((minx, miny, maxx, maxy), crs=bbox_crs)

    def retrieve_worldcover_raster_for_aoi(self):
        dst_crs = "EPSG:3857"
        dataset = self.worldcover

        # Original bounds in source CRS
        left, bottom, right, top = dataset.bounds

        # Transform bounds to EPSG:3857
        transformer = Transformer.from_crs(dataset.crs, dst_crs, always_xy=True)
        minx, miny = transformer.transform(left, bottom)
        maxx, maxy = transformer.transform(right, top)

        width = math.ceil((maxx - minx) / self.resolution)
        height = math.ceil((maxy - miny) / self.resolution)

        transform, width, height = calculate_default_transform(
            dataset.crs,
            dst_crs,
            dataset.width,
            dataset.height,
            left,
            bottom,
            right,
            top,
            dst_width=width,
            dst_height=height,
        )

        full_array = np.empty((height, width), dtype=dataset.dtypes[0])

        reproject(
            source=dataset.read(1),
            destination=full_array,
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

        minx, miny, maxx, maxy = self.aoi_bbox
        window = from_bounds(minx, miny, maxx, maxy, transform)
        window = window.round_offsets().round_lengths()

        cropped = full_array[
            int(window.row_off) : int(window.row_off + window.height),
            int(window.col_off) : int(window.col_off + window.width),
        ]

        cropped_transform = rasterio.windows.transform(window, transform)

        self.aoi_worldcover = cropped
        return cropped, cropped_transform, dst_crs
