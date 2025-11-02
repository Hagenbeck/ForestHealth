import numpy as np
import rasterio
from rasterio.transform import Affine, from_bounds
from rasterio.warp import (
    Resampling,
    reproject,
    transform_bounds,
)
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

    def retrieve_worldcover_raster_for_aoi(self) -> tuple[np.ndarray, Affine, CRSType]:
        """_summary_

        Returns:
            tuple[np.ndarray, Affine, CRSType]: _description_
        """
        dst_crs = "EPSG:3857"
        dataset = self.worldcover

        minx_aoi, miny_aoi, maxx_aoi, maxy_aoi = self.aoi_bbox

        width_px = int((maxx_aoi - minx_aoi) / self.resolution)
        height_px = int((maxy_aoi - miny_aoi) / self.resolution)

        target_transform = from_bounds(
            minx_aoi, miny_aoi, maxx_aoi, maxy_aoi, width_px, height_px
        )

        full_array = np.empty((height_px, width_px), dtype=dataset.dtypes[0])

        reproject(
            source=dataset.read(1),
            destination=full_array,
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            dst_transform=target_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

        self.aoi_worldcover = full_array
        return full_array, target_transform, dst_crs
