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
from core.logger import Logger, LogSegment
from core.paths import get_data_path
from data_processing.band_dto import BandDTO
from data_sourcing.data_models import CRSType
from data_sourcing.geometry_toolkit import GeometryToolkit


class GeometryProcessor:
    aoi_geometry: any
    aoi_crs: CRSType
    aoi_bbox: BBox
    aoi_worldcover: np.ndarray | None
    monthly_observations: np.ndarray
    worldcover: rasterio.io.DatasetReader
    resolution: int
    logger: Logger
    pixel_coords: np.ndarray | None

    def __init__(self, data_file: str = cf.OBSERVATION_SAVE_FILE):
        self.logger = Logger.get_instance()
        self.logger.info(
            LogSegment.DATA_PROCESSING,
            f"Initializing GeometryProcessor with data file: {data_file}",
        )
        self.monthly_observations = np.load(get_data_path(data_file))
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
        self.aoi_worldcover = None
        self.pixel_coords = None

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
        """Extracts the Bounding Box from a geometry in geojson format

        Args:
            geometry (dict): geojson where the bounding box should be extracted
            geometry_crs (CRSType): CRS of the geometry
            bbox_crs (CRSType): CRS that the bounding box should be in
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
        """extract the worldcover raster that fits exactly to the AOI

        Returns:
            tuple[np.ndarray, Affine, CRSType]: returns the Worldcover Raster with its Transform and the CRS-Type
        """
        dst_crs = "EPSG:3857"
        dataset = self.worldcover

        full_array, target_transform = self.transform_and_clip_raster_to_aoi(
            dataset=dataset, dst_crs=dst_crs, resampling=Resampling.nearest
        )

        self.aoi_worldcover = full_array
        return full_array, target_transform, dst_crs

    def transform_and_clip_raster_to_aoi(
        self,
        dataset: rasterio.io.DatasetReader,
        dst_crs: CRSType,
        resampling: Resampling,
        band_index: int = 1,
    ) -> tuple[np.ndarray, Affine]:
        """transforms a raster to match the pixels of the AOI and clips it

        Args:
            dataset (rasterio.io.DatasetReader): Raster Dataset to be transformed
            dst_crs (CRSType): CRS of the AOI
            resampling (Resampling): Resampling Method
            band_index (int, optional): index of the band of the raster that should be transformed. Defaults to 1.

        Returns:
            tuple[np.ndarray, Affine]: transformed raster with its transform
        """
        target_transform, height_px, width_px = self.get_target_transform()

        full_array = np.empty((height_px, width_px), dtype=dataset.dtypes[0])

        reproject(
            source=dataset.read(band_index),
            destination=full_array,
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            dst_transform=target_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )

        return full_array, target_transform

    def get_target_transform(self) -> tuple[Affine, int, int]:
        """Extract the target transform from

        Returns:
            tuple[Affine, int, int]: returns the transform, the height (px) and width (px)
        """

        minx_aoi, miny_aoi, maxx_aoi, maxy_aoi = self.aoi_bbox

        width_px = int((maxx_aoi - minx_aoi) / self.resolution)
        height_px = int((maxy_aoi - miny_aoi) / self.resolution)

        target_transform = from_bounds(
            minx_aoi, miny_aoi, maxx_aoi, maxy_aoi, width_px, height_px
        )

        return target_transform, height_px, width_px

    def _create_forest_mask_from_worldcover_raster(self) -> np.ndarray:
        """Returns the forest mask based on the worldcover raster

        Returns:
            np.ndarray: Forest Mask
        """

        if self.aoi_worldcover is None:
            self.retrieve_worldcover_raster_for_aoi()

        return self.aoi_worldcover == 10

    def flatten_and_filter_monthly_data(self) -> BandDTO:
        """
        Filter spatial data by boolean mask, flatten to pixel list, and include coordinates.

        Returns:
            pixel_data : np.ndarray
                Shape (n_months, n_forest_pixels, bands)
            pixel_coords : np.ndarray
                Shape (n_forest_pixels, 2) with columns [row, col]
                These are the (y, x) indices in the original spatial grid
        """

        forest_mask = self._create_forest_mask_from_worldcover_raster()

        n_months, bands, height, width = self.monthly_observations.shape

        rows, cols = np.where(forest_mask)
        self.pixel_coords = np.column_stack([rows, cols])
        self.output_shape = (height, width)

        data_flat = self.monthly_observations.reshape(n_months, bands, -1)
        mask_flat = forest_mask.flatten()
        pixel_data = data_flat[:, :, mask_flat].transpose(0, 2, 1)

        return BandDTO(
            pixel_list=pixel_data,
            pixel_coords=self.pixel_coords,
            spatial_data=self.monthly_observations,
        )

    def reconstruct_2d(self, values: np.ndarray) -> np.ndarray:
        """Reconstruct 2D array from flat values and coordinates.

        Args:
            values (np.ndarray): Shape (n_forest_pixels,) - predictions or features for each pixel

        Returns:
            np.ndarray: Shape (height, width) with values placed at coordinates, NaN elsewhere
        """
        result = np.full(self.output_shape, np.nan)
        result[self.pixel_coords[:, 0], self.pixel_coords[:, 1]] = values
        return result

    def export_reconstruction_as_geotiff(
        self, values: np.ndarray, output_path: str, nodata_value: float = -9999.0
    ) -> None:
        """Export reconstructed 2D array as a GeoTIFF with proper georeferencing.

        Args:
            values (np.ndarray): Shape (n_forest_pixels,) - predictions or features for each pixel
            output_path (str): Path where the GeoTIFF should be saved
            nodata_value (float): Value to use for NoData pixels (default: -9999.0)
        """
        reconstructed = self.reconstruct_2d(values)

        reconstructed = np.where(np.isnan(reconstructed), nodata_value, reconstructed)

        if self.aoi_worldcover is None:
            _, transform, crs = self.retrieve_worldcover_raster_for_aoi()
        else:
            minx_aoi, miny_aoi, maxx_aoi, maxy_aoi = self.aoi_bbox
            width_px = int((maxx_aoi - minx_aoi) / self.resolution)
            height_px = int((maxy_aoi - miny_aoi) / self.resolution)
            transform = from_bounds(
                minx_aoi, miny_aoi, maxx_aoi, maxy_aoi, width_px, height_px
            )
            crs = "EPSG:3857"

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=reconstructed.shape[0],
            width=reconstructed.shape[1],
            count=1,
            dtype=reconstructed.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata_value,
        ) as dst:
            dst.write(reconstructed, 1)

        self.logger.info(
            LogSegment.DATA_PROCESSING, f"GeoTIFF exported to: {output_path}"
        )
