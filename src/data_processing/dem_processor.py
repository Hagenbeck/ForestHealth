import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder
from rasterio.warp import Resampling
from scipy.ndimage import distance_transform_edt, generic_filter

import config as cf
from core.logger import Logger, LogSegment
from data_processing.geometry_processor import GeometryProcessor


class DEMProcessor:
    dem_grid: Grid
    dem_raster: Raster
    dem_raster_hydrology: Raster | None
    dem_raster_sink_filled: Raster | None
    dem_rio: rasterio.io.DatasetReader
    dem_transform: Affine
    cluster_labels_raster: rasterio.io.DatasetReader
    geometry_processor: GeometryProcessor
    logger: Logger

    def __init__(self, cluster_labels_path: str = None):
        self.logger = Logger.get_instance()
        self.logger.info(LogSegment.DEM_PROCESSOR, "Initializing DEM Processor")
        self.geometry_processor = GeometryProcessor()
        raw_dem_data = self.geometry_processor.load_raster_layer(cf.DEM_FILE)

        self.dem_rio, self.dem_transform = (
            self.geometry_processor.transform_and_clip_raster_to_aoi(
                dataset=raw_dem_data,
                dst_crs="EPSG:3857",
                resampling=Resampling.bilinear,
            )
        )

        viewfinder = ViewFinder(
            affine=self.dem_transform,
            shape=self.dem_rio.shape,
            crs="EPSG:3857",
            nodata=self.dem_rio.dtype.type(-9999),
        )

        self.dem_raster = Raster(self.dem_rio, viewfinder=viewfinder)
        self.dem_grid = Grid.from_raster(self.dem_raster)

        self.dem_raster_sink_filled, self.dem_raster_hydrology = (
            self.__resolve_sinks_and_flats()
        )

        self.cluster_labels_raster = self.geometry_processor.load_raster_layer(
            cf.CLUSTER_LABEL_OUTPUT_FILE
            if cluster_labels_path is None
            else cluster_labels_path
        )

    def extract_features_as_df(self) -> pd.DataFrame:
        """Extract the DEM Features and combine them with the cluster labels

        Returns:
            pd.DataFrame: Feature Dataframe containing all the topographic features and the cluster labels
        """
        features = self.generate_topographic_features()

        if self.geometry_processor.pixel_coords is None:
            self.geometry_processor.flatten_and_filter_monthly_data()

        coords = self.geometry_processor.pixel_coords
        cluster_labels_raster = self.cluster_labels_raster.read(1)

        rows, cols = coords[:, 0], coords[:, 1]

        df = pd.DataFrame(
            {
                "height": features["height"][rows, cols],
                "slope_deg": features["slope"][rows, cols],
                "aspect_deg": features["aspect"][rows, cols],
                "northness": features["northness"][rows, cols],
                "eastness": features["eastness"][rows, cols],
                "tpi": features["tpi"][rows, cols],
                "twi": features["twi"][rows, cols],
                "uca": features["uca"][rows, cols],
                "log_uca": np.log10(features["uca"][rows, cols]),
                "distance_to_stream": features["distance_to_stream"][rows, cols],
                "label": cluster_labels_raster[rows, cols],
            }
        )

        return df

    def generate_topographic_features(self) -> dict:
        """Generate all the topographic features

        Returns:
            dict: Dictionary containing all the features calculated
        """
        dem_filled_np = np.asarray(self.dem_raster_sink_filled)

        flow_direction = self.dem_grid.flowdir(self.dem_raster_hydrology)
        flow_accumulation = self.dem_grid.accumulation(flow_direction)

        stream_mask = self.__generate_stream_mask(flow_accumulation=flow_accumulation)

        slope_radian, slope_degree, aspect_radian, aspect_degree = (
            self.__compute_slope_and_aspect(dem_filled_np=dem_filled_np)
        )

        northness, eastness = self.__compute_northness_eastness(
            aspect_radian=aspect_radian
        )

        tpi = self.__compute_topographic_position_index(
            dem_filled_np=dem_filled_np, radius_m=250
        )

        uca = self.__compute_upslope_contribution_area(
            flow_accumulation=flow_accumulation
        )

        twi = self.__compute_topographic_wetness_index(
            flow_accumulation=flow_accumulation, slope_radian=slope_radian
        )

        distance_to_stream = self.__compute_stream_distance(stream_mask=stream_mask)

        return {
            "height": dem_filled_np,
            "slope": slope_degree,
            "aspect": aspect_degree,
            "northness": northness,
            "eastness": eastness,
            "tpi": tpi,
            "twi": twi,
            "uca": uca,
            "distance_to_stream": distance_to_stream,
        }

    def __resolve_sinks_and_flats(self) -> tuple[Raster, Raster]:
        """Resolve pits, depressions and flats in the DEM data

        Returns:
            tuple[Raster, Raster]: Raster with filled depressions and raster with filled depressions and resolves flats
        """
        dem_raster_pits = self.dem_grid.fill_pits(self.dem_raster)
        dem_raster_depressions = self.dem_grid.fill_depressions(dem_raster_pits)
        dem_raster_flats = self.dem_grid.resolve_flats(dem_raster_depressions)
        return dem_raster_depressions, dem_raster_flats

    @staticmethod
    def __compute_slope_and_aspect(
        dem_filled_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the slope and aspect per pixel

        Args:
            dem_filled_np (np.ndarray): array containing the DEM with filled pits and depressions

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: arrays with radian and degree of the slope and radian and degree of the aspect
        """
        res = cf.DEM_FILE_RESOLUTION

        dz_dx, dz_dy = np.gradient(dem_filled_np, res, res)

        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)

        aspect_rad = np.arctan2(-dz_dy, dz_dx)
        aspect_deg = (np.degrees(aspect_rad) + 360) % 360

        return slope_rad, slope_deg, aspect_rad, aspect_deg

    @staticmethod
    def __compute_northness_eastness(
        aspect_radian: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the northness and eastness of the aspect

        Args:
            aspect_radian (np.ndarray): array of the radian of the aspect per pixel

        Returns:
            tuple[np.ndarray, np.ndarray]: arrays with northness and eastness of aspect per pixel
        """
        return np.cos(aspect_radian), np.sin(aspect_radian)

    @staticmethod
    def __compute_upslope_contribution_area(
        flow_accumulation: np.ndarray,
    ) -> np.ndarray:
        """Compute the Upslope Contribution Area (UCA) in square meters

        Args:
            flow_accumulation (np.ndarray): array containing the flow accumulation per pixel

        Returns:
            np.ndarray: returns the flow accumulation in square meters per pixel
        """
        return flow_accumulation * cf.DEM_FILE_RESOLUTION**2

    @staticmethod
    def __compute_topographic_position_index(
        dem_filled_np: np.ndarray, radius_m: int
    ) -> np.ndarray:
        """Compute the Topographic Position Index (TPI) per Pixel

        Args:
            dem_filled_np (np.ndarray): array containing the DEM with filled pits and depressions
            radius_m (int): Radius for the index computation

        Returns:
            np.ndarray: array of the TPI per pixel
        """
        radius_px = max(1, int(radius_m / cf.DEM_FILE_RESOLUTION))
        footprint_size = 2 * radius_px + 1

        tpi = generic_filter(
            dem_filled_np,
            function=lambda w: (
                w[footprint_size**2 // 2]
                - np.mean(np.delete(w, footprint_size**2 // 2))
            ),
            size=footprint_size,
            mode="reflect",
        )
        return tpi

    @staticmethod
    def __compute_topographic_wetness_index(
        flow_accumulation: np.ndarray, slope_radian: np.ndarray
    ) -> np.ndarray:
        """Compute the Topographic Wetness Index (TWI) per Pixel

        Args:
            flow_accumulation (np.ndarray): array containing the flow accumulation per pixel
            slope_radian (np.ndarray): array of the radian of the slope per pixel

        Returns:
            np.ndarray: Array containing the TWI per pixel
        """
        specific_uca = flow_accumulation * cf.DEM_FILE_RESOLUTION
        TWI = np.log((specific_uca + 1e-6) / (np.tan(slope_radian) + 1e-6))
        TWI = np.clip(TWI, -10, 30)
        return TWI

    @staticmethod
    def __generate_stream_mask(flow_accumulation: np.ndarray) -> np.ndarray:
        """Generate an array containing a mask of all the streams

        Args:
            flow_accumulation (np.ndarray): array containing the flow accumulation per pixel

        Returns:
            np.ndarray: mask of all the streams
        """
        stream_mask = (
            flow_accumulation * cf.DEM_FILE_RESOLUTION**2
        ) >= cf.MIN_CATCHMENT_AREA_M2
        return stream_mask

    @staticmethod
    def __compute_stream_distance(stream_mask: np.ndarray) -> np.ndarray:
        """Compute the distance to the nearest stream

        Args:
            stream_mask (np.ndarray): array containing a mask of all the streams

        Returns:
            np.ndarray: array containing the distance to the nearest stream
        """
        distance = distance_transform_edt(~stream_mask) * cf.DEM_FILE_RESOLUTION
        return distance
