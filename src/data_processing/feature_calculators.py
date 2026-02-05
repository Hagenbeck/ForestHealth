from abc import ABC, abstractmethod

import numpy as np
from scipy.ndimage import gaussian_filter, generic_filter, sobel

from data_processing.band_dto import BandDTO
from pydantic_models.feature_setting import Feature
from pydantic_models.feature_setting_spatial import (
    SpatialCVFeature,
    SpatialEdgeStrengthFeature,
    SpatialRangeFeature,
    SpatialStdDifferenceFeature,
    SpatialStdFeature,
)
from pydantic_models.feature_setting_temporal import (
    DeseasonalizedDiffFeature,
    DeseasonalizedDiffSpecificMonthFeature,
    DifferenceInMeanBetweenIntervalsFeature,
    MeanFeature,
    RawFeature,
    StdFeature,
)


class FeatureCalculator(ABC):
    """Base class for all feature calculators with self-registration"""

    _registry = {}
    feature_type: str

    def __init_subclass__(cls, **kwargs):
        """Automatically register calculator subclasses"""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "feature_type"):
            FeatureCalculator._registry[cls.feature_type] = cls()

    def create_feature(self, feature: Feature, input_data: BandDTO) -> np.ndarray:
        """Template method that handles consideration intervals, then delegates to calculation"""

        if "consideration_interval_start" in type(feature).model_fields:
            sliced_data = self._apply_consideration_intervals(
                input_data,
                feature.consideration_interval_start,
                feature.consideration_interval_end,
            )
        else:
            sliced_data = input_data

        return self._calculate(sliced_data, feature)

    @abstractmethod
    def _calculate(self, input_data: BandDTO, feature: Feature) -> np.ndarray:
        """Subclasses implement only the core calculation logic"""
        pass

    def _apply_consideration_intervals(self, input_data: BandDTO, start: int, end: int):
        """Shared utility for slicing time intervals"""
        if start is None and end is None:
            return input_data

        return BandDTO(
            pixel_list=input_data.pixel_list[start:end, :, :],
            spatial_data=input_data.spatial_data[start:end, :, :, :],
            pixel_coords=input_data.pixel_coords,
        )  # TODO Test


class RawCalculator(FeatureCalculator):
    """Raw band data - no processing"""

    feature_type = "raw"

    def _calculate(self, input_data: BandDTO, feature: RawFeature) -> np.ndarray:
        """Return raw band data without processing."""
        return input_data.pixel_list[:, :, feature.band_id]


class MeanCalculator(FeatureCalculator):
    """Mean across time periods"""

    feature_type = "mean"

    def _calculate(self, input_data: BandDTO, feature: MeanFeature) -> np.ndarray:
        return input_data.pixel_list[:, :, feature.band_id].mean(axis=0)


class StdCalculator(FeatureCalculator):
    """Standard deviation across time periods"""

    feature_type = "std"

    def _calculate(self, input_data: BandDTO, feature: StdFeature) -> np.ndarray:
        """Calculate standard deviation across time periods."""
        return input_data.pixel_list[:, :, feature.band_id].std(axis=0)


class DeseasonalizedDiffCalculator(FeatureCalculator):
    """Deseasonalized differences (lag differences)"""

    feature_type = "deseasonalized_diff"

    def _calculate(
        self, input_data: BandDTO, feature: DeseasonalizedDiffFeature
    ) -> np.ndarray:
        """Calculate differences between time points at a fixed lag."""
        return np.diff(
            input_data.pixel_list[:, :, feature.band_id], axis=0, n=feature.lag
        ).mean(axis=0)


class DeseasonalizedDiffSpecificMonthCalculator(FeatureCalculator):
    """Year-over-year difference for a specific month"""

    feature_type = "deseasonalized_diff_specific_month"

    def _calculate(
        self, input_data: BandDTO, feature: DeseasonalizedDiffSpecificMonthFeature
    ) -> np.ndarray:
        """Calculate year-over-year differences for a specific month."""
        return np.diff(
            input_data.pixel_list[feature.month :: 12, :, feature.band_id],
            axis=0,
            n=feature.lag,
        ).mean(axis=0)


class DifferenceInMeanBetweenIntervalsCalculator(FeatureCalculator):
    """The difference between the means of two time intervals"""

    feature_type = "difference_in_mean_between_intervals"

    def _calculate(
        self, input_data: BandDTO, feature: DifferenceInMeanBetweenIntervalsFeature
    ) -> np.ndarray:
        """Calculate difference between two time interval means."""
        return input_data.pixel_list[
            feature.interval_two_start : feature.interval_two_end, :, feature.band_id
        ].mean(axis=(0)) - input_data.pixel_list[
            feature.interval_one_start : feature.interval_one_end, :, feature.band_id
        ].mean(axis=(0))


class SpatialCVCalculator(FeatureCalculator):
    """Local coefficient of variation"""

    feature_type = "spatial_cv"

    def cv_func(self, arr):
        mean = np.mean(arr)
        std = np.std(arr)
        return std / mean if mean != 0 else 0

    def _calculate(self, input_data: BandDTO, feature: SpatialCVFeature) -> np.ndarray:
        """Calculate local coefficient of variation within a window."""

        index_data = generic_filter(
            input_data.spatial_data.mean(axis=0)[feature.band_id],
            self.cv_func,
            size=feature.window_size,
            mode="constant",
            cval=0,
        )
        return index_data[input_data.pixel_coords[:, 0], input_data.pixel_coords[:, 1]]


class SpatialStdCalculator(FeatureCalculator):
    """Local standard deviation"""

    feature_type = "spatial_std"

    def _calculate(self, input_data: BandDTO, feature: SpatialStdFeature) -> np.ndarray:
        """Calculate local standard deviation within a window."""
        index_data = generic_filter(
            input_data.spatial_data.mean(axis=0)[feature.band_id],
            np.std,
            size=feature.window_size,
            mode="constant",
            cval=0,
        )
        return index_data[input_data.pixel_coords[:, 0], input_data.pixel_coords[:, 1]]


class SpatialStdDifferenceCalculator(FeatureCalculator):
    """Spatial STD over the difference of the means of two time intervals"""

    feature_type = "spatial_std_difference"

    def _calculate(
        self, input_data: BandDTO, feature: SpatialStdDifferenceFeature
    ) -> np.ndarray:
        """Calculate spatial STD of difference between two time interval means."""

        diff_data = input_data.spatial_data[
            feature.interval_two_start : feature.interval_two_end,
            feature.band_id,
            :,
            :,
        ].mean(axis=(0)) - input_data.spatial_data[
            feature.interval_one_start : feature.interval_one_end,
            feature.band_id,
            :,
            :,
        ].mean(axis=(0))

        index_data = generic_filter(
            diff_data,
            np.std,
            size=feature.window_size,
            mode="constant",
            cval=0,
        )
        return index_data[input_data.pixel_coords[:, 0], input_data.pixel_coords[:, 1]]


class SpatialRangeCalculator(FeatureCalculator):
    """Local range (peak-to-peak)"""

    feature_type = "spatial_range"

    def range_func(self, arr):
        return np.ptp(arr) if len(arr) > 0 else 0

    def _calculate(
        self, input_data: BandDTO, feature: SpatialRangeFeature
    ) -> np.ndarray:
        """Calculate local range (max - min) within a window."""
        index_data = generic_filter(
            input_data.spatial_data.mean(axis=0)[feature.band_id],
            self.range_func,
            size=feature.window_size,
            mode="constant",
            cval=0,
        )
        return index_data[input_data.pixel_coords[:, 0], input_data.pixel_coords[:, 1]]


class SpatialEdgeStrengthCalculator(FeatureCalculator):
    """Edge strength using Sobel gradient magnitude"""

    feature_type = "spatial_edge_strength"

    def get_edge(self, data):
        sigma = 1.0
        if sigma > 0:
            grid_2d = gaussian_filter(data, sigma=sigma)

        # Sobel operators for x and y gradients
        grad_x = sobel(grid_2d, axis=1)  # Horizontal edges
        grad_y = sobel(grid_2d, axis=0)  # Vertical edges

        # Gradient magnitude (edge strength)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return gradient_magnitude

    def _calculate(
        self, input_data: BandDTO, feature: SpatialEdgeStrengthFeature
    ) -> np.ndarray:
        """Calculate edge strength using Sobel gradient magnitude."""
        index_data = self.get_edge(
            input_data.spatial_data.mean(axis=0)[feature.band_id]
        )
        return index_data[input_data.pixel_coords[:, 0], input_data.pixel_coords[:, 1]]
