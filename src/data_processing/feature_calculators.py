from abc import ABC, abstractmethod

import numpy as np

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
            self._apply_consideration_intervals(
                input_data,
                feature.consideration_interval_start,
                feature.consideration_interval_end,
            )

        return self._calculate(input_data, feature)

    @abstractmethod
    def _calculate(self, input_data: BandDTO, feature: Feature) -> np.ndarray:
        """Subclasses implement only the core calculation logic"""
        pass

    def _apply_consideration_intervals(self, input_data: BandDTO, start: int, end: int):
        """Shared utility for slicing time intervals"""
        if start is not None or end is not None:
            input_data.pixel_list = input_data.pixel_list[:, start:end, :] # TODO Fix


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
        return input_data.pixel_list[:, :, feature.band_id].mean(axis=1)


class StdCalculator(FeatureCalculator):
    """Standard deviation across time periods"""

    feature_type = "std"

    def _calculate(self, input_data: BandDTO, feature: StdFeature) -> np.ndarray:
        """Calculate standard deviation across time periods."""
        pass


class DeseasonalizedDiffCalculator(FeatureCalculator):
    """Deseasonalized differences (lag differences)"""

    feature_type = "deseasonalized_diff"

    def _calculate(
        self, input_data: BandDTO, feature: DeseasonalizedDiffFeature
    ) -> np.ndarray:
        """Calculate differences between time points at a fixed lag."""
        pass


class DeseasonalizedDiffSpecificMonthCalculator(FeatureCalculator):
    """Year-over-year difference for a specific month"""

    feature_type = "deseasonalized_diff_specific_month"

    def _calculate(
        self, input_data: BandDTO, feature: DeseasonalizedDiffSpecificMonthFeature
    ) -> np.ndarray:
        """Calculate year-over-year differences for a specific month."""
        pass


class DifferenceInMeanBetweenIntervalsCalculator(FeatureCalculator):
    """The difference between the means of two time intervals"""

    feature_type = "difference_in_mean_between_intervals"

    def _calculate(
        self, input_data: BandDTO, feature: DifferenceInMeanBetweenIntervalsFeature
    ) -> np.ndarray:
        """Calculate difference between two time interval means."""
        pass


class SpatialCVCalculator(FeatureCalculator):
    """Local coefficient of variation"""

    feature_type = "spatial_cv"

    def _calculate(
        self, input_data: BandDTO, feature: SpatialCVFeature
    ) -> np.ndarray:
        """Calculate local coefficient of variation within a window."""
        pass


class SpatialStdCalculator(FeatureCalculator):
    """Local standard deviation"""

    feature_type = "spatial_std"

    def _calculate(
        self, input_data: BandDTO, feature: SpatialStdFeature
    ) -> np.ndarray:
        """Calculate local standard deviation within a window."""
        pass


class SpatialStdDifferenceCalculator(FeatureCalculator):
    """Spatial STD over the difference of the means of two time intervals"""

    feature_type = "spatial_std_difference"

    def _calculate(
        self, input_data: BandDTO, feature: SpatialStdDifferenceFeature
    ) -> np.ndarray:
        """Calculate spatial STD of difference between two time interval means."""
        pass


class SpatialRangeCalculator(FeatureCalculator):
    """Local range (peak-to-peak)"""

    feature_type = "spatial_range"

    def _calculate(
        self, input_data: BandDTO, feature: SpatialRangeFeature
    ) -> np.ndarray:
        """Calculate local range (max - min) within a window."""
        pass


class SpatialEdgeStrengthCalculator(FeatureCalculator):
    """Edge strength using Sobel gradient magnitude"""

    feature_type = "spatial_edge_strength"

    def _calculate(
        self, input_data: BandDTO, feature: SpatialEdgeStrengthFeature
    ) -> np.ndarray:
        """Calculate edge strength using Sobel gradient magnitude."""
        pass
