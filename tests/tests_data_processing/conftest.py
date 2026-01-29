"""Shared pytest fixtures for data_processing tests"""

import numpy as np
import pytest

from data_processing.band_dto import BandDTO
from data_processing.feature_calculators import (
    DeseasonalizedDiffCalculator,
    DeseasonalizedDiffSpecificMonthCalculator,
    DifferenceInMeanBetweenIntervalsCalculator,
    MeanCalculator,
    RawCalculator,
    SpatialCVCalculator,
    SpatialEdgeStrengthCalculator,
    SpatialRangeCalculator,
    SpatialStdCalculator,
    SpatialStdDifferenceCalculator,
    StdCalculator,
)
from data_processing.feature_service import FeatureService
from pydantic_models.feature_setting import FeatureSetting
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

# ============================================================================
# Data Fixtures
# ============================================================================


@pytest.fixture
def sample_pixel_data():
    """
    Standard sample data for testing.
    Shape: (n_months=24, n_pixels=5, n_bands=7)
    """
    return np.random.rand(24, 5, 7)


@pytest.fixture
def sample_spatial_data():
    """
    Standard spatial data for testing.
    Shape: (n_months=24, n_bands=7, height=10, width=10)
    """
    return np.random.rand(24, 7, 10, 10)


@pytest.fixture
def sample_pixel_coords():
    """Pixel coordinates for 5 forest pixels"""
    return np.array([[2, 3], [4, 5], [6, 7], [8, 9], [1, 2]])


@pytest.fixture
def sample_band_dto(sample_pixel_data, sample_spatial_data, sample_pixel_coords):
    """Complete BandDTO with all three components"""
    return BandDTO(
        pixel_list=sample_pixel_data,
        spatial_data=sample_spatial_data,
        pixel_coords=sample_pixel_coords,
    )


@pytest.fixture
def sample_data_with_pattern():
    """
    Sample data with predictable values for testing calculations.
    Each pixel has a different linear pattern over time.
    Shape: (n_months=24, n_pixels=5, n_bands=7)
    """
    n_months, n_pixels, n_bands = 24, 5, 7
    data = np.zeros((n_months, n_pixels, n_bands))

    for pixel in range(n_pixels):
        for band in range(n_bands):
            # Each pixel-band combination has a linear trend
            data[:, pixel, band] = np.arange(n_months) * (pixel + 1) * (band + 1)

    return data


@pytest.fixture
def band_dto_with_pattern(sample_data_with_pattern):
    """BandDTO with predictable pattern data"""
    n_months, n_pixels, n_bands = sample_data_with_pattern.shape
    spatial_data = np.random.rand(n_months, n_bands, 10, 10)
    pixel_coords = np.array([[i, i] for i in range(n_pixels)])

    return BandDTO(
        pixel_list=sample_data_with_pattern,
        spatial_data=spatial_data,
        pixel_coords=pixel_coords,
    )


@pytest.fixture
def small_band_dto():
    """
    Smaller BandDTO for quick tests.
    Shape: (n_months=12, n_pixels=3, n_bands=7)
    """
    return BandDTO(
        pixel_list=np.random.rand(12, 3, 7),
        spatial_data=np.random.rand(12, 7, 5, 5),
        pixel_coords=np.array([[0, 0], [1, 1], [2, 2]]),
    )


@pytest.fixture
def tiny_band_dto():
    """
    Tiny BandDTO for minimal testing.
    Shape: (n_months=2, n_pixels=2, n_bands=2)
    """
    return BandDTO(
        pixel_list=np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        spatial_data=np.random.rand(2, 2, 3, 3),
        pixel_coords=np.array([[0, 0], [1, 1]]),
    )


# ============================================================================
# Calculator Fixtures
# ============================================================================


@pytest.fixture
def raw_calculator():
    """RawCalculator instance"""
    return RawCalculator()


@pytest.fixture
def mean_calculator():
    """MeanCalculator instance"""
    return MeanCalculator()


@pytest.fixture
def std_calculator():
    """StdCalculator instance"""
    return StdCalculator()


@pytest.fixture
def deseasonalized_diff_calculator():
    """DeseasonalizedDiffCalculator instance"""
    return DeseasonalizedDiffCalculator()


@pytest.fixture
def deseasonalized_diff_specific_month_calculator():
    """DeseasonalizedDiffSpecificMonthCalculator instance"""
    return DeseasonalizedDiffSpecificMonthCalculator()


@pytest.fixture
def difference_in_mean_between_intervals_calculator():
    """DifferenceInMeanBetweenIntervalsCalculator instance"""
    return DifferenceInMeanBetweenIntervalsCalculator()


@pytest.fixture
def spatial_cv_calculator():
    """SpatialCVCalculator instance"""
    return SpatialCVCalculator()


@pytest.fixture
def spatial_std_calculator():
    """SpatialStdCalculator instance"""
    return SpatialStdCalculator()


@pytest.fixture
def spatial_std_difference_calculator():
    """SpatialStdDifferenceCalculator instance"""
    return SpatialStdDifferenceCalculator()


@pytest.fixture
def spatial_range_calculator():
    """SpatialRangeCalculator instance"""
    return SpatialRangeCalculator()


@pytest.fixture
def spatial_edge_strength_calculator():
    """SpatialEdgeStrengthCalculator instance"""
    return SpatialEdgeStrengthCalculator()


# ============================================================================
# Feature Fixtures
# ============================================================================


@pytest.fixture
def raw_feature():
    """RawFeature with band_id=3"""
    return RawFeature(band_id=3)


@pytest.fixture
def mean_feature():
    """MeanFeature with band_id=1"""
    return MeanFeature(band_id=1)


@pytest.fixture
def std_feature():
    """StdFeature with band_id=1"""
    return StdFeature(band_id=1)


@pytest.fixture
def deseasonalized_diff_feature():
    """DeseasonalizedDiffFeature with band_id=2, lag=12"""
    return DeseasonalizedDiffFeature(band_id=2, lag=12)


@pytest.fixture
def deseasonalized_diff_specific_month_feature():
    """DeseasonalizedDiffSpecificMonthFeature with band_id=2, month=8"""
    return DeseasonalizedDiffSpecificMonthFeature(band_id=2, month=8)


@pytest.fixture
def difference_in_mean_between_intervals_feature():
    """DifferenceInMeanBetweenIntervalsFeature with standard intervals"""
    return DifferenceInMeanBetweenIntervalsFeature(
        band_id=2,
        interval_one_start=0,
        interval_one_end=11,
        interval_two_start=-12,
        interval_two_end=-1,
    )


@pytest.fixture
def spatial_cv_feature():
    """SpatialCVFeature with band_id=2, window_size=5"""
    return SpatialCVFeature(band_id=2, window_size=5)


@pytest.fixture
def spatial_std_feature():
    """SpatialStdFeature with band_id=1, window_size=5"""
    return SpatialStdFeature(band_id=1, window_size=5)


@pytest.fixture
def spatial_std_difference_feature():
    """SpatialStdDifferenceFeature with standard configuration"""
    return SpatialStdDifferenceFeature(
        band_id=2,
        window_size=5,
        interval_one_start=0,
        interval_one_end=11,
        interval_two_start=-12,
        interval_two_end=-1,
    )


@pytest.fixture
def spatial_range_feature():
    """SpatialRangeFeature with band_id=3, window_size=5"""
    return SpatialRangeFeature(band_id=3, window_size=5)


@pytest.fixture
def spatial_edge_strength_feature():
    """SpatialEdgeStrengthFeature with band_id=2, sigma=1.0"""
    return SpatialEdgeStrengthFeature(band_id=2, sigma=1.0)


# ============================================================================
# Feature Setting Fixtures
# ============================================================================


@pytest.fixture
def single_mean_feature_setting():
    """Feature setting with single mean feature"""
    return FeatureSetting(features=[{"type": "mean", "band_id": 2}])


@pytest.fixture
def multi_feature_setting():
    """Feature setting with multiple different features"""
    return FeatureSetting(
        features=[
            {"type": "mean", "band_id": 5},
            {"type": "std", "band_id": 3},
        ]
    )


@pytest.fixture
def duplicate_feature_setting():
    """Feature setting with duplicate feature types"""
    return FeatureSetting(
        features=[
            {"type": "mean", "band_id": 5},
            {"type": "mean", "band_id": 3},
            {"type": "mean", "band_id": 2},
        ]
    )


@pytest.fixture
def mixed_feature_setting():
    """Feature setting with mixed feature types for deduplication testing"""
    return FeatureSetting(
        features=[
            {"type": "mean", "band_id": 5},
            {"type": "std", "band_id": 3},
            {"type": "mean", "band_id": 2},
        ]
    )


# ============================================================================
# FeatureService Fixtures
# ============================================================================


@pytest.fixture
def feature_service_default(sample_band_dto):
    """FeatureService with default features"""
    return FeatureService(sample_band_dto)


@pytest.fixture
def feature_service_custom(small_band_dto, single_mean_feature_setting):
    """FeatureService with custom features"""
    return FeatureService(small_band_dto, feature_settings=single_mean_feature_setting)
