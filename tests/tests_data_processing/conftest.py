"""Shared pytest fixtures for data_processing tests"""

import numpy as np
import pytest

from src.data_processing.feature_calculators import (
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
from src.data_processing.feature_service import FeatureService
from src.pydantic_models.feature_setting import FeatureSetting
from src.pydantic_models.feature_setting_spatial import (
    SpatialCVFeature,
    SpatialEdgeStrengthFeature,
    SpatialRangeFeature,
    SpatialStdDifferenceFeature,
    SpatialStdFeature,
)
from src.pydantic_models.feature_setting_temporal import (
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
def sample_data():
    """Standard sample data: 5 indices, 24 months, 7 bands"""
    return np.random.rand(5, 24, 7)


@pytest.fixture
def sample_data_with_pattern():
    """Sample data with predictable values for testing"""
    data = np.arange(5 * 24 * 7).reshape(5, 24, 7).astype(float)
    return data


@pytest.fixture
def raw_data_default():
    """Standard raw data: 10 indices, 24 months, 7 bands"""
    return np.random.rand(10, 24, 7)


@pytest.fixture
def raw_data_small():
    """Smaller raw data: 5 indices, 12 months, 7 bands"""
    return np.random.rand(5, 12, 7)


@pytest.fixture
def raw_data_tiny():
    """Tiny raw data for minimal testing"""
    return np.array([[[1, 2], [3, 4]]])


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
def feature_service_default(raw_data_default):
    """FeatureService with default features"""
    return FeatureService(raw_data_default)


@pytest.fixture
def feature_service_custom(raw_data_small, single_mean_feature_setting):
    """FeatureService with custom features"""
    return FeatureService(raw_data_small, feature_settings=single_mean_feature_setting)
