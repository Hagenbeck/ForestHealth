import numpy as np

from pydantic_models.feature_setting_temporal import (
    DifferenceInMeanBetweenIntervalsFeature,
    MeanFeature,
    RawFeature,
)
from src.data_processing.feature_calculators import (
    FeatureCalculator,
    MeanCalculator,
    RawCalculator,
    StdCalculator,
)


class TestRawCalculator:
    """Tests for RawCalculator"""

    def test_raw_calculator_registered(self):
        """Test that RawCalculator is registered"""
        assert "raw" in FeatureCalculator._registry
        assert isinstance(FeatureCalculator._registry["raw"], RawCalculator)

    def test_raw_returns_correct_band(self, raw_calculator, raw_feature, sample_data):
        """Test that raw calculator returns the correct band data"""
        result = raw_calculator.create_feature(raw_feature, sample_data)

        expected = sample_data[:, :, 3]
        assert result.shape == (5, 24)
        assert np.array_equal(result, expected)

    def test_raw_with_consideration_interval(self, raw_calculator, sample_data):
        """Test raw calculator with consideration intervals"""
        feature = RawFeature(
            band_id=2, consideration_interval_start=0, consideration_interval_end=12
        )
        result = raw_calculator.create_feature(feature, sample_data)

        expected = sample_data[:, 0:12, 2]
        assert result.shape == (5, 12)
        assert np.array_equal(result, expected)


class TestMeanCalculator:
    """Tests for MeanCalculator"""

    def test_mean_calculator_registered(self):
        """Test that MeanCalculator is registered"""
        assert "mean" in FeatureCalculator._registry
        assert isinstance(FeatureCalculator._registry["mean"], MeanCalculator)

    def test_mean_returns_1d_array(self, mean_calculator, mean_feature, sample_data):
        """Test that mean calculator returns 1D array"""
        result = mean_calculator.create_feature(mean_feature, sample_data)

        assert result.shape == (5,)
        assert result.dtype == np.float64

    def test_mean_calculation_correctness(
        self, mean_calculator, sample_data_with_pattern
    ):
        """Test that mean is calculated correctly"""
        feature = MeanFeature(band_id=0)
        result = mean_calculator.create_feature(feature, sample_data_with_pattern)

        expected = sample_data_with_pattern[:, :, 0].mean(axis=1)
        assert np.array_equal(result, expected)

    def test_mean_with_consideration_interval(self, mean_calculator, sample_data):
        """Test mean with consideration intervals"""
        feature = MeanFeature(
            band_id=2, consideration_interval_start=-12, consideration_interval_end=None
        )
        result = mean_calculator.create_feature(feature, sample_data)

        expected = sample_data[:, -12:, 2].mean(axis=1)
        assert result.shape == (5,)
        assert np.allclose(result, expected)


class TestStdCalculator:
    """Tests for StdCalculator"""

    def test_std_calculator_registered(self):
        """Test that StdCalculator is registered"""
        assert "std" in FeatureCalculator._registry
        assert isinstance(FeatureCalculator._registry["std"], StdCalculator)

    def test_std_returns_1d_array(self, std_calculator, std_feature, sample_data):
        """Test that std calculator returns 1D array"""
        result = std_calculator.create_feature(std_feature, sample_data)

        assert result.shape == (5,)


class TestDeseasonalizedDiffCalculator:
    """Tests for DeseasonalizedDiffCalculator"""

    def test_deseasonalized_diff_registered(self):
        """Test that DeseasonalizedDiffCalculator is registered"""
        assert "deseasonalized_diff" in FeatureCalculator._registry

    def test_deseasonalized_diff_returns_1d_array(
        self, deseasonalized_diff_calculator, deseasonalized_diff_feature, sample_data
    ):
        """Test that deseasonalized diff returns 1D array"""
        result = deseasonalized_diff_calculator.create_feature(
            deseasonalized_diff_feature, sample_data
        )

        assert result.shape == (5,)


class TestDeseasonalizedDiffSpecificMonthCalculator:
    """Tests for DeseasonalizedDiffSpecificMonthCalculator"""

    def test_deseasonalized_diff_specific_month_registered(self):
        """Test that calculator is registered"""
        assert "deseasonalized_diff_specific_month" in FeatureCalculator._registry

    def test_deseasonalized_diff_specific_month_returns_1d_array(
        self,
        deseasonalized_diff_specific_month_calculator,
        deseasonalized_diff_specific_month_feature,
        sample_data,
    ):
        """Test that it returns 1D array"""
        result = deseasonalized_diff_specific_month_calculator.create_feature(
            deseasonalized_diff_specific_month_feature, sample_data
        )

        assert result.shape == (5,)


class TestDifferenceInMeanBetweenIntervalsCalculator:
    """Tests for DifferenceInMeanBetweenIntervalsCalculator"""

    def test_difference_in_mean_registered(self):
        """Test that calculator is registered"""
        assert "difference_in_mean_between_intervals" in FeatureCalculator._registry

    def test_difference_in_mean_returns_1d_array(
        self,
        difference_in_mean_between_intervals_calculator,
        difference_in_mean_between_intervals_feature,
        sample_data,
    ):
        """Test that it returns 1D array"""
        result = difference_in_mean_between_intervals_calculator.create_feature(
            difference_in_mean_between_intervals_feature, sample_data
        )

        assert result.shape == (5,)

    def test_difference_in_mean_no_consideration_intervals(
        self, difference_in_mean_between_intervals_calculator, sample_data
    ):
        """Test that it works without consideration intervals"""
        feature = DifferenceInMeanBetweenIntervalsFeature(band_id=0)
        result = difference_in_mean_between_intervals_calculator.create_feature(
            feature, sample_data
        )

        assert result.shape == (5,)


class TestSpatialCVCalculator:
    """Tests for SpatialCVCalculator"""

    def test_spatial_cv_registered(self):
        """Test that SpatialCVCalculator is registered"""
        assert "spatial_cv" in FeatureCalculator._registry

    def test_spatial_cv_returns_correct_shape(
        self, spatial_cv_calculator, spatial_cv_feature, sample_data
    ):
        """Test that spatial cv returns correct shape"""
        result = spatial_cv_calculator.create_feature(spatial_cv_feature, sample_data)

        # Should return 3D array (indices, height, width) after spatial operations
        assert len(result.shape) >= 1


class TestSpatialStdCalculator:
    """Tests for SpatialStdCalculator"""

    def test_spatial_std_registered(self):
        """Test that SpatialStdCalculator is registered"""
        assert "spatial_std" in FeatureCalculator._registry

    def test_spatial_std_returns_array(
        self, spatial_std_calculator, spatial_std_feature, sample_data
    ):
        """Test that spatial std returns array"""
        result = spatial_std_calculator.create_feature(spatial_std_feature, sample_data)

        assert result is not None
        assert len(result.shape) >= 1


class TestSpatialStdDifferenceCalculator:
    """Tests for SpatialStdDifferenceCalculator"""

    def test_spatial_std_difference_registered(self):
        """Test that calculator is registered"""
        assert "spatial_std_difference" in FeatureCalculator._registry

    def test_spatial_std_difference_returns_array(
        self,
        spatial_std_difference_calculator,
        spatial_std_difference_feature,
        sample_data,
    ):
        """Test that it returns array"""
        result = spatial_std_difference_calculator.create_feature(
            spatial_std_difference_feature, sample_data
        )

        assert result is not None
        assert len(result.shape) >= 1


class TestSpatialRangeCalculator:
    """Tests for SpatialRangeCalculator"""

    def test_spatial_range_registered(self):
        """Test that SpatialRangeCalculator is registered"""
        assert "spatial_range" in FeatureCalculator._registry

    def test_spatial_range_returns_array(
        self, spatial_range_calculator, spatial_range_feature, sample_data
    ):
        """Test that spatial range returns array"""
        result = spatial_range_calculator.create_feature(
            spatial_range_feature, sample_data
        )

        assert result is not None
        assert len(result.shape) >= 1


class TestSpatialEdgeStrengthCalculator:
    """Tests for SpatialEdgeStrengthCalculator"""

    def test_spatial_edge_strength_registered(self):
        """Test that SpatialEdgeStrengthCalculator is registered"""
        assert "spatial_edge_strength" in FeatureCalculator._registry

    def test_spatial_edge_strength_returns_array(
        self,
        spatial_edge_strength_calculator,
        spatial_edge_strength_feature,
        sample_data,
    ):
        """Test that spatial edge strength returns array"""
        result = spatial_edge_strength_calculator.create_feature(
            spatial_edge_strength_feature, sample_data
        )

        assert result is not None
        assert len(result.shape) >= 1


class TestCalculatorRegistry:
    """Tests for calculator registry functionality"""

    def test_all_calculators_registered(self):
        """Test that all 11 calculators are registered"""
        expected_types = {
            "raw",
            "mean",
            "std",
            "deseasonalized_diff",
            "deseasonalized_diff_specific_month",
            "difference_in_mean_between_intervals",
            "spatial_cv",
            "spatial_std",
            "spatial_std_difference",
            "spatial_range",
            "spatial_edge_strength",
        }

        assert expected_types.issubset(FeatureCalculator._registry.keys())
        assert len(FeatureCalculator._registry) == 11

    def test_all_calculators_are_instances(self):
        """Test that all registered calculators are instances, not classes"""
        for feature_type, calculator in FeatureCalculator._registry.items():
            assert isinstance(calculator, FeatureCalculator), (
                f"{feature_type} is not an instance of FeatureCalculator"
            )

    def test_all_calculators_have_create_feature_method(self):
        """Test that all calculators have create_feature method"""
        for feature_type, calculator in FeatureCalculator._registry.items():
            assert hasattr(calculator, "create_feature"), (
                f"{feature_type} calculator missing create_feature method"
            )
            assert callable(calculator.create_feature), (
                f"{feature_type} create_feature is not callable"
            )


class TestConsiderationIntervals:
    """Tests for consideration interval handling"""

    def test_feature_with_consideration_intervals(self, mean_calculator, sample_data):
        """Test that features with consideration_interval_start work"""
        feature = MeanFeature(
            band_id=0,
            consideration_interval_start=5,
            consideration_interval_end=15,
        )
        result = mean_calculator.create_feature(feature, sample_data)

        assert result.shape == (5,)

    def test_feature_without_consideration_intervals(
        self, difference_in_mean_between_intervals_calculator, sample_data
    ):
        """Test that features without consideration intervals use full data"""
        feature = DifferenceInMeanBetweenIntervalsFeature(band_id=0)
        result = difference_in_mean_between_intervals_calculator.create_feature(
            feature, sample_data
        )

        assert result.shape == (5,)
