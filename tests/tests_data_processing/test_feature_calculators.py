"""Tests for FeatureCalculator implementations"""

import numpy as np
import pytest

from data_processing.feature_calculators import FeatureCalculator
from pydantic_models.feature_setting_temporal import (
    DifferenceInMeanBetweenIntervalsFeature,
    MeanFeature,
    RawFeature,
)


class TestFeatureCalculatorRegistry:
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

    def test_apply_consideration_intervals_returns_new_dto(
        self, mean_calculator, sample_band_dto
    ):
        """Test that applying intervals returns a new BandDTO, not mutating original"""
        original_shape = sample_band_dto.pixel_list.shape

        sliced = mean_calculator._apply_consideration_intervals(sample_band_dto, 0, 12)

        # Original unchanged
        assert sample_band_dto.pixel_list.shape == original_shape
        assert sample_band_dto.pixel_list.shape[0] == 24

        # Sliced has correct shape
        assert sliced.pixel_list.shape[0] == 12
        assert sliced.pixel_list.shape[1:] == sample_band_dto.pixel_list.shape[1:]

    def test_apply_consideration_intervals_creates_views(
        self, mean_calculator, sample_band_dto
    ):
        """Test that slicing creates views, not copies (memory efficient)"""
        sliced = mean_calculator._apply_consideration_intervals(sample_band_dto, 0, 12)

        # Views share the same underlying data buffer
        assert (
            sliced.pixel_list.base is sample_band_dto.pixel_list
            or sliced.pixel_list.base is sample_band_dto.pixel_list.base
        )

    def test_apply_consideration_intervals_with_none_returns_original(
        self, mean_calculator, sample_band_dto
    ):
        """Test that None intervals return the original DTO unchanged"""
        result = mean_calculator._apply_consideration_intervals(
            sample_band_dto, None, None
        )

        # Should return the exact same object
        assert result is sample_band_dto

    def test_apply_consideration_intervals_slices_both_arrays(
        self, mean_calculator, sample_band_dto
    ):
        """Test that both pixel_list and spatial_data are sliced consistently"""
        sliced = mean_calculator._apply_consideration_intervals(sample_band_dto, 5, 15)

        assert sliced.pixel_list.shape[0] == 10  # 15 - 5
        assert sliced.spatial_data.shape[0] == 10  # Same for spatial
        assert np.array_equal(sliced.pixel_coords, sample_band_dto.pixel_coords)

    def test_apply_consideration_intervals_negative_indexing(
        self, mean_calculator, sample_band_dto
    ):
        """Test that negative indices work correctly (e.g., last 12 months)"""
        sliced = mean_calculator._apply_consideration_intervals(
            sample_band_dto, -12, None
        )

        assert sliced.pixel_list.shape[0] == 12
        # Check it's actually the last 12 months
        assert np.array_equal(sliced.pixel_list[0], sample_band_dto.pixel_list[-12])


class TestRawCalculator:
    """Tests for RawCalculator"""

    def test_raw_calculator_registered(self):
        """Test that RawCalculator is registered"""
        from data_processing.feature_calculators import RawCalculator

        assert "raw" in FeatureCalculator._registry
        assert isinstance(FeatureCalculator._registry["raw"], RawCalculator)

    def test_raw_returns_correct_band(
        self, raw_calculator, raw_feature, sample_band_dto
    ):
        """Test that raw calculator returns the correct band data"""
        result = raw_calculator.create_feature(raw_feature, sample_band_dto)

        # Shape should be (n_months, n_pixels)
        assert result.shape == (24, 5)
        # Should be the same as directly indexing
        expected = sample_band_dto.pixel_list[:, :, 3]
        assert np.array_equal(result, expected)

    def test_raw_with_consideration_interval(self, raw_calculator, sample_band_dto):
        """Test raw calculator with consideration intervals"""
        feature = RawFeature(
            band_id=2, consideration_interval_start=0, consideration_interval_end=12
        )
        result = raw_calculator.create_feature(feature, sample_band_dto)

        # Should return only first 12 months
        assert result.shape == (12, 5)
        expected = sample_band_dto.pixel_list[0:12, :, 2]
        assert np.array_equal(result, expected)


class TestMeanCalculator:
    """Tests for MeanCalculator"""

    def test_mean_calculator_registered(self):
        """Test that MeanCalculator is registered"""
        from data_processing.feature_calculators import MeanCalculator

        assert "mean" in FeatureCalculator._registry
        assert isinstance(FeatureCalculator._registry["mean"], MeanCalculator)

    def test_mean_returns_1d_array(
        self, mean_calculator, mean_feature, sample_band_dto
    ):
        """Test that mean calculator returns 1D array (averaged over time)"""
        result = mean_calculator.create_feature(mean_feature, sample_band_dto)

        # Should be (n_pixels,) - averaged over time (axis 0)
        assert result.shape == (5,)
        assert result.dtype == np.float64

    def test_mean_calculation_correctness(self, mean_calculator, band_dto_with_pattern):
        """Test that mean is calculated correctly with known data"""
        feature = MeanFeature(band_id=0)
        result = mean_calculator.create_feature(feature, band_dto_with_pattern)

        # Manual calculation
        expected = band_dto_with_pattern.pixel_list[:, :, 0].mean(axis=0)
        assert np.allclose(result, expected)

    def test_mean_with_consideration_interval(self, mean_calculator, sample_band_dto):
        """Test mean with consideration intervals (e.g., last year only)"""
        feature = MeanFeature(
            band_id=2, consideration_interval_start=-12, consideration_interval_end=None
        )
        result = mean_calculator.create_feature(feature, sample_band_dto)

        # Should average only last 12 months
        expected = sample_band_dto.pixel_list[-12:, :, 2].mean(axis=0)
        assert result.shape == (5,)
        assert np.allclose(result, expected)

    def test_mean_does_not_mutate_input(self, mean_calculator, sample_band_dto):
        """Test that computing mean doesn't change original data"""
        original_shape = sample_band_dto.pixel_list.shape
        original_data = sample_band_dto.pixel_list.copy()

        feature = MeanFeature(
            band_id=1, consideration_interval_start=0, consideration_interval_end=12
        )
        _ = mean_calculator.create_feature(feature, sample_band_dto)

        # Original should be completely unchanged
        assert sample_band_dto.pixel_list.shape == original_shape
        assert np.array_equal(sample_band_dto.pixel_list, original_data)


class TestStdCalculator:
    """Tests for StdCalculator"""

    def test_std_calculator_registered(self):
        """Test that StdCalculator is registered"""
        from data_processing.feature_calculators import StdCalculator

        assert "std" in FeatureCalculator._registry
        assert isinstance(FeatureCalculator._registry["std"], StdCalculator)

    def test_std_returns_1d_array(self, std_calculator, std_feature, sample_band_dto):
        """Test that std calculator returns 1D array"""
        result = std_calculator.create_feature(std_feature, sample_band_dto)

        assert result.shape == (5,)


class TestDeseasonalizedDiffCalculator:
    """Tests for DeseasonalizedDiffCalculator"""

    def test_deseasonalized_diff_registered(self):
        """Test that DeseasonalizedDiffCalculator is registered"""
        assert "deseasonalized_diff" in FeatureCalculator._registry

    def test_deseasonalized_diff_returns_1d_array(
        self,
        deseasonalized_diff_calculator,
        deseasonalized_diff_feature,
        sample_band_dto,
    ):
        """Test that deseasonalized diff returns 1D array"""
        result = deseasonalized_diff_calculator.create_feature(
            deseasonalized_diff_feature, sample_band_dto
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
        sample_band_dto,
    ):
        """Test that it returns 1D array"""
        result = deseasonalized_diff_specific_month_calculator.create_feature(
            deseasonalized_diff_specific_month_feature, sample_band_dto
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
        sample_band_dto,
    ):
        """Test that it returns 1D array"""
        result = difference_in_mean_between_intervals_calculator.create_feature(
            difference_in_mean_between_intervals_feature, sample_band_dto
        )

        assert result.shape == (5,)

    def test_difference_in_mean_no_consideration_intervals(
        self, difference_in_mean_between_intervals_calculator, sample_band_dto
    ):
        """Test that it works without consideration intervals in feature config"""
        feature = DifferenceInMeanBetweenIntervalsFeature(band_id=0)
        result = difference_in_mean_between_intervals_calculator.create_feature(
            feature, sample_band_dto
        )

        assert result.shape == (5,)


class TestSpatialCalculators:
    """Tests for spatial feature calculators"""

    def test_spatial_cv_registered(self):
        """Test that SpatialCVCalculator is registered"""
        assert "spatial_cv" in FeatureCalculator._registry

    def test_spatial_cv_returns_correct_shape(
        self, spatial_cv_calculator, spatial_cv_feature, sample_band_dto
    ):
        """Test that spatial cv returns correct shape"""
        result = spatial_cv_calculator.create_feature(
            spatial_cv_feature, sample_band_dto
        )

        # Should return flattened spatial features
        assert len(result.shape) == 1

    def test_spatial_std_registered(self):
        """Test that SpatialStdCalculator is registered"""
        assert "spatial_std" in FeatureCalculator._registry

    def test_spatial_std_difference_registered(self):
        """Test that calculator is registered"""
        assert "spatial_std_difference" in FeatureCalculator._registry

    def test_spatial_range_registered(self):
        """Test that SpatialRangeCalculator is registered"""
        assert "spatial_range" in FeatureCalculator._registry

    def test_spatial_edge_strength_registered(self):
        """Test that SpatialEdgeStrengthCalculator is registered"""
        assert "spatial_edge_strength" in FeatureCalculator._registry


class TestImmutabilityAndThreadSafety:
    """Tests to ensure the refactored design is safe for parallel processing"""

    def test_multiple_features_dont_interfere(self, mean_calculator, sample_band_dto):
        """Test that computing multiple features with different intervals doesn't interfere"""
        feature1 = MeanFeature(
            band_id=0, consideration_interval_start=0, consideration_interval_end=12
        )
        feature2 = MeanFeature(
            band_id=0, consideration_interval_start=-12, consideration_interval_end=None
        )

        result1 = mean_calculator.create_feature(feature1, sample_band_dto)
        result2 = mean_calculator.create_feature(feature2, sample_band_dto)

        # Results should be different (first 12 vs last 12 months)
        assert not np.allclose(result1, result2)

        # Original DTO unchanged
        assert sample_band_dto.pixel_list.shape[0] == 24

    def test_band_dto_is_frozen(self, sample_band_dto):
        """Test that BandDTO is actually frozen and can't be mutated"""
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            sample_band_dto.pixel_list = np.zeros((10, 5, 7))
