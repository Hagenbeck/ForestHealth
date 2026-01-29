"""Integration tests for FeatureService"""

import pytest

from data_processing.feature_service import FeatureService


class TestFeatureServiceInitialization:
    """Test FeatureService initialization with different configurations"""

    def test_init_with_default_features(self, sample_band_dto):
        """Test initialization with default features from default_features.json"""
        service = FeatureService(sample_band_dto)

        assert service.input_data.pixel_list.shape == (24, 5, 7)
        assert service.feature_setting is not None
        assert len(service.feature_setting.features) > 0

    def test_init_with_custom_features(self, sample_band_dto, multi_feature_setting):
        """Test initialization with custom feature settings"""
        service = FeatureService(
            sample_band_dto, feature_settings=multi_feature_setting
        )

        assert service.feature_setting is multi_feature_setting
        assert len(service.feature_setting.features) == 2

    def test_init_stores_input_data(self, tiny_band_dto):
        """Test that input_data is stored correctly as BandDTO"""
        service = FeatureService(tiny_band_dto)

        assert service.input_data is tiny_band_dto
        assert service.input_data.pixel_list.shape == (2, 2, 2)


class TestFeatureCalculation:
    """Test feature calculation logic"""

    def test_calculate_features_returns_dataframe(self, feature_service_default):
        """Test that calculate_features_for_monthly_data returns a DataFrame"""
        result = feature_service_default.calculate_features_for_monthly_data()

        assert result is not None
        # Should have one row per pixel
        assert len(result) == 5

    def test_calculate_features_with_single_feature(
        self, small_band_dto, single_mean_feature_setting
    ):
        """Test feature calculation with a single feature"""
        service = FeatureService(
            small_band_dto, feature_settings=single_mean_feature_setting
        )
        result = service.calculate_features_for_monthly_data()

        assert "mean" in result.columns
        assert len(result) == 3  # 3 pixels in small_band_dto

    def test_calculate_features_with_multiple_features(
        self, small_band_dto, multi_feature_setting
    ):
        """Test feature calculation with multiple features"""
        service = FeatureService(small_band_dto, feature_settings=multi_feature_setting)
        result = service.calculate_features_for_monthly_data()

        assert "mean" in result.columns
        assert "std" in result.columns
        assert len(result.columns) == 2

    def test_calculate_features_correct_number_of_rows(self, sample_band_dto):
        """Test that result has one row per pixel"""
        service = FeatureService(sample_band_dto)
        result = service.calculate_features_for_monthly_data()

        # Number of rows should equal number of pixels
        n_pixels = sample_band_dto.pixel_list.shape[1]
        assert len(result) == n_pixels


class TestFeatureNameDeduplication:
    """Test feature name deduplication logic"""

    def test_duplicate_feature_names_get_numbered(
        self, small_band_dto, duplicate_feature_setting
    ):
        """Test that duplicate feature types get numbered suffixes"""
        service = FeatureService(
            small_band_dto, feature_settings=duplicate_feature_setting
        )
        result = service.calculate_features_for_monthly_data()

        assert "mean" in result.columns
        assert "mean2" in result.columns
        assert "mean3" in result.columns

    def test_feature_name_deduplication_preserves_order(
        self, small_band_dto, mixed_feature_setting
    ):
        """Test that feature names are deduplicated in order"""
        service = FeatureService(small_band_dto, feature_settings=mixed_feature_setting)
        result = service.calculate_features_for_monthly_data()

        column_order = list(result.columns)
        assert column_order[0] == "mean"
        assert column_order[1] == "std"
        assert column_order[2] == "mean2"


class TestCalculatorRegistry:
    """Test calculator registry functionality"""

    def test_calculators_registry_exists(self):
        """Test that CALCULATORS registry is populated"""
        assert FeatureService.CALCULATORS is not None
        assert len(FeatureService.CALCULATORS) > 0

    def test_all_feature_types_have_calculators(self):
        """Test that all expected feature types have registered calculators"""
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

        assert expected_types.issubset(FeatureService.CALCULATORS.keys())

    def test_calculator_has_create_feature_method(self):
        """Test that calculators have the required create_feature method"""
        for feature_type, calculator in FeatureService.CALCULATORS.items():
            assert hasattr(calculator, "create_feature"), (
                f"{feature_type} calculator missing create_feature method"
            )


class TestInputDataImmutability:
    """Test that FeatureService doesn't mutate the input BandDTO"""

    def test_calculating_features_doesnt_mutate_input(self, sample_band_dto):
        """Test that feature calculation doesn't change the input data"""
        original_pixel_shape = sample_band_dto.pixel_list.shape
        original_spatial_shape = sample_band_dto.spatial_data.shape

        service = FeatureService(sample_band_dto)
        _ = service.calculate_features_for_monthly_data()

        # Input data should be completely unchanged
        assert sample_band_dto.pixel_list.shape == original_pixel_shape
        assert sample_band_dto.spatial_data.shape == original_spatial_shape

    def test_multiple_feature_calculations_consistent(self, sample_band_dto):
        """Test that calling calculate multiple times gives same results"""
        service = FeatureService(sample_band_dto)

        result1 = service.calculate_features_for_monthly_data()
        result2 = service.calculate_features_for_monthly_data()

        # Results should be identical
        assert result1.equals(result2)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_service_with_minimal_data(self, tiny_band_dto):
        """Test service works with minimal data (2 months, 2 pixels, 2 bands)"""
        from pydantic_models.feature_setting import FeatureSetting

        # Create feature setting compatible with only 2 bands
        feature_setting = FeatureSetting(
            features=[
                {"type": "mean", "band_id": 0},
                {"type": "mean", "band_id": 1},
            ]
        )

        service = FeatureService(tiny_band_dto, feature_settings=feature_setting)
        result = service.calculate_features_for_monthly_data()

        assert result is not None
        assert len(result) == 2  # 2 pixels
        assert len(result.columns) == 2  # 2 features

    def test_feature_setting_with_no_features_fails(self, sample_band_dto):
        """Test that FeatureSetting requires at least one feature"""
        from pydantic_models.feature_setting import FeatureSetting

        with pytest.raises(Exception):  # ValidationError from Pydantic
            FeatureSetting(features=[])
