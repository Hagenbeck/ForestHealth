"""Integration tests for FeatureService"""

import numpy as np

from data_processing.feature_service import FeatureService


class TestFeatureServiceInitialization:
    """Test FeatureService initialization with different configurations"""

    def test_init_with_default_features(self, raw_data_default):
        """Test initialization with default features from default_features.json"""
        service = FeatureService(raw_data_default)

        assert service.raw_data.shape == (10, 24, 7)
        assert service.feature_setting is not None
        assert len(service.feature_setting.features) > 0

    def test_init_with_custom_features(self, raw_data_default, multi_feature_setting):
        """Test initialization with custom feature settings"""
        service = FeatureService(
            raw_data_default, feature_settings=multi_feature_setting
        )

        assert service.feature_setting is multi_feature_setting
        assert len(service.feature_setting.features) == 2

    def test_init_stores_raw_data(self, raw_data_tiny):
        """Test that raw_data is stored correctly"""
        service = FeatureService(raw_data_tiny)

        assert np.array_equal(service.raw_data, raw_data_tiny)


class TestFeatureCalculation:
    """Test feature calculation logic"""

    def test_calculate_features_returns_dataframe(self, feature_service_default):
        """Test that calculate_features_for_monthly_data returns a DataFrame"""
        result = feature_service_default.calculate_features_for_monthly_data()

        assert result is not None
        assert len(result) == 10

    def test_calculate_features_with_single_feature(
        self, raw_data_small, single_mean_feature_setting
    ):
        """Test feature calculation with a single feature"""
        service = FeatureService(
            raw_data_small, feature_settings=single_mean_feature_setting
        )
        result = service.calculate_features_for_monthly_data()

        assert "mean" in result.columns
        assert len(result) == 5

    def test_calculate_features_with_multiple_features(
        self, raw_data_small, multi_feature_setting
    ):
        """Test feature calculation with multiple features"""
        service = FeatureService(raw_data_small, feature_settings=multi_feature_setting)
        result = service.calculate_features_for_monthly_data()

        assert "mean" in result.columns
        assert "std" in result.columns
        assert len(result.columns) == 2


class TestFeatureNameDeduplication:
    """Test feature name deduplication logic"""

    def test_duplicate_feature_names_get_numbered(
        self, raw_data_small, duplicate_feature_setting
    ):
        """Test that duplicate feature types get numbered suffixes"""
        service = FeatureService(
            raw_data_small, feature_settings=duplicate_feature_setting
        )
        result = service.calculate_features_for_monthly_data()

        assert "mean" in result.columns
        assert "mean2" in result.columns
        assert "mean3" in result.columns

    def test_feature_name_deduplication_preserves_order(
        self, raw_data_small, mixed_feature_setting
    ):
        """Test that feature names are deduplicated in order"""
        service = FeatureService(raw_data_small, feature_settings=mixed_feature_setting)
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
