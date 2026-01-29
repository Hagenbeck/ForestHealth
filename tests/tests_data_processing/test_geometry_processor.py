"""Tests for GeometryProcessor"""

import numpy as np
import pytest

from data_processing.band_dto import BandDTO
from data_processing.geometry_processor import GeometryProcessor


@pytest.fixture
def geometry_processor():
    """Create a GeometryProcessor instance for testing"""
    # This will load actual data files, so it's an integration test
    return GeometryProcessor()


class TestGeometryProcessorInitialization:
    """Test GeometryProcessor initialization and data loading"""

    def test_init_loads_monthly_observations(self, geometry_processor):
        """Test that monthly observations are loaded correctly"""
        assert geometry_processor.monthly_observations is not None
        assert isinstance(geometry_processor.monthly_observations, np.ndarray)
        # Should have 4 dimensions: (n_months, bands, height, width)
        assert geometry_processor.monthly_observations.ndim == 4

    def test_init_loads_geometry(self, geometry_processor):
        """Test that AOI geometry is loaded"""
        assert geometry_processor.aoi_geometry is not None
        assert isinstance(geometry_processor.aoi_geometry, dict)

    def test_init_loads_worldcover(self, geometry_processor):
        """Test that worldcover data is loaded"""
        assert geometry_processor.worldcover is not None

    def test_init_extracts_bbox(self, geometry_processor):
        """Test that AOI bounding box is calculated"""
        assert geometry_processor.aoi_bbox is not None
        from sentinelhub import BBox

        assert isinstance(geometry_processor.aoi_bbox, BBox)


class TestBBoxExtraction:
    """Test extract_bbox_from_geometry static method"""

    def test_extract_bbox_from_geometry(self):
        """Test bbox extraction from MultiPolygon geometry"""
        geometry = {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [8.188143814749436, 48.61627297089052],
                        [8.39026475367633, 48.61627297089052],
                        [8.39026475367633, 48.542578737075878],
                        [8.188143814749436, 48.542578737075878],
                        [8.188143814749436, 48.61627297089052],
                    ]
                ]
            ],
        }

        from sentinelhub import BBox

        expected = BBox(
            (911499.9999999999, 6197600, 934000, 6209999.999999998),
            crs="EPSG:3857",
        )

        result = GeometryProcessor.extract_bbox_from_geometry(
            geometry=geometry, geometry_crs="EPSG:4326", bbox_crs="EPSG:3857"
        )

        # Compare bbox coordinates with some tolerance for floating point
        assert np.allclose(result.min_x, expected.min_x, rtol=1e-5)
        assert np.allclose(result.min_y, expected.min_y, rtol=1e-5)
        assert np.allclose(result.max_x, expected.max_x, rtol=1e-5)
        assert np.allclose(result.max_y, expected.max_y, rtol=1e-5)


class TestWorldcoverRetrieval:
    """Test worldcover raster retrieval and processing"""

    def test_retrieve_worldcover_raster_for_aoi(self, geometry_processor):
        """Test that worldcover raster is retrieved and reprojected correctly"""
        raster, transform, crs = geometry_processor.retrieve_worldcover_raster_for_aoi()

        assert raster is not None
        assert isinstance(raster, np.ndarray)
        assert raster.ndim == 2  # 2D spatial array
        assert transform is not None
        assert crs == "EPSG:3857"

    def test_worldcover_raster_sets_cache(self, geometry_processor):
        """Test that worldcover raster is stored in aoi_worldcover after retrieval"""

        raster, transform, crs = geometry_processor.retrieve_worldcover_raster_for_aoi()

        assert geometry_processor.aoi_worldcover is not None
        assert np.array_equal(geometry_processor.aoi_worldcover, raster)


class TestForestMaskCreation:
    """Test forest mask creation from worldcover data"""

    def test_create_forest_mask_from_worldcover_raster(self, geometry_processor):
        """Test that forest mask is created correctly"""
        mask = geometry_processor._create_forest_mask_from_worldcover_raster()

        assert mask is not None
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.ndim == 2

    def test_forest_mask_has_true_values(self, geometry_processor):
        """Test that forest mask contains some forest pixels"""
        mask = geometry_processor._create_forest_mask_from_worldcover_raster()

        # Should have at least some forest pixels (value 10 in worldcover)
        assert np.any(mask)

    def test_forest_mask_retrieves_worldcover_if_needed(self):
        """Test that forest mask creation triggers worldcover retrieval if not cached"""
        gp = GeometryProcessor()
        # Don't call retrieve_worldcover_raster_for_aoi first

        mask = gp._create_forest_mask_from_worldcover_raster()

        # Should have retrieved worldcover automatically
        assert gp.aoi_worldcover is not None
        assert mask is not None


class TestFlattenAndFilterMonthlyData:
    """Test the main data transformation method"""

    def test_flatten_and_filter_returns_band_dto(self, geometry_processor):
        """Test that method returns a BandDTO object"""
        result = geometry_processor.flatten_and_filter_monthly_data()

        assert isinstance(result, BandDTO)
        assert result.pixel_list is not None
        assert result.spatial_data is not None
        assert result.pixel_coords is not None

    def test_flatten_and_filter_correct_shapes(self, geometry_processor):
        """Test that returned BandDTO has correct array shapes"""
        result = geometry_processor.flatten_and_filter_monthly_data()

        n_months, bands, height, width = geometry_processor.monthly_observations.shape
        n_pixels = result.pixel_list.shape[1]

        # pixel_list: (n_months, n_pixels, bands)
        assert result.pixel_list.shape[0] == n_months
        assert result.pixel_list.shape[2] == bands

        # spatial_data: (n_months, bands, height, width)
        assert result.spatial_data.shape == (n_months, bands, height, width)

        # pixel_coords: (n_pixels, 2)
        assert result.pixel_coords.shape == (n_pixels, 2)

    def test_flatten_and_filter_only_forest_pixels(self, geometry_processor):
        """Test that only forest pixels are included in pixel_list"""
        result = geometry_processor.flatten_and_filter_monthly_data()

        forest_mask = geometry_processor._create_forest_mask_from_worldcover_raster()
        expected_n_pixels = np.sum(forest_mask)

        assert result.pixel_list.shape[1] == expected_n_pixels

    def test_pixel_coords_within_bounds(self, geometry_processor):
        """Test that pixel coordinates are valid indices"""
        result = geometry_processor.flatten_and_filter_monthly_data()

        _, _, height, width = geometry_processor.monthly_observations.shape

        # All row indices should be < height
        assert np.all(result.pixel_coords[:, 0] < height)
        # All col indices should be < width
        assert np.all(result.pixel_coords[:, 1] < width)
        # All indices should be >= 0
        assert np.all(result.pixel_coords >= 0)

    def test_pixel_coords_correspond_to_forest_pixels(self, geometry_processor):
        """Test that pixel_coords point to actual forest pixels in the mask"""
        result = geometry_processor.flatten_and_filter_monthly_data()
        forest_mask = geometry_processor._create_forest_mask_from_worldcover_raster()

        # Every coordinate should point to a forest pixel
        for row, col in result.pixel_coords:
            assert forest_mask[row, col]


class TestReconstruct2D:
    """Test reconstruction of 2D spatial arrays from pixel lists"""

    def test_reconstruct_2d_returns_correct_shape(self, geometry_processor):
        """Test that reconstruction returns correct spatial dimensions"""
        band_dto = geometry_processor.flatten_and_filter_monthly_data()

        # Create dummy values for each pixel
        n_pixels = band_dto.pixel_list.shape[1]
        values = np.arange(n_pixels, dtype=float)

        result = geometry_processor.reconstruct_2d(values)

        assert result.shape == geometry_processor.output_shape

    def test_reconstruct_2d_preserves_values(self, geometry_processor):
        """Test that reconstruction preserves values at correct locations"""
        band_dto = geometry_processor.flatten_and_filter_monthly_data()

        # Use sequential values
        n_pixels = band_dto.pixel_list.shape[1]
        values = np.arange(n_pixels, dtype=float)

        result = geometry_processor.reconstruct_2d(values)

        # Check that values appear at their coordinate locations
        for i, (row, col) in enumerate(band_dto.pixel_coords):
            assert result[row, col] == values[i]

    def test_reconstruct_2d_fills_non_forest_with_nan(self, geometry_processor):
        """Test that non-forest pixels are filled with NaN"""
        band_dto = geometry_processor.flatten_and_filter_monthly_data()

        n_pixels = band_dto.pixel_list.shape[1]
        values = np.ones(n_pixels)

        result = geometry_processor.reconstruct_2d(values)

        # Non-forest pixels should be NaN
        forest_mask = geometry_processor._create_forest_mask_from_worldcover_raster()
        assert np.all(np.isnan(result[~forest_mask]))

    def test_reconstruct_2d_roundtrip(self, geometry_processor):
        """Test that flatten -> reconstruct is consistent"""
        band_dto = geometry_processor.flatten_and_filter_monthly_data()

        # Take first time step, first band from pixel_list
        original_values = band_dto.pixel_list[0, :, 0]

        # Reconstruct to 2D
        reconstructed_2d = geometry_processor.reconstruct_2d(original_values)

        # Extract values back at the same coordinates
        extracted_values = reconstructed_2d[
            band_dto.pixel_coords[:, 0], band_dto.pixel_coords[:, 1]
        ]

        # Should match original values
        assert np.allclose(extracted_values, original_values)


class TestIntegration:
    """Integration tests for the full workflow"""

    def test_full_workflow_geometry_to_features(self, geometry_processor):
        """Test the complete workflow from geometry to feature-ready data"""
        from data_processing.feature_service import FeatureService
        from pydantic_models.feature_setting import FeatureSetting

        # Get BandDTO
        band_dto = geometry_processor.flatten_and_filter_monthly_data()

        # Create simple feature configuration
        feature_setting = FeatureSetting(features=[{"type": "mean", "band_id": 5}])

        # Calculate features
        service = FeatureService(band_dto, feature_settings=feature_setting)
        features = service.calculate_features_for_monthly_data()

        # Should have one feature per pixel
        assert len(features) == band_dto.pixel_list.shape[1]
        assert "mean" in features.columns

    def test_data_integrity_through_pipeline(self, geometry_processor):
        """Test that data maintains integrity through the pipeline"""
        band_dto = geometry_processor.flatten_and_filter_monthly_data()

        # Original spatial data should be unchanged
        assert np.array_equal(
            band_dto.spatial_data, geometry_processor.monthly_observations
        )

        # Pixel list should contain subset of spatial data
        # (can't easily verify without reconstructing, but check shapes are consistent)
        n_months, bands = band_dto.spatial_data.shape[0], band_dto.spatial_data.shape[1]
        assert band_dto.pixel_list.shape[0] == n_months
        assert band_dto.pixel_list.shape[2] == bands
