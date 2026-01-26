from typing import Literal

from pydantic import Field

from pydantic_models.feature_setting_base import FeatureBase


class SpatialCVFeature(FeatureBase):
    """Local coefficient of variation"""

    type: Literal["spatial_cv"] = "spatial_cv"
    band_id: int = Field(..., ge=0, le=8)
    window_size: int = Field(default=5, ge=1)


class SpatialStdFeature(FeatureBase):
    """Local standard deviation"""

    type: Literal["spatial_std"] = "spatial_std"
    band_id: int = Field(..., ge=0, le=8)
    window_size: int = Field(default=5, ge=1)


class SpatialStdDifferenceFeature(FeatureBase):
    """Spatial STD of 2020 vs 2025 difference"""

    type: Literal["spatial_std_difference"] = "spatial_std_difference"
    band_id: int = Field(..., ge=0, le=8)
    window_size: int = Field(default=5, ge=1)


class SpatialRangeFeature(FeatureBase):
    """Local range (peak-to-peak)"""

    type: Literal["spatial_range"] = "spatial_range"
    band_id: int = Field(..., ge=0, le=8)
    window_size: int = Field(default=5, ge=1)


class SpatialEdgeStrengthFeature(FeatureBase):
    """Edge strength using Sobel gradient magnitude"""

    type: Literal["spatial_edge_strength"] = "spatial_edge_strength"
    band_id: int = Field(..., ge=0, le=8)
    sigma: float = Field(default=1.0, gt=0)
