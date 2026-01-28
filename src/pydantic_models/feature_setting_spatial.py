from typing import Literal

from pydantic import Field

from pydantic_models.feature_setting_base import FeatureBase


class SpatialCVFeature(FeatureBase):
    """Local coefficient of variation"""

    type: Literal["spatial_cv"] = "spatial_cv"
    band_id: int = Field(..., ge=0)
    window_size: int = Field(default=5, ge=1)
    consideration_interval_start: int | None = Field(default=None)
    consideration_interval_end: int | None = Field(default=None)


class SpatialStdFeature(FeatureBase):
    """Local standard deviation"""

    type: Literal["spatial_std"] = "spatial_std"
    band_id: int = Field(..., ge=0)
    window_size: int = Field(default=5, ge=1)
    consideration_interval_start: int | None = Field(default=None)
    consideration_interval_end: int | None = Field(default=None)


class SpatialStdDifferenceFeature(FeatureBase):
    """Spatial STD over the difference of the means of two time intervals"""

    type: Literal["spatial_std_difference"] = "spatial_std_difference"
    band_id: int = Field(..., ge=0)
    window_size: int = Field(default=5, ge=1)
    interval_one_start: int = Field(default=0)
    interval_one_end: int = Field(default=11)
    interval_two_start: int = Field(default=-12)
    interval_two_end: int = Field(default=-1)


class SpatialRangeFeature(FeatureBase):
    """Local range (peak-to-peak)"""

    type: Literal["spatial_range"] = "spatial_range"
    band_id: int = Field(..., ge=0)
    window_size: int = Field(default=5, ge=1)
    consideration_interval_start: int | None = Field(default=None)
    consideration_interval_end: int | None = Field(default=None)


class SpatialEdgeStrengthFeature(FeatureBase):
    """Edge strength using Sobel gradient magnitude"""

    type: Literal["spatial_edge_strength"] = "spatial_edge_strength"
    band_id: int = Field(..., ge=0)
    sigma: float = Field(default=1.0, gt=0)
    consideration_interval_start: int | None = Field(default=None)
    consideration_interval_end: int | None = Field(default=None)
