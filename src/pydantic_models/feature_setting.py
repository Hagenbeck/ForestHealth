from enum import StrEnum

from pydantic import Field

from pydantic_models.foresthealth_base_model import ForestHealthBaseModel


class FeatureType(StrEnum):
    MEAN_ALL = "mean_all"
    MEAN_LAST_YEAR = "mean_last_year"
    STD_ALL_MONTHS = "std_all_months"
    DIFFERENCE_2020_VS_2025 = "difference_2020_vs_2025"
    YEAR_OVER_YEAR_DIFF = "year_over_year_diff"
    DESEASONALIZED_DIFF = "deseasonalized_diff"
    LOCAL_CV_SPATIAL = "local_cv_spatial"
    LOCAL_STD_SPATIAL = "local_std_spatial"
    LOCAL_RANGE_SPATIAL = "local_range_spatial"
    EDGE_STRENGTH = "edge_strength"
    SPATIAL_STD_DIFFERENCE = "spatial_std_difference"


class Setting(StrEnum):
    BAND_ID = "band_id"
    MONTH = "month"
    LAG = "lag"
    WINDOW_SIZE = "window_size"
    SIGMA = "sigma"


class FeatureParameter(ForestHealthBaseModel):
    setting: Setting
    value: int | float = Field(..., gt=0)


class Feature(ForestHealthBaseModel):
    type: FeatureType
    parameters: list[FeatureParameter] = Field(default_factory=list)


class FeatureSetting(ForestHealthBaseModel):
    features: list[Feature] = Field(..., min_items=1)
