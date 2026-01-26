from typing import Literal

from pydantic import Field

from pydantic_models.feature_setting_base import FeatureBase


class RawFeature(FeatureBase):
    """Raw band data - no parameters needed"""

    type: Literal["raw"] = "raw"


class MeanFeature(FeatureBase):
    """Mean across all time periods"""

    type: Literal["mean"] = "mean"
    band_id: int = Field(..., ge=0, le=8)


class StdFeature(FeatureBase):
    """Standard deviation across all time periods"""

    type: Literal["std"] = "std"
    band_id: int = Field(..., ge=0, le=8)


class DeseasonalizedDiffFeature(FeatureBase):
    """Deseasonalized differences (lag differences)"""

    type: Literal["deseasonalized_diff"] = "deseasonalized_diff"
    band_id: int = Field(..., ge=0, le=8)
    lag: int = Field(default=12, ge=1)


class DeseasonalizedDiffSpecificMonthFeature(FeatureBase):
    """Year-over-year difference for a specific month"""

    type: Literal["deseasonalized_diff_specific_month"] = (
        "deseasonalized_diff_specific_month"
    )
    band_id: int = Field(..., ge=0, le=8)
    month: int = Field(..., ge=0, le=11)
    lag: int = Field(default=12, ge=1)
