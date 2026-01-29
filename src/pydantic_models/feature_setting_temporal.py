from typing import Literal

from pydantic import Field

from pydantic_models.feature_setting_base import FeatureBase


class RawFeature(FeatureBase):
    """Raw band data - no parameters needed"""

    type: Literal["raw"] = "raw"
    band_id: int = Field(..., ge=0)
    consideration_interval_start: int | None = Field(default=None)
    consideration_interval_end: int | None = Field(default=None)


class MeanFeature(FeatureBase):
    """Mean across time periods"""

    type: Literal["mean"] = "mean"
    band_id: int = Field(..., ge=0)
    consideration_interval_start: int | None = Field(default=None)
    consideration_interval_end: int | None = Field(default=None)


class StdFeature(FeatureBase):
    """Standard deviation across time periods"""

    type: Literal["std"] = "std"
    band_id: int = Field(..., ge=0)
    consideration_interval_start: int | None = Field(default=None)
    consideration_interval_end: int | None = Field(default=None)


class DeseasonalizedDiffFeature(FeatureBase):
    """Deseasonalized differences (lag differences)"""

    type: Literal["deseasonalized_diff"] = "deseasonalized_diff"
    band_id: int = Field(..., ge=0)
    lag: int = Field(default=12, ge=1)
    consideration_interval_start: int | None = Field(default=None)
    consideration_interval_end: int | None = Field(default=None)


class DeseasonalizedDiffSpecificMonthFeature(FeatureBase):
    """Year-over-year difference for a specific month"""

    type: Literal["deseasonalized_diff_specific_month"] = (
        "deseasonalized_diff_specific_month"
    )
    band_id: int = Field(..., ge=0)
    month: int = Field(..., ge=0, le=11)
    lag: int = Field(default=12, ge=1)
    consideration_interval_start: int | None = Field(default=None)
    consideration_interval_end: int | None = Field(default=None)


class DifferenceInMeanBetweenIntervalsFeature(FeatureBase):
    """The difference between the means of two time intervals"""

    type: Literal["difference_in_mean_between_intervals"] = (
        "difference_in_mean_between_intervals"
    )
    band_id: int = Field(..., ge=0)
    interval_one_start: int = Field(default=0)
    interval_one_end: int = Field(default=11)
    interval_two_start: int = Field(default=-12)
    interval_two_end: int = Field(default=-1)
