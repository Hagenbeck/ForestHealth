from typing import Annotated, Union

from pydantic import Discriminator, Field

from pydantic_models.feature_setting_spatial import (
    SpatialCVFeature,
    SpatialEdgeStrengthFeature,
    SpatialRangeFeature,
    SpatialStdDifferenceFeature,
    SpatialStdFeature,
)
from pydantic_models.feature_setting_temporal import (
    DeseasonalizedDiffFeature,
    DeseasonalizedDiffSpecificMonthFeature,
    DifferenceInMeanBetweenIntervals,
    MeanFeature,
    RawFeature,
    StdFeature,
)
from pydantic_models.foresthealth_base_model import ForestHealthBaseModel

Feature = Annotated[
    Union[
        RawFeature,
        MeanFeature,
        StdFeature,
        DeseasonalizedDiffFeature,
        DeseasonalizedDiffSpecificMonthFeature,
        DifferenceInMeanBetweenIntervals,
        SpatialCVFeature,
        SpatialStdFeature,
        SpatialStdDifferenceFeature,
        SpatialRangeFeature,
        SpatialEdgeStrengthFeature,
    ],
    Discriminator("type"),
]


class FeatureSetting(ForestHealthBaseModel):
    """Container for feature configuration"""

    features: list[Feature] = Field(..., min_items=1)
