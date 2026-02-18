import json
import pathlib as pl

import pandas as pd

from core.logger import Logger, LogSegment
from data_processing.band_dto import BandDTO
from data_processing.feature_calculators import FeatureCalculator
from pydantic_models.feature_setting import Feature, FeatureSetting


class FeatureService:
    CALCULATORS = FeatureCalculator._registry
    input_data: BandDTO
    feature_setting: FeatureSetting
    created_features: list[str]
    logger: Logger

    def __init__(self, input_data: BandDTO, feature_settings: FeatureSetting = None):
        """Initialize the FeatureService with multi-dimensional array data.

        Args:
            raw_data (np.ndarray): Input array with shape (index, time, bands)
                containing monthly satellite imagery data across multiple bands.
        """
        self.logger = Logger.get_instance()
        self.logger.info(LogSegment.DATA_PROCESSING, "Initializing FeatureService")
        self.input_data = input_data

        if feature_settings is not None:
            self.feature_setting = feature_settings
        else:
            self.feature_setting = FeatureSetting(
                **json.loads(
                    (
                        pl.Path(__file__).parent.parent / "default_features.json"
                    ).read_text()
                )
            )

    def calculate_features_for_monthly_data(self) -> pd.DataFrame:
        """Calculate vegetation and water indices features from monthly satellite data.

        Computes mean values across all months for NDRE740, and year-over-year
        differences for September measurements of NDRE705, NDVI, and NDWI indices.

        Returns:
            pd.DataFrame: DataFrame containing calculated features with columns
        """
        self.logger.info(
            LogSegment.DATA_PROCESSING,
            f"Calculating features with {len(self.feature_setting.features)} feature definitions",
        )
        feature_df = pd.DataFrame()
        self.created_features = []

        for feature in self.feature_setting.features:
            calculator: FeatureCalculator = self.CALCULATORS[feature.type]
            feature_df[self.__get_feature_name(feature)] = calculator.create_feature(
                feature, self.input_data
            )

        self.logger.info(
            LogSegment.DATA_PROCESSING,
            f"Feature calculation completed. Generated {len(self.created_features)} features",
        )
        self.logger._flush_logs()

        return feature_df

    def __get_feature_name(self, feature: Feature) -> str:
        """Method that calculates the feature name that should
        be used when adding this feature to the feature dataframe with deduplication logic

        Args:
            feature (Feature): feature that will be added to the feature dataframe

        Returns:
            str: name to be used in the feature dataframe
        """

        feature_name = feature.type
        i = 2

        while feature_name in self.created_features:
            feature_name = feature.type + str(i)
            i += 1

        self.created_features = self.created_features + [feature_name]

        return feature_name
