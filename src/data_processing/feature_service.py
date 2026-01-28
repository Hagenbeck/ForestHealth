import json
import pathlib as pl

import numpy as np
import pandas as pd

from pydantic_models.feature_setting import FeatureSetting

# from scipy.ndimage import gaussian_filter, generic_filter, sobel


class FeatureService:
    raw_data: np.ndarray
    feature_setting: FeatureSetting

    def __init__(self, raw_data: np.ndarray, feature_settings: FeatureSetting = None):
        """Initialize the FeatureService with multi-dimensional array data.

        Args:
            raw_data (np.ndarray): Input array with shape (index, time, bands)
                containing monthly satellite imagery data across multiple bands.
        """
        self.raw_data = raw_data
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
        feature_df = pd.DataFrame()

        feature_df["Mean_All_NDRE740"] = self._get_mean_all_feature(band_id=3)
        feature_df["Mean_Diff_Sept_NDRE705"] = self._get_mean_diff__feature(band_id=2, month=8)  # fmt: skip
        feature_df["Mean_Diff_Sept_NDVI"] = self._get_mean_diff__feature(band_id=5, month=8)  # fmt: skip
        feature_df["Mean_Diff_Sept_NDWI"] = self._get_mean_diff__feature(band_id=6, month=8)  # fmt: skip

        return feature_df

    def _get_mean_all_feature(self, band_id: int) -> np.ndarray:
        """Calculate the mean value across all time periods for a specific band.

        Args:
            band_id (int): Index of the band to extract (e.g., 3 for NDRE740).

        Returns:
            np.ndarray: 1D array of shape (index,) containing the temporal
                mean values for the specified band across all indices.
        """
        return self.raw_data[:, :, band_id].mean(axis=1)

    def _get_mean_diff__feature(self, band_id: int, month: int) -> np.ndarray:
        """Calculate the mean year-over-year difference for a specific band and month.

        Computes the difference between corresponding months in consecutive years,
        then averages across all year pairs for the specified month.

        Args:
            band_id (int): Index of the band to extract (e.g., 2 for NDRE705,
                5 for NDVI, 6 for NDWI).
            month (int): Zero-indexed month number (0=January, 8=September, etc.)
                to calculate differences for.

        Returns:
            np.ndarray: 1D array of shape (index,) containing the mean
                year-over-year difference values for the specified band and month.
        """
        return (self.raw_data[:, 12:, band_id] - self.raw_data[:, :-12, band_id])[:, month::12].mean(axis=1)  # fmt: skip
